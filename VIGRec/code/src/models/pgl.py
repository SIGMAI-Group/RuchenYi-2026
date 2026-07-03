r"""
PGL
Reference:
    https://github.com/demonph10/PGL
    AAAI'2025: [Mind Individual Information! Principal Graph Learning for Multimedia Recommendation]
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        if self.mode == 'global':
            self.sub_graph = self.global_subgraph_extraction(L)
            self.sub_graph = self.sparse_mx_to_torch_sparse_tensor(self.sub_graph).to(self.device)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def global_subgraph_extraction(self, adj):
        norm_adj = adj.tocsc()
        ut, s, vt = sparsesvd(norm_adj, self.embedding_dim)

        num_top_bottom = int(0.25 * self.embedding_dim)
        top_singular_values = s[:num_top_bottom]
        bottom_singular_values = s[-num_top_bottom:]

        product_singular_values = top_singular_values * bottom_singular_values

        product_matrix = np.diag(product_singular_values)
        product_sparse_matrix = ut.T[:, :num_top_bottom] @ product_matrix @ vt[:num_top_bottom, :]
        product_sparse_matrix = sp.csr_matrix(product_sparse_matrix * (abs(product_sparse_matrix) >= 1e-3))
        return product_sparse_matrix

    def alignment(self, x, y):
        user, item = self.interaction_matrix.nonzero()
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x[user] - y[item]).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self, x, t=2):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def save(self):
        pass

    def pre_epoch_processing(self):
        if self.mode == 'local':
            degree_len = int(self.edge_values.size(0) * 0.3)
            degree_idx = torch.multinomial(self.edge_values, degree_len)
            keep_indices = self.edge_indices[:, degree_idx]
            keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
            all_values = torch.cat((keep_values, keep_values))
            keep_indices[1] += self.n_users
            all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
            self.sub_graph = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).to(self.device)

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def forward(self, adj):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)

        image_feats, text_feats = F.normalize(image_feats), F.normalize(text_feats)
        user_embeds = torch.cat([self.user_image.weight, self.user_text.weight], dim=1)
        item_embeds = torch.cat([image_feats, text_feats], dim=1)

        h = item_embeds
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)

        ego_embeddings = torch.cat((user_embeds, item_embeds), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings + h

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings = self.forward(self.sub_graph)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        cl_loss = (self.InfoNCE(self.dropoutf(u_g_embeddings), self.dropoutf(u_g_embeddings), 0.2)
                   + self.InfoNCE(self.dropoutf(pos_i_g_embeddings), self.dropoutf(pos_i_g_embeddings), 0.2)) / 2
        return batch_mf_loss + self.reg_weight * cl_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
