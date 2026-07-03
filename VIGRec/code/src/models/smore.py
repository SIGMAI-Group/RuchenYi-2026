r"""
SMORE - Multi-modal Recommender System
Reference:
    ACM WSDM 2025: Spectrum-based Modality Representation Fusion Graph Convolutional Network for Multimodal Recommendation

Reference Code:
    https://github.com/kennethorq/SMORE
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def spectrum_convolution(self, image_embeds, text_embeds):
        image_fft = torch.fft.rfft(image_embeds, dim=1, norm='ortho')
        text_fft = torch.fft.rfft(text_embeds, dim=1, norm='ortho')

        image_complex_weight = torch.view_as_complex(self.image_complex_weight)
        text_complex_weight = torch.view_as_complex(self.text_complex_weight)
        fusion_complex_weight = torch.view_as_complex(self.fusion_complex_weight)

        image_conv = torch.fft.irfft(image_fft * image_complex_weight, n=image_embeds.shape[1], dim=1, norm='ortho')
        text_conv = torch.fft.irfft(text_fft * text_complex_weight, n=text_embeds.shape[1], dim=1, norm='ortho')

        fusion_conv = torch.fft.irfft(text_fft * image_fft * fusion_complex_weight, n=text_embeds.shape[1], dim=1, norm='ortho')

        return image_conv, text_conv, fusion_conv

    def forward(self, adj, train=False):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)

        image_conv, text_conv, fusion_conv = self.spectrum_convolution(image_feats, text_feats)
        image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_v(image_conv))
        text_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_t(text_conv))
        fusion_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_f(fusion_conv))

        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]

        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_embeds = all_embeddings

        if self.sparse:
            for i in range(self.n_layers):
                image_item_embeds = torch.sparse.mm(self.image_original_adj, image_item_embeds)
        else:
            for i in range(self.n_layers):
                image_item_embeds = torch.mm(self.image_original_adj, image_item_embeds)
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)

        if self.sparse:
            for i in range(self.n_layers):
                text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
        else:
            for i in range(self.n_layers):
                text_item_embeds = torch.mm(self.text_original_adj, text_item_embeds)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

        if self.sparse:
            for i in range(self.n_layers):
                fusion_item_embeds = torch.sparse.mm(self.fusion_adj, fusion_item_embeds)
        else:
            for i in range(self.n_layers):
                fusion_item_embeds = torch.mm(self.fusion_adj, fusion_item_embeds)
        fusion_user_embeds = torch.sparse.mm(self.R, fusion_item_embeds)
        fusion_embeds = torch.cat([fusion_user_embeds, fusion_item_embeds], dim=0)

        fusion_att_v, fusion_att_t = self.query_v(fusion_embeds), self.query_t(fusion_embeds)
        fusion_soft_v = self.softmax(fusion_att_v)
        agg_image_embeds = fusion_soft_v * image_embeds

        fusion_soft_t = self.softmax(fusion_att_t)
        agg_text_embeds = fusion_soft_t * text_embeds

        image_prefer = self.gate_image_prefer(content_embeds)
        text_prefer = self.gate_text_prefer(content_embeds)
        fusion_prefer = self.gate_fusion_prefer(content_embeds)
        image_prefer, text_prefer, fusion_prefer = self.dropout(image_prefer), self.dropout(text_prefer), self.dropout(fusion_prefer)

        agg_image_embeds = torch.multiply(image_prefer, agg_image_embeds)
        agg_text_embeds = torch.multiply(text_prefer, agg_text_embeds)
        fusion_embeds = torch.multiply(fusion_prefer, fusion_embeds)

        side_embeds = torch.mean(torch.stack([agg_image_embeds, agg_text_embeds, fusion_embeds]), dim=0)

        all_embeds = content_embeds + side_embeds

        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

        if train:
            return all_embeddings_users, all_embeddings_items, side_embeds, content_embeds

        return all_embeddings_users, all_embeddings_items

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

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

        ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.forward(
            self.norm_adj, train=True)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)

        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        cl_loss = self.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items], 0.2) + self.InfoNCE(
            side_embeds_users[users], content_embeds_user[users], 0.2)

        return batch_mf_loss + batch_emb_loss + batch_reg_loss + self.cl_loss * cl_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
