r"""
LightGCN

Reference:
    Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.

Reference code:
    https://github.com/kuandeng/LightGCN
    def __init__(self, config, dataset):
        super(LightGCN, self).__init__(config, dataset)

        self.interaction_matrix = dataset.inter_matrix(
            form='coo').astype(np.float32)

        self.latent_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.embedding_dict = self._init_model()

        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)


    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_users, self.latent_dim))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_items, self.latent_dim)))
        })

        return embedding_dict

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings = lightgcn_all_embeddings[:self.n_users, :]
        item_all_embeddings = lightgcn_all_embeddings[self.n_users:, :]

        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user, :]
        posi_embeddings = item_all_embeddings[pos_item, :]
        negi_embeddings = item_all_embeddings[neg_item, :]

        pos_scores = torch.mul(u_embeddings, posi_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, negi_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        u_ego_embeddings = self.embedding_dict['user_emb'][user, :]
        posi_ego_embeddings = self.embedding_dict['item_emb'][pos_item, :]
        negi_ego_embeddings = self.embedding_dict['item_emb'][neg_item, :]

        reg_loss = self.reg_loss(u_ego_embeddings, posi_ego_embeddings, negi_ego_embeddings)
        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        restore_user_e, restore_item_e = self.forward()
        u_embeddings = restore_user_e[user, :]

        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))

        return scores
