import copy
import math

import numpy as np
import torch
import torch.nn as nn
from common.abstract_recommender import GeneralRecommender
import scipy.sparse as sp


class LightGCN_Encoder(GeneralRecommender):
    def __init__(self, config, dataset):
        super(LightGCN_Encoder, self).__init__(config, dataset)
        self.interaction_matrix = dataset.inter_matrix(
            form='coo').astype(np.float32)

        self.user_count = self.n_users
        self.item_count = self.n_items
        self.latent_size = config['embedding_size']
        self.n_layers = 3 if config['n_layers'] is None else config['n_layers']
        self.layers = [self.latent_size] * self.n_layers

        self.drop_ratio = 1.0
        self.drop_flag = True

        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = self.get_norm_adj_mat().to(self.device)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.user_count, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.item_count, self.latent_size)))
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
