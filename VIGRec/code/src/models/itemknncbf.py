r"""
ItemKNNCBF
Reference:
    https://github.com/CRIPAC-DIG/LATTICE
    Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches, ACM RecSys'19
        分块计算物品相似矩阵并显示进度条。

        :param features: Tensor, 物品特征向量，形状为 (num_items, feature_dim)
        :param block_size: int, 分块大小，默认 1000
        :return: Tensor, 权重邻接矩阵
