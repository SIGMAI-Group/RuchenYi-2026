
from logging import getLogger

import numpy as np


def recall_(pos_index, pos_len):
    rec_ret = np.cumsum(pos_index, axis=1) / pos_len.reshape(-1, 1)
    return rec_ret.mean(axis=0)


def recall2_(pos_index, pos_len):
    r"""
    All hits are summed up and then averaged for recall.
    :param pos_index:
    :param pos_len:
    :return:
    len_rank = np.full_like(pos_len, pos_index.shape[1])
    idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

    iranks = np.zeros_like(pos_index, dtype=np.float)
    iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
    for row, idx in enumerate(idcg_len):
        idcg[row, idx:] = idcg[row, idx - 1]

    ranks = np.zeros_like(pos_index, dtype=np.float)
    ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    dcg = 1.0 / np.log2(ranks + 1)
    dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

    result = dcg / idcg
    return result.mean(axis=0)


def map_(pos_index, pos_len):
    r"""MAP_ (also known as Mean Average Precision) The MAP is meant to calculate Avg. Precision for the relevant items.
    Note:
        In this case the normalization factor used is :math:`\frac{1}{\min (m,N)}`, which prevents your AP score from
        being unfairly suppressed when your number of recommendations couldn't possibly capture all the correct ones.

    .. _map: http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html#MAP-for-Recommender-Algorithms

    .. math::
        \begin{align*}
        \mathrm{AP@N} &= \frac{1}{\mathrm{min}(m,N)}\sum_{k=1}^N P(k) \cdot rel(k) \\
        \mathrm{MAP@N}& = \frac{1}{|U|}\sum_{u=1}^{|U|}(\mathrm{AP@N})_u
        \end{align*}
    rec_ret = pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)
    return rec_ret.mean(axis=0)
def target_hr_(pos_index, pos_len=None):
    r"""
    Target Hit Ratio (TargetHR)
    物理含义：衡量有多少比例的正常用户，其推荐列表中出现了【至少一个】目标物品。
    :param pos_index: 形状为 (num_users, max_k) 的布尔矩阵，True 代表该位置是 Target Item
    :param pos_len: 在此处不需要用到，为了保持接口统一保留
    Target NDCG
    物理含义：衡量目标物品在受害者推荐列表中的曝光顺位质量。
    复用原生的 NDCG 逻辑，只需外部传入正确的 pos_len (即目标物品的总数) 即可。
metrics_dict = {
    'ndcg': ndcg_,
    'recall': recall_,
    'recall2': recall2_,
    'precision': precision_,
    'map': map_,

    'target_hr': target_hr_,
    'target_ndcg': target_ndcg_,
}
