# coding: utf-8
# @email: enoche.chow@gmail.com
#
# updated: Mar. 25, 2022
# Filled non-existing raw features with non-zero after encoded from encoders

"""
Data pre-processing
##########################
"""
from logging import getLogger
from collections import Counter
import os
import pandas as pd
import numpy as np
import torch
from utils.data_utils import (ImageResize, ImagePad, image_to_tensor, load_decompress_img_from_lmdb_value)
import lmdb


class RecDataset(object):
    def __init__(self, config, df=None):
        self.config = config
        self.logger = getLogger()

        # data path & files
        self.dataset_name = config['dataset']
        self.dataset_path = os.path.abspath(config['data_path']+self.dataset_name)

        # dataframe
        self.uid_field = self.config['USER_ID_FIELD']
        self.iid_field = self.config['ITEM_ID_FIELD']
        self.splitting_label = self.config['inter_splitting_label']

        if df is not None:
            self.df = df
            return
        # if all files exists
        check_file_list = [self.config['inter_file_name']]
        for i in check_file_list:
            file_path = os.path.join(self.dataset_path, i)
            if not os.path.isfile(file_path):
                raise ValueError('File {} not exist'.format(file_path))

        # load rating file from data path?
        self.load_inter_graph(config['inter_file_name'])
        self.item_num = int(max(self.df[self.iid_field].values)) + 1
        self.user_num = int(max(self.df[self.uid_field].values)) + 1

    def load_inter_graph(self, file_name):
        inter_file = os.path.join(self.dataset_path, file_name)
        cols = [self.uid_field, self.iid_field, self.splitting_label]
        self.df = pd.read_csv(inter_file, usecols=cols, sep=self.config['field_separator'])
        if not self.df.columns.isin(cols).all():
            raise ValueError('File {} lost some required columns.'.format(inter_file))

    def split(self):
        dfs = []
        # splitting into training/validation/test
        for i in range(3):
            temp_df = self.df[self.df[self.splitting_label] == i].copy()
            temp_df.drop(self.splitting_label, inplace=True, axis=1)        # no use again
            dfs.append(temp_df)
        if self.config['filter_out_cod_start_users']:
            # filtering out new users in val/test sets
            train_u = set(dfs[0][self.uid_field].values)
            for i in [1, 2]:
                dropped_inter = pd.Series(True, index=dfs[i].index)
                dropped_inter ^= dfs[i][self.uid_field].isin(train_u)
                dfs[i].drop(dfs[i].index[dropped_inter], inplace=True)

        # wrap as RecDataset
        full_ds = [self.copy(_) for _ in dfs]
        return full_ds

    def copy(self, new_df):
        """Given a new interaction feature, return a new :class:`Dataset` object,
                whose interaction feature is updated with ``new_df``, and all the other attributes the same.

                Args:
                    new_df (pandas.DataFrame): The new interaction feature need to be updated.

                Returns:
                    :class:`~Dataset`: the new :class:`~Dataset` object, whose interaction feature has been updated.
                """
        nxt = RecDataset(self.config, new_df)

        nxt.item_num = self.item_num
        nxt.user_num = self.user_num
        return nxt

    def get_user_num(self):
        return self.user_num

    def get_item_num(self):
        return self.item_num

    def shuffle(self):
        """Shuffle the interaction records inplace.
        """
        self.df = self.df.sample(frac=1, replace=False).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Series result
        return self.df.iloc[idx]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        info = [self.dataset_name]
        self.inter_num = len(self.df)
        uni_u = pd.unique(self.df[self.uid_field])
        uni_i = pd.unique(self.df[self.iid_field])
        tmp_user_num, tmp_item_num = 0, 0
        if self.uid_field:
            tmp_user_num = len(uni_u)
            avg_actions_of_users = self.inter_num/tmp_user_num
            info.extend(['The number of users: {}'.format(tmp_user_num),
                         'Average actions of users: {}'.format(avg_actions_of_users)])
        if self.iid_field:
            tmp_item_num = len(uni_i)
            avg_actions_of_items = self.inter_num/tmp_item_num
            info.extend(['The number of items: {}'.format(tmp_item_num),
                         'Average actions of items: {}'.format(avg_actions_of_items)])
        info.append('The number of inters: {}'.format(self.inter_num))
        if self.uid_field and self.iid_field:
            sparsity = 1 - self.inter_num / tmp_user_num / tmp_item_num
            info.append('The sparsity of the dataset: {}%'.format(sparsity * 100))
        return '\n'.join(info)
    def inject_fake_data(self, fake_data_path):
        """
        将攻击框架生成的 .npz 稀疏矩阵格式假数据，
        无缝转化为 MMRec 原生的 DataFrame 表格格式，并注入训练集。
        """
        import scipy.sparse as sp
        import pandas as pd
        
        self.logger.info(f"========== 正在注入多模态攻击假数据 ==========")
        self.logger.info(f"读取路径: {fake_data_path}")
        
        # 【格式A处理：解析 .npz 稀疏矩阵】
        # 这里的 fake_data_path 指向的是 Attack 框架输出的 best.npz 文件。
        # 它的物理形态是一个被高度压缩的 CSR 稀疏矩阵。
        # 使用 sp.load_npz 将其加载到内存中，此时它依然是纯粹的全局数学矩阵形态。
        fake_data = sp.load_npz(fake_data_path)
        
        # 获取矩阵的行数，即假用户的数量 (例如 131 个)
        n_fakes = fake_data.shape[0] 
        
        # 记录当前系统中的真实用户总数，作为假用户 ID 的起始编号。
        # 这是为了在后续合并时，让假用户顺延排在真实用户群体之后，确保绝对不覆盖真实的历史记录。
        start_fake_uid = self.user_num
        
        # 【格式转换核心：从“矩阵坐标”到“交互流水”】
        # 调用 .nonzero() 方法，直接提取稀疏矩阵中所有非零元素（即有点击行为）的坐标。
        # fake_u 得到的是行索引数组（代表假用户的局部 ID：0, 1, 2...）
        # fake_i 得到的是列索引数组（代表被点击物品的真实 ID）
        fake_u, fake_i = fake_data.nonzero()
        
        # 将假用户的局部 ID 加上真实的起始编号，将其映射为全局系统中合法的绝对 ID。
        fake_u = fake_u + start_fake_uid  
        
        # 【格式B处理：组装为 MMRec 识别的 DataFrame 表格】
        # 此时，高维稀疏矩阵已经被拆解成了平面的横纵坐标对。
        # 接下来，严格按照数据集原有的结构，用这批坐标对凭空构建出一个 Pandas DataFrame。
        # self.uid_field 和 self.iid_field 直接复用系统表头中的用户列名和物品列名。
        fake_df = pd.DataFrame({
            self.uid_field: fake_u,
            self.iid_field: fake_i,
            # 核心设定：0 代表将这批伪造的交互流水强制划入训练集 (Training Dataset)
            self.splitting_label: 0  
        })
        
        # 【数据合并：物理层面的缝合】
        # self.df 是系统之前读取原始文件生成的全局真实数据表。
        # 利用 pd.concat 将构建好的 fake_df 直接追加到全局表的最末尾。
        # 至此，从稀疏矩阵到结构化数据表的形态跨越完成。
        self.df = pd.concat([self.df, fake_df], ignore_index=True)
        
        # 更新数据集的宏观统计信息，使推荐系统的底层逻辑完全接纳这批新数据。
        self.user_num += n_fakes
        self.inter_num = len(self.df)
        
        self.logger.info(f"✅ 成功注入 {n_fakes} 个假用户，共 {len(fake_df)} 条假点击。")
        self.logger.info(f"更新后用户总数: {self.user_num}, 交互总数: {self.inter_num}")
        self.logger.info(f"==============================================")