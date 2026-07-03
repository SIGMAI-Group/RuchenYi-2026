import os
import pandas as pd
from scipy import sparse
import numpy as np

class DataLoader(object):

  def __init__(self, path):
    self.path = path

    self.inter_file = self._get_inter_file()

    self.n_users, self.n_items = self._init_boundaries()

    self.user2id = {}
    self.item2id = {}

    self.train_data = None
    self.test_data = None

  def _get_inter_file(self):
    for f in os.listdir(self.path):
        if f.endswith('.inter'):
            return os.path.join(self.path, f)
    raise FileNotFoundError(f"未在 {self.path} 找到 .inter 文件")

  def _init_boundaries(self):
    df_head = pd.read_csv(self.inter_file, sep='\t', nrows=5)
    self.user_col = [col for col in df_head.columns if 'user' in col.lower()][0]
    self.item_col = [col for col in df_head.columns if 'item' in col.lower()][0]
    self.split_col = [col for col in df_head.columns if 'split' in col.lower() or 'label' in col.lower()][0]

    df_lite = pd.read_csv(self.inter_file, sep='\t', usecols=[self.user_col, self.item_col])
    n_users = int(df_lite[self.user_col].max()) + 1
    n_items = int(df_lite[self.item_col].max()) + 1

    return n_users, n_items

  def load_train_data(self):
    if self.train_data is None:
      df = pd.read_csv(self.inter_file, sep='\t')

      train_df = df[df[self.split_col] == 0]

      rows = train_df[self.user_col]
      cols = train_df[self.item_col]

      self.train_data = sparse.csr_matrix(
        (np.ones_like(rows), (rows, cols)), dtype='float64',
        shape=(self.n_users, self.n_items))

    return self.train_data

  def load_test_data(self):
    if self.test_data is None:
      df = pd.read_csv(self.inter_file, sep='\t')

      test_df = df[df[self.split_col] == 2]

      rows = test_df[self.user_col]
      cols = test_df[self.item_col]

      self.test_data = sparse.csr_matrix(
        (np.ones_like(rows), (rows, cols)), dtype='float64',
        shape=(self.n_users, self.n_items))

    return self.test_data
