import copy
import os
import platform
import traceback
from typing import Union

import math
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from models.DLinear import series_decomp
from util.functional import ridge_regression, instance_norm
from util.lead_estimate import estimate_indicator, accurate_indicator, accurate_strict_indicator_coef, \
    estimate_strict_indicator_coef, cross_corr_coef, shifted_leader_seq
from util.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


def get_alldata(filename='electricity.csv', root_path='./'):
    path = os.path.join(root_path, filename)
    if filename.endswith('.csv'):
        df = pd.read_csv(path)
        if filename.startswith('wind'):
            df['date'] = pd.date_range(start='2000-01-01', periods=len(df), freq='H')
    else:
        if filename.startswith('nyc'):
            import h5py
            x = h5py.File(path, 'r')
            data = list()
            for key in x.keys():
                data.append(x[key][:])
            ts = np.stack(data, axis=1)
            ts = ts.reshape((ts.shape[0], ts.shape[1] * ts.shape[2]))
            df = pd.DataFrame(ts)
            df['date'] = pd.date_range(start='2007-04-01', periods=len(df), freq='30T')
        elif filename.endswith('.npz'):
            ts = np.load(path)['data'].astype(np.float32)
            ts = ts.reshape((ts.shape[0], ts.shape[1] * ts.shape[2]))
            df = pd.DataFrame(ts)
            if filename == 'PeMSD4':
                df['date'] = pd.date_range(start='2017-07-01', periods=len(df), freq='5T')
            else:
                df['date'] = pd.date_range(start='2012-03-01', periods=len(df), freq='5T')
        elif filename.endswith('.h5'):
            df = pd.read_hdf(path)
            df['date'] = df.index.values
        elif filename.endswith('.txt'):
            df = pd.read_csv(path, header=None)
            df['date'] = pd.date_range(start='1/1/2007', periods=len(df), freq='10T')
        df = df[[df.columns[-1]] + list(df.columns[:-1])]
    return df


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False, border=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.border = border

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        if self.border is None:
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        else:
            border1s, border2s = self.border
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, self.timeenc, freq=self.freq)

        self.data = data[:border2s[-1]].astype(np.float32)
        self.border = (border1, border2)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp.astype(np.float32)
        self.data_x = torch.tensor(self.data_x).float()
        self.data_y = torch.tensor(self.data_y).float()
        self.data_stamp = torch.tensor(self.data_stamp).float()

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_end:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', train_only=False, border=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.border = border

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        if self.border is None:
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            border1s, border2s = self.border
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, self.timeenc, freq=self.freq)

        self.data = data[:border2s[-1]].astype(np.float32)
        self.border = (border1, border2)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp.astype(np.float32)
        self.data_x = torch.tensor(self.data_x).float()
        self.data_y = torch.tensor(self.data_y).float()
        self.data_stamp = torch.tensor(self.data_stamp).float()

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_end:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_Meta(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='Weather.csv',
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False, border=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        if len(size) > 3:
            self.meta_len = size[3]
        else:
            self.meta_len = self.pred_len + 48
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = get_alldata(self.data_path, self.root_path)

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('date')
        # print(cols)
        num_train = int(len(df_raw) * (0.7 if not self.train_only else 1))
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len - self.meta_len, len(df_raw) - num_test - self.seq_len - self.meta_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.features == 'MS':
            self.data_y = data[:, -1][border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp.astype(np.float32)
        self.data_x = torch.tensor(self.data_x).float()
        self.data_y = torch.tensor(self.data_y).float()
        self.data_stamp = torch.tensor(self.data_stamp).float()

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        pre_x = self.data_x[torch.arange(self.seq_len).unsqueeze(0) + torch.arange(self.meta_len - self.pred_len + 1).unsqueeze(-1) + index]
        pre_y = self.data_y[torch.arange(self.pred_len).unsqueeze(0) + torch.arange(self.meta_len - self.pred_len + 1).unsqueeze(-1) + index + self.seq_len]
        seq_x = self.data_x[s_begin + self.meta_len:s_end + self.meta_len]
        seq_y = self.data_y[s_end + self.meta_len:r_end + self.meta_len]
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, pre_x, pre_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1 - self.meta_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False, border=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only

        self.split_ratio = (0.7, 0.2) if border is None else border
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = get_alldata(self.data_path, self.root_path)

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('date')
        # print(cols)
        num_train = int(len(df_raw) * (self.split_ratio[0] if not self.train_only else 1))
        num_test = int(len(df_raw) * self.split_ratio[1])
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, self.timeenc, freq=self.freq)

        data = data.astype(np.float32)
        self.data = data
        self.border = (border1, border2)

        self.data_x = data[border1:border2]
        if self.features == 'MS':
            self.data_y = data[:, -1][border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp.astype(np.float32)
        self.data_x = torch.tensor(self.data_x).float()
        self.data_y = torch.tensor(self.data_y).float()
        self.data_stamp = torch.tensor(self.data_stamp).float()

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_end:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom_Perturb(Dataset_Custom):
    def __init__(self, root_path, target_variable=-1, min_shift=1, **kwargs):
        super().__init__(root_path, **kwargs)
        self.target_variable = target_variable
        self.data_y = self.data_y[..., self.target_variable]
        self.min_shift = min_shift
        self.max_shift = self.pred_len
        self.const_indices = torch.arange(self.seq_len, dtype=torch.long).unsqueeze(-1)
        self.shift = torch.randint(self.min_shift, self.max_shift,
                                   size=self.__len__() * self.data_x.shape[-1]).reshape(self.__len__(), -1)
        self.shift[:, target_variable] = 0

    def __getitem__(self, index):
        s_begin = index + self.max_shift
        shift_indices = self.const_indices + s_begin - self.shift[index]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x.gather(0, shift_indices)
        seq_y = self.data_y[s_end:r_end]
        seq_x_mark = self.data_stamp.gather(0, shift_indices)
        seq_y_mark = self.data_stamp.gather(0, shift_indices - s_begin + r_begin)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return super().__len__() - self.max_shift

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_ETT_hour_CI(Dataset_ETT_hour):
    def __init__(self, root_path, *args, **kwargs):
        super().__init__(root_path, *args, **kwargs)
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = super().__len__()

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[s_end:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return super().__len__() * self.enc_in


class Dataset_ETT_minute_CI(Dataset_ETT_minute):
    def __init__(self, root_path, *args, **kwargs):
        super().__init__(root_path, *args, **kwargs)
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = super().__len__()

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[s_end:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return super().__len__() * self.enc_in


class Dataset_Custom_CI(Dataset_Custom):
    def __init__(self, root_path, *args, **kwargs):
        super().__init__(root_path, *args, **kwargs)
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = super().__len__()

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[s_end:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return super().__len__() * self.enc_in


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv', target='OT',
                 scale=True, inverse=False, timeenc=0, freq='15min', cols=None, train_only=False, border=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = get_alldata(self.data_path, self.root_path)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
        else:
            cols = list(df_raw.columns)
            self.cols = cols.copy()
            cols.remove('date')
        if self.features == 'S':
            cols.remove(self.target)

        num_train = int(len(df_raw) * (0.7 if not self.train_only else 1))
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len - self.meta_len, len(df_raw) - num_test - self.seq_len - self.meta_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = 0
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values.astype(np.float32)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp.astype(np.float32)
        self.data_x = torch.tensor(self.data_x).float()
        self.data_y = torch.tensor(self.data_y).float()
        self.data_stamp = torch.tensor(self.data_stamp).float()

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[s_end:r_end]
        else:
            seq_y = self.data_y[s_end:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_Lead(Dataset):
    def __init__(self, dataset, prefetch_path=None,
                 leader_num=4, local_max=True,
                 prefetch_batch_size=32, device='cuda', pin_gpu=False,
                 trunc_tail=12, variable_batch_size=32, efficient=True,
                 trend_based=False, seasonal_based=False, segment_based=False, **kwargs):
        self.moving_average_kernel = 25
        self.dataset = dataset
        self.seq_len = dataset.seq_len
        self.pred_len = dataset.pred_len
        self.cache_size = len(dataset)
        self.C = dataset.data_x.shape[-1]
        self.K = min(self.C, leader_num)
        self.prefetch_batch_size = prefetch_batch_size
        self.device = torch.device(device)
        self.pin_gpu = pin_gpu
        self.variable_batch_size = variable_batch_size
        self.efficient = efficient
        if trunc_tail == 0:
            trunc_tail = -dataset.seq_len
        self.trunc_tail = trunc_tail
        self.trend_based = trend_based
        self.seasonal_based = seasonal_based
        self.segment_based = segment_based
        self.local_max = local_max
        self.cache = {}
        self._load_prefetch_files(prefetch_path, lambda x: instance_norm(x.permute(0, 2, 1), -1))
        if trend_based or seasonal_based:
            self.decomposition = series_decomp(self.moving_average_kernel)
        if seasonal_based:
            self._load_prefetch_files(prefetch_path[:-4] + f'_season{self.moving_average_kernel}.npz',
                                      lambda x: instance_norm(self.decomposition(x)[0].permute(0, 2, 1), -1),
                                      suffix=f'_season{self.moving_average_kernel}')
        if trend_based:
            self._load_prefetch_files(prefetch_path[:-4] + f'_trend{self.moving_average_kernel}.npz',
                                      lambda x: instance_norm(self.decomposition(x)[1].permute(0, 2, 1), -1),
                                      suffix=f'_trend{self.moving_average_kernel}')
        if segment_based:
            self.trunc_tail = 2
            seg_len = self.moving_average_kernel - 1
            self._load_prefetch_files(prefetch_path[:-4] + f'_segment{seg_len}.npz',
                                      lambda x: instance_norm(
                                          x.view(x.shape[0], x.shape[1] // seg_len, seg_len, x.shape[2]).
                                          mean(-2).permute(0, 2, 1),
                                          -1),
                                      suffix=f'_segment{seg_len}')

    def _load_prefetch_files(self, prefetch_path,
                             process_x_func,
                             suffix=''):
        try:
            print('Loading prefetch files from', prefetch_path)
            assert prefetch_path and os.path.exists(prefetch_path)
            prefetch = np.load(prefetch_path)
            assert prefetch['leader_ids' + suffix].shape[0] == len(self.dataset) + self.pred_len
            assert prefetch['leader_ids' + suffix].shape[-1] >= self.K
            self.cache['leader_ids' + suffix] = torch.tensor(prefetch['leader_ids' + suffix][:self.cache_size, :, :self.K])
            if self.pin_gpu:
                self.cache['leader_ids' + suffix] = self.cache['leader_ids' + suffix].to(self.device)
            for k in ['shift', 'corr']:
                assert prefetch[k + suffix].shape[-2] >= self.K
                self.cache[k + suffix] = torch.tensor(prefetch[k + suffix][:self.cache_size, :, :self.K])
                if self.pin_gpu:
                    self.cache[k + suffix] = self.cache[k + suffix].to(self.device)
        except Exception as e:
            traceback.print_exc()
            print('Fail to load prefetch files')
            if not os.path.exists(os.path.dirname(prefetch_path)):
                os.mkdir(os.path.dirname(prefetch_path))
            self._generate_prefetch_files(process_x_func=process_x_func, suffix=suffix, K=self.K)
            print('Generate new prefetch files to', prefetch_path)
            np.savez(prefetch_path,
                     **{k + suffix: self.cache[k + suffix].cpu().numpy() for k in ['leader_ids', 'shift', 'corr']})

    def _generate_prefetch_files(self, process_x_func, K=16, suffix=''):
        K = min(self.C, K)
        device = self.device if self.pin_gpu else torch.device('cpu')
        self.cache.update(**{
            'leader_ids' + suffix: torch.empty((self.cache_size + self.pred_len, self.C, K), dtype=torch.long, device=device),
            'corr' + suffix: torch.empty((self.cache_size + self.pred_len, self.C, K), dtype=torch.float32, device=device),
            'shift' + suffix: torch.empty((self.cache_size + self.pred_len, self.C, K), dtype=torch.long, device=device),
        })
        self.dataset.data_x = torch.tensor(self.dataset.data_x, device=self.device)

        indices = torch.arange(self.dataset.seq_len).unsqueeze(0) + torch.arange(self.prefetch_batch_size).unsqueeze(-1)

        if self.efficient:
            estimate_num = self.cache_size + self.pred_len
        else:
            estimate_num = self.seq_len - self.dataset.border[0]

        if self.efficient or estimate_num > 0:
            for i in tqdm(range(0, estimate_num, self.prefetch_batch_size)):
                _idx = (indices + i)[:min(estimate_num - i, self.prefetch_batch_size)]
                x = self.dataset.data_x[_idx].to(self.device)
                res = estimate_indicator(process_x_func(x), K, trunc_tail=self.trunc_tail, local_max=self.local_max)
                for ei, (k, v) in enumerate(zip(['leader_ids', 'shift', 'corr'], res)):
                    self.cache[k + suffix][i: i+len(_idx)] = v if self.pin_gpu else v.cpu()

        if not self.efficient:
            data_x = self.dataset.data[max(0, self.dataset.border[0]-self.seq_len): self.dataset.border[1]]
            _idx = torch.arange(self.seq_len).unsqueeze(0) + torch.arange(len(data_x) - self.seq_len + 1).unsqueeze(-1)
            x = torch.tensor(data_x[_idx], device=self.device, dtype=torch.float32)
            x = process_x_func(x)
            for j in tqdm(range(self.C)):
                res = accurate_indicator(x, j, K, local_max=self.local_max)
                for ei, (k, v) in enumerate(zip(['leader_ids', 'shift', 'corr'], res)):
                    self.cache[k + suffix][max(0, estimate_num):, j] = v if self.pin_gpu else v.cpu()
                print(k)
                print(v[[0, 5, 10]])

    def __getitem__(self, index):
        # if not self.trend_based and not self.seasonal_based:
        res = [self.cache[k][index] for k in ['leader_ids', 'shift', 'corr']]
        if self.trend_based:
            suffix = f'_trend{self.moving_average_kernel}'
            res += [self.cache[k + suffix][index] for k in ['leader_ids', 'shift', 'corr']]
        if self.segment_based:
            suffix = f'_segment{self.moving_average_kernel - 1}'
            res += [self.cache[k + suffix][index] for k in ['leader_ids', 'shift', 'corr']]
        return self.dataset[index] + tuple(res)

    def __len__(self):
        return len(self.dataset)


class Dataset_Lead_Stat(Dataset_Lead):
    def __init__(self, dataset, threshold=0, **kwargs):
        self.threshold = threshold
        super().__init__(dataset, **kwargs)

    def _load_prefetch_files(self, prefetch_path,
                             process_x_func,
                             suffix=''):
        super()._load_prefetch_files(prefetch_path, process_x_func, suffix='')
        self.evaluate_prefetch_files(prefetch_path,
                                     process_x_func=process_x_func,
                                     suffix=suffix, K=self.K)

    def evaluate_prefetch_files(self, prefetch_path, process_x_func, K=16, suffix=''):
        K = min(self.C, K)
        device = self.device if self.pin_gpu else torch.device('cpu')
        # self.cache.update(**{
        #     'corr' + suffix: torch.empty((self.cache_size + self.pred_len, self.C, self.C, self.seq_len - 2), dtype=torch.float32),
        # })

        print('mean corr', self.cache['corr'].mean())

        indices = torch.arange(self.dataset.seq_len).unsqueeze(0) + torch.arange(self.prefetch_batch_size).unsqueeze(-1)

        self.efficient = True
        if self.efficient:
            estimate_num = self.cache_size
        else:
            estimate_num = self.seq_len - self.dataset.border[0]

        delta, future_rs = [], []

        def _instance_norm(ts, dim):
            mu = ts.mean(dim, keepdims=True)
            ts = ts - mu
            std = ((ts ** 2).mean(dim, keepdims=True) + 1e-8) ** 0.5
            return ts / std, mu, std

        const_indices = torch.arange(self.seq_len, self.seq_len + self.pred_len, dtype=torch.int, device=self.device).unsqueeze(0).unsqueeze(0)
        if self.efficient or estimate_num > 0:
            self.prefetch_batch_size = self.prefetch_batch_size // 2
            for i in tqdm(range(0, len(self.dataset), self.prefetch_batch_size)):
                batch = [[], [], [], [], []]
                for _i in range(i, min(i+self.prefetch_batch_size, len(self.dataset))):
                    _item = self.__getitem__(_i)
                    for j, _x in enumerate(_item[:2] + _item[4:]):
                        batch[j].append(_x)
                x, y, leader_ids, shift, r = [torch.stack(_data, 0).to(self.device) for _data in batch]

                x, y = x.permute(0, 2, 1), y.permute(0, 2, 1) # [B, C, H]
                x, mu, std = _instance_norm(x, -1)
                y = (y - mu) / std

                seq_shifted, r_abs = shifted_leader_seq(x, y, self.K, leader_ids, shift, r,
                                                    const_indices, trunc_tail=self.trunc_tail) # [B, C, K, H]
                future_r = (instance_norm(seq_shifted, -1) @ instance_norm(y, -1).unsqueeze(-1)).squeeze(-1) / self.pred_len

                delta.append((future_r - r_abs)[r_abs > self.threshold].view(-1).cpu())
                future_rs.append(future_r[r_abs > self.threshold].view(-1).cpu())
            delta = torch.cat(delta, 0)
            future_rs = torch.cat(future_rs, 0).numpy()
            delta = np.sort(delta.numpy())

            print(delta.mean())
            print('75%', np.quantile(delta, 0.25))
            print('50%', np.median(delta))
            print(future_rs.mean())
        self.changes = delta
        self.future_rs = future_rs

        if not self.efficient:
            data_x = self.dataset.data[max(0, self.dataset.border[0]-self.seq_len): self.dataset.border[1]]
            _idx = torch.arange(self.seq_len).unsqueeze(0) + torch.arange(len(data_x) - self.seq_len + 1).unsqueeze(-1)
            x = torch.tensor(data_x[_idx], device=self.device, dtype=torch.float32)
            x = process_x_func(x)
            for j in tqdm(range(self.C)):
                v = accurate_strict_indicator_coef(x, j)
                self.cache['corr'][max(0, estimate_num):, j] = v.cpu()


class Dataset_Lead_Pretrain(Dataset_Lead):
    def __init__(self, dataset, pred_path=None, **kwargs):
        super().__init__(dataset, **kwargs)
        pred = np.load(pred_path).astype(np.float32)
        if self.dataset.border[1] == len(self.dataset.data):
            pred = pred[-len(self.dataset):]
        else:
            begin_index = max(0, self.dataset.border[0] - self.pred_len + 1)
            pred = pred[begin_index: begin_index + len(dataset)]
        self.pred = torch.tensor(pred, device=self.device if self.pin_gpu else 'cpu', dtype=torch.float32)

    def __getitem__(self, index):
        return super().__getitem__(index) + (self.pred[index], )


class Dataset_Recent(Dataset):
    def __init__(self, dataset, gap: Union[int, tuple, list], recent_num=1, **kwargs):
        super().__init__()
        self.dataset = dataset
        self.gap = gap
        self.recent_num = recent_num

    def _stack(self, data):
        if isinstance(data[0], np.ndarray):
            return np.vstack(data)
        else:
            return torch.stack(data, 0)

    def __getitem__(self, index):
        if self.recent_num == 1:
            return self.dataset[index], self.dataset[index + self.gap]
        else:
            current_data = self.dataset[index + self.gap + self.recent_num - 1]
            if not isinstance(current_data, tuple):
                recent_data = tuple(self.dataset[index + n] for n in range(self.recent_num))
                recent_data = self._stack(recent_data)
                return current_data, recent_data
            else:
                recent_data = tuple([] for _ in range(len(current_data)))
                for past in range(self.recent_num):
                    for j, past_data in enumerate(self.dataset[index + past]):
                        recent_data[j].append(past_data)
                recent_data = tuple(self._stack(recent_d) for recent_d in recent_data)
            return current_data, recent_data

    def __len__(self):
        return len(self.dataset) - self.gap - self.recent_num + 1


class Dataset_Concept(Dataset):
    def __init__(self, dataset, span, concept_path=None, bias=True, norm='instance', penalty=0,
                 device='cuda', general_stat=True, concept=None, **kwargs):
        self.dataset = dataset
        self.seq_len = dataset.seq_len
        self.pred_len = dataset.pred_len
        self.channel_num = dataset.data_x.shape[-1]
        self.span = span
        self.norm = norm
        self.bias = bias
        self.penalty = penalty
        self.general_stat = general_stat
        self.return_key = ['Y|X', 'mu']
        self.device = torch.device(device)
        self.cache = None
        self.max_pred_len = max(96, self.pred_len)
        if concept is None:
            if concept_path is None:
                data_name = dataset.data_path.split("/")[-1].split(".")[0]
                if platform.system() != 'Windows':
                    path = './'
                else:
                    path = 'D:/data/'
                concept_path = path + f'concept/{data_name}_{self.seq_len}_{self.max_pred_len}' \
                               f'_span{span}_{norm}Norm_bias{bias}_penalty{penalty}.npz'
                # concept_path = f'./concept/{data_name}_{self.seq_len}_{self.pred_len}' \
                #                f'_span{span}_{norm}Norm_bias{bias}_penalty{penalty}.npz'
            concept = self._load_concepts(concept_path)
        self.cache = {}
        cache_size = self.dataset.border[1] - self.seq_len - self.pred_len + 1
        for k in self.return_key:
            if k == 'Y|X':
                self.cache[k] = torch.tensor(concept[k][:cache_size, :self.pred_len])
            else:
                self.cache[k] = torch.tensor(concept[k][:cache_size])

    def _load_concepts(self, concept_path):
        try:
            print('Loading concepts from', concept_path)
            assert concept_path and os.path.exists(concept_path)
            return np.load(concept_path)
        except Exception as e:
            traceback.print_exc()
            print('Fail to load concepts')
            if not os.path.exists(os.path.dirname(concept_path)):
                os.mkdir(os.path.dirname(concept_path))
            print('Generate new prefetch files to', concept_path)
            concept = self._learning_concept(self.max_pred_len)

            np.savez(concept_path, **{k: v.cpu().numpy() for k, v in concept.items()})
            return concept

    def _learning_concept(self, pred_len=96):
        cache_size = len(self.dataset.data) - self.seq_len
        concept = {'Y|X': torch.empty((cache_size, pred_len, self.seq_len + self.bias), dtype=torch.float32),
                      # 'mean': torch.empty((cache_size, self.seq_len), dtype=torch.float32),
                      # 'std': torch.empty((cache_size, self.seq_len), dtype=torch.float32),
                      'mu': torch.empty((cache_size, self.channel_num), dtype=torch.float32),
                      # 'sigma': torch.empty((cache_size, self.channel_num), dtype=torch.float32)
                   }

        data = torch.tensor(self.dataset.data, device=self.device, dtype=torch.float32)
        indices = torch.arange(self.seq_len + pred_len).unsqueeze(0)
        for i in tqdm(range(0, cache_size)):
            X = data[indices[:, :min(indices.shape[-1], len(data) - i)] + torch.arange(max(0, i-self.span+1), i+1).unsqueeze(-1)]
            X = X.permute(0, 2, 1)
            X, Y = X[..., :self.seq_len], X[..., self.seq_len:]
            mean = X.mean(-1, keepdims=True)
            var = ((X - mean) ** 2).mean(-1, keepdims=True)
            mask = (var > 1e-5).reshape(-1, 1)
            std = (var + 1e-5) ** 0.5
            concept['mu'][i] = mean.mean(0).squeeze(-1).cpu()
            # concept['sigma'][i] = std.mean(0).squeeze(-1).cpu()

            # concept['mean'][i] = X.reshape(-1, X.shape[-1]).mean(0).cpu()
            # concept['std'][i] = X.reshape(-1, X.shape[-1]).std(0).cpu()

            if self.norm == 'last':
                X, Y = X - X[..., [-1]], Y - X[..., [-1]]
            elif self.norm == 'instance':
                X, Y = (X - mean) / std, (Y - mean) / std
                # X, Y = instance_norm(X, -1, Y)

            X, Y = X.reshape(-1, X.shape[-1]) * mask, Y.reshape(-1, Y.shape[-1]) * mask
            concept['Y|X'][i, :Y.shape[-1]] = ridge_regression(X, Y, lamda=self.penalty, bias=self.bias).transpose(-1, -2).cpu()
        return concept

    def __getitem__(self, index, keys=None):
        if keys is None:
            keys = self.return_key
        elif not isinstance(keys, list):
            keys = [keys]
        return tuple(self.cache[k][index + self.dataset.border[0]] for k in keys)

    def __len__(self):
        return len(self.cache[self.return_key[-1]])

class Dataset_Concept_Window(Dataset):
    def __init__(self, dataset, prompt_len, span, concept_path=None, offset=0, kind='whole',
                 bias=True, norm='instance', penalty=0, device='cuda', general_stat=True, concept=None, **kwargs):
        self.dataset = dataset
        self.seq_len = dataset.seq_len
        self.pred_len = dataset.pred_len
        self.prompt_len = prompt_len
        self.history_num = prompt_len * span
        if offset < 0:
            offset = span // 2
        self.offset = offset
        self.span = span
        self.kind = kind
        self.return_key = ['Y|X']
        # self.return_key = ['Y|X', 'mean', 'std'] if general_stat else ['Y|X', 'mu', 'sigma']
        # self.concept_label = Dataset_Concept(self.dataset, 1, concept_path, bias, norm, penalty, device, general_stat)
        self.concept = Dataset_Concept(self.dataset, self.span, concept_path, bias, norm, penalty, device, general_stat,
                                       concept=concept)
        reserve_history_num = (self.prompt_len - 1) * self.span + self.pred_len
        # assert self.pred_len % self.span == 0
        # if self.pred_len % self.span == 0: reserve_history_num += self.span - self.pred_len % self.span

        self.concept_indices_X = torch.arange(reserve_history_num)
        if self.kind == 'whole':
            self.concept_indices_YX = torch.arange(self.prompt_len - 1 + math.ceil(self.pred_len / self.span)) * self.span
        else:
            self.concept_indices_YX = torch.arange(self.prompt_len) * self.span
        # self.concept_indices = torch.arange(self.prompt_len) * self.span + torch.arange(self.pred_len).unsqueeze(-1)
        # self.concept_indices = self.concept_indices.unsqueeze(-1).expand(-1, -1, self.seq_len)

        """ Truncate """
        if self.dataset.border[0] == 0:
            self.dataset.border = (reserve_history_num, self.dataset.border[1])
            for k in ['data_x', 'data_y', 'data_stamp']:
                setattr(self.dataset, k, getattr(self.dataset, k)[self.dataset.border[0]:])
        # self.concept_label.cache = {k: v[self.dataset.border[0]:
        #                                  self.dataset.border[1] - self.seq_len - self.pred_len + 1]
        #                             for k, v in self.concept_label.cache.items()}
        for k, v in self.concept.cache.items():
            self.concept.cache[k] = v[self.dataset.border[0] - reserve_history_num:
                                      self.dataset.border[1] - self.seq_len - self.pred_len]

        # assert self.concept_label.cache['Y|X'].shape[0] == len(self.dataset)

        if self.kind == 'whole':
            """ Align """
            shift = 0
            for i in range(1, self.pred_len):
                shift = i % self.span
                if shift > 0:
                    self.concept.cache['Y|X'][shift:, i] = self.concept.cache['Y|X'][:-shift, i].clone()
            self.concept.cache['Y|X'] = self.concept.cache['Y|X'][shift:]
            assert self.concept.cache['Y|X'].shape[0] - self.concept_indices_YX[-1] == len(self.dataset)
        elif self.kind == 'partial':
            """ Align """
            for i in range(1, self.pred_len):
                self.concept.cache['Y|X'][i:, i] = self.concept.cache['Y|X'][:-i, i].clone()
            self.concept.cache['Y|X'] = self.concept.cache['Y|X'][self.pred_len - 1:]
            assert self.concept.cache['Y|X'].shape[0] - self.concept_indices_YX[-1] == len(self.dataset)
        else:
            assert self.concept.cache['Y|X'].shape[0] - self.concept_indices_YX[-1] == len(self.dataset) + self.pred_len - 1


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        current_data = self.dataset[index]
        if not isinstance(current_data, tuple):
            current_data = (current_data, )
        concept_past_YX = (self.concept.cache['Y|X'][index + self.concept_indices_YX], )
        # concept_past_X = tuple(self.concept.cache[k][index + self.concept_indices_X] for k in self.return_key[1:])
        # concept_label = tuple(self.concept_label.cache[k][index] for k in self.return_key)
        return current_data + concept_past_YX


class Dataset_Concept_Pred(Dataset_Concept):
    def __init__(self, dataset, prompt_len, span, concept_prediction_path, concept_path=None, bias=True, norm='instance',
                 penalty=0, device='cuda', general_stat=True, prediction=None, use_mean=True, **kwargs):
        super().__init__(dataset, 1, concept_path, bias, norm, penalty, device, general_stat, **kwargs)
        self.prompt_len = prompt_len
        self.history_num = prompt_len * span
        self.span = span
        self.reserve_history_num = (self.prompt_len - 1) * self.span + self.pred_len

        for k, v in self.cache.items():
            self.cache[k] = v[self.dataset.border[0]: self.dataset.border[1] - self.seq_len - self.pred_len + 1]
        if prediction is None:
            print('Load predicted metadata from', concept_prediction_path)
            prediction = torch.tensor(np.load(concept_prediction_path))
        assert len(prediction) == len(self.dataset.data) - self.seq_len - self.pred_len * 3 + 3 - self.reserve_history_num, \
            (len(prediction), len(self.dataset.data) - self.seq_len - self.pred_len * 3 + 3 - self.reserve_history_num)
        # self.cache['pred'] = self.cache['Y|X'].clone()
        self.use_mean = use_mean
        if self.use_mean:
            self.return_key += ['pred']
        else:
            self.return_key = ['Y|X', 'pred']
            self.cache.pop('mu')
        self.cleared = False
        self.reload_concept(prediction)
        # self.cache = {k: v.to(device) for k, v in self.cache.items()}

    def clear_pred(self):
        self.cache['pred'] = self.cache['pred'][:-self._len_pred]
        self.cleared = True

    def reload_concept(self, all_predictions):
        predictions = all_predictions[self.dataset.border[0] - (0 if self.dataset.border[0] == 0 else self.reserve_history_num):
                                          self.dataset.border[1] - self.seq_len - self.pred_len + 1 - self.reserve_history_num]
        self._len_pred = len(predictions)
        if self.cleared:
            self.cache['pred'] = torch.cat([self.cache['pred'], predictions.to(self.cache['pred'].device)])
        else:
            self.cache['pred'] = torch.cat([self.cache['Y|X'][:-self._len_pred].clone(), predictions.to(self.cache['Y|X'].device)])

    def __getitem__(self, index):
        current_data = self.dataset[index]
        if not isinstance(current_data, tuple):
            current_data = (current_data, )
        if self.use_mean:
            concept_YX, concept_mean, concept_pred = tuple(self.cache[k][index] for k in self.return_key)
            concept_pred = torch.cat([concept_pred.reshape(*concept_pred.shape[:-2], -1), concept_mean], -1)
            concept = torch.cat([concept_YX.reshape(*concept_YX.shape[:-2], -1), concept_mean], -1)
        else:
            concept, concept_pred = tuple(self.cache[k][index].reshape(*self.cache[k][index].shape[:-2], -1) for k in self.return_key)
        return current_data + (concept, concept_pred)


class Dataset_Concept_X(Dataset_Concept):
    def __init__(self, dataset, prompt_len, span, concept_path=None, bias=True, norm='instance',
                 penalty=0, device='cuda', general_stat=True, prediction=None, use_mean=True, **kwargs):
        super().__init__(dataset, 1, concept_path, bias, norm, penalty, device, general_stat, **kwargs)
        self.prompt_len = prompt_len
        self.history_num = prompt_len * span
        self.span = span
        self.reserve_history_num = (self.prompt_len - 1) * self.span + self.pred_len

        self.cache = self.cache['mu'][self.dataset.border[0]: self.dataset.border[1] - self.seq_len - self.pred_len + 1].to(device)

    def __getitem__(self, index):
        current_data = self.dataset[index]
        if not isinstance(current_data, tuple):
            current_data = (current_data, )
        concept_mean = self.cache[index]
        return current_data + (concept_mean, )

    def __len__(self):
        return len(self.cache)

