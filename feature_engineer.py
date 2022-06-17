# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 22:10:41 2022

@author: tongj
"""

import pandas as pd

import warnings
warnings.filterwarnings("ignore")


def fe_rolling_stat(data, id_col, time_col, time_varying_cols, window_size):
    print('[+] fe_rolling_stat')
    df = data.copy()
    result = df[[id_col, time_col]]
    df = df.sort_values(by = time_col)
    add_feas = []
    key = id_col
    for cur_ws in window_size:
        for val in time_varying_cols:
            for op in ['mean', 'max', 'min', 'std', 'median', 'kurt', 'skew']:
                name = f'{key}|{val}|{op}|{cur_ws}'
                add_feas.append(name)
                if op == 'mean':
                    df[name] = df.groupby(key)[val].transform(
                        lambda x: x.rolling(window=cur_ws).mean())
                if op == 'std':
                    df[name] = df.groupby(key)[val].transform(
                        lambda x: x.rolling(window=cur_ws).std())
                if op == 'median':
                    df[name] = df.groupby(key)[val].transform(
                        lambda x: x.rolling(window=cur_ws).median())
                if op == 'max':
                    df[name] = df.groupby(key)[val].transform(
                        lambda x: x.rolling(window=cur_ws).max())
                if op == 'min':
                    df[name] = df.groupby(key)[val].transform(
                        lambda x: x.rolling(window=cur_ws).min())
                if op == 'kurt':
                    df[name] = df.groupby(key)[val].transform(
                        lambda x: x.rolling(window=cur_ws).kurt())
                if op == 'skew':
                    df[name] = df.groupby(key)[val].transform(
                        lambda x: x.rolling(window=cur_ws).skew())
    result = result.merge(df[[id_col, time_col] + add_feas], on = [id_col, time_col], how = 'left')
    data = pd.merge(data, result, on = [id_col, time_col], how = 'left')
    return data

def fe_ewm_stat(data, id_col, time_col, time_varying_cols, window_size):
    print('[+] fe_ewm_stat')
    df = data.copy()
    result = df[[id_col, time_col]]
    df = df.sort_values(by = time_col)
    add_feas = []
    key = id_col
    for cur_ws in window_size:
        for val in time_varying_cols:
            for op in ['mean', 'std']:
                name = f'{key}|{val}|ewm_{op}|{cur_ws}'
                add_feas.append(name)
                if op == 'mean':
                    df[name] = df.groupby(key)[val].transform(
                        lambda x: x.ewm(span=cur_ws).mean())
                if op == 'std':
                    df[name] = df.groupby(key)[val].transform(
                        lambda x: x.ewm(span=cur_ws).std())
    result = result.merge(df[[id_col, time_col] + add_feas], on = [id_col, time_col], how = 'left')
    data = pd.merge(data, result, on = [id_col, time_col], how = 'left')
    return data
    
def fe_lag(data, id_col, time_col, time_varying_cols, lag):
    print('[+] fe_lag')
    df = data.copy()
    result = df[[id_col, time_col]].copy()
    df = df.sort_values(by = time_col)
    add_feas = []
    key = id_col
    for value in time_varying_cols:
        for cur_lag in lag:
            name = f'{key}|{value}|lag|{cur_lag}'
            add_feas.append(name)
            df[name] = df.groupby(key)[value].shift(cur_lag)
    result = result.merge(df[[id_col, time_col] + add_feas], on = [id_col, time_col], how = 'left')
    data = pd.merge(data, result, on = [id_col, time_col], how = 'left')
    return data

def fe_diff(data, id_col, time_col, time_varying_cols, lag):
    print('[+] fe_diff')
    df = data.copy()
    result = df[[id_col, time_col]].copy()
    df = df.sort_values(by = time_col)
    add_feas = []
    key = id_col
    for value in time_varying_cols:
        for cur_lag in lag:
            name = f'{key}|{value}|diff|{cur_lag}'
            add_feas.append(name)
            df[name] = df[value] - df.groupby(key)[value].shift(cur_lag)
    result = result.merge(df[[id_col, time_col] + add_feas], on = [id_col, time_col], how = 'left')
    data = pd.merge(data, result, on = [id_col, time_col], how = 'left')
    return data

def fe_standardizing(data):
    df = data[data['cid_day_vv_t'] != 0]
    df['nth_day_standardize'] = df.groupby('nth_day')['cid_day_vv_t'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    df['seriesId_t_standardize'] = df.groupby(['nth_day', 'seriesId_t'])['cid_day_vv_t'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    df['channelId_t_standardize'] = df.groupby(['nth_day', 'channelId_t'])['cid_day_vv_t'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    df['leader_t_standardize'] = df.groupby(['nth_day', 'leader_t'])['cid_day_vv_t'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    df['kind_t_standardize'] = df.groupby(['nth_day', 'kind_t'])['cid_day_vv_t'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    
    df = df[['cid_t', 'nth_day', 'nth_day_standardize',
            'seriesId_t_standardize', 'channelId_t_standardize',
            'leader_t_standardize', 'kind_t_standardize']]    
    data = pd.merge(data, df, on = ['cid_t', 'nth_day'], how = 'left')
    return data
    
def fe_date(data):
    df = data.copy()
    # is_holiday	当天是否节假日
    df['is_holiday'] = df['is_holiday'].astype(int)
    
    # 是否是周末
    df['is_wknd'] = df['weekday'] // 6
    return df

def fe_update(data):
    df = data.copy()
    
    # date_has_update	当天是否有更新
    df['date_has_update'] = df['date_has_update'].astype(int)

    # 近几天新增播放视频个数
    for cur_lag in [1, 7]:
        name = f'cid_t|vv_vid_cnt|diff|{cur_lag}'
        df[name] = df['vv_vid_cnt'] - df.groupby('cid_t')['vv_vid_cnt'].shift(cur_lag)   

    # 近几天新增上线视频个数
    for cur_lag in [1, 7]:
        name = f'cid_t|online_vid_cnt|diff|{cur_lag}'
        df[name] = df['online_vid_cnt'] - df.groupby('cid_t')['online_vid_cnt'].shift(cur_lag)   
    return df

def make_feature(sample = 'train'):
    data = pd.read_csv('comp_2022_all_rank_b_data.tsv', sep="\t", encoding="utf-8")
    data = data.sort_values(by = ['cid_t','nth_day']).reset_index(drop = True)
    
    # 数据预处理
    data['leader_t'] = data['leader_t'].map(lambda x: x.split(',')[0])
    data['kind_t'] = data['kind_t'].map(lambda x: x.split(',')[0])   
           
    print('[+] feature engineer')
    # rolling 特征
    data = fe_rolling_stat(data, id_col = 'cid_t', time_col= 'nth_day', time_varying_cols = ['cid_day_vv_t'], window_size=[7])

    # ewm 特征
    data = fe_ewm_stat(data, id_col = 'cid_t', time_col= 'nth_day',  time_varying_cols = ['cid_day_vv_t'], window_size=[7])

    # lag 特征
    data = fe_lag(data, id_col = 'cid_t', time_col= 'nth_day', time_varying_cols = ['cid_day_vv_t'], lag=[7])

    # diff特征
    data = fe_diff(data, id_col = 'cid_t', time_col= 'nth_day', time_varying_cols=['cid_day_vv_t'], lag=[7])

    # 标准化特征
    data = fe_standardizing(data)
    
    # 日期特征
    data = fe_date(data)
    
    # 视频更新特征
    data = fe_update(data)

    # 构造样本
    print('[+] construct data')
    train_test = pd.DataFrame()
    for k_step in range(1, 7 + 1):
        df_shift = data.copy()
        df_shift['t'] = df_shift.groupby('cid_t')['nth_day'].shift(-k_step)
        df_shift['y'] = df_shift.groupby('cid_t')['cid_day_vv_t'].shift(-k_step)
        shift_features = ['is_holiday', 'date_has_update', 'weekday', 'is_wknd',
                          'cid_t|vv_vid_cnt|diff|1', 'cid_t|vv_vid_cnt|diff|7',
                          'cid_t|online_vid_cnt|diff|1', 'cid_t|online_vid_cnt|diff|7']
        for i in shift_features:
            df_shift[i] = df_shift.groupby('cid_t')[i].shift(-k_step)

        df_shift = df_shift[df_shift['cid_day_vv_t'] > 0]
        df_shift = df_shift.groupby('cid_t').tail(35)        
        train_test = train_test.append(df_shift.copy())
    
    if sample == 'train':
        train = train_test.loc[train_test['y'] > 0].reset_index(drop = True)
        return train
    elif sample == 'test':
        test = train_test.loc[train_test['y'] == 0].reset_index(drop = True)
        return test        