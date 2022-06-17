# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 18:06:51 2022

@author: tongj
"""

import datetime
import lightgbm as lgb
from feature_engineer import make_feature
from sklearn.metrics import mean_absolute_percentage_error

import warnings
warnings.filterwarnings("ignore")

        
def main():
    train = make_feature(sample = 'train')

    # 特征选择
    used_features = list(train.columns)
    not_used = ['nth_day', 'seriesNo', 'cid_t',
                'seriesId_t', 'channelId_t', 'leader_t', 'kind_t',
                't', 'y']
    used_features = [x for x in used_features if x not in not_used]
    print(used_features)

    # 模型训练
    print('[+] Training on model')
    
    params = {'learning_rate': 0.1,
              'boosting_type': 'gbdt',
              'objective': 'mape',
              'lambda_l1': 0.1,
              'lambda_l2': 0.0,
              'max_depth': 0,
              'num_leaves': 402,
              'min_data_in_leaf': 130,
              'n_jobs': -1,
              'seed': 2022,
              'verbosity': -1}

    # 切线下验证集: 最后一部分的数据作为线下验证
    valid_idx = train.groupby('cid_t').tail(7).index
    train_idx = train.drop(valid_idx).index
    
    x_train = train.iloc[train_idx][used_features]
    y_train = train.iloc[train_idx]['y']
    
    x_valid = train.iloc[valid_idx][used_features]
    y_valid = train.iloc[valid_idx]['y']
    y_val = train.iloc[valid_idx][['cid_t', 't', 'y']]

    ## All data
    trn_data = lgb.Dataset(x_train, label = y_train)
    val_data = lgb.Dataset(x_valid, label = y_valid)

    clf = lgb.train(params, trn_data, valid_sets=[trn_data, val_data],
                    num_boost_round = 30000,
                    verbose_eval = 300, early_stopping_rounds = 300)
    # 模型保存
    clf.save_model('model.txt')
    
    y_val['y_val'] = clf.predict(train.iloc[valid_idx][used_features])
    cur_metric = mean_absolute_percentage_error(y_val['y'], y_val['y_val'])
    print(f'valid score: {cur_metric}')
    
    mape_val = y_val.groupby('cid_t').apply(lambda x:mean_absolute_percentage_error(x["y"], x["y_val"])).reset_index()     
    mape_val.columns = ['cid_t', 'mape_val']
    mape_val.to_csv('mape_val.csv', index = False)

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    print('Model  finished in {}'.format(str(datetime.datetime.now() - start_time)))
