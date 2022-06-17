# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 18:09:56 2022

@author: tongj
"""

import datetime
import pandas as pd
import lightgbm as lgb
from feature_engineer import make_feature

import warnings
warnings.filterwarnings("ignore")


def main():
    data = pd.read_csv('comp_2022_all_rank_b_data.tsv', sep="\t", encoding="utf-8")    
    mape_val = pd.read_csv('mape_val.csv')
    
    test = make_feature(sample = 'test')
    
    used_features = list(test.columns)
    not_used = ['nth_day', 'seriesNo', 'cid_t',
                'seriesId_t', 'channelId_t', 'leader_t', 'kind_t',
                't', 'y']
    used_features = [x for x in used_features if x not in not_used]
    
    # 模型加载
    clf = lgb.Booster(model_file='model.txt')   
    
    y_pred = test[['cid_t', 't']].copy()
    y_pred['VV'] = clf.predict(test[used_features])
    y_pred = y_pred.groupby(['cid_t', 't'])['VV'].mean().reset_index()

    # 提交结果
    print('submit')
    sub = test[['cid_t', 't']].drop_duplicates()
    sub = pd.merge(sub, y_pred, on = ['cid_t', 't'])
    sub.rename(columns = {'t':'nth_day'}, inplace = True) 

    # 取最后七天最小值
    y_min = data[data['cid_day_vv_t'] != 0]
    y_min = y_min.groupby('cid_t').tail(7)
    y_min = y_min.groupby('cid_t')['cid_day_vv_t'].min().reset_index()
    y_min.rename(columns = {'cid_day_vv_t':'VV_min'}, inplace = True)    
    sub = pd.merge(sub, y_min, on = 'cid_t')
    sub.loc[sub['cid_t'].isin(mape_val[mape_val['mape_val'] > 0.4]['cid_t']), 'VV'] = sub['VV_min']

    sub = sub[['cid_t', 'nth_day', 'VV']]
    sub.to_csv('submit.tsv', sep="\t", encoding="utf-8", index=False)
     
if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    print('Model  finished in {}'.format(str(datetime.datetime.now() - start_time)))
