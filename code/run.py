#!/usr/bin/env python
# coding: utf-8
from pycaret.datasets import get_data
import tqdm
from pycaret.classification import *
import pandas as pd
import pickle
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
embedding = pd.DataFrame(load_pkl('../user_data/node2vec_64_32_8_4_1_399_0.001_0.8617328519855596.pkl'))

data_files = '../xfdata/'
train_df = pd.read_csv(data_files + 'train.csv')
test_df = pd.read_csv(data_files + 'test.csv')
target_df = train_df.append(test_df)
target_df = target_df.fillna(-1)

embedding['id'] = target_df['id']
embedding['target'] =   target_df['target']
data_df = embedding

exp = ClassificationExperiment()
train_df = data_df[data_df.target.isin([0,1])]
test_df = data_df[data_df.target.isin([-1])]

random_state = 38
exp.set_config('seed', random_state)
exp_name = exp.setup(data = train_df,  target = 'target',index='id',verbose=False)
best_model = exp.compare_models(sort = 'acc',n_select = 1,budget_time=360,verbose=True,include=['gpc'])

result = exp.predict_model(best_model,data=test_df)# generate predictions
result = result.reset_index()
result['target'] = result['prediction_label']
result[['id','target']].to_csv(f'../prediction_result/predict.csv',index=False)