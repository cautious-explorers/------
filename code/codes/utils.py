import pandas as pd
import numpy as np
import os
import yaml
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import re
import string
import pickle
import datetime
import networkx as nx
from powerlaw import Fit, plot_pdf
import pathlib
import random
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm # , # tqdm_notebook, # tnrange
tqdm.pandas(desc='Progress')

data_files = '/storage/data/yangxsh/network_graph/data/'

def save_pkl(path,obj):
    with open(path, 'wb') as f:
        pickle.dump(obj,f)
        
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
class mylog():
    def __init__(self, config):
        config['config_path'] = os.path.join(config['current_path'], 'result', config['version'])
        os.makedirs(config['config_path'], exist_ok=False)
        #log文件
        config['log_path'] = os.path.join(config['config_path'], 'log.txt')
        self.path = config['log_path']
        open(config['log_path'] , 'a').close()
        #log基本参数
        configs = ''
        for i in config:
            if(i.find('path')<0):
                configs+=f'{i}:{config[i]},'
        self.logger(configs)
    def logger(self, text):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = current_time + ',' + text
        with open(self.path, "a", encoding="utf-8") as file:
            file.write(text + "\n")
def save_model(model, config):
    file_name = os.path.join(config['config_path'], 'model.pt')
    with open(file_name, 'wb') as f:
        torch.save(model.state_dict(), f)  #state_dict只保存参数,所以需要初始化.不然就不需要


def load_model(config):
    file_name = os.path.join(config['config_path'], 'model.pt')
    if not os.path.exists(file_name):
        print(file_name+' not exits')
        return
    with open(file_name, 'rb') as f:
        model = torch.load(f)
    return model

    
def get_data():
    edges = pd.read_csv(data_files + 'edges.csv')

    ids = list(set(list(edges.id_1.unique()) + list(edges.id_2.unique())))
    ids.sort()
    id_map_dict = dict(zip(ids, range(len(ids))))
    edges['index_1'] =edges['id_1'].apply(lambda x: id_map_dict[x])
    edges['index_2'] =edges['id_2'].apply(lambda x: id_map_dict[x])
    
    print(len(id_map_dict))

    #不存在指回自己的，所以可以默认为是无向的
    # t = edges.merge(edges,left_on=['index_2'],right_on=['index_1'])
    # t[t.index_1_x == t.index_2_y].shape[0]

    train_df = pd.read_csv(data_files + 'train.csv')
    train_df['index'] =train_df['id'].apply(lambda x: id_map_dict[x])

    test_df = pd.read_csv(data_files + 'test.csv')
    test_df['index'] =test_df['id'].apply(lambda x: id_map_dict[x])
    
    return edges,train_df,test_df


def construct_graph(edges):
    G = nx.Graph()
    for _,edge_i in edges.iterrows():
        G.add_edge(edge_i['index_1'], edge_i['index_2'])
    return G

#字典格式
def load_config(path):
    with open(path,'r',encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    config['device'] = ("cuda" if torch.cuda.is_available() else "cpu")
    config['device_num'] = 0
    if torch.cuda.device_count() > 0:
        config['device_num'] = torch.cuda.device_count()
    
    # if(config['device']=='cpu'):
    #     1/0
    
    config['out_name'] = path.split('/')[-1].split('.')[0]
    
    return config


def seed_everything(seed):
    '''
    固定随机种子
    :param random_seed: 随机种子数目
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    torch.backends.cudnn.benchmark = False  # 确定性的选择一个算法,可复现
    torch.backends.cudnn.deterministic = True  # 固定网络结构
    # torch.use_deterministic_algorithms(True) 明令禁止非确定算法
    # https://pytorch.org/docs/stable/notes/randomness.html

