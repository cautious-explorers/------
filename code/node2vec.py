#!/usr/bin/env python
# coding: utf-8


import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec

from codes.utils import *
from codes.dateset import SocialDataset
import argparse
import itertools


import os

def check_custom_start_file_exists(folder_path, prefix):
    file_list = os.listdir(folder_path)  # 获取文件夹中的所有文件列表
    for file_name in file_list:
        if file_name.startswith(prefix):  # 检查文件名是否以指定的前缀开头
            return True  # 存在符合条件的文件
    return False  # 不存在符合条件的文件


# In[2]:

parse = argparse.ArgumentParser()


config_path= './config/1.yaml'
config = load_config(config_path)
dataset = SocialDataset(config['datasetname'],config=config)


# In[3]:


data = dataset[0]
device = config['device']
seed_everything(config['seed'] )


# In[5]:

qlist = [1]  
plist = [1] 
embedding_dim_list = [64]
walk_length_list = [32]  
context_size_list = [8]
walks_per_node_list = [4]
num_negative_samples_list = [1]
epoch_list = [400]#
lr_list = [0.001] 



hyperparameter_combinations = list(itertools.product(qlist,plist,embedding_dim_list, walk_length_list,context_size_list, walks_per_node_list, num_negative_samples_list,epoch_list,lr_list))


# In[6]:


def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def test():
    model.eval()
    z = model()
    acc = model.test(z[data.train_mask], data.y[data.train_mask],z[data.val_mask], data.y[data.val_mask], max_iter=150) # 使用train_mask训练一个分类器，用test_mask分类
    return acc



for q,p,embedding_dim, walk_length, context_size,walks_per_node, num_negative_samples,epoch,lr in hyperparameter_combinations:
    #条件
    folder_path = "../user_data"  # 替换为实际的文件夹路径
    prefix = f'node2vec_{embedding_dim}_{walk_length}_{context_size}_{walks_per_node}_{epoch}_{lr}_'  # 替换为实际的前缀
    print(prefix)
    exists = check_custom_start_file_exists(folder_path, prefix)
    if exists:
        print(f"存在以'{prefix}'开头的文件")
        continue
    
    
    print(device)
    model = Node2Vec(data.edge_index, embedding_dim=embedding_dim, walk_length=walk_length,
                 context_size=context_size, walks_per_node=walks_per_node, p=p,q=q,num_negative_samples=num_negative_samples).to(device)
    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    
    data = data.to(device)
    model.to(device).reset_parameters()

    optimizer = torch.optim.Adam (
        model.parameters(), 
        lr=lr
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    

    for epoch in (range(1, epoch+1)):
        try:
            loss = train()
        except:
            continue
        if(epoch%10==0):
            print(f'Epoch:{epoch:02d}, Loss: {loss:.4f}')
        
    model.eval()
    result = model().cpu().detach().numpy()
    acc = test()
    name = f'{embedding_dim}_{walk_length}_{context_size}_{walks_per_node}_{epoch}_{lr}_{acc}'
    save_pkl(f'../user_data/node2vec_{name}.pkl',result)





