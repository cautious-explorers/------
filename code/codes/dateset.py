import pandas as pd
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import warnings
warnings.filterwarnings("ignore", category=Warning)
from torch_geometric.datasets import TUDataset
from codes.utils import *

#https://pytorch-geometric.readthedocs.io/en/2.3.1/tutorial/create_dataset.html
#适合小规模:InMemoryDataset
#大规模
class SocialDataset(InMemoryDataset):
    '''
    1. raw_file_names 没有就要去下载
    2. 查找processed_file_names是否有处理好的
    3. 没有处理好的调用process函数
    
    '''
    
    #tranform数据增强方法
    def __init__(self, root, config,transform=None, pre_transform=None):
        self.data_file = config['source_data_path']
        self.feature = config['feature']
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0]) #load作为加载项
        self.get_split(config)
        self.get_basic_info()
        
    @property
    def raw_file_names(self):
        # 返回原始文件的文件名列表
        file_list = []
        for i in ['edges.csv', 'train_with_centrality.csv', 'test_with_centrality.csv','centrality.pkl']:
            file_list.append( self.data_file + i )
        return file_list
    
    @property
    def processed_file_names(self):
        return ['data.pt']  #会默认加上root作为前缀
    
    def download(self):
        pass
    
    def get_split(self,config):
        train_nodes = int((self.data.y>=0).sum())
        index_list = torch.randperm(train_nodes)
        self.data.val_mask = index_list[: int(train_nodes*(config['valsize']))   ]
        self.data.train_mask = index_list[int(train_nodes*(config['valsize']))  : ]
        self.data.test_mask =  torch.range(train_nodes , self.data.y.shape[0]-1, dtype=torch.long)
        self.data.out = pd.read_csv( self.raw_file_names[2])
        
        class_counts = torch.bincount(self.data.y[index_list])
        self.data.weights = 1.0 / class_counts.float()

    
    def get_basic_info(self):
        
        data = self.data
        print(f'Number of nodes: {data.num_nodes}')
        print(f'Number of edges: {data.num_edges}')
        print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
        # print(f'Number of training nodes: {data.train_mask.sum()}')
        # print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
        print(f'Has isolated nodes: {data.has_isolated_nodes()}')
        print(f'Has self-loops: {data.has_self_loops()}')
        print(f'Is undirected: {data.is_undirected()}')
        

    def process(self):

        links = pd.read_csv(self.raw_file_names[0])
        labels = pd.read_csv( self.raw_file_names[1])\
                .append( pd.read_csv(self.raw_file_names[2]))
        
        
        ids = list(labels['id'])
        id_map_dict = dict(zip(ids, range(len(ids))))
        id_map_verse_dict = dict(zip(ids, range(len(ids))))
        links['index_1'] =links['id_1'].apply(lambda x: id_map_dict[x])
        links['index_2'] =links['id_2'].apply(lambda x: id_map_dict[x])
        labels['index'] =labels['id'].apply(lambda x: id_map_dict[x])
        labels.fillna(-1,inplace=True)
        
        #edges,这里需要设置为双向边（无向边）
        link_left_list = list(links['index_1'])+list(links['index_2'])
        link_right_list = list(links['index_2'])+list(links['index_1'])
        Edge_index = torch.tensor([link_left_list,
                                   link_right_list], dtype=torch.long)
        
        #print(labels.head(10))
        labels.sort_values(['index'],inplace=True)
        
        #节点特征没有：可以有，也可以没有
        if(self.feature is None):
            labels['A'] = 1
            columns = ['A']
            Xfeatures = torch.tensor((np.array(labels[columns])), dtype=torch.float)
        elif(self.feature=='node2vec'):
            node2vec_matrix = load_pkl('/storage/data/yangxsh/network_graph/data/node2vec/node2vec_64_32_8_4_1_399_0.001_0.8617328519855596.pkl')
            Xfeatures = torch.tensor((node2vec_matrix), dtype=torch.float)
        else:
            columns = ['B','C','D','E']
            Xfeatures = torch.tensor((np.array(labels[columns])), dtype=torch.float)
        
        #节点标签从0开始
        Y = torch.tensor(list(labels['target']),dtype=torch.long)
        
        data = Data(x=Xfeatures,edge_index=Edge_index, y=Y)  #可以放节点和边的数量，链接情况和labels
        data_list = [data] #一张图
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            
        data, slices = self.collate(data_list)  #将list划分为不同的slices去保存数据
        torch.save((data, slices), self.processed_paths[0]) 