#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import dropout_adj

from tqdm import tqdm

from torch_geometric.datasets import Planetoid, CitationFull
import torch_geometric.transforms as T

from torch_geometric.nn import SAGEConv, GCNConv

import numpy as np
import functools


from GCL.eval import get_split, LREvaluator

from pathlib import Path


# In[2]:


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


# In[3]:


def train(x, edge_index, topo_embedding, lam, batch_size,
          model, optimizer, drop_edge_rate_1, drop_edge_rate_2,
          drop_feature_rate_1, drop_feature_rate_2, data):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    z1, z2 = model(x_1, edge_index_1, x_2, edge_index_2)
    if batch_size == 0:
        feat_cor = data.x @ data.x.t()
        topo_cor = topo_embedding @ topo_embedding.t()
        loss = model.loss(z1, z2, feat_cor, topo_cor, lam)
    else:
        loss = model.batch_loss(z1, z2, data.x, topo_embedding, lam, batch_size)
    
    loss.backward()
    optimizer.step()

    return loss.item()


# In[4]:


def test(model, x, edge_index, y, final=False):
    model.eval()
    z = model.get_embedding(x, edge_index)
    result = label_classification(z, y, ratio=0.1)
    #print(result)
    return result


# In[5]:


def pyg_test(encoder_model, x, edge_index, y, test_times = 10):
    encoder_model.eval()
    z = encoder_model.get_embedding(x, edge_index) #, data.edge_attr
    mi = []
    ma = []
    for i in range(test_times):
        split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
        result = LREvaluator()(z, y, split)
        mi.append(result['micro_f1'])
        ma.append(result['macro_f1'])
    print(
        'Mi_mean: ',str( torch.tensor(mi).mean().item() ),
        ' Mi_std: ',str( torch.tensor(mi).std().item() ),
        ' Ma_mean: ',str( torch.tensor(ma).mean().item() ),
        ' Ma_std: ',str( torch.tensor(ma).std().item() ),
    )
    
    return torch.tensor(mi).mean().item()


# In[6]:


from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')

from Dataset_Load import load_dataset
from Model import DC

from Test import label_classification

from Topo_embedding import get_GraRep_topo_embedding, get_PPR_topo_embedding

import json

import time

import random

#env_load
def parameters_test(config):
    
    torch.manual_seed(config['seed'])
    random.seed(12345)
    
    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]
    num_layers = config['num_layers']
    batch_size = config['batch_size']

    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']

    topo_demension = config['topo_demension']
    topo_order = config['topo_order']
    topo_iter = config['topo_iter']
    topo_seed = config['topo_seed']
    topo_lam = config['topo_lam']
    topo_self_loop = config['topo_self_loop']
    topo_alpha = config['topo_alpha']   

    dataset_name = config['dataset_name']
    dataset_path = config['dataset_dir']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    #begin
    dataset = load_dataset(dataset_name, dataset_path)
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    model = DC(dataset.num_features, num_hidden, num_layers).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    
    if Path(config['topo_dir']+config['dataset_name']+config['topo_method']+'.pt').is_file():
        topo_embedding = torch.load(config['topo_dir']+config['dataset_name']+config['topo_method']+'.pt')
    else:
        print('Culculate Topo_Embedding')

        if config['topo_method'] == 'GraRep':
            topo_embedding = get_GraRep_topo_embedding(data.clone().to(torch.device('cpu')),
                                                       dimensions = topo_demension, 
                                                       iterations = topo_iter, 
                                                       order = topo_order, 
                                                       seed = topo_seed).to(device)
        elif config['topo_method'] == 'PPR':
            topo_embedding = get_PPR_topo_embedding(data.clone().to(torch.device('cpu')),
                                                       demensions = topo_demension,
                                                       self_loop = topo_self_loop,
                                                       iterations = topo_iter,
                                                       alpha = topo_alpha).to(device)
        topo_embedding = F.normalize(topo_embedding)
        torch.save(topo_embedding,
                  config['topo_dir']+config['dataset_name']+config['topo_method']+'.pt'
                  )
        

    for epoch in tqdm(range(1, num_epochs + 1)):
        loss = train(data.x, data.edge_index, topo_embedding, topo_lam, batch_size,
                     model, optimizer, drop_edge_rate_1, drop_edge_rate_2, 
                     drop_feature_rate_1, drop_feature_rate_2, data)

        # print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f} ')

    print("=== Final ===")
    if config['test'] == 'GRACE_TEST':
        return test(model, data.x, data.edge_index, data.y, final=True)
    elif config['test'] == 'PYG_TEST':
        return pyg_test(model, data.x, data.edge_index, data.y)


# In[7]:


config = {
    'dataset_name': 'CiteSeer',
    'dataset_dir': './datasets',
    'batch_size': 0,
    
    'learning_rate': 0.001,
    'num_hidden': 256,
    'num_proj_hidden': 256,
    'activation': 'relu',
    'base_model': 'GCNConv',
    'num_layers': 2,
    'drop_edge_rate_1': 0.2,
    'drop_edge_rate_2': 0.0,
    'drop_feature_rate_1': 0.3,
    'drop_feature_rate_2': 0.2,
    'num_epochs': 200,
    'weight_decay': 0.00001,
    
    'topo_method':'GraRep',
    'topo_dir':'./topo_embedding/',
    #PPR
    #'topo_demension': 50, #shared
    'topo_self_loop': True,
    'topo_alpha':0.2, 
    #'topo_iter':20, #shared
    
    #GraRep
    'topo_demension': 16,#shared
    'topo_order': 5,
    'topo_iter': 20,#shared
    'topo_seed': 42,
    
    
    'topo_lam': 0.005,
    
    'log_root': './log/',
    
    'test': 'GRACE_TEST',
    
    'seed': 38108
}


# In[8]:

print('#######################1######################')
for topo_lam in [0.005]:
    for num_epochs in [500]:
        for weight_decay in [0.00005]:
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
            config['topo_lam'] = topo_lam
            config['num_epochs'] = num_epochs            
            config['weight_decay'] = weight_decay
            
            print(json.dumps(config, indent=4, ensure_ascii=False))
            
            for i in range(10):
                result = parameters_test(config)
                
                with open(config['log_root']+config['dataset_name']+'.txt', 'a') as f:
                    if i == 0:
                        f.write('\n')
                        f.write(str(config.items()) + '\n')
                    
                    f.write(str(result.items()) + '\n')
                    
                    
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
'''
            with open(config['log_root']+config['dataset_name']+'.txt', 'a') as f:
                f.write('Lam: ' + str(topo_lam)+
                        ' E: ' + str(num_epochs)+
                        ' WD: ' + str(weight_decay)+
                        
                        ' Result: '+str(result) + '\n')
'''


