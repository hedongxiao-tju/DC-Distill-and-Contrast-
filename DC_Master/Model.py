#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, GCNConv

# In[4]:


class DC(nn.Module):
    '''
    Model Distill & Contrast for Graph Representation Learning
    '''
    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers - 1):
            in_channels = in_channels if i == 0 else hidden_channels*2
            self.convs.append(GCNConv(in_channels, hidden_channels*2))
        self.convs.append(GCNConv(hidden_channels*2, hidden_channels))
        
        self.act = nn.ReLU()
        
    def encode(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.act(self.convs[i](x, edge_index))
            
        return x
    
    def forward(self, x_a, edge_a, x_b, edge_b):
        
        z_a = self.encode(x_a, edge_a)
        z_b = self.encode(x_b, edge_b)
        
        return z_a, z_b
    
    def loss(self, z_a, z_b, fea_target, topo_target, lam):
        sim_matrix = z_a @ z_b.t()
        loss_1 = (sim_matrix - fea_target).pow(2).sum()
        loss_2 = (sim_matrix - topo_target).pow(2).sum()
        #print(loss_1.item(), (lam*loss_2).item(), (loss_1 + lam*loss_2).item())
        return loss_1 + lam*loss_2
        
        
    def batch_loss(self, z_a, z_b, fea, topo, lam, batch_size):
        node_perturb = torch.arange(0, z_a.shape[0])
        num_batch = z_a.shape[0] // batch_size + 1
        loss = []
        
        for i in range(num_batch):
            mask = node_perturb[i*batch_size: (i+1)*batch_size]
            fea_tar = fea[mask] @ fea[mask].t()
            topo_tar = topo[mask] @ topo[mask].t()
            sim_matrix = z_a[mask] @ z_b[mask].t()
            
            loss_1 = (sim_matrix - fea_tar).pow(2).sum()
            loss_2 = (sim_matrix - topo_tar).pow(2).sum()
            
            loss.append((loss_1 + lam*loss_2).unsqueeze(0))
        #print(loss)
        
        return torch.cat(loss).mean()
        
    def get_embedding(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.act(self.convs[i](x, edge_index))
            
        return x


# In[ ]:




