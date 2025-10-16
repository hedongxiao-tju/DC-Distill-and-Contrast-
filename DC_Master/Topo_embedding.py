#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Dataset_Load import load_dataset

from GraRep.utils import read_graph
from GraRep.grarep import GraRep
from GraRep.param_parser import parameter_parser

import torch

from scipy.linalg import fractional_matrix_power, inv

from sklearn.decomposition import TruncatedSVD

import torch

# In[3]:

def CiteSeerEx(edge, node_num):
    e_n = torch.unique(edge[0,:]).tolist()
    add_n = []
    for i in range(node_num):
        if i in e_n:
            continue
        else:
            add_n.append(i)
    
    a_n = torch.cat([torch.tensor(add_n).unsqueeze(1),torch.tensor(add_n).unsqueeze(1)], axis=1)
    a_n = torch.cat([a_n.t(),edge], axis=1)
    indices = torch.sort(a_n[0,:],descending=False).indices
    new_edge = a_n[:,indices]
    return new_edge
    
def get_GraRep_topo_embedding_lost_node(data,
                           dimensions = 16, iterations = 20, order = 5, seed = 42                           
                          ):

    args = parameter_parser()
    args.dimensions = dimensions
    args.edge_path = './rubbish/'
    args.iterations = iterations
    args.order = order
    args.output_path = './rubbish/'
    args.seed = seed
    
    new_edge = CiteSeerEx(data.edge_index, data.x.size()[0])

    A = read_graph(new_edge.t().tolist())
    model = GraRep(A, args)
    model.optimize()
    embedding = model.get_embedding()
    embedding = torch.Tensor(embedding.to_numpy()[:,1:])
    #print(embedding.size())
    
    #target = embedding @ embedding.t()
    
    return embedding

    
    
def get_GraRep_topo_embedding(data,
                           dimensions = 16, iterations = 20, order = 5, seed = 42                           
                          ):

    args = parameter_parser()
    args.dimensions = dimensions
    args.edge_path = './rubbish/'
    args.iterations = iterations
    args.order = order
    args.output_path = './rubbish/'
    args.seed = seed

    A = read_graph(data.edge_index.t().tolist())
    model = GraRep(A, args)
    model.optimize()
    embedding = model.get_embedding()
    embedding = torch.Tensor(embedding.to_numpy()[:,1:])
    #print(embedding.size())
    
    #target = embedding @ embedding.t()
    
    return embedding

                           


def get_PPR_topo_embedding(data, demensions, iterations = 20, self_loop=True, alpha=0.2):

    I = torch.eye(data.x.size()[0])
    A = torch.sparse_coo_tensor(data.edge_index, 
                                torch.ones([data.edge_index.size()[1]]),
                                size = [data.x.size()[0],data.x.size()[0]]
                               ).to_dense()
    if self_loop == True:
        A = A+I
    D_INV = torch.tensor(fractional_matrix_power(torch.diag(A.sum(1)), -0.5) )
    A = D_INV @ A @ D_INV
    trans_A = alpha * torch.inverse(I-(1-alpha)*A)

    if demensions != 0:
        del A, D_INV
        trans_A = trans_A.numpy()
        svd = TruncatedSVD(n_components=demensions,
                           n_iter=iterations)
        svd.fit(trans_A)
        embedding = svd.transform(trans_A)
        #print(embedding)
        return torch.from_numpy(embedding)
    else:
        return trans_A
