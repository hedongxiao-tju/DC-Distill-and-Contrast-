#!/usr/bin/env python
# coding: utf-8

# In[1]:

from GraRep.utils import read_graph
from GraRep.grarep import GraRep
from GraRep.param_parser import parameter_parser

import torch

from scipy.linalg import fractional_matrix_power, inv

from sklearn.decomposition import TruncatedSVD

# In[3]:


def get_GraRep_topo_embedding(x,edge_index,
                           dimensions = 16, iterations = 20, order = 5, seed = 42                           
                          ):

    args = parameter_parser()
    args.dimensions = dimensions
    args.edge_path = './rubbish/'
    args.iterations = iterations
    args.order = order
    args.output_path = './rubbish/'
    args.seed = seed
    A = read_graph(edge_index.t().tolist())
    model = GraRep(A, args)
    model.optimize()
    embedding = model.get_embedding()
    embedding = torch.Tensor(embedding.to_numpy()[:,1:])
    #print(embedding.size())
    
    #target = embedding @ embedding.t()
    
    return embedding

def get_PPR_topo_embedding(x, edge_index,demensions, iterations = 20, self_loop=True, alpha=0.2):

    I = torch.eye(x.size()[0])
    A = torch.sparse_coo_tensor(edge_index, 
                                torch.ones([edge_index.size()[1]]),
                                size = [x.size()[0],x.size()[0]]
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
