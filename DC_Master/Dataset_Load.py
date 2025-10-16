#!/usr/bin/env python
# coding: utf-8

# In[17]:


from torch_geometric.datasets import Planetoid, CitationFull, Amazon, Coauthor
from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
import torch_geometric.transforms as T

def load_dataset(dataset_name, dataset_dir):

    print('Dataloader: Loading Dataset', dataset_name)
    assert dataset_name in ['Cora', 'CiteSeer', 'PubMed',
                            'dblp', 'Photo','Computers', 
                            'CS','Physics', 
                            'ogbn-products', 'ogbn-arxiv']
    
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(dataset_dir, name=dataset_name, 
                            transform=T.NormalizeFeatures())
        
    elif dataset_name == 'dblp':
        dataset = CitationFull(dataset_dir, name=dataset_name, 
                               transform=T.NormalizeFeatures()
                              )
        
    elif dataset_name in ['Photo','Computers']:
        dataset = Amazon(dataset_dir, name=dataset_name, 
                         transform=T.NormalizeFeatures())
        
    elif dataset_name in ['CS','Physics']:
        dataset = Coauthor(dataset_dir, name=dataset_name, 
                           transform=T.NormalizeFeatures())
        
    elif dataset_name in ['ogbn-products', 'ogbn-arxiv']:
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=dataset_dir,
                                         transform=T.NormalizeFeatures())
    #print(dataset)
    print('Dataloader: Loading success.')
    print(dataset[0])
    
    return dataset


# In[ ]:




