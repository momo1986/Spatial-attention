#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn


# In[ ]:

def result_show():
    print('rmse reslt\n')
    true_tra=np.load(r'./result/true_tra.npy')
    base_pre=np.load(r'./result/seq2seq_beam_search_predict_tra.npy')
    true_tra=true_tra.transpose(1,0,2)
    true_tra=true_tra[:,:28672,:]
    print(true_tra.shape)
    print(base_pre.shape)

    true=torch.from_numpy(true_tra).float()
    base=torch.from_numpy(base_pre).float()

    cre=nn.MSELoss()

    xloss=np.zeros(5)
    yloss=np.zeros(5)





    for i in range(5):
        xloss[i]=cre(base[:5*(i+1),:,0],true[:5*(i+1),:,0])
        yloss[i]=cre(base[:5*(i+1),:,1],true[:5*(i+1),:,1])
    loss=np.sqrt(xloss+yloss)
    longitudinal_loss=np.sqrt(xloss)
    lateral_loss=np.sqrt(yloss)
    print('base line\n')
    print('position loss as 1 seconds increment')
    print(loss)
    print('\n')
    print('longitudinal position loss')
    
    print(longitudinal_loss)
    print('\n')
    print('lateral position loss')
    print(lateral_loss)

