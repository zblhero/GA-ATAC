
import logging
import sys
import time
import copy

from abc import abstractmethod
from collections import defaultdict, OrderedDict
from itertools import cycle, chain

import numpy as np
import torch

from sklearn.model_selection._split import _validate_shuffle_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torch.optim import RMSprop, Adam, SGD
from tqdm import trange
from utils import *


def loss(model, kl_weight, sample_batch, local_l_mean, local_l_var, batch_index, t):  # tensors is local data
    loss = model(sample_batch, local_l_mean, local_l_var, batch_index)
        
    reconst_loss, kl_divergence, g_loss, d_loss, z_rec_loss = loss
    return torch.mean(reconst_loss + kl_weight * kl_divergence), g_loss, d_loss, z_rec_loss



def train(model, gene_dataset, epoch_num=1000, eps=0.01, weight_decay=1e-6, seed=42):
    model.train()
    n_epochs_kl_warmup = 400
    
    random_state = np.random.RandomState(seed=seed)
    permutation = random_state.permutation(len(gene_dataset))
    indices_train = permutation[0 : len(gene_dataset)]
            
    data_loader_kwargs = {'X': np.float32, 'local_means': np.float32, 'local_vars': np.float32, 'batch_indices': np.int64, 'labels': np.int64}
            
    data_loader_kwargs = copy.copy(data_loader_kwargs)
    sampler = SequentialSampler(gene_dataset)
    data_loader_kwargs.update(
        {"collate_fn": gene_dataset.collate_fn_builder(), "sampler": sampler}
    )
    train_set = DataLoader(gene_dataset, **data_loader_kwargs)
    
    
    
    optimizer_eg = torch.optim.Adam(chain(model.z_encoder.parameters(), 
                                          model.l_encoder.parameters(),
                                          model.decoder.parameters()), 
                                            lr=1e-4, eps=eps, weight_decay=weight_decay)
    optimizer_d = torch.optim.Adam(model.discriminator.parameters(), lr=1e-4, eps=eps, weight_decay=weight_decay)
    
    for epoch in range(epoch_num):
        print('epoch', epoch)
        kl_weight = min(1, epoch/n_epochs_kl_warmup)
        
        
        for tensors_list in train_set:
            for i in range(len(tensors_list)):
                tensors_list[i] = tensors_list[i].cuda()
                
            loss = loss(model, kl_weight, *tensors_list)
            
            reconst_loss, g_loss, d_loss, z_rec_loss = loss
                        
            if self.epoch > 0:
                for _ in range(1):
                    # autoencoder loss, freeze D, update E and G
                    self.model.z_encoder.requires_grad = True
                    self.model.l_encoder.requires_grad = True
                    self.model.decoder.requires_grad = True
                    self.model.discriminator.requires_grad = False
                    optimizer_eg.zero_grad()
                    (reconst_loss+g_loss+z_rec_loss).backward(retain_graph=True)   # decoder to reconstruct
                    optimizer_eg.step()
                        
            if self.epoch > 0:
                for _ in range(1):
                    # discriminator loss, freeze E and G, update D
                    self.model.z_encoder.requires_grad = False
                    self.model.l_encoder.requires_grad = False
                    self.model.decoder.requires_grad = False
                    self.model.discriminator.requires_grad = True
                    optimizer_d.zero_grad()
                    (d_loss).backward()    # discriminate between fake and real
                    optimizer_d.step()
    model.eval()