import copy

import matplotlib.pyplot as plt
import torch
import logging
import sys
import time

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

#from posterior import Posterior
from trainer import Trainer
from utils import *


plt.switch_backend("agg")




class UnsupervisedTrainer(Trainer):
    default_metrics_to_monitor = ["elbo"]

    def __init__(
        self,
        model,
        gene_dataset,
        train_size=0.8,
        test_size=None,
        n_epochs_kl_warmup=400,
        **kwargs
    ):
        super().__init__(model, gene_dataset, **kwargs)
        self.n_epochs_kl_warmup = n_epochs_kl_warmup
        if type(self) is UnsupervisedTrainer:
            #self.train_set, self.test_set, self.validation_set = self.train_test_validation(
            #    model, gene_dataset, train_size, test_size
            #)
            random_state = np.random.RandomState(seed=self.seed)
            permutation = random_state.permutation(len(gene_dataset))
            indices_train = permutation[0 : len(gene_dataset)]
            #self.train_set = self.create_posterior(
            #    model, gene_dataset, indices=indices_train, type_class=Posterior
            #)
            '''self.train_set = Posterior(
                model,
                gene_dataset,
                shuffle=False,
                indices=indices_train,
                use_cuda=True,
                data_loader_kwargs=self.data_loader_kwargs,
            )'''
            
            data_loader_kwargs = {'X': np.float32, 'local_means': np.float32, 'local_vars': np.float32, 'batch_indices': np.int64, 'labels': np.int64}
            
            self.data_loader_kwargs = copy.copy(self.data_loader_kwargs)
            sampler = SequentialSampler(gene_dataset)
            self.data_loader_kwargs.update(
                {"collate_fn": gene_dataset.collate_fn_builder(), "sampler": sampler}
            )
            print('post', self.data_loader_kwargs)
            self.train_set = DataLoader(gene_dataset, **self.data_loader_kwargs)
            self.train_set.to_monitor = ["elbo"]
            #self.test_set.to_monitor = ["elbo"]
            #self.validation_set.to_monitor = ["elbo"]
            
    def __iter__(self):
        return map(self.to_cuda, iter(self.train_set))

    def to_cuda(self, tensors):
        return [t.cuda() if self.use_cuda else t for t in tensors]

    @property
    def posteriors_loop(self):
        return ["train_set"]

    def loss(self, sample_batch, local_l_mean, local_l_var, batch_index, t):  # tensors is local data
        #print('loss', self.n_epochs_kl_warmup, self.kl_weight)
        #sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors
        loss = self.model(
            sample_batch, local_l_mean, local_l_var, batch_index
        )
        
        reconst_loss, kl_divergence, g_loss, d_loss, z_rec_loss = loss
        return torch.mean(reconst_loss + self.kl_weight * kl_divergence), g_loss, d_loss, z_rec_loss
        

    def on_epoch_begin(self):
        if self.n_epochs_kl_warmup is not None:
            self.kl_weight = min(1, self.epoch / self.n_epochs_kl_warmup)
        else:
            self.kl_weight = 1.0
            #self.kl_weight = 0
        
        self.gan_weight = 1.0
        
        










