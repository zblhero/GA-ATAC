
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






class Trainer:
    default_metrics_to_monitor = []

    def __init__(
        self,
        model,
        gene_dataset,
        use_cuda=True,
        metrics_to_monitor=None,
        benchmark=False,
        frequency=None,
        weight_decay=1e-6,
        early_stopping_kwargs=None,
        data_loader_kwargs=None,
        show_progbar=True,
        seed=0,
        reconstruction_loss='zinb'
    ):
        # handle mutable defaults
        early_stopping_kwargs = (
            early_stopping_kwargs if early_stopping_kwargs else dict()
        )
        data_loader_kwargs = data_loader_kwargs if data_loader_kwargs else dict()

        self.model = model
        self.gene_dataset = gene_dataset
        self._posteriors = OrderedDict()
        self.seed = seed

        self.data_loader_kwargs = {"batch_size": 128, "pin_memory": use_cuda}
        self.data_loader_kwargs.update(data_loader_kwargs)

        self.weight_decay = weight_decay
        self.benchmark = benchmark
        self.epoch = -1  # epoch = self.epoch + 1 in compute metrics
        self.training_time = 0
        self.reconstruction_loss = reconstruction_loss

        if metrics_to_monitor is not None:
            self.metrics_to_monitor = set(metrics_to_monitor)
        else:
            self.metrics_to_monitor = set(self.default_metrics_to_monitor)


        
        self.use_cuda = use_cuda and torch.cuda.is_available()
        if self.use_cuda:
            self.model.cuda()

        self.frequency = frequency if not benchmark else None

        self.history = defaultdict(list)

        self.best_state_dict = self.model.state_dict()
        self.best_epoch = self.epoch

        self.show_progbar = show_progbar
        
    def data_loaders_loop(
        self
    ):  # returns an zipped iterable corresponding to loss signature
        data_loaders_loop = [self._posteriors[name] for name in ["train_set"]]
        return zip(
            data_loaders_loop[0],
            *[cycle(data_loader) for data_loader in data_loaders_loop[1:]]
        )
    
    def register_posterior(self, name, value):
        name = name.strip("_")
        self._posteriors[name] = value

    def train(self, n_epochs=20, lr=1e-4, eps=0.01, decay_lr=0.75, params=None):
        begin = time.time()
        self.model.train()
        
        optimizer_eg = torch.optim.Adam(chain(self.model.z_encoder.parameters(), 
                                                                   self.model.l_encoder.parameters(),
                                                                  self.model.decoder.parameters()), 
                                                             lr=1e-4, eps=eps, weight_decay=self.weight_decay)
        optimizer_d = torch.optim.Adam(self.model.discriminator.parameters(), lr=1e-4, eps=eps, weight_decay=self.weight_decay)

        self.compute_metrics_time = 0
        self.n_epochs = n_epochs
        #self.compute_metrics()

        with trange(
            n_epochs, desc="training", file=sys.stdout, disable=not self.show_progbar
        ) as pbar:
            for self.epoch in pbar:
                self.on_epoch_begin()
                pbar.update(1)
                #for tensors_list in self.data_loaders_loop():
                for tensors_list in self.train_set:
                    if self.use_cuda:
                        for i in range(len(tensors_list)):
                            tensors_list[i] = tensors_list[i].cuda() 
                    if tensors_list[0][0].shape[0] < 3:
                        continue
                    loss = self.loss(*tensors_list)
                    
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
                        
                    

                if not self.on_epoch_end():
                    break


        self.model.eval()
        self.training_time += (time.time() - begin) - self.compute_metrics_time

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        return True
    
    
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
            
            data_loader_kwargs = {'X': np.float32, 'local_means': np.float32, 'local_vars': np.float32, 'batch_indices': np.int64, 'labels': np.int64}
            
            self.data_loader_kwargs = copy.copy(self.data_loader_kwargs)
            sampler = SequentialSampler(gene_dataset)
            self.data_loader_kwargs.update(
                {"collate_fn": gene_dataset.collate_fn_builder(), "sampler": sampler}
            )
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
        loss = self.model(sample_batch, local_l_mean, local_l_var, batch_index)
        
        reconst_loss, kl_divergence, g_loss, d_loss, z_rec_loss = loss
        return torch.mean(reconst_loss + self.kl_weight * kl_divergence), g_loss, d_loss, z_rec_loss
        

    def on_epoch_begin(self):
        if self.n_epochs_kl_warmup is not None:
            self.kl_weight = min(1, self.epoch / self.n_epochs_kl_warmup)
        else:
            self.kl_weight = 1.0
            #self.kl_weight = 0
        
        self.gan_weight = 1.0

