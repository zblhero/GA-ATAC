'''import copy
import os
import logging

from typing import List, Optional, Union, Tuple

from sklearn.decomposition import TruncatedSVD

import numpy as np
import pandas as pd
import scipy
import torch
import torch.distributions as distributions
import community
import networkx as nx

from matplotlib import pyplot as plt
from scipy.stats import kde, entropy
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import silhouette_score, homogeneity_score
from sklearn.mixture import GaussianMixture as GMM
from sklearn import cluster
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.utils.linear_assignment_ import linear_assignment
from torch.utils.data import DataLoader
from torch.utils.data.sampler import (
    SequentialSampler,
    SubsetRandomSampler,
    RandomSampler,
)

#from dataset import GeneExpressionDataset
from log_likelihood import (
    compute_elbo,
    compute_reconstruction_error,
    compute_marginal_log_likelihood,
)

logger = logging.getLogger(__name__)


class SequentialSubsetSampler(SubsetRandomSampler):
    def __iter__(self):
        return iter(self.indices)


class Posterior:
    def __init__(
        self,
        model,
        gene_dataset=None,
        shuffle=False,
        indices=None,
        use_cuda=True,
        data_loader_kwargs=dict(),
    ):
        self.model = model
        self.gene_dataset = gene_dataset
        self.to_monitor = []
        self.use_cuda = use_cuda

        if indices is not None and shuffle:
            raise ValueError("indices is mutually exclusive with shuffle")
        if indices is None:
            if shuffle:
                sampler = RandomSampler(gene_dataset)
            else:
                sampler = SequentialSampler(gene_dataset)
        else:
            if hasattr(indices, "dtype") and indices.dtype is np.dtype("bool"):
                indices = np.where(indices)[0].ravel()
            sampler = SubsetRandomSampler(indices)
        self.data_loader_kwargs = copy.copy(data_loader_kwargs)
        self.data_loader_kwargs.update(
            {"collate_fn": gene_dataset.collate_fn_builder(), "sampler": sampler}
        )
        print('post', self.data_loader_kwargs)
        self.data_loader = DataLoader(gene_dataset, **self.data_loader_kwargs)

    def accuracy(self):
        pass

    accuracy.mode = "max"

    @property
    def indices(self):
        if hasattr(self.data_loader.sampler, "indices"):
            return self.data_loader.sampler.indices
        else:
            return np.arange(len(self.gene_dataset))


    def __iter__(self):
        return map(self.to_cuda, iter(self.data_loader))

    def to_cuda(self, tensors):
        return [t.cuda() if self.use_cuda else t for t in tensors]

    def update(self, data_loader_kwargs):
        posterior = copy.copy(self)
        posterior.data_loader_kwargs = copy.copy(self.data_loader_kwargs)
        posterior.data_loader_kwargs.update(data_loader_kwargs)
        posterior.data_loader = DataLoader(
            self.gene_dataset, **posterior.data_loader_kwargs
        )
        return posterior

    def sequential(self, batch_size=128):
        return self.update(
            {
                "batch_size": batch_size,
                "sampler": SequentialSubsetSampler(indices=self.indices),
            }
        )

    @torch.no_grad()
    def elbo(self):
        elbo = compute_elbo(self.model, self)
        logger.debug("ELBO : %.4f" % elbo)
        return elbo

    elbo.mode = "min"

    @torch.no_grad()
    def reconstruction_error(self):
        reconstruction_error = compute_reconstruction_error(self.model, self)
        logger.debug("Reconstruction Error : %.4f" % reconstruction_error)
        return reconstruction_error

    reconstruction_error.mode = "min"


    @torch.no_grad()
    def get_latent(self, sample=False):
        latent = []
        batch_indices = []
        labels = []
        for tensors in self:
            sample_batch, local_l_mean, local_l_var, batch_index, label = tensors
            give_mean = not sample
            latent += [
                self.model.sample_from_posterior_z(
                    sample_batch, give_mean=give_mean
                ).cpu()
            ]
            batch_indices += [batch_index.cpu()]
            labels += [label.cpu()]
        return (
            np.array(torch.cat(latent)),
            np.array(torch.cat(batch_indices)),
            np.array(torch.cat(labels)).ravel(),
        )


    @torch.no_grad()
    def sample_scale_from_batch(self, n_samples, batchid=None, selection=None):
        px_scales = []
        if selection is None:
            raise ValueError("selections should be a list of cell subsets indices")
        else:
            if selection.dtype is np.dtype("bool"):
                selection = np.asarray(np.where(selection)[0].ravel())
        old_loader = self.data_loader
        for i in batchid:
            idx = np.random.choice(
                np.arange(len(self.gene_dataset))[selection], n_samples
            )
            sampler = SubsetRandomSampler(idx)
            self.data_loader_kwargs.update({"sampler": sampler})
            self.data_loader = DataLoader(self.gene_dataset, **self.data_loader_kwargs)
            px_scales.append(self.get_harmonized_scale(i))
        self.data_loader = old_loader
        px_scales = np.concatenate(px_scales)
        return px_scales'''







