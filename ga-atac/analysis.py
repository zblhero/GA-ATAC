import torch

import community
import networkx as nx
import os, os.path

import numpy as np
from matplotlib import pyplot as plt
import leidenalg
import igraph as ig
import pandas as pd

from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from scipy.optimize import linear_sum_assignment as linear_assignment
#from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import TSNE

from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import silhouette_score, homogeneity_score

from torch.utils.data import DataLoader
from torch.utils.data.sampler import (
    SequentialSampler,
    SubsetRandomSampler,
    RandomSampler,
)

use_cuda = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor
LONG = torch.cuda.LongTensor



def analysis():
    df_gan = pd.read_csv('result/pbmc_two_batch-cluster_result.csv', delimiter=',')
    df_lsi = pd.read_csv('result/pbmc_two_batch-LSI-cluster_result.csv', delimiter=',')
    
    labels = df_lsi['predicted label'].values
    labels_pred = df_gan['predicted label'].values
    
    nmi_score = NMI(labels, labels_pred)
    ari_score = ARI(labels, labels_pred)
    print("Clustering Scores:\nNMI: %.4f\nARI: %.4f\n"% (nmi_score, ari_score))
    
analysis()