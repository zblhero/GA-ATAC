import torch

import community
import networkx as nx
import os, os.path

import numpy as np
from matplotlib import pyplot as plt

from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from scipy.optimize import linear_sum_assignment as linear_assignment
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

def _freeze(*args):
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False
def _unfreeze(*args):
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True
                
                
class SequentialSubsetSampler(SubsetRandomSampler):
    def __iter__(self):
        return iter(self.indices)
                
                
def get_latent(gene_dataset, model, use_cuda):
    latent = []
    batch_indices = []
    labels = []
    
    sampler = SequentialSampler(gene_dataset)
    data_loader_kwargs = {"collate_fn": gene_dataset.collate_fn_builder(), "sampler": sampler}
    data_loader = DataLoader(gene_dataset, **data_loader_kwargs)
    
    for tensors in data_loader:
        sample_batch, local_l_mean, local_l_var, batch_index, label = tensors
        if use_cuda:
            latent += [model.sample_from_posterior_z(sample_batch.cuda(), give_mean=True).cpu()]
        else:
            latent += [model.sample_from_posterior_z(sample_batch, give_mean=True).cpu()]
    return np.array(torch.cat(latent).detach())


def clustering_scores(latent, cells, labels, dataset, tlabels, n_hidden, n_latent, louvain_num, seed=42):

    from scipy.spatial import distance
    
    mat = kneighbors_graph(latent, louvain_num, mode='distance', include_self=True).todense()
    G = nx.from_numpy_matrix(mat)
    partition = community.best_partition(G, random_state=seed)

    labels_pred = []
    for i in range(mat.shape[0]):
        labels_pred.append(partition[i])

    labels_pred = np.array(labels_pred)
    
    if not os.path.exists('result/%s'%(dataset)):
        os.mkdir('result/%s'%(dataset))
                    
    if len(labels) > 0:
        
        asw_score = silhouette_score(latent, labels)
        nmi_score = NMI(labels, labels_pred)
        ari_score = ARI(labels, labels_pred)
        homo_score = homogeneity_score(labels, labels_pred) 
        #uca_score = unsupervised_clustering_accuracy(labels, labels_pred)[0]
        uca_score = 0

        print("Clustering Scores:\nSilhouette: %.4f\nNMI: %.4f\nARI: %.4f\nUCA: %.4f\nHOMO:%.4f"
                        % (asw_score, nmi_score, ari_score, uca_score, homo_score))
        
        vec = latent
        tsne = TSNE(random_state=seed).fit_transform(vec)
        show_tsne(tsne, labels_pred, 'result/%s/%s-%d-%d-%d-pred.png'%(dataset, dataset, n_hidden, n_latent, louvain_num), tlabels=tlabels) 
        
        with open('result/%s/%s-%d-%d-%d-cluster_result.csv'%(dataset, dataset, n_hidden, n_latent, louvain_num), 'w') as f:
            f.write('cell,predicted label,tsne-1,tsne-2\n')
            for cell, pred, t in zip(cells, labels_pred, tsne):
                f.write('%s,%d,%f,%f\n'%(cell, pred, t[0], t[1]))
        return asw_score, nmi_score, ari_score, uca_score
    else:
        vec = latent
        tsne = TSNE(random_state=seed).fit_transform(vec)
        show_tsne(tsne, labels_pred, 'result/%s/%s-%d-%d-%d-pred.png'%(dataset, dataset, n_hidden, n_latent, louvain_num), tlabels=None) 
        
        with open('result/%s/%s-%d-%d-%d-cluster_result.csv'%(dataset, dataset, n_hidden, n_latent, louvain_num), 'w') as f:
            f.write('cell,predicted label,tsne-1,tsne-2\n')
            for cell, pred, t in zip(cells, labels, tlabels, labels_pred, tsne):
                f.write('%s,%d,%s,%d,%f,%f\n'%(cell, label, tlabel, pred, t[0], t[1]))


        


def unsupervised_clustering_accuracy(y, y_pred):
    assert len(y_pred) == len(y)
    u = np.unique(np.concatenate((y, y_pred)))
    n_clusters = len(u)
    mapping = dict(zip(u, range(n_clusters)))
    reward_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for y_pred_, y_ in zip(y_pred, y):
        if y_ in mapping:
            reward_matrix[mapping[y_pred_], mapping[y_]] += 1
    cost_matrix = reward_matrix.max() - reward_matrix
    ind = linear_assignment(cost_matrix)
    return sum([reward_matrix[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind

def show_tsne(tsne, labels, filename, tlabels=None):
    n_components = len(np.unique(labels))
    
    vis_x = tsne[:, 0]
    vis_y = tsne[:, 1]
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'yellow', 'black', 'teal', 'plum', 'tan', 'bisque', 'beige', 'slategray', 'brown', 'darkred', 'salmon', 'coral', 'olive', 'lightpink', 'teal', 'darkcyan']

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    for i, y in enumerate(range(n_components)):

        indexes = [j for j in range(len(labels)) if labels[j]==y]
        vis_x1 = tsne[indexes, 0]
        vis_y1 = tsne[indexes, 1]
        c = colors[i]

        print('show_tsne', tlabels)
        if tlabels is None:
            sc = plt.scatter(vis_x1, vis_y1, c=c, marker='.', cmap='hsv', label=y)
        else:
            sc = plt.scatter(vis_x1, vis_y1, c=c, marker='.', cmap='hsv', label=tlabels[indexes[0]])
        
    ax.legend()
    plt.clim(-0.5, 9.5)
    plt.savefig(filename)
    plt.clf()


def one_hot(index, n_cat):
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)

def log_nb_positive(x, mu, theta, eps=1e-8):
    if theta.ndimension() == 1:
        theta = theta.view(
            1, theta.size(0)
        )  # In this case, we reshape theta for broadcasting

    log_theta_mu_eps = torch.log(theta + mu + eps)

    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    return torch.sum(res, dim=-1)



