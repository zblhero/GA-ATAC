import torch

import community
import networkx as nx
import os, os.path

import numpy as np
from matplotlib import pyplot as plt
import leidenalg
import igraph as ig
import umap

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


def batch_removal(X, batchids):
    for b in [0, 1, 2]:
        indices = [i for i in range(X.shape[0]) if int(batchids[i])==b]
        T = X[indices][:]
        
        mean, std = np.mean(X[indices]), np.std(X[indices])        
        X[indices] = (X[indices]-mean)/std
    return X
        


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
    return sum([reward_matrix[i, j] for i, j in ind]) * 1.0 / y_pred.size

def show_tsne(embedding, labels, filename, tlabels=None):
    n_components = len(np.unique(labels))
    
    vis_x = embedding[:, 0]
    vis_y = embedding[:, 1]
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'yellow', 'black', 'teal', 'plum', 'tan', 'bisque', 'beige', 'slategray', 'brown', 'darkred', 'salmon', 'coral', 'olive', 'lightpink', 'teal', 'darkcyan', 'BlueViolet', 'CornflowerBlue', 'DarkKhaki', 'DarkTurquoise']

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    for i, y in enumerate(range(n_components)):

        indexes = [j for j in range(len(labels)) if labels[j]==y]
        vis_x1 = embedding[indexes, 0]
        vis_y1 = embedding[indexes, 1]
        try: 
            c = colors[i]
        except IndexError:
            c = colors[0]

        if tlabels is None:
            sc = plt.scatter(vis_x1, vis_y1, c=c, marker='.', cmap='hsv', label=y)
        else:
            sc = plt.scatter(vis_x1, vis_y1, c=c, marker='.', cmap='hsv', label=tlabels[indexes[0]])
        
    ax.legend()
    #plt.clim(-0.5, 9.5)
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

def clustering_scores(args, latent, labels, cells, dataset, suffix, tlabels, louvain_num=15, prediction_algorithm="knn", X_tf=None, ensemble=False, batch_indices=None, save_cluster=False, seed=42):
    from scipy.spatial import distance


    vec = latent
    mat = kneighbors_graph(latent, louvain_num, mode='distance', include_self=True).todense()
    print('mat', mat.shape)

    
    
    alg = 'louvain'
    if alg == 'louvain':
        labels_pred = []
        G = nx.from_numpy_matrix(mat)
        partition = community.best_partition(G, random_state=seed)
        for i in range(mat.shape[0]):
            labels_pred.append(partition[i])
    elif alg == 'leiden':
        vcount = max(mat.shape)
        sources, targets = mat.nonzero()
        edgelist = zip(sources.tolist(), targets.tolist())
        g = ig.Graph(vcount, edgelist)
        partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
        print(partition.membership)
        
        labels_pred = partition.membership

    labels_pred = np.array(labels_pred)

    if args.plot == 'tsne':
        embedding = TSNE(random_state=seed, perplexity=50).fit_transform(vec)  
    elif args.plot == 'umap':
        embedding = umap.UMAP(random_state=42).fit_transform(vec)  
        
        
    print('pred labels is', labels_pred.shape, np.max(labels_pred), vec[0,:5], embedding[:5], labels_pred[:100])
    print('labels is', np.array(labels).shape)
    show_tsne(embedding, labels_pred, 'result/%s/%s-GMVAE-%s-%s-pred.png'%(dataset, suffix, 'alpha-gan', ensemble))
    np.savetxt('result/%s/labels.txt'%(dataset), labels_pred)

    #if labels is not None:   
    if len(labels) == 0:
        with open('result/%s-cluster_result.csv'%(dataset), 'w') as f:
            f.write('cell,predicted label,tsne-1,tsne-2\n')
            for cell, pred, t in zip(cells, labels_pred, embedding):
                f.write('%s,%d,%f,%f\n'%(cell, pred, t[0], t[1]))
        if batch_indices is not None:
            print('batch', batch_indices)
            show_tsne(embedding, batch_indices, 'result/%s/%s-%s-batch.png'%(dataset, suffix, 'alpha-gan'), tlabels=batch_indices)
    else:
        show_tsne(embedding, labels, 'result/%s/%s-GMVAE-%s-%s-true.png'%(dataset, suffix, 'alpha-gan', ensemble), tlabels=tlabels)
        if batch_indices is None:
            with open('result/%s-cluster_result.csv'%(dataset), 'w') as f:
                f.write('cell,tlabel id,label,predicted label,tsne-1,tsne-2\n')
                for cell, label, tlabel, pred, t in zip(cells, labels, tlabels, labels_pred, embedding):
                    f.write('%s,%d,%s,%d,%f,%f\n'%(cell, label, tlabel, pred, t[0], t[1]))
        else:
            with open('result/%s-cluster_result.csv'%(dataset), 'w') as f:
                f.write('cell,tlabel id,label,predicted label,tsne-1,tsne-2,batch\n')
                for cell, label, tlabel, pred, t, batch in zip(cells, labels, tlabels, labels_pred, embedding, batch_indices):
                    f.write('%s,%d,%s,%d,%f,%f,%d\n'%(cell, label, tlabel, pred, t[0], t[1], batch))

        #print(labels, labels_pred, latent)
        #asw_score = silhouette_score(latent, labels)
        asw_score = 0
        nmi_score = NMI(labels, labels_pred)
        ari_score = ARI(labels, labels_pred)
        homo_score = homogeneity_score(labels, labels_pred) 
        #uca_score = unsupervised_clustering_accuracy(labels, labels_pred)
        print("Clustering Scores:\nHOMO: %.4f\nNMI: %.4f\nARI: %.4f\nUCA: %.4f"%(homo_score, nmi_score, ari_score, 0))

        if batch_indices is not None:
            show_tsne(embedding, batch_indices, 'result/%s/%s-%s-batch.png'%(dataset, suffix, 'alpha-gan'), tlabels=batch_indices)
        return asw_score, nmi_score, ari_score, 0



