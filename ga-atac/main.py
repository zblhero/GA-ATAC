
import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from trainer import UnsupervisedTrainer
#from train import train
from datasetfilter import SingleCellDataset

import torch
from scDataset import SCDataset
from model import *
import pickle
import random
import torchsummary
from torchvision.models.resnet import *

from utils import *
import extract
import argparse

save_path = 'models/'

plt.switch_backend("agg")




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    
def cluster(dataset, n_hidden, n_latent, louvain_num, ratio=0.1, seed=6, min_peaks=100, min_cells=0.05, max_cells=0.95, n_epochs=1000, 
            is_labeled=True, gpu=-1):
    set_seed(seed)
    use_batches = False
    
    if gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%(gpu)
        use_cuda = True if torch.cuda.is_available() else False
    else:
        use_cuda = False
    
    X, cells, peaks, labels, cell_types, tlabels, = extract.extract_simulated(dataset, is_labeled)   # downloaded  
    X = np.where(X>0, 1, 0) 
    
    d = SingleCellDataset(X, peaks, cells, low=min_cells, high=max_cells, min_peaks=min_peaks)
    labels = [labels[i] for i in d.barcode if labels is not None]
    tlabels = [tlabels[i] for i in d.barcode if tlabels is not None]
    gene_dataset = SCDataset('models/', mat=d.data, ylabels=labels, tlabels=tlabels, cell_types=cell_types)   
    
    model = GAATAC(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * use_batches, X=gene_dataset.X,
             n_hidden=n_hidden, n_latent=n_latent, dropout_rate=0, reconst_ratio=ratio, use_cuda=use_cuda)
    if use_cuda:
        model.cuda()
    trainer = UnsupervisedTrainer(
        model,
        gene_dataset,
        train_size=1.0,
        use_cuda=use_cuda,
        frequency=5
    )
    model_name = '%s/%s-%d-%d.pkl'%(save_path, dataset, n_hidden, n_latent)
    

    trainer.train(n_epochs=n_epochs)
    #torch.save(trainer.model.state_dict(), model_name)
        
    latent = get_latent(gene_dataset, trainer.model, use_cuda)
    
    print('get clustering', n_hidden, n_latent, louvain_num)
    clustering_scores(latent, cells, labels, dataset, gene_dataset.tlabels, n_hidden, n_latent, louvain_num, seed)
     
    
params = {
    'GSE99172': [512, 10, 50],
    'GSE96769': [1024, 20, 200],
    'GSE112091': [128, 40, 150],
    'forebrain-scale':[1024, 32, 50],
    'GM12878vsHEK': [128, 8, 150],
    'GM12878vsHL': [512, 24, 150],
    'Splenocyte':[512, 16, 50],
    'atac_pbmc_1k_merge': [128, 16, 50],
    'scChip-seq': [128, 16, 50],
    'scRNA_cortex': [64, 14, 50],
    'ZY_bin_cell_matrix': [128, 16, 50]
}
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GA-ATAC: Generative Adversarial ATAC-seq Analysis')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--n_hidden', type=int, help='hidden unit number', default=128)
    parser.add_argument('--n_latent', type=int, help='latent size', default=16)
    parser.add_argument('--n_louvain', type=int, help='louvain number', default=30)
    parser.add_argument('--seed', type=int, default=6, help='Random seed for repeat results')
    parser.add_argument('--gpu', default=0, type=int, help='Select gpu device number when training')
    parser.add_argument('--min_peaks', type=float, default=100, help='Remove low quality cells with few peaks')
    parser.add_argument('--min_cells', type=float, default=0.05, help='Remove low quality peaks')
    parser.add_argument('--max_cells', type=float, default=0.95, help='Remove low quality peaks')
    parser.add_argument('--n_epochs', type=int, default=1000, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--labeled', type=int, default=1, help='has label data (cell type file)') # 1 stands for existing of celltype file
    

    args = parser.parse_args()    
    cluster(args.dataset, args.n_hidden, args.n_latent, args.n_louvain, 
            seed=args.seed, min_peaks=args.min_peaks, min_cells=args.min_cells, max_cells=args.max_cells, n_epochs=args.n_epochs,
            is_labeled=args.labeled, gpu=args.gpu)
    