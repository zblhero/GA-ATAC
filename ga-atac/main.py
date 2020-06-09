
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

save_path = 'models/'

plt.switch_backend("agg")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    
def main(n_hidden, n_latent, dropout, louvain_num, ratio=0.1, seed=42):
    set_seed(seed)
    
    suffix = 'clean' if dataset=='GSE112091' else ''
    if dataset == 'forebrain-scale':
        X, cells, peaks, labels, cell_types, tlabels, = extract.extract_data(dataset)   # downloaded
    else:
        X, cells, peaks, labels, cell_types, tlabels, = extract.extract_simulated(dataset, suffix)   
    X = np.where(X>0, 1, 0)
        
    print(X.shape, np.sum(X), np.sum(X[0]))
        
    use_batches = False
    use_cuda = True
    
    
    filter = True if dataset in ['GSE99172' , 'GSE112091', 'GSE96769', 'human_Occipital', 'atac_pbmc_1k_merge'] else False
    if filter:
        d = SingleCellDataset(X, peaks, cells, low=0.05, high=0.95, min_peaks=100)
        labels = [labels[i] for i in d.barcode if labels is not None]
        tlabels = [tlabels[i] for i in d.barcode if tlabels is not None]
        if dataset == 'atac_pbmc_1k_merge':
            gene_dataset = SCDataset('models/', mat=d.data)
        else:
            gene_dataset = SCDataset('models/', mat=d.data, ylabels=labels, tlabels=tlabels, cell_types=cell_types)
    else:
        gene_dataset = SCDataset('models/', mat=X, ylabels=labels, tlabels=tlabels, cell_types=cell_types)#, batch_ids_file=batch_ids_file)
    
    model = GAATAC(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * use_batches, X=gene_dataset.X,
             n_hidden=n_hidden, n_latent=n_latent, dropout_rate=dropout, reconst_ratio=ratio).cuda()
    trainer = UnsupervisedTrainer(
        model,
        gene_dataset,
        train_size=1.0,
        use_cuda=use_cuda,
        frequency=5
    )
    model_name = '%s/%s.pkl'%(save_path, dataset)
    
    #train(model, gene_dataset)

    if os.path.isfile(model_name):
        trainer.model.load_state_dict(torch.load(model_name))
        trainer.model.eval()
    else:
        trainer.train(n_epochs=1000)
        torch.save(trainer.model.state_dict(), model_name)
        
    latent = get_latent(gene_dataset, trainer.model)
    clustering_scores(latent, labels, cells, dataset, suffix, gene_dataset.tlabels, louvain_num, seed)
     
    
params = {
    'GSE99172': [512, 10, 0, 50],
    'GSE96769': [1024, 32, 0, 30],
    'GSE112091': [128, 40, 0, 150],
    'forebrain-scale':[1024, 32, 0, 50],
    'GM12878vsHEK': [128, 8, 0, 150],
    'GM12878vsHL': [512, 24, 0, 150],
    'Splenocyte':[512, 16, 0, 50],
    'atac_pbmc_1k_merge': [128, 16, 0, 50]
}
    
    
if __name__ == "__main__":
    #dataset = 'GM12878vsHEK'
    #dataset = 'GSE96769'
    #dataset = 'GSE99172'
    #dataset = 'GSE112091'
    dataset = 'forebrain-scale'
    #dataset = 'GM12878vsHEK'
    #dataset = 'GM12878vsHL'
    #dataset = 'Splenocyte'
    #dataset = 'atac_pbmc_1k_merge'
    main(*params[dataset])