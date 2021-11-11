
import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD
from pprint import pprint

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn import mixture, cluster
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from datasetfilter import SingleCellDataset
from scDataset import SCDataset

#import lda


import matplotlib as mpl
import matplotlib.pylab as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.switch_backend('agg')
from matplotlib.colors import ListedColormap

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


import community
import networkx as nx
import random

import sys
sys.path.append("..")
import utils, extract



#filtered_dir = '/dataset/user/858f2ba0-e230-11e8-b0aa-fa163ee59f29/1802/simulated_data/'
#filtered_dir = '/dataset/user/858f2ba0-e230-11e8-b0aa-fa163ee59f29/1802/simulated_data2/GSE96772/'
#filtered_dir = '/dataset/user/858f2ba0-e230-11e8-b0aa-fa163ee59f29/1802/simulated_data2/GSE74310/'
#filtered_dir = '/dataset/user/858f2ba0-e230-11e8-b0aa-fa163ee59f29/1802/simulated_data2/GSE99172/'
filtered_dir = '/home/zhangbolei/dataset/1802/simulated_data2/GSE99172/'


    
def clustering(Xsvd, cells, dataset, suffix, labels=None, tlabels=None, method='knn', istsne=True, name='', batch_labels=None, seed=42):
    tsne = TSNE(n_jobs=24).fit_transform(Xsvd)
    
    for n_components in [15]:
        if method == 'gmm':
            clf = mixture.GaussianMixture(n_components= n_components).fit(mat)
            labels_pred = clf.predict(tsne)
        elif method == 'knn':
            labels_pred = KMeans(n_components, n_init=200).fit_predict(tsne)  # n_jobs>1 ?
        elif method == 'dbscan':
            labels_pred = DBSCAN(eps=0.3, min_samples=10).fit(tsne).labels_
        elif method == 'spectral':
            spectral = cluster.SpectralClustering(
                n_clusters=n_components, eigen_solver='arpack',
                affinity="nearest_neighbors")
            labels_pred = spectral.fit_predict(tsne)
        elif method == 'louvain':
            from scipy.spatial import distance
                    
            for louvain in [30]:
                print('****', louvain)
                mat = kneighbors_graph(Xsvd, louvain, mode='distance', include_self=True).todense()
                
                G = nx.from_numpy_matrix(mat)
                partition = community.best_partition(G, random_state=seed)


                labels_pred = []
                for i in range(mat.shape[0]):
                    labels_pred.append(partition[i])

                labels_pred = np.array(labels_pred)
                print('louvain', louvain, tsne[:5], len(labels), len(labels_pred))
                #print(np.unique(labels_pred))

                if labels is not None:
                    nmi_score = NMI(labels, labels_pred)
                    ari_score = ARI(labels, labels_pred)
                    print(n_components, method, "Clustering Scores:\nNMI: %.4f\nARI: %.4f\n"% (nmi_score, ari_score))
    
    if istsne:
        n_components = len(np.unique(labels_pred))
        vis_x = tsne[:, 0]
        vis_y = tsne[:, 1]
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'yellow', 'black', 'teal', 'plum', 'tan', 'bisque', 'beige', 'slategray', 'brown', 'darkred', 'salmon', 'coral', 'olive', 'lightpink', 'teal', 'darkcyan', 'BlueViolet', 'CornflowerBlue', 'DarkKhaki', 'DarkTurquoise']

        show_tsne(tsne, labels, 'result/%s/%s-%s-LSI-true.png'%(dataset, name, suffix), tlabels=tlabels)
        show_tsne(tsne, labels_pred, 'result/%s/%s-%s-LSI-pred.png'%(dataset, name, suffix))
        
        with open('result/%s-LSI-cluster_result.csv'%(dataset), 'w') as f:
            f.write('cell,predicted label,tsne-1,tsne-2\n')
            for cell, pred, t in zip(cells, labels_pred, tsne):
                f.write('%s,%d,%f,%f\n'%(cell, pred, t[0], t[1]))
                
    if batch_labels is not None:
        show_tsne(tsne, batch_labels, 'result/%s/%s-GMVAE-%s-%s-batch.png'%(dataset, dataset, suffix, name))

    
def show_tsne(tsne, labels, filename, tlabels=None):
    
    n_components = len(np.unique(labels))

    #print('tsne', tsne.shape, n_components, labels[:5])
    
    vis_x = tsne[:, 0]
    vis_y = tsne[:, 1]
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'yellow', 'black', 'teal', 'plum', 'tan', 'bisque', 'beige', 'slategray', 'brown']

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    for i, y in enumerate(range(n_components)):

        indexes = [j for j in range(len(labels)) if labels[j]==y]
        vis_x1 = tsne[indexes, 0]
        vis_y1 = tsne[indexes, 1]
        c = colors[i]

        if tlabels is None:
            sc = plt.scatter(vis_x1, vis_y1, c=c, marker='.', cmap='hsv', label=y)
        else:
            sc = plt.scatter(vis_x1, vis_y1, c=c, marker='.', cmap='hsv', label=tlabels[indexes[0]])
    ax.legend()
    #plt.clim(-0.5, 9.5)
    #plt.savefig('models/lsi-clustering.png')
    plt.savefig(filename)
    plt.clf()
    

def histgram(X, figname):
    X = X.reshape(1, -1)
    print(X.shape)
    
    data = [x for x in X[0] if x!=0]
    print(len(data), np.max(data), np.min(data))
    n, bins, patches = plt.hist(data, 100, facecolor='g', alpha=0.75)

    plt.xlabel('Comment ratio')
    plt.ylabel('Count')
    #plt.title('Histogram of ')
    plt.grid(True)
    plt.savefig(figname)
    plt.clf()
    
def tfidf(X):
    X_rowsums = np.array([np.sum(X[i, :])+1 for i in range(X.shape[0])])
    X_colsums = np.array([np.sum(X[:, i]) for i in range(X.shape[1])])
    
    X_tf = np.array([X[i, :]/X_rowsums[i] for i in range(X_rowsums.shape[0])])
    X_tf = np.log(X_tf*100000+1)
    
    X_idf = np.log(1+X.shape[0]/(1+X_colsums))
    
    #X_tfidf = np.zeros(X.shape)
    #for i in range(X.shape[1]):
    #    X_tfidf[:, i] = X_tf[:, i]*X_idf[i]
    #return X_tfidf
    return X_tf
    
    

def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    
    #dataset = 'GSE99172'
    dataset = 'pbmc_two_batch'
    suffix = ''
    alg = 'LSI'
    name = 'bin'
    min_peaks = 5000
    batch = False
    
    if batch:
        batch_ids_file = '../../simulated_data/%s/batch_index.csv'%(dataset)
        batch_indices = pd.read_csv('../simulated_data/%s/batch_index.csv'%(dataset), header=0, delimiter='\t')[['barcode', 'info']]
        batch_indices['info'] = batch_indices['info'].apply(lambda x: int(x[-1])-1)
        batch_labels = [x for x in batch_indices['info'].values]
        print('batch indices', batch_indices)
    else: 
        batch_labels = None
    
    #X, cells, peaks, labels, cell_types, tlabels, = extract.extract_data(dataset)
    #X, cells, peaks, labels, cell_types, tlabels, = extract.extract_data(dataset)
    X, cells, peaks, labels, cell_types, tlabels, batches = extract.extract_simulated(dataset, suffix=suffix, is_labeled=True, batch=2)
    print(X.shape, len(labels), len(tlabels))
    
    filter = True
    if filter:
        d = SingleCellDataset(X, peaks, cells, low=0.2, high=0.9, min_peaks=min_peaks)
        #d = SingleCellDataset(X, peaks, cells, low=0.005, high=1, min_peaks=0)
        labels = [labels[i] for i in d.barcode if labels is not None]
        tlabels = [tlabels[i] for i in d.barcode if tlabels is not None]
        #print('filter data info', gene_dataset.mat.shape, gene_dataset.mat.max(), gene_dataset.mat.min())
        if dataset in ['atac_pbmc_1k_merge', 'ZY_bin_cell_matrix']:
            gene_dataset = SCDataset('models/', mat=d.data)
        else:
            gene_dataset = SCDataset('models/', mat=d.data, ylabels=labels, tlabels=tlabels, cell_types=cell_types)
    else:
        gene_dataset = SCDataset('models/', mat=X, ylabels=labels, tlabels=tlabels, cell_types=cell_types)
        
        
    #print('dataset', len(gene_dataset.ylabels), gene_dataset.mat.shape, len(gene_dataset))
    
    if name == 'bin':
        X = np.where(gene_dataset.X>0, 1, 0)
    elif name == 'tf':
        X = tfidf(gene_dataset.X)
    print(X.shape)
    
    if alg == 'LSI':
    
        #for i in [10, 20, 30, 50, 80, 100, 200]:
        for i in [50]:
            clf = TruncatedSVD(i, random_state=seed)
            Xsvd = clf.fit_transform(X)[:, 1:]
            print(i, 'SVD', Xsvd.shape, Xsvd.max(), Xsvd.min(), X[0, :5], Xsvd[0, :5])
            Xsvd = normalize(Xsvd)
            print(i, 'SVD_norm', Xsvd.shape, Xsvd.max(), Xsvd.min(), Xsvd[0, :5], labels)

            clustering(Xsvd, cells, dataset, suffix, labels=labels, tlabels=tlabels, method='louvain', name=name+str(min_peaks), batch_labels=labels)
    elif alg == 'LDA':
        n_topics = 50
        X = tfidf(X)
        model = lda.LDA(n_topics=n_topics, n_iter=1000, random_state=1)
        model.fit(X)

        doc_topic = model.doc_topic_
        print(len(doc_topic), doc_topic.shape, doc_topic.max(), doc_topic.min())
    
    
if __name__ == "__main__":
    main()