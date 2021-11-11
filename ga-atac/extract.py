
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, coo_matrix
import gc
import os, os.path
import pysam
#from datasets.cortex import CortexDataset


def read_barcodes(filename='/GSE99172/GSE99172_barcode.txt'):
    cells, tlabels, y, cell_types = [], [], [], []
    
    with open(os.path.dirname(__file__) +filename) as f:
        for line in f.readlines():
            cell = line.strip('\n').replace('"', '').split(',')[0]
            cells.append(cell)
    return cells


def read_peaks(filename='/GSE99172/peak.bed'):
    peaks = []
    i = 0
    with open(os.path.dirname(__file__) +filename) as f:
        for line in f.readlines():
            values = line.strip('\n').split('\t')
            peaks.append(values)

    return peaks
    
    
def read_labels(filename, cells, dataset):
    df = pd.read_csv(os.path.dirname(__file__) +filename, delimiter=',')
    df = df[df.barcode.isin(cells)]
    df['cell_types'] = df['celltype'].apply(lambda x: x)
   
    
    
    cell_types = list(df.cell_types.value_counts().keys())
    y = []
    for i, cell in enumerate(cells):
        data = df[df.barcode==cell]
        if not pd.isna(data.cell_types.values[0]):
            y.append(cell_types.index(data.cell_types.values[0]))
        
    tlabels = df.cell_types.values
    return y, cell_types, tlabels
    
    
def read_pos(filename):
    peaks = []
    
    with open(os.path.dirname(__file__) +filename) as f:
        chr_lines = f.readlines()
        for line in chr_lines:
            
            sp = ':' if 'txt' in filename else '\t'
            values = line.strip().split(sp)
            if len(values) > 3:
                key, start, end = '_'.join(values[:-2]), int(values[-2]), int(values[-1])
            elif len(values) == 2:
                offsets = values[1].split('-')
                key, start, end = values[0], int(offsets[0]), int(offsets[1])
            else:
                key, start, end = values[0], int(values[1]), int(values[2])
            
            peaks.append((key, start, end))
    return peaks

def load_data(filename, cell_num):
    #if '65361' in filename:
    #    return np.load(os.path.dirname(__file__)+filename)
    row, col, data = [], [], []
    with open(os.path.dirname(__file__)+filename) as f:
        for line in f.readlines():
            values = line.strip('\n').split()
            if len(values) != 3:
                continue
            try:
                if int(values[1])-1 < cell_num:
                    col.append(int(values[0]))
                    row.append(int(values[1])-1)
                    data.append(int(values[2]))
            except ValueError:
                #print(line)
                pass
    print('load data', len(data), len(row), len(col), np.unique(row), np.unique(col))
    X = coo_matrix((data, (row, col))).toarray()
    del data
    del row
    del col
    return X

def read_batches(filename):
    df = pd.read_csv(os.path.dirname(__file__)+filename, delimiter=',')
    df['batch'] = df['batch'].apply(lambda x: int(x[-1])-1)
    return df.batch.values
    
    


def extract_simulated(dataset='GSE65360', is_labeled=True, suffix='clean', batch=False):
    dirname = '/../data/%s/'%(dataset) 
    
    
    cells = read_barcodes(dirname+'%s_barcode.txt'%(dataset))
    if dataset == 'pbmc_two_batch':
        X = load_data(dirname+'%s_ATAC_matrix.txt'%(dataset), len(cells))
        print('ATAC binary: ', X.shape, X.max(), X.min())  # (14623, 277809)
        X1 = load_data(dirname+'%s_RNA_matrix.txt'%(dataset), len(cells))
        X1 = X1/7545.0
        print('RNA not binary: ', X1.shape, X1.max(), X1.min())  # (14623, 36601)
        
        X = np.concatenate((X, X1), axis=1)
    else:
        X = load_data(dirname+'%s_SparseMatrix.txt'%(dataset), len(cells))
    
    try:
        peaks = read_pos(dirname+'%s_peak.bed'%(dataset))
    except FileNotFoundError:
        peaks = range(X.shape[1])
    
    
    if is_labeled:
        labels, cell_types, tlabels = read_labels(dirname+'%s_celltype_info.csv'%(dataset), cells, dataset)
    else:
        labels, cell_types, tlabels = None, None, None
        
    if dataset == 'pbmc3k':
        print(X.shape, 'RNA', X[:, :36572].shape, X[:, :36572].max(), X[:, :36572].min(), 'ATAC', X[:, 36572:].shape, X[:, 36572:].max(), X[:, 36572:].min())
        X[:, :36572] = X[:, :36572]/7545.0
        X[:, 36572:] = np.where(X[:, 36572:]>0, 1, 0)
        #X = X[:, 36572:]
        print("load X", X.shape, X.max(), X.min())
    
    if batch:
        batches = read_batches(dirname + '%s_batch_info.csv'%(dataset))
        print('batch', len(batches))
        return X, cells, peaks, labels, cell_types, tlabels, batches
    
    
    print(X.shape, len(cells)), len(peaks)
    return X, cells, peaks, labels, cell_types, tlabels, None
    


if __name__ == "__main__":
    #extract()
    #mat, cells, peaks, labels, cell_types, tlabels = extract_data()
    #load_data('simulated_data/GSE65360/simulated.true.nodups.sort.bam')
    
    #extract_simulated(dataset='For_specific_peak', suffix='')
    
    extract_simulated(dataset='pbmc_two_batch', suffix='', is_labeled=False, batch=True)