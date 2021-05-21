
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
            cell = line.strip('\n').replace('"', '')
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

def load_data(filename):
    #if '65361' in filename:
    #    return np.load(os.path.dirname(__file__)+filename)
    row, col, data = [], [], []
    with open(os.path.dirname(__file__)+filename) as f:
        for line in f.readlines():
            values = line.strip('\n').split()
            #values = line.strip('\n').split('\t')
            #print(values)
            if len(values) != 3:
                continue
            try:
                col.append(int(values[0]))
                row.append(int(values[1])-1)
                data.append(int(values[2]))
            except ValueError:
                print(line)
    print('load data', len(data), len(row), len(col))
    X = coo_matrix((data, (row, col))).toarray()
    return X


def extract_simulated(dataset='GSE65360', is_labeled=True, suffix='clean'):
    dirname = '/../../scATAC/data/%s/'%(dataset) 
    
    peaks = read_pos(dirname+'%s_peak.bed'%(dataset))
    cells = read_barcodes(dirname+'%s_barcode.txt'%(dataset))
    
    if is_labeled:
        labels, cell_types, tlabels = read_labels(dirname+'%s_celltype_info.csv'%(dataset), cells, dataset)
    else:
        labels, cell_types, tlabels = None, None, None
    X = load_data(dirname+'%s_SparseMatrix.txt'%(dataset))
    
    return X, cells, peaks, labels, cell_types, tlabels
    


if __name__ == "__main__":
    #extract()
    #mat, cells, peaks, labels, cell_types, tlabels = extract_data()
    #load_data('simulated_data/GSE65360/simulated.true.nodups.sort.bam')
    
    #extract_simulated(dataset='For_specific_peak', suffix='')
    
    extract_simulated(dataset='scRNA_cortex', suffix='')