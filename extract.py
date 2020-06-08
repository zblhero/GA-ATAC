
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, coo_matrix
import gc
import os, os.path
import pysam


def read_barcodes(filename='/GSE99172/GSE99172_barcode.txt'):
    cells = []
    with open(os.path.dirname(__file__) +filename) as f:
        for line in f.readlines():
            cell = line.strip('\n')
            if 'forebrain' in filename:
                cell = cell[4:]
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

def read_bam_mat(dirname, filename, regions, cells, suffix):
    bamfile = pysam.Samfile(os.path.join(dirname, filename), 'rb')
      
    cellsdic = {}
    for x, cell in enumerate(cells):
        cellsdic[cell] = x+1
        
    regionsdic = {}
    for x, region in enumerate(regions):
        regionsdic[region] = x+1
    
    row = []
    col = []
    data = []
    outter = open(dirname+'test_bin_%s.txt'%(suffix), 'w')
    outter.write('Region\tbarcode\tcount\n')
    #for region in regionsdic.keys():
    for i, rec in enumerate(regions):
        #rec = region.split('.')[0:3]
        reads = bamfile.fetch(rec[0], int(rec[1]), int(rec[2]))
        currcounts = {}
        for read in reads:
            readname = read.qname.split('.')[0]
            #print(readname, cellsdic[readname])
            try:
                cellsdic[readname]  # make sure readname in cell barcode
                try:
                    currcounts[readname] += 1   # currcounts save read count(in each cell) in current region
                except KeyError:
                    currcounts[readname] = 1
            except KeyError:
                continue
        if i %10000 == 0:
            print(i, len(currcounts))
        if len(currcounts)>0:
            curr_col = []
            curr_data = []
            curr_row = regionsdic[region] * len(currcounts)

            for key in currcounts.keys():
                outter.write(str(regionsdic[rec]) + '\t' + str(cellsdic[key]) + '\t' + str(currcounts[key])+'\n')
    outter.close()
    
    
def read_labels(filename, cells, dataset):
    if dataset in ['GSE99172', '74310', 'forebrain', 'GSE96769', 'GSE112091', 'GM12878vsHEK', 'GM12878vsHL', 'Splenocyte', 'human_Occipital']:
        if dataset in ['forebrain', 'GSE96769', 'GSE112091']:
            df = pd.read_csv(os.path.dirname(__file__) +filename, delimiter='\t')
        else:
            df = pd.read_csv(os.path.dirname(__file__) +filename, delimiter=',')
        print(df.columns)
        df = df[df.barcode.isin(cells)]
        df['cell_types'] = df['celltype'].apply(lambda x: x)
    else:
        df = pd.read_csv(os.path.dirname(__file__) +filename, delimiter='\t')
        
        df = df[df.Run.isin(cells)]
        df['cell_types'] = df['cell_line'].apply(lambda x: x)
    
    
    cell_types = list(df.cell_types.value_counts().keys())
    y = []
    for i, cell in enumerate(cells):
        if dataset in ['GSE99172', '74310', 'forebrain', 'GSE96769', 'GSE112091', 'GM12878vsHEK', 'GM12878vsHL', 'Splenocyte', 'human_Occipital']:
            data = df[df.barcode==cell]
        else:
            data = df[df.Run==cell]
        if not pd.isna(data.cell_types.values[0]):
            y.append(cell_types.index(data.cell_types.values[0]))
        
    tlabels = df.cell_types.values
    return y, cell_types, tlabels
    
    
def read_pos(filename):
    bases, regions = {}, []
    
    with open(os.path.dirname(__file__) +filename) as f:
        chr_lines = f.readlines()
        for line in chr_lines:
            
            if 'human_Occipital' in filename:
                values = line.strip().split(':')
                vs = values[1].split('-')
                key, start, end = values[0], int(vs[0]), int(vs[1])
            else:
                sp = '_' if 'txt' in filename else '\t'
                values = line.strip().split(sp)
                key, start, end = values[0], int(values[1]), int(values[2])
            
            if values[0] in bases:    
                bases[values[0]].append((start, end))
            else:
                bases[values[0]] = [(start, end)]
            regions.append((key, start, end))
    return bases, regions

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
    
    

    
def extract(dataset='GSE99172'):
    dirname = '/data/%s/'%(dataset)

    bases, peaks = read_pos(dirname+'peaks.bed')
    #bases, peaks = read_pos(dirname+'SelectBin.bed')
    print('bases peaks', len(bases), len(peaks), peaks[:10])
    
    cells = read_barcodes(dirname+'barcode.txt')
    print('num of cells', len(cells))
    
    labels, cell_types, tlabels = read_labels(dirname+'SraRunTable.txt', cells, dataset)
    print(len(labels), labels[:10])
    

    mat = read_bam_mat(dirname, 'simulated.true.nodups.sort.bam', peaks, cells)
    X = load_data(dirname+'test_bin.txt')
    print(X.shape, np.sum(X))
    
    return X, cells, peaks, labels, cell_types, tlabels

def extract_data(dataset='GSE65360'):
    dirname = '../simulated_data/%s/'%(dataset)
    
    i = 0
    mat, peaks, labels, cell_types, tlabels = [], [], [], [], []
    with open(dirname +'data.txt') as f:
        for line in f.readlines():
            values = line.strip('\n').split()
            if i == 0:
                cells = [x.split('.')[0] for x in values]
            else:
                cell_genes = values[1:]
                peaks.append(values[0])
                mat.append([int(x) for x in cell_genes])
            i+=1
    i = 0
    with open(dirname+'labels.txt') as f:
        for line in f.readlines():
            values = line.strip('\n').split()
            tlabels.append(values[1])
            if values[1] not in cell_types:
                cell_types.append(values[1])
            labels.append(cell_types.index(values[1]))
            i += 1
    return np.array(mat).T, cells, peaks, labels, cell_types, tlabels


def extract_simulated(dataset='GSE65360', suffix='clean'):
    dirname = '/../simulated_data/%s/'%(dataset)
    
    
    
    if dataset in ['GSE99172', 'forebrain', 'GM12878vsHEK', 'GM12878vsHL', 'Splenocyte', 'human_Occipital']:
        if dataset in ['GM12878vsHEK', 'GM12878vsHL', 'Splenocyte', 'human_Occipital']:
            bases, peaks = read_pos(dirname+'%s_peak.txt'%(dataset))
        else:
            bases, peaks = read_pos(dirname+'%s_peak.bed'%(dataset))
        #bases, peaks = read_pos(dirname+'bin_hg19_2.5kb_filtered.bed')
        print('bases peaks', len(bases), len(peaks), peaks[:10])

        cells = read_barcodes(dirname+'barcode.txt')
        print('num of cells', len(cells), cells[:10])
        if dataset in ['GSE99172', 'GM12878vsHEK', 'GM12878vsHL', 'Splenocyte', 'human_Occipital']:
            labels, cell_types, tlabels = read_labels(dirname+'%s_celltype_info.csv'%(dataset), cells, dataset)
            X = load_data(dirname+'%s_SparseMatrix.txt'%(dataset))
        elif dataset == 'forebrain':
            labels, cell_types, tlabels = read_labels(dirname+'celltype_info.txt', cells, dataset)
            X = load_data(dirname+'forebrain_matrix.txt')
        
    elif dataset == 'GSE96769':
        cells = read_barcodes(dirname+'barcode.txt')
        labels, cell_types, tlabels = read_labels(dirname+'GSE96769_info_select8type.txt', cells, dataset)
        X = load_data(dirname+'GSE96769_SParseMatrix_select8type.txt')
        peaks = None
    elif dataset == 'For_specific_peak':
        cells = read_barcodes(dirname+'barcode.txt')
        print(len(cells))
        bases, peaks = read_pos(dirname+'peaks.bed')
        print(len(peaks))
        labels, cell_types, tlabels = None, None, None
        #labels, cell_types, tlabels = read_labels(dirname+'GSE96769_info_select8type.txt', cells, dataset)
        X = load_data(dirname+'ZY_SparseMatrix.txt')
    elif dataset == 'atac_pbmc_1k_merge':
        cells = read_barcodes(dirname+'%s_barcode.txt'%(dataset))
        bases, peaks = read_pos(dirname+'peak_merge.bed')
        labels, cell_types, tlabels = None, None, None
        X = load_data(dirname+'%s_SparseMatrix.txt'%(dataset))
    else:
        bases, peaks = read_pos(dirname+'%s_peaks_%s.bed'%(dataset, suffix))
        #bases, peaks = read_pos(dirname+'bin_hg19_2.5kb_filtered.bed')
        print('bases peaks', len(bases), len(peaks), peaks[:10])

        cells = read_barcodes(dirname+'barcode.txt')
        print('num of cells', len(cells), cells[:10])
        labels, cell_types, tlabels = read_labels(dirname+'SraRunTable.txt', cells, dataset)
        
        #mat = read_bam_mat(dirname, 'simulated.true.nodups.sort.bam', peaks, cells, suffix)
    
        X = load_data(dirname+'test_bin_%s.txt'%(suffix))
    #print('labels', len(labels), labels[:10], len(tlabels), tlabels[:10])
    print(X.shape, np.sum(X))
    
    return X, cells, peaks, labels, cell_types, tlabels
    


if __name__ == "__main__":
    #extract()
    #mat, cells, peaks, labels, cell_types, tlabels = extract_data()
    #load_data('simulated_data/GSE65360/simulated.true.nodups.sort.bam')
    
    #extract_simulated(dataset='For_specific_peak', suffix='')
    
    extract_simulated(dataset='atac_pbmc_1k_merge', suffix='')