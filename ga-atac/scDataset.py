import logging
import os
from typing import Iterable, Union

import numpy as np
import pandas as pd
from dataset import *

#import sys
#sys.path.append("..")
#import load

logger = logging.getLogger(__name__)


class SCDataset(DownloadableDataset):

    def __init__(
        self,
        filename: str,
        save_path: str = "data/",
        url: str = None,
        new_n_genes: int = None,
        subset_genes: Iterable[Union[int, str]] = None,
        compression: str = None,
        sep: str = ",",
        gene_by_cell: bool = True,
        labels_file: str = None,
        batch_ids_file: str = None,
        delayed_populating: bool = False,
        mat = None,
        ylabels = None,
        tlabels = None,
        cell_types = None
    ):
        self.compression = compression
        self.sep = sep
        self.gene_by_cell = (
            gene_by_cell
        )  # Whether the original dataset is genes by cells
        self.labels_file = labels_file
        self.batch_ids_file = batch_ids_file
        self.mat = mat
        self.ylabels = ylabels
        self.tlabels = tlabels
        self.cell_types = cell_types
        
        super().__init__(
            urls=url,
            filenames=filename,
            save_path=save_path,
            delayed_populating=delayed_populating,
        )
        self.subsample_genes(new_n_genes, subset_genes)

    def populate(self):
        logger.info("Preprocessing dataset")

        #gene_names = np.asarray(data.columns, dtype=str)
        gene_names = [str(i) for i in range(self.mat.shape[1])]
        batch_indices = None

        if self.batch_ids_file is not None:
            batch_indices = pd.read_csv(
                os.path.join(self.save_path, self.batch_ids_file), header=0, delimiter='\t'
            )[['barcode', 'info']]
            batch_indices['info'] = batch_indices['info'].apply(lambda x: int(x[-1])-1)
            batch_indices = batch_indices.values
            
            

        #print('labels', self.ylabels, self.cell_types, batch_indices, gene_names)
        self.populate_from_data(
            X=self.mat,
            batch_indices=batch_indices,
            labels=None,
            gene_names=gene_names,
            cell_types=self.cell_types,
        )
        #self.filter_cells_by_count()
