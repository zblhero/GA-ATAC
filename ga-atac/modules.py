import collections
from typing import Iterable, List

import torch
from torch import nn as nn
from torch.distributions import Normal
from torch.nn import ModuleList
import torch.nn.functional as F

from utils import *


def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()


# Encoder
class Encoder(nn.Module):

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.BatchNorm1d(n_hidden, momentum=0.01, eps=0.001),
            nn.ReLU()
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int):
        
        # Parameters for latent distribution
        q = self.encoder(x)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent


# Decoder
class DecoderSCVI(nn.Module):

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.px_decoder = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.BatchNorm1d(n_hidden, momentum=0.01, eps=0.001),
            nn.ReLU()
        )

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output), 
            nn.Softmax(dim=-1)
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Sequential(
                    nn.Linear(n_hidden, n_output), 
                    nn.ReLU()
        )

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)
        
        # alpha for Beta
        self.px_alpha_decoder = nn.Sequential(
                    nn.Linear(n_hidden, n_output), 
                    nn.ReLU()
        )

    def forward(
        self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int
    ):  
        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z)   # cat list includes batch index
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        
        px_alpha = self.px_alpha_decoder(px)+1
        library = torch.clamp(library, max=12)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  
        px_rate = torch.clamp(px_rate, max=12)
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        
        #print('output px_rate', px_rate.shape, torch.max(px_rate), torch.min(px_rate), library.shape, torch.max(library), torch.min(library), px_scale.shape, torch.max(px_scale), torch.min(px_scale))
        return px_scale, px_r, px_rate, px_dropout, px_alpha
    


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_size, n_batch_gan, gan_loss):
        super(Discriminator, self).__init__()
        
        self.n_batch_gan = n_batch_gan
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.model = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512, momentum=0.01, eps=0.001),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.4),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256, momentum=0.01, eps=0.001),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.4),
                nn.Linear(256, 1+n_batch_gan),
                #nn.Sigmoid(),
        )

    def forward(self, x):
        validity = self.model(x)
        
        if self.n_batch_gan == 0:
            out = self.sigmoid(validity)
            return out, None
        else:
            out1 = self.sigmoid(validity[:, :1])
            #out2 = self.softmax(validity[:, 1:])
            out2 = validity[:, 1:]
            

            return out1, out2