

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence as kl

#from log_likelihood import log_nb_positive
from modules import Encoder, DecoderSCVI, Discriminator, reparameterize_gaussian
from utils import *
import math
import numpy as np
from sklearn.mixture import GaussianMixture

from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True
FLOAT = torch.cuda.FloatTensor
adversarial_loss = torch.nn.BCELoss()
l1loss = torch.nn.L1Loss()

_eps = 1e-15

# VAE model
class GAATAC(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 256,
        n_latent: int = 18,
        n_layers: int = 1,
        dropout_rate: float = 0.,
        dispersion: str = "gene",
        log_variational: bool = False,
        reconstruction_loss: str = "alpha-gan",
        n_centroids = 12,
        X = None,
        gan_loss = 'gan',
        reconst_ratio = 1
    ):
        super().__init__()
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.log_variational = log_variational
        self.reconstruction_loss = reconstruction_loss
        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_centroids = n_centroids
        self.dropout_rate = dropout_rate
        self.X = X
        self.gan_loss = gan_loss
        self.reconst_ratio = reconst_ratio
        
        
        self.px_r = torch.nn.Parameter(torch.randn(n_input))
        

        
        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate
        )
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input, 1, n_layers=1, n_hidden=n_hidden, dropout_rate=dropout_rate
        )
        self.discriminator = Discriminator(n_input, n_hidden, gan_loss)
        
        self.decoder = DecoderSCVI(
                n_latent,
                n_input,
                n_cat_list=[n_batch],
                n_layers=n_layers,
                n_hidden=n_hidden,
        )
        
    def sample_from_posterior_z(self, x, y=None, give_mean=False):
        if self.log_variational:
            x = torch.log(1 + x)
        qz_m, qz_v, z = self.z_encoder(x, y)  # y only used in VAEC
        if give_mean:
            z = qz_m
        return z

    def sample_from_posterior_l(self, x):
        if self.log_variational:
            x = torch.log(1 + x)
        ql_m, ql_v, library = self.l_encoder(x)
        return library
    

    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout, px_alpha):
        # Reconstruction Loss
        reconst_loss = -log_nb_positive(x, px_rate, px_r)
        return reconst_loss


    def inference(self, x, batch_index=None, y=None, n_samples=1):
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        # Sampling
        qz_m, qz_v, z = self.z_encoder(x_, y)
        ql_m, ql_v, library = self.l_encoder(x_)   

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            z = Normal(qz_m, qz_v.sqrt()).sample()
            ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
            ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
            library = Normal(ql_m, ql_v.sqrt()).sample()

        px_scale, px_r, px_rate, px_dropout, px_alpha = self.decoder(
             self.dispersion, z, library, batch_index, y
        ) 
        
        px_r = torch.exp(self.px_r)

        return dict(
            px_scale=px_scale,
            px_r=px_r,
            px_rate=px_rate,
            px_dropout=px_dropout,
            px_alpha = px_alpha,
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            ql_m=ql_m,
            ql_v=ql_v,
            library=library,
        )
    

    def forward(self, x, local_l_mean, local_l_var, batch_index=None, y=None):

        # Parameters for z latent distribution
        outputs = self.inference(x, batch_index, y)
        qz_m = outputs["qz_m"]
        qz_v = outputs["qz_v"]
        ql_m = outputs["ql_m"]
        ql_v = outputs["ql_v"]    
        px_rate = outputs["px_rate"]
        px_r = outputs["px_r"]
        px_dropout = outputs["px_dropout"]
        px_alpha = outputs['px_alpha']
        z = outputs['z']

        # KL Divergence
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(dim=1) 
        kl_divergence_l = kl(Normal(ql_m, torch.sqrt(ql_v)), Normal(local_l_mean, torch.sqrt(local_l_var))).sum(dim=1)

        reconst_loss = self.get_reconstruction_loss(x, px_rate, px_r, px_dropout, px_alpha)
        
        #d_loss = 0
        #g_loss = 0
        
        valid = Variable(FLOAT(x.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(FLOAT(x.size(0), 1).fill_(0.0), requires_grad=False)
        idx = torch.randperm(self.X.shape[0])
        x_real = FLOAT(self.X[idx[:x.size(0)]])
            
        # generate sample from random priors
        z_prior = reparameterize_gaussian(torch.zeros(qz_m.shape).cuda(), torch.ones(qz_v.shape).cuda())
        l_prior = reparameterize_gaussian(torch.zeros(ql_m.shape).cuda(), torch.ones(ql_v.shape).cuda())
        _, _, x_fake, x_dropout, _ = self.decoder(self.dispersion, z_prior, l_prior, batch_index, y)
        _, _, z_rec = self.z_encoder(x_fake, y)
            
        z_rec_loss = l1loss(z_rec, z_prior)
        #z_rec_loss = FLOAT([0])
            
            
        # GAN loss
        g_loss = adversarial_loss(self.discriminator(px_rate), valid)# + adversarial_loss(self.discriminator(x_fake), valid)
        d_loss = adversarial_loss(self.discriminator(x), valid) +  adversarial_loss(self.discriminator(px_rate), fake) + adversarial_loss(self.discriminator(x_fake), fake)
                
            
        return self.reconst_ratio*reconst_loss, kl_divergence_l+kl_divergence_z, g_loss, d_loss, z_rec_loss
