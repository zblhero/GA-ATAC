

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence as kl

from log_likelihood import log_zinb_positive, log_nb_positive, log_bnb_positive
from modules import Encoder, DecoderSCVI, Discriminator, reparameterize_gaussian
from utils import one_hot
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
    r"""Variational auto-encoder model.

    :param n_input: Number of input genes
    :param n_batch: Number of batches
    :param n_labels: Number of labels
    :param n_hidden: Number of nodes per hidden layer
    :param n_latent: Dimensionality of the latent space
    :param n_layers: Number of hidden layers used for encoder and decoder NNs
    :param dropout_rate: Dropout rate for neural networks
    :param dispersion: One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell

    :param log_variational: Log(data+1) prior to encoding for numerical stability. Not normalization.
    :param reconstruction_loss:  One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'bnb'`` - Beta negative binomial distribution

    Examples:
        >>> gene_dataset = CortexDataset()
        >>> vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels)

    """

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
        reconst_ratio = 1,
        use_cuda: bool = True
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
        self.use_cuda = use_cuda
        
        
        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":  # batch is different times of exp
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label', 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )
            
        self.pi = nn.Parameter(torch.ones(n_centroids)/n_centroids)  # pc
        self.mu_c = nn.Parameter(torch.zeros(n_latent, n_centroids)) # mu
        self.var_c = nn.Parameter(torch.ones(n_latent, n_centroids)) # sigma^2

        
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
        
        if self.reconstruction_loss == 'vae-gan':
            self.decoder = Decoder(
                n_latent,
                n_input,
                n_cat_list=[n_batch],
                n_layers=n_layers,
                n_hidden=n_hidden,
            )
            
        else:
            # decoder goes from n_latent-dimensional space to n_input-d data
            self.decoder = DecoderSCVI(
                n_latent,
                n_input,
                n_cat_list=[n_batch],
                n_layers=n_layers,
                n_hidden=n_hidden,
            )

    def get_latents(self, x, y=None):
        r""" returns the result of ``sample_from_posterior_z`` inside a list

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :return: one element list of tensor
        :rtype: list of :py:class:`torch.Tensor`
        """
        return [self.sample_from_posterior_z(x, y)]

    def sample_from_posterior_z(self, x, y=None, give_mean=False):
        r""" samples the tensor of latent values from the posterior
        #doesn't really sample, returns the means of the posterior distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param give_mean: is True when we want the mean of the posterior  distribution rather than sampling
        :return: tensor of shape ``(batch_size, n_latent)``
        :rtype: :py:class:`torch.Tensor`
        """
        if self.log_variational:
            x = torch.log(1 + x)
        qz_m, qz_v, z = self.z_encoder(x, y)  # y only used in VAEC
        if give_mean:
            z = qz_m
        return z

    def sample_from_posterior_l(self, x):
        r""" samples the tensor of library sizes from the posterior
        #doesn't really sample, returns the tensor of the means of the posterior distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :return: tensor of shape ``(batch_size, 1)``
        :rtype: :py:class:`torch.Tensor`
        """
        
        if self.log_variational:
            x = torch.log(1 + x)
        ql_m, ql_v, library = self.l_encoder(x)
        return library

    def get_sample_scale(self, x, batch_index=None, y=None, n_samples=1):
        r"""Returns the tensor of predicted frequencies of expression

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param n_samples: number of samples
        :return: tensor of predicted frequencies of expression with shape ``(batch_size, n_input)``
        :rtype: :py:class:`torch.Tensor`
        """
        return self.inference(x, batch_index=batch_index, y=y, n_samples=n_samples)[
            "px_scale"
        ]

    def get_sample_rate(self, x, batch_index=None, y=None, n_samples=1):
        r"""Returns the tensor of means of the negative binomial distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param n_samples: number of samples
        :return: tensor of means of the negative binomial distribution with shape ``(batch_size, n_input)``
        :rtype: :py:class:`torch.Tensor`
        """
        return self.inference(x, batch_index=batch_index, y=y, n_samples=n_samples)[
            "px_rate"
        ]

    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout, px_alpha):
        # Reconstruction Loss
        if self.reconstruction_loss == "zinb" or self.reconstruction_loss == "zinb-gmm":
            reconst_loss = -log_zinb_positive(x, px_rate, px_r, px_dropout)
            #l1loss = nn.MSELoss()
            #reconst_loss = l1loss(x, px_rate)
        elif self.reconstruction_loss == "vae-gan":
            reconst_loss = mse_loss(x, px_rate) 
        elif self.reconstruction_loss == "nb" or self.reconstruction_loss == "alpha-gan":
            reconst_loss = -log_nb_positive(x, px_rate, px_r)
        elif self.reconstruction_loss == "bnb":
            reconst_loss = -log_bnb_positive(x, px_rate, px_r, px_alpha)
        return reconst_loss

    def scale_from_z(self, sample_batch, fixed_batch):
        if self.log_variational:
            sample_batch = torch.log(1 + sample_batch)
        qz_m, qz_v, z = self.z_encoder(sample_batch)
        batch_index = fixed_batch * torch.ones_like(sample_batch[:, [0]])
        library = 4.0 * torch.ones_like(sample_batch[:, [0]])
        px_scale, _, _, _ = self.decoder("gene", z, library, batch_index)
        return px_scale

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
        
        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)

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
    
    def get_gamma(self, z):
        """
        Inference c from z

        gamma is q(c|x)
        q(c|x) = p(c|z) = p(c)p(c|z)/p(z)
        """
        n_centroids = self.n_centroids

        N = z.size(0)
        z = z.unsqueeze(2).expand(z.size(0), z.size(1), n_centroids)
        pi = self.pi.repeat(N,1) # NxK
        mu_c = self.mu_c.repeat(N,1,1) # NxDxK
        var_c = self.var_c.repeat(N,1,1) # NxDxK

        # p(c,z) = p(c)*p(z|c) as p_c_z
        p_c_z = torch.exp(torch.log(pi) - torch.sum(0.5*torch.log(2*math.pi*var_c) + (z-mu_c)**2/(2*var_c), dim=1)) + 1e-10
        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

        return gamma, mu_c, var_c, pi

    def init_gmm_params(self, latent, device='cuda'):
        """
        Init SCALE model with GMM model parameters
        """
        gmm = GaussianMixture(n_components=self.n_centroids, covariance_type='diag')
        #z = self.encodeBatch(dataset.mat, device)
        z = latent
        gmm.fit(z)
        print('init gmm', z.shape, gmm.means_.shape)
        self.mu_c.data.copy_(torch.from_numpy(gmm.means_.T.astype(np.float32)))
        self.var_c.data.copy_(torch.from_numpy(gmm.covariances_.T.astype(np.float32)))
        
    

    def forward(self, x, local_l_mean, local_l_var, batch_index=None, y=None):
        r""" Returns the reconstruction loss and the Kullback divergences

        :param x: tensor of values with shape (batch_size, n_input)
        :param local_l_mean: tensor of means of the prior distribution of latent variable l
         with shape (batch_size, 1)
        :param local_l_var: tensor of variancess of the prior distribution of latent variable l
         with shape (batch_size, 1)
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param y: tensor of cell-types labels with shape (batch_size, n_labels)
        :return: the reconstruction loss and the Kullback divergences
        :rtype: 2-tuple of :py:class:`torch.FloatTensor`
        """
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

        if self.reconstruction_loss == "zinb-gmm":
            gamma, mu_c, var_c, pi = self.get_gamma(z) #, self.n_centroids, c_params)
            kl_divergence_z = kl_gmm(x, gamma, (mu_c, var_c, pi), (qz_m, torch.sqrt(qz_v)))
        else:
            kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(dim=1)
            
        kl_divergence_l = kl(
            Normal(ql_m, torch.sqrt(ql_v)),
            Normal(local_l_mean, torch.sqrt(local_l_var)),
        ).sum(dim=1)

        reconst_loss = self.get_reconstruction_loss(x, px_rate, px_r, px_dropout, px_alpha)
        
        d_loss = 0
        g_loss = 0
        if self.reconstruction_loss == 'alpha-gan':
            valid = Variable(FLOAT(x.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(FLOAT(x.size(0), 1).fill_(0.0), requires_grad=False)
            idx = torch.randperm(self.X.shape[0])
            x_real = FLOAT(self.X[idx[:x.size(0)]])
            
            # generate sample from random priors
            z_prior = reparameterize_gaussian(torch.zeros(qz_m.shape).cuda(), torch.ones(qz_v.shape).cuda())
            l_prior = reparameterize_gaussian(torch.zeros(ql_m.shape).cuda(), torch.ones(ql_v.shape).cuda())
            _, _, x_fake, x_dropout, _ = self.decoder(self.dispersion, z_prior, l_prior, batch_index, y)
            _, _, z_rec = self.z_encoder(x_fake, y)
            x_fake = Normal(torch.zeros(x.shape), torch.ones(x.shape)).rsample().cuda()
            
            z_rec_loss = l1loss(z_rec, z_prior)
            #z_rec_loss = FLOAT([0])
            
            
            # GAN loss
            if self.gan_loss == 'gan':
                g_loss = adversarial_loss(self.discriminator(px_rate), valid) + adversarial_loss(self.discriminator(x_fake), valid)
                d_loss = adversarial_loss(self.discriminator(x), valid) +  adversarial_loss(self.discriminator(px_rate), fake) + adversarial_loss(self.discriminator(x_fake), fake)
                
            elif self.gan_loss == 'wgan':
                g_loss = -(self.discriminator(px_rate)+_eps).mean() -(self.discriminator(x_fake)+_eps).mean()    
                d_loss = -(self.discriminator(x) + _eps).mean() - (1 - self.discriminator(x_fake) + _eps).mean() - (1 - self.discriminator(px_rate) + _eps).mean() + self.gradient_penalty(self.discriminator, x, px_rate)
            
            return self.reconst_ratio*reconst_loss, kl_divergence_l+kl_divergence_z, g_loss, d_loss, z_rec_loss
        return reconst_loss, kl_divergence_l+kl_divergence_z
        
    
    def predict(self, latent, device='cuda', method='kmeans'):
        """
        Predict assignments applying k-means on latent feature

        Input: 
            x, data matrix
        Return:
            predicted cluster assignments
        """
        #feature = self.encodeBatch(gene_dataset.mat, device)
        feature = latent
        if method == 'kmeans':
            from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
            
            #print('feature', feature.shape, gene_dataset.mat.shape)
            kmeans = KMeans(n_clusters=self.n_centroids, n_init=20, random_state=0)
            pred = kmeans.fit_predict(feature)
        elif method == 'gmm':
            print('logits', feature.shape)
            pred = np.argmax(feature, axis=1)
        elif method == 'louvain':
            import networkx as nx
            import community
            from sklearn.neighbors import kneighbors_graph
            mat = kneighbors_graph(feature, 15, mode='distance', include_self=True).todense()
            G = nx.from_numpy_matrix(mat)
            partition = community.best_partition(G)
                    
            pred = []
            for i in range(feature.shape[0]):
                pred.append(partition[i])

        return pred
    
    def gradient_penalty(self, model, x, x_gen, w=10):
        """WGAN-GP gradient penalty"""
        assert x.size()==x_gen.size(), "real and sampled sizes do not match"
        alpha_size = tuple((len(x), *(1,)*(x.dim()-1)))
        alpha_t = torch.cuda.FloatTensor if x.is_cuda else torch.Tensor
        alpha = alpha_t(*alpha_size).uniform_()
        x_hat = x.data*alpha + x_gen.data*(1-alpha)
        x_hat = Variable(x_hat, requires_grad=True)

        def eps_norm(x):
            x = x.view(len(x), -1)
            return (x*x+_eps).sum(-1).sqrt()
        def bi_penalty(x):
            return (x-1)**2

        grad_xhat = torch.autograd.grad(
            model(x_hat).sum(), x_hat, create_graph=True, only_inputs=True
        )[0]
    
        penalty = w*bi_penalty(eps_norm(grad_xhat)).mean()
        return penalty
    


def kl_gmm(x, gamma, c_params, z_params):
    """
    L elbo(x) = Eq(z,c|x)[ log p(x|z) ] - KL(q(z,c|x)||p(z,c))
              = Eq(z,c|x)[ log p(x|z) + log p(z|c) + log p(c) - log q(z|x) - log q(c|x) ]
    """
    mu_c, var_c, pi = c_params; #print(mu_c.size(), var_c.size(), pi.size())
    n_centroids = pi.size(1)
    mu, logvar = z_params
    mu_expand = mu.unsqueeze(2).expand(mu.size(0), mu.size(1), n_centroids)
    logvar_expand = logvar.unsqueeze(2).expand(logvar.size(0), logvar.size(1), n_centroids)

    # log p(z|c)
    logpzc = -0.5*torch.sum(gamma*torch.sum(math.log(2*math.pi) + \
                                           torch.log(var_c) + \
                                           torch.exp(logvar_expand)/var_c + \
                                           (mu_expand-mu_c)**2/var_c, dim=1), dim=1)
    # log p(c)
    logpc = torch.sum(gamma*torch.log(pi), 1)

    # log q(z|x) or q entropy    
    qentropy = -0.5*torch.sum(1+logvar+math.log(2*math.pi), 1)

    # log q(c|x)
    logqcx = torch.sum(gamma*torch.log(gamma), 1)

    kld = -logpzc - logpc + qentropy + logqcx

    return torch.sum(kld)

def calc_gradient_penalty(netD, real_data, generated_data):
    # GP strength
    LAMBDA = 10

    b_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(b_size, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.cuda()
    
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = netD(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(b_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return LAMBDA * ((gradients_norm - 1) ** 2).mean()