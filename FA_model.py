import pyro
from pyro.nn import PyroSample, PyroModule
from pyro.infer import SVI, Trace_ELBO, autoguide
import torch
from torch.nn.functional import softplus
from sklearn.metrics import mean_squared_error
import random
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ann
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
import numpy as np
import tempfile
from numpyro.distributions import TransformedDistribution, transforms
from torch.distributions.transforms import SoftmaxTransform
from scipy.stats import spearmanr, pearsonr
import scanpy as sc
import anndata as ad
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# simulate data
n_obs = 100
n_features1 = 20
n_features2 = 30
n_factors = 20

random.seed(2024)
torch.manual_seed(2024)
np.random.seed(2024)

percentage = 0.3 # percentage of values set to zero

Z_in = torch.randn(n_obs, n_factors)
W1_in = torch.randn(n_features1, n_factors)
W2_in = torch.randn(n_features2, n_factors)

# create observated values from the simulated factor and weight matrix with some random noise\n,
Y1 = torch.matmul(Z_in, W1_in.t()) #+ 0.2 * torch.randn(n_obs, n_features1)
Y2 = torch.matmul(Z_in, W2_in.t()) #+ 0.2 * torch.randn(n_obs, n_features2)

#try creating shared factors which are correlating
Y1[:,:9] = torch.sort(Y1[:,:9]).values
Y2[:,:9] = torch.sort(Y2[:,:9]).values

#introducing NANs/0 into dataframe
Y1_indices = np.random.choice(Y1.shape[1]*Y1.shape[0], replace=False, size=int(Y1.shape[1]*Y1.shape[0]*percentage))
Y1[np.unravel_index(Y1_indices, Y1.shape)] = 0 

Y2_indices = np.random.choice(Y2.shape[1]*Y2.shape[0], replace=False, size=int(Y2.shape[1]*Y2.shape[0]*percentage))
Y2[np.unravel_index(Y2_indices, Y2.shape)] = 0 

Y = torch.cat((Y1.T, Y2.T)).T


class FA(PyroModule):
    def __init__(self, dat, n_features1, n_features2, K):
        """
        Args:
            Y: Tensor (Samples x Features)
            K: Number of Latent Factors
        """
        super().__init__()
        pyro.clear_param_store()
        
        # data
        self.num_features1 = n_features1
        self.num_features2 = n_features2
        self.Y1 = dat[:,:n_features1]
        self.Y2 = dat[:,n_features1:]
        self.K = K
        
        self.num_samples = self.Y1.shape[0]
        
        self.sample_plate = pyro.plate("sample", self.num_samples)
        self.feature_plate1 = pyro.plate("feature1", self.num_features1)
        self.feature_plate2 = pyro.plate("feature2", self.num_features2)
        self.latent_factor_plate = pyro.plate("latent factors", self.K)
        
    def model(self):
        """
        how to generate a matrix
        """
        with self.latent_factor_plate:
            with self.feature_plate1:
                # sample weight matrix with Normal prior distribution
                W1_hat = pyro.sample("W1", pyro.distributions.Normal(0., 1.))  
                #theta1 = pyro.sample("theta1", pyro.distributions.Beta(1., 1.))
                #alpha1 = pyro.sample("alpha1", pyro.distributions.Gamma(torch.tensor(10.), torch.tensor(10.)))
                #W1_hat = pyro.sample("W1", pyro.distributions.Normal(0., 1./alpha1))*pyro.sample("s1", pyro.distributions.Bernoulli(theta1))

            with self.feature_plate2:
                # sample weight matrix with Normal prior distribution
                W2_hat = pyro.sample("W2", pyro.distributions.Normal(0., 1.))  
                #theta2 = pyro.sample("theta2", pyro.distributions.Beta(1., 1.))
                #alpha2 = pyro.sample("alpha2", pyro.distributions.Gamma(torch.tensor(10.), torch.tensor(10.)))
                #W2_hat = pyro.sample("W2",pyro.distributions.Normal(0., 1./alpha2))*pyro.sample("s2",pyro.distributions.Bernoulli(theta2))
                
            with self.sample_plate:
                # sample factor matrix with Normal prior distribution
                Z = pyro.sample("Z", pyro.distributions.Normal(0., 1.))
        
        # estimate for Y
        Y1_hat = torch.matmul(Z, W1_hat.t())
        Y2_hat = torch.matmul(Z, W2_hat.t())
        
        with pyro.plate("feature1_", self.Y1.shape[1]), pyro.plate("sample_", self.Y1.shape[0]):
            # masking the NA values such that they are not considered in the distributions
            obs_mask = torch.ones_like(self.Y1, dtype=torch.bool)
            if self.Y1 is not None:
                obs_mask = torch.logical_not(torch.isnan(self.Y1))
            with pyro.poutine.mask(mask=obs_mask):
                if self.Y1 is not None:
                    # a valid value for the NAs has to be defined even though these samples will be ignored later
                    self.Y1 = torch.nan_to_num(self.Y1, nan=0)
            
                    # sample scale parameter for each feature-sample pair with LogNormal prior (has to be positive)
                    precision1 = pyro.sample("precision1", pyro.distributions.Gamma(torch.tensor(20.), torch.tensor(20.)))
                    #precision1 = 0.6
                    #print(precision1)
                    scale = pyro.sample("scale1", pyro.distributions.LogNormal(0., 1./(precision1)))
                    #print(scale)
                    # compare sampled estimation to the true observation Y
                    pyro.sample("obs1", pyro.distributions.Normal(Y1_hat, scale), obs=self.Y1)
                    #Y1_sampled = pyro.distributions.Normal(Y1_hat, scale)
                    #pyro.sample("obs1", pyro.distributions.TransformedDistribution(Y1_sampled, SoftmaxTransform()), obs=self.Y1)


        with pyro.plate("feature2_", self.Y2.shape[1]), pyro.plate("sample2_", self.Y2.shape[0]):
            # masking the NA values such that they are not considered in the distributions
            obs_mask = torch.ones_like(self.Y2, dtype=torch.bool)
            if self.Y2 is not None:
                obs_mask = torch.logical_not(torch.isnan(self.Y2))
            with pyro.poutine.mask(mask=obs_mask):
                if self.Y2 is not None:
                    # a valid value for the NAs has to be defined even though these samples will be ignored later
                    self.Y2 = torch.nan_to_num(self.Y2, nan=0) 
            
                    # sample scale parameter for each feature-sample pair with LogNormal prior (has to be positive)
                    precision2 = pyro.sample("precision2", pyro.distributions.Gamma(torch.tensor(20.), torch.tensor(20.)))
                    #precision2 = 0.6
                    scale = pyro.sample("scale2", pyro.distributions.LogNormal(0., 1./(precision2)))
                    # compare sampled estimation to the true observation Y
                    pyro.sample("obs2", pyro.distributions.Normal(Y2_hat, scale), obs=self.Y2)
                    #Y2_sampled = pyro.distributions.Normal(Y2_hat, scale)
                    #pyro.sample("obs2", pyro.distributions.TransformedDistribution(Y2_sampled, SoftmaxTransform()), obs=self.Y2)


    def train(self):
        # set training parameters
        optimizer = pyro.optim.Adam({"lr": 0.02})
        elbo = Trace_ELBO()
        #guide = autoguide.AutoNormal(pyro.poutine.block(self.model, hide=['s1', "s2"]))
        guide = autoguide.AutoGuideList(self.model)
        guide.append(autoguide.AutoNormal(pyro.poutine.block(self.model, hide=['s1', "s2"])))
        guide.append(autoguide.AutoDiscreteParallel(pyro.poutine.block(self.model, expose=["s1", "s2"])))
        
        # initialize stochastic variational inference
        svi = SVI(
            model = self.model,
            guide = guide,
            optim = optimizer,
            loss = elbo
        )
        
        num_iterations = 5000
        train_loss = []
        for j in range(num_iterations):
        #for j in enumerate(self.train_dataloader):
            # calculate the loss and take a gradient step
            loss = svi.step()

            train_loss.append(loss/self.Y1.shape[0])
            #    test_loss.append(elbo.loss(self.model, guide, test_data))
            if j % 200 == 0:
                print("[iteration %04d] loss: %.4f" % (j + 1, loss / self.Y1.shape[0]))
            
            #with torch.no_grad():  # for logging only
                #train_loss = elbo.loss(self.model, guide, self.train_data) # or average over batch_loss
                #test_loss = elbo.loss(self.model, guide, self.test_data)
            #print(train_loss, test_loss)
        torch.save({"model": self.state_dict(), "guide" : guide}, "/mnt/storage/thien/FA_model.pt")
        pyro.get_param_store().save("/mnt/storage/thien/FA_model_params.pt")

            # Obtain maximum a posteriori estimates for W and Z
        map_estimates = guide(Y)
            
        return train_loss, map_estimates




FA_model = FA(Y,20, 30, n_factors)
losses, estimates = FA_model.train()


Y1_hat = torch.matmul(torch.Tensor(estimates["Z"].detach().numpy()),torch.Tensor(estimates["W1"].detach().numpy()).t())
correlations_pearson = []
correlations_spearman = []
for observation in range(n_obs):
    correlations_pearson_temp = []
    correlations_spearman_temp = []
    for observation2 in range(n_obs):
        correlations_pearson_temp.append(pearsonr(Y1[observation,:], Y1_hat[observation2,:]))
        correlations_spearman_temp.append(spearmanr(Y1[observation,:], Y1_hat[observation2,:], axis=0))
    correlations_pearson.append(max(correlations_pearson_temp))
    correlations_spearman.append(max(correlations_spearman_temp))

correlations_pearson
correlations_spearman



#load data
#Cite_GEX = sc.read_h5ad("/mnt/storage/anna/data/Cite_GEX_preprocessed.h5ad")
#Cite_ADT = sc.read_h5ad("/mnt/storage/anna/data/Cite_ADT_preprocessed.h5ad")

#Y = ad.concat([Cite_ADT, Cite_GEX], axis = 1, merge="same")
#Y = torch.tensor(Y.X.A)
