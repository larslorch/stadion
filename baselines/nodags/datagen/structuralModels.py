from matplotlib.style import available
import networkx as nx
import numpy as np 
import torch
import math

from baselines.nodags.models.functions import indMLPFunction, nonlinearMLP
from baselines.nodags.models.resblock import iResBlock

def standard_normal_logprob(z, noise_scale=0.5):
    logZ = -0.5 * np.log(2 * math.pi * noise_scale**2)
    return logZ - z**2 / (2 * noise_scale**2)

def make_non_cotractive(weights):
    s = np.linalg.svd(weights, compute_uv=False)
    scale = 1.0
    if s[0] <= 1.0:
        scale = 2/s[0]
    
    return scale * weights 

def make_contractive(weights):
    s = np.linalg.svd(weights, compute_uv=False)
    scale=1.1
    if s[0] >= 1.0:
        scale = 1.1 * s[0]
    
    return weights/scale


class linearSEM:

    """
    -------------------------------------------------------------------
    This class models a Linear Structural Equation Model (Linear SEM)
    -------------------------------------------------------------------
    The model is initialized with the number of nodes in the graph and
    the absolute minimum and maximum weights for the edges. 
    """
    def __init__(self, graph, weights=None, abs_weight_low=0.2, abs_weight_high=0.9, noise_scale=0.5, contractive=True):
        self.graph = graph
        self.abs_weight_low = abs_weight_low 
        self.abs_weight_high = abs_weight_high
        self.contractive = contractive
        self.noise_scale = np.array(noise_scale)

        self.n_nodes = len(graph.nodes)

        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.uniform(self.abs_weight_low, self.abs_weight_high, size=(self.n_nodes, self.n_nodes))
            self.weights *= 2 * np.random.binomial(1, 0.5, size=self.weights.shape) - 1
            self.weights *= nx.to_numpy_array(self.graph)

            if not self.contractive:
                self.weights = make_non_cotractive(self.weights)
            else:
                self.weights = make_contractive(self.weights)

    def generateData(self, rng, n_samples, intervention_set=[None], fixed_intervention=False, intervention=None):
        observed_set = np.setdiff1d(np.arange(self.n_nodes), intervention_set)
        U = np.zeros((self.n_nodes, self.n_nodes))
        U[observed_set, observed_set] = 1

        C = np.zeros((self.n_nodes, n_samples))
        if intervention_set[0] != None:
            if fixed_intervention:
                C_interv = rng.normal(size=len(intervention_set))
            else:
                C_interv = rng.normal(size=(len(intervention_set), n_samples))

            if intervention is not None:
                assert C_interv.shape == intervention.shape, f"{C_interv.shape} {intervention.shape}"
                C_interv = intervention

            if C_interv.ndim == 1:
                C_interv = C_interv[:, np.newaxis]

            C[intervention_set, :] = C_interv

        I = np.eye(self.n_nodes)
        E = rng.normal(size=(self.n_nodes, n_samples)) * self.noise_scale[..., None]
        X = np.linalg.inv(I - U @ self.weights.T) @ (U @ E + C)

        # return: [N, d]
        return X.T

    def compute_log_likelihood(self, x, intervention_set):
        # masks intervened variables
        d = x.shape[-1]
        I = np.eye(d)
        observed_set = np.setdiff1d(np.arange(d), intervention_set)
        U = np.zeros((d, d))
        U[observed_set, observed_set] = 1

        e = x @ (I - self.weights @ U)
        logpe = standard_normal_logprob(e[..., observed_set],
                                        noise_scale=self.noise_scale[observed_set]
                                        if self.noise_scale.ndim > 0 else self.noise_scale).sum(axis=-1)
        det = np.linalg.det(I - self.weights.T)
        logdetgrad = math.log(np.abs(det))
        logdetgrad_vec = np.ones(logpe.shape) * logdetgrad
        logpx = logpe + logdetgrad_vec
        return logpx


class nonlinearSEM:
    """
    ----------------------------------------------------------------------
    This class models a Nonlinear Structural Equation Model (Linear SEM)
    ----------------------------------------------------------------------
    The nonlinear function is taken from models.functions 
    """

    def __init__(self, graph, lip_const=0.9, fun_type='sin-mlp', act_fun='tanh', device=None, noise_scale=0.5, n_hidden=1, bias=False, contractive=True):
        self.lip_const = lip_const 
        self.graph = graph 
        self.n_nodes = len(graph.nodes)
        self.act_fun = act_fun
        self.n_hidden = n_hidden
        self.bias = bias

        self.contractive = contractive 
        if self.contractive:
            self.lip_const = 2.0

        if fun_type == 'mul-mlp':
            self.f = indMLPFunction(n_nodes=self.n_nodes, 
                                    lip_constant=self.lip_const,
                                    activation=self.act_fun,
                                    n_layers=n_hidden,
                                    full_input=False,
                                    graph_given=True,
                                    graph=self.graph, 
                                    bias=self.bias)
        else:
            self.f = nonlinearMLP(n_nodes=self.n_nodes, 
                                  lip_constant=self.lip_const,
                                  n_layers=self.n_hidden, 
                                  bias=self.bias,
                                  activation_fn=self.act_fun, 
                                  graph_given=True, 
                                  graph=self.graph)
            
        if device != None:
            self.f = self.f.to(device)
        self.device = device
        self.noise_scale = noise_scale
        
        
    def generateData(self, rng, n_samples, intervention_set=[None], noise_provided=False, noise_vec=None, n_iter=30,
                     fixed_intervention=False):

        with torch.no_grad():
            observed_set = np.setdiff1d(np.arange(self.n_nodes), intervention_set)
            U = torch.zeros(self.n_nodes, self.n_nodes, device=self.device).float()
            U[observed_set, observed_set] = 1
            
            C = torch.zeros(n_samples, self.n_nodes, device=self.device)
            if intervention_set[0] != None:
                if fixed_intervention:
                    C[:, intervention_set] = torch.tensor(rng.normal(size=(1, len(intervention_set))), device=self.device).float()
                else:
                    C[:, intervention_set] = torch.tensor(rng.normal(size=(n_samples, len(intervention_set))), device=self.device).float()

            if noise_provided:
                E = noise_vec
                if E.device != self.device:
                    E = E.to(self.device)
            else:
                E = self.noise_scale * torch.tensor(rng.normal(size=(n_samples, self.n_nodes)), device=self.device).float()

            X = torch.randn(n_samples, self.n_nodes, device=self.device).float()
            for _ in range(n_iter):
                X = self.f(X) @ U + E @ U + C 
        
        return X.cpu().numpy()

    def computeNLL(self, x, intervention_set):
        I = np.eye(x.shape[1])
        observed_set = np.setdiff1d(np.arange(x.shape[1]), intervention_set)
        U = np.zeros((x.shape[1], x.shape[1]))
        U[observed_set, observed_set] = 1

        x = torch.tensor(x).float().to(self.device)
        resb = iResBlock(self.f, n_power_series=None)
        e, logdetgrad = resb(x, intervention_set, logdet=True, neumann_grad=False)
        logpe = standard_normal_logprob(e[:, observed_set], noise_scale=self.noise_scale).sum(axis=1)
        
        logpx = logpe + logdetgrad
        # The final data matrix is dimensions - n_samples X self.nodes
        return -1 * logpx.mean()


