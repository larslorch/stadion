import os
import argparse
import time
import math
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD

from baselines.nodags.datagen.generateDataset import Dataset
from baselines.nodags.datagen.torchDataset import experimentDataset, experimentDatasetStrat

from baselines.nodags.models.functions import indMLPFunction, linearFunction, nonlinearMLP, factorMLPFunction, gumbelSoftMLP
from baselines.nodags.models.resblock import iResBlock
from baselines.nodags.models.layers.mlpLipschitz import linearLipschitz
from baselines.nodags.utils import *

# Helper functions

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def standard_normal_logprob(z, noise_scales):
    logZ = -0.5 * torch.log(2 * math.pi * (noise_scales.pow(2)))
    return logZ - z.pow(2) / (2 * (noise_scales.pow(2)))

def computeNLL(noise, intervention_set, logdetgrad, noise_scales):
    observed_set = np.setdiff1d(np.arange(noise.shape[1]), intervention_set)
    logpe = standard_normal_logprob(noise[:, observed_set], noise_scales=noise_scales).sum(1, keepdim=True)
    logpx = logpe + logdetgrad
    return -torch.mean(logpx).detach().cpu().numpy()

def compute_loss(model, x, intervention_set, l1_regularize=False, lambda_c=1e-2, fun_type=None):
    e, logdetgrad = model(x, intervention_set, logdet=True)
    observed_set = np.setdiff1d(np.arange(x.shape[1]), intervention_set)
    lat_var = torch.exp(model.var[observed_set])
    logpe = standard_normal_logprob(e[:, observed_set], noise_scales=lat_var).sum(1, keepdim=True)
    logpx = logpe + logdetgrad        
    loss = -torch.mean(logpx)
    if l1_regularize:
        if fun_type == 'fac-mlp':
            l1_norm = model.f.get_w_adj().abs().sum()
        elif fun_type == 'gst-mlp':
            l1_norm = model.f.get_w_adj().abs().sum()
            # print(l1_norm)
        else:
            l1_norm = sum(p.abs().sum() for p in model.parameters())

        loss += lambda_c * l1_norm
    return loss, torch.mean(logpe), torch.mean(logdetgrad)
    
def update_lipschitz(model, n_iterations):
    for m in model.modules():
        if isinstance(m, linearLipschitz):
            m.compute_weight(update=True, n_iterations=n_iterations)

class resflow_train_test_wrapper:
    def __init__(self,
                 n_nodes,
                 batch_size=64,
                 l1_reg=False,
                 lambda_c=1e-2,
                 n_lip_iter=5,
                 fun_type='mul-mlp',
                 lip_const=0.9,
                 act_fun='tanh',
                 lr=1e-3,
                 epochs=10,
                 optim='sgd',
                 v=False, 
                 inline=False,
                 upd_lip=True,
                 full_input=False, 
                 n_hidden=1, 
                 n_factors=10, 
                 var=None,
                 n_power_series=None, 
                 init_var=0.5,
                 lin_logdet=False, 
                 dag_input=False, 
                 thresh_val=1e-2, 
                 centered=True):
        
        self.n_nodes = n_nodes
        self.batch_size = batch_size
        self.l1_reg = l1_reg
        self.lambda_c = lambda_c
        self.n_lip_iter = n_lip_iter
        self.fun_type = fun_type
        self.lip_const = lip_const
        self.act_fun = act_fun
        self.lr = lr
        self.epochs = epochs
        self.optim = optim
        self.v = v
        self.inline = inline
        self.upd_lip = upd_lip
        self.full_input = full_input
        self.n_hidden = n_hidden
        self.n_factors = n_factors
        self.var = var 
        self.n_power_series = n_power_series
        self.lin_logdet = lin_logdet
        self.thresh_val = thresh_val
        self.centered = centered

        if self.v or self.inline:
            print("Initializing the model")

        if self.fun_type == 'mul-mlp':
            self.f = indMLPFunction(n_nodes=self.n_nodes, lip_constant=self.lip_const, activation=self.act_fun, full_input=self.full_input, n_layers=n_hidden)
        elif self.fun_type == 'lin-mlp':
            self.f = linearFunction(n_nodes=self.n_nodes, lip_constant=self.lip_const, full_input=self.full_input)
        elif self.fun_type == 'nnl-mlp':
            self.f = nonlinearMLP(n_nodes=self.n_nodes, lip_constant=self.lip_const, n_layers=self.n_hidden, full_input=self.full_input, activation_fn=self.act_fun)
        elif self.fun_type == 'fac-mlp':
            self.f = factorMLPFunction(n_nodes=self.n_nodes, n_factors=self.n_factors, lip_constant=self.lip_const, n_hidden=self.n_hidden, activation=self.act_fun)
        elif self.fun_type == 'gst-mlp':
            self.f = gumbelSoftMLP(n_nodes=self.n_nodes, lip_constant=self.lip_const, n_hidden=self.n_hidden, activation=self.act_fun)
        
        self.model = iResBlock(self.f, n_power_series=self.n_power_series, var=self.var, init_var=init_var, dag_input=dag_input, lin_logdet=self.lin_logdet, centered=self.centered)
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            # print("Available CUDA devices: {}".format(torch.cuda.device_count()))
            # if torch.cuda.device_count() > 1:
            #     self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        if self.v or self.inline:
            print("Number of Parameters : {}".format(count_parameters(self.model)))
    
    def n_parameters(self):
        return count_parameters(self.model)

    def train(self, datasets, intervention_sets, return_time=False, return_loss=False, batch_size=64):
        # training
        training_dataset = [experimentDataset(data, intervention_sets[i]) for i, data in enumerate(datasets)]
        train_dataloader = [DataLoader(training_data, batch_size=batch_size) for training_data in training_dataset]
        if self.inline:
            print("Starting Training")
        if self.optim == 'sgd':
            optimizer = SGD(self.model.parameters(), lr=self.lr)
        else:
            optimizer = Adam(self.model.parameters(), lr=self.lr)

        loss_rep = 0
        count = 0
        start_time = time.time()
        for epoch in range(self.epochs):
            for exp, dataloader in enumerate(train_dataloader):
                for batch, x in enumerate(dataloader):
                    optimizer.zero_grad()
                    
                    intervention_set = intervention_sets[exp]
                    x = x.float().to(self.device)
                    loss, logpe, logdetgrad = compute_loss(self.model, x, intervention_set, l1_regularize=self.l1_reg, fun_type=self.fun_type, lambda_c=self.lambda_c)
                    loss_rep += loss.item()
                    count += 1

                    if batch % 20 == 0:
                        if self.v: 
                            print("Exp: {}/{}, Epoch: {}/{}, Batch: {}/{}, Loss: {}, Log(pe): {}, logdetjac: {}".format(exp+1, len(train_dataloader), epoch+1, self.epochs, batch, len(dataloader), loss.item(), logpe.item(), logdetgrad.item()))
                        elif self.inline:
                            loss_rep /= count
                            count = 0
                            print("Exp: {}/{}, Epoch: {}/{}, Batch: {}/{}, Loss: {}".format(exp+1, len(train_dataloader), epoch+1, self.epochs, batch, len(dataloader), loss_rep), end='\r', flush=True)
                            loss_rep = 0
                    
                    loss.backward()
                    optimizer.step()
                    if self.upd_lip:
                        update_lipschitz(self.model, self.n_lip_iter)
                        
        stop_time = time.time()
        seconds = int(stop_time - start_time)

        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if return_time and return_loss:
            return (h, m, s), (loss.item(), logdetgrad.item()) 
        elif return_time:
            return h, m, s
        elif return_loss:
            return loss.item(), logdetgrad.item()
        
    def threshold(self):
        # Threshold the adjacency matrix and set graph_given = True, and store the threhosld adjacency to self.model.f.graph_adj = adj_mat >= t
        W = self.get_adjacency()
        adj_mat = W >= self.thresh_val 
        self.model.f.graph_given = True 
        self.model.f.graph_adj = adj_mat 
    
    def get_adjacency(self):
  
        if self.fun_type == 'mul-mlp':
            W, _ = get_adjacency_from_func(self.model.f, threshold=1.1, full_input=self.full_input)
        elif self.fun_type == 'lin-mlp':
            W = get_adj_from_single_func(self.model.f, device=self.device)
            # W = np.abs(W.T)
        elif self.fun_type == 'nnl-mlp':
            W = get_adj_from_single_func(self.model.f, device=self.device)
        elif self.fun_type == 'fac-mlp':
            W = np.abs(self.f.get_w_adj().detach().cpu().numpy())
        elif self.fun_type == 'gst-mlp':
            W = np.abs(self.f.get_w_adj().detach().cpu().numpy())
        
        if self.model.f.graph_given:
            return self.model.f.graph_adj * W       
        
        return W
    
    def get_auroc(self, W_gt):

        _, _, area = compute_auroc(W_gt, self.get_adjacency())
        return area

    def get_shd(self, W_gt):
        W_est = self.model.f.graph_adj
        shd, _ = compute_shd(W_gt, W_est)
        return shd

    def get_auprc(self, W_gt, n_points=50):
        baseline, area = compute_auprc(W_gt, self.get_adjacency(), n_points=n_points)
        return baseline, area

    def store_figure(self, graph, generative_model, output_path="figures", gid=1):
        fig, axs = plt.subplots(1, 3)
        fig.set_size_inches(12, 4)
        pos = nx.circular_layout(graph)
        nx.draw(graph, pos=pos, with_labels=True, ax=axs[0])
        axs[0].set_title("Graph")

        axs[1].set_title("Ground Truth - Adj")
        axs[1].imshow(np.abs(generative_model.weights) > 0)

        W = self.get_adjacency()
        axs[2].set_title("Estimated - Adj")
        axs[2].imshow(W)
        plt.savefig(os.path.join(output_path, 'd_{}_g_{}_f_{}_af_{}.png'.format(self.n_nodes, gid, self.fun_type, self.act_fun)))

    def predict(self, noises, intervention_sets, n_iter=10, init_provided=False, x_init=None):
        pred_datasets = list()
        i = 0
        for noise, intervention_set in zip(noises, intervention_sets):
            lat_t = torch.tensor(noise).float().to(self.model.device)
            data_pred = self.model.predict_from_noise(lat_t, n_iter, intervention_set=intervention_set, init_provided=init_provided, x_init=x_init[i]) 
            i += 1
            data_pred = data_pred.detach().cpu().numpy()
            pred_datasets.append(data_pred)
        return pred_datasets    
    
    def forwardPass(self, datasets):
        predictions = list()
        for dataset in datasets:
            data_t = torch.tensor(dataset).float().to(self.device)
            f_x = self.model.f(data_t)
            predictions.append(f_x.detach().cpu().numpy())
        
        return predictions
    
    def predictLikelihood(self, datasets, intervention_sets):
        likelihood_list = list()
        for dataset, intervention_set in zip(datasets, intervention_sets):
            data_t = torch.tensor(dataset).float().to(self.device)
            noises, logdetgrad = self.model(data_t, intervention_set, logdet=True, neumann_grad=False)
            observed_set = np.setdiff1d(np.arange(dataset.shape[1]), intervention_set)
            lat_var = torch.exp(self.model.var[observed_set])
            nll = computeNLL(noises, intervention_set, logdetgrad, noise_scales=lat_var)
            likelihood_list.append(nll.item()/self.n_nodes)
        
        return likelihood_list
    
    def predictSamples(self, intervention, intervention_set, n_samples=5000, noise_vec=None, imperfect=False):
        if noise_vec is None:
            noise_scale = np.diag(np.exp(self.model.var.detach().cpu().numpy()))
            noise_vec = np.random.randn(n_samples, self.n_nodes) @ noise_scale
        else:
            assert noise_vec.shape == (n_samples, self.n_nodes)
        noise_vec = torch.tensor(noise_vec).float().to(self.device)
        x = self.model.predict_from_noise(noise_vec, intervention_set=intervention_set, x_init=intervention, imperfect=imperfect)
        return x  

    def _pred_cond_mean_for_intervention(self, dataset, intervention_set):
        noise_scale = np.diag(np.exp(self.model.var.detach().cpu().numpy()))
        noise_vec = np.random.randn(dataset.shape[0], dataset.shape[1]) @ noise_scale
        noise_vec = torch.tensor(noise_vec).float().to(self.device)
        x_init = torch.tensor(dataset).float().to(self.device)
        x = self.model.predict_from_noise(noise_vec, intervention_set=intervention_set, x_init=x_init)
        return x.detach().cpu().numpy().mean(axis=0)

    def predictConditionalMean(self, datasets, intervention_sets):
        cond_mean_list = list()
        for dataset, intervention_set in zip(datasets, intervention_sets):
            cond_mean = self._pred_cond_mean_for_intervention(dataset, intervention_set)
            cond_mean_list.append(cond_mean)
        return cond_mean_list

if __name__ == '__main__':

    # Parsing command line arguments

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_nodes', type=int, default=5)
    parser.add_argument('--exp_dens', type=int, default=1)
    parser.add_argument('--n_samples', type=int, default=5000)
    parser.add_argument('--gen_model', type=str, choices=['lin', 'nnl'], default='lin')
    parser.add_argument('--n_exp', type=int, default=5)
    parser.add_argument('--mode', type=str, choices=['indiv-node', 'no-constraint', 'sat-pair-condition'], default='indiv-node')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--l1_reg', action='store_true', default=False)
    parser.add_argument('--lambda_c', type=float, default=1e-2)
    parser.add_argument('--n_lip_iter', type=int, default=5)
    parser.add_argument('--fun_type', type=str, choices=['mul-mlp', 'lin-mlp', 'nnl-mlp'], default='mul-mlp')
    parser.add_argument('--lip_const', type=float, default=0.9)
    parser.add_argument('--act_fun', type=str, choices=['tanh', 'relu', 'sigmoid'], default='tanh')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--optim', type=str, choices=['adam', 'sgd'], default='sgd')
    parser.add_argument('--gid', type=int, default=1)
    parser.add_argument('--v', action='store_true', default=False)
    parser.add_argument('--store_fig', action='store_true', default=False)
    parser.add_argument('--inline', action='store_true', default=False)
    parser.add_argument('--upd_lip', action='store_true', default=False)
    parser.add_argument('--full_input', action='store_true', default=False)
    parser.add_argument('--dag-input', action='store_true', default=False)
    parser.add_argument('--no-inter', action='store_false', default=True) 
    parser.add_argument('--lin-logdet', action='store_true', default=False)
    
    args = parser.parse_args()



    # Generate the Graph and the dataset. 

    print("Generating the graph and the dataset")

    dataset_gen = Dataset(n_nodes=args.n_nodes, 
                        expected_density=args.exp_dens, 
                        n_samples=args.n_samples, 
                        n_experiments=args.n_exp, 
                        mode=args.mode, 
                        enforce_dag=True)
    dataset = dataset_gen.generate()
    graph = dataset_gen.graph
    generative_model = dataset_gen.gen_model


    resblock = resflow_train_test_wrapper(n_nodes=args.n_nodes,
                               batch_size=args.batch_size,
                               l1_reg=args.l1_reg,
                               lambda_c=args.lambda_c,
                               n_lip_iter=args.n_lip_iter,
                               fun_type=args.fun_type,
                               lip_const=args.lip_const,
                               act_fun=args.act_fun,
                               lr=args.lr,
                               epochs=args.epochs, 
                               optim=args.optim,
                               v=args.v,
                               inline=args.inline,
                               upd_lip=args.upd_lip,
                               full_input=args.full_input,
                               lin_logdet=args.lin_logdet,
                               dag_input=args.dag_input)
    h, m, s = resblock.train(dataset, dataset_gen.targets, return_time=True, batch_size=args.batch_size)

    if args.store_fig:
        resblock.store_figure(graph, generative_model, gid=args.gid)

    area = resblock.get_auprc(np.abs(generative_model.weights) > 0)
    print()
    print("ID: {}, Elapsed time: {:d}:{:02d}:{:02d}, AUPRC: {}".format(args.gid, h, m, s, area))
    lat_var = np.exp(resblock.model.var.detach().cpu().numpy())
    print("Estimated Latent variance: {}".format(lat_var))

    val_dataset_gen = Dataset(n_nodes=args.n_nodes,
                             expected_density=1, 
                             n_samples=1000, 
                             n_experiments=10, 
                             mode='no-constraint',
                             min_targets=2,
                             max_targets=2,
                             graph_provided=True,
                             graph=graph,
                             gen_model_provided=True,
                             gen_model=generative_model)
    val_datasets = val_dataset_gen.generate(fixed_interventions=True)
    nll_list = resblock.predictLikelihood(val_datasets, val_dataset_gen.targets)
    print("Average NLL: {}".format(np.mean(nll_list)))