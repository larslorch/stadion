# coding=utf-8
"""
GraN-DAG

Copyright © 2019 Sébastien Lachapelle, Philippe Brouillard, Tristan Deleu

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
from functools import partial
import copy
import os
import math
import torch
import numpy as np
from types import SimpleNamespace
import warnings
warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"

from stadion.definitions import IS_CLUSTER
from stadion.utils.graph import is_acyclic, break_cycles_randomly, topological_ordering

try:
    from baselines.dcdi.learnables import LearnableModel_NonLinGaussANM
    from baselines.dcdi.flows import DeepSigmoidalFlowModel
    from .train import train, compute_loss
    from .data import CustomManagerFile
    from .utils.save import dump

except ImportError:
    from baselines.dcdi.learnables import LearnableModel_NonLinGaussANM
    from baselines.dcdi.flows import DeepSigmoidalFlowModel
    from train import train, compute_loss
    from data import CustomManagerFile
    from utils.save import dump


def _print_metrics(stage, step, metrics, throttle=None):
    for k, v in metrics.items():
        print("    %s:" % k, v)

def file_exists(prefix, suffix):
    return os.path.exists(os.path.join(prefix, suffix))

def main(opt, x, interv_mask, metrics_callback=_print_metrics, plotting_callback=None):
    """
    :param opt: a Bunch-like object containing hyperparameter values
    :param metrics_callback: a function of the form f(step, metrics_dict)
        used to log metric values during training
    """

    # Control as much randomness as possible
    torch.manual_seed(opt.random_seed)
    np.random.seed(opt.random_seed)

    if opt.lr_reinit is not None:
        assert opt.lr_schedule is None, "--lr-reinit and --lr-schedule are mutually exclusive"

    # Dump hyperparameters to disk
    # dump(opt.__dict__, opt.exp_path, 'opt')

    # Initialize metric logger if needed
    if metrics_callback is None:
        metrics_callback = _print_metrics

    # adjust some default hparams
    if opt.lr_reinit is None: opt.lr_reinit = opt.lr

    # Use GPU
    if opt.gpu and IS_CLUSTER:
        if opt.float:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    else:
        if opt.float:
            torch.set_default_tensor_type('torch.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.DoubleTensor')

    # # create experiment path
    # if not os.path.exists(opt.exp_path):
    #     os.makedirs(opt.exp_path)

    # raise error if not valid setting
    if not(not opt.intervention or \
    (opt.intervention and opt.intervention_type == "perfect" and opt.intervention_knowledge == "known") or \
    (opt.intervention and opt.intervention_type == "perfect" and opt.intervention_knowledge == "unknown") or \
    (opt.intervention and opt.intervention_type == "imperfect" and opt.intervention_knowledge == "known")):
        raise ValueError("Not implemented")

    # if observational, force interv_type to perfect/known
    if not opt.intervention:
        print("No intervention")
        opt.intervention_type = "perfect"
        opt.intervention_knowledge = "known"

    # create DataManager for training
    train_data = CustomManagerFile(x, interv_mask, opt.train_samples, opt.test_samples, train=True,
                                 normalize=opt.normalize_data,
                                 random_seed=opt.random_seed,
                                 intervention=opt.intervention,
                                 intervention_knowledge=opt.intervention_knowledge,
                                 dcd=opt.dcd,
                                 regimes_to_ignore=opt.regimes_to_ignore)
    test_data = CustomManagerFile(x, interv_mask, opt.train_samples, opt.test_samples, train=False,
                                normalize=opt.normalize_data, mean=train_data.mean, std=train_data.std,
                                random_seed=opt.random_seed,
                                intervention=opt.intervention,
                                intervention_knowledge=opt.intervention_knowledge,
                                dcd=opt.dcd,
                                regimes_to_ignore=opt.regimes_to_ignore)

    # create learning model and ground truth model
    if opt.model == "DCDI-G":
        model = LearnableModel_NonLinGaussANM(opt.num_vars,
                                              opt.num_layers,
                                              opt.hid_dim,
                                              nonlin=opt.nonlin,
                                              intervention=opt.intervention,
                                              intervention_type=opt.intervention_type,
                                              intervention_knowledge=opt.intervention_knowledge,
                                              num_regimes=train_data.num_regimes)
    elif opt.model == "DCDI-DSF":
        model = DeepSigmoidalFlowModel(num_vars=opt.num_vars,
                                       cond_n_layers=opt.num_layers,
                                       cond_hid_dim=opt.hid_dim,
                                       cond_nonlin=opt.nonlin,
                                       flow_n_layers=opt.flow_num_layers,
                                       flow_hid_dim=opt.flow_hid_dim,
                                       intervention=opt.intervention,
                                       intervention_type=opt.intervention_type,
                                       intervention_knowledge=opt.intervention_knowledge,
                                       num_regimes=train_data.num_regimes)
    else:
        raise ValueError("opt.model has to be in {DCDI-G, DCDI-DSF}")


    # save gt adjacency
    # dump(train_data.adjacency.detach().cpu().numpy(), opt.exp_path, 'gt-adjacency')

    # train until constraint is sufficiently close to being satisfied
    train(model, train_data.gt_interv, train_data, test_data, opt, metrics_callback, plotting_callback)

    # get predicted graph
    g_possibly_cyclic = model.adjacency.detach().cpu().numpy()
    g_edge_probs = (model.adjacency * model.get_w_adj()).detach().cpu().numpy()

    if np.allclose(g_possibly_cyclic, 1 - np.eye(g_possibly_cyclic.shape[-1])):
        warnings.warn("DCDI predicted all ones, probably not converged yet")

    # get graph and find topological ordering
    rng = np.random.default_rng(opt.random_seed)
    was_acyclic = is_acyclic(g_possibly_cyclic)
    if not was_acyclic:
        g = break_cycles_randomly(rng, g_possibly_cyclic.copy())
        warnings.warn(f"DCDI sampler got cyclic graph. Breaking cycles randomly.\n"
                      f"Given acyclic: {is_acyclic(g_possibly_cyclic)}:\n{g_possibly_cyclic}\n"
                      f"DAG acyclic: {is_acyclic(g)}:\n{g}\n")
        assert is_acyclic(g), "DAG forcing failed."

    else:
        g = g_possibly_cyclic

    toporder = topological_ordering(g)

    # construct output
    pred = dict(dag=g, g_edges=g_possibly_cyclic, g_edge_probs=g_edge_probs, is_acyclic=was_acyclic,
                theta=model.weights[0][..., 0].detach().cpu().numpy()) # save first layer of observational regime (0)

    assert np.allclose(train_data.gt_interv.T[0], 0), "Regime 0 should be observational"
    assert np.any(~np.isclose(train_data.gt_interv.T[1:], 0), axis=-1).all(0), "Any other regime should have at least 1 intervention"
    assert model.intervention_type == 'imperfect', "Sampler assumes interventions are imperfect to " \
                                                   "extract correct MLP from dimension 0 of weights"

    sampler = partial(model.sampler,
                      g=g,
                      toporder=toporder,
                      weights=copy.deepcopy(model.weights),
                      biases=copy.deepcopy(model.biases),
                      extra_params=copy.deepcopy(model.extra_params),
                      weight_sel=0)  # take regime 0 MLP for prediction

    print("Exiting run_dcdi succesfully. ", flush=True)

    return sampler, pred


def run_dcdi(seed, targets, config):
    """
    The below code is minimally modified from https://github.com/slachapelle/dcdi
    to accept the hyperparameters without argparse and our data format
    """

    # create interv mask for all observations
    interv_mask = []
    for data, intv in zip(targets.data, targets.intv):
        assert intv.shape[-1] == data.shape[-1]
        interv_mask.append(np.ones_like(data) * intv)

    # concatenate all observations
    x = np.concatenate([data for data in targets.data], axis=0)
    interv_mask = np.concatenate(interv_mask, axis=0)
    assert x.shape == interv_mask.shape

    # copy of the argparse arguments in https://github.com/slachapelle/dcdi/blob/master/main.py
    args = SimpleNamespace()
    args.random_seed = seed
    args.gpu = config.get("use_gpu", False)
    args.float = False

    args.num_vars = x.shape[1]
    args.train_samples = 0.8
    args.test_samples = None
    args.train_batch_size = min(config["train_batch_size"], math.floor(x.shape[0] * args.train_samples))
    args.num_train_iter = config["num_train_iter"]
    args.normalize_data = False
    args.regimes_to_ignore = None

    args.model = config["model"]
    args.num_layers = config["num_layers"]
    args.hid_dim = config["hid_dim"]
    args.nonlin = "leaky-relu"
    args.flow_num_layers = config["num_layers"]
    args.flow_hid_dim = config["hid_dim"]

    args.intervention = True
    args.dcd = False # Use DCD (DCDI with a loss not taking into account the intervention)
    args.intervention_type = "imperfect"
    args.intervention_knowledge = "known"
    args.coeff_interv_sparsity = 1e-8

    args.optimizer = "rmsprop"
    args.lr = config["learning_rate"]
    args.lr_reinit = None
    args.lr_schedule = None
    args.stop_crit_win = 100
    args.reg_coeff = config["reg_coeff"]

    args.omega_gamma = 1e-4
    args.omega_mu = 0.9
    args.mu_init = 1e-8
    args.mu_mult_factor = 2
    args.gamma_init = 0.0
    args.h_threshold = 1e-8

    args.patience = 10
    args.train_patience = 5
    args.train_patience_post = 5
    args.lr_schedule = None
    args.lr_schedule = None

    args.no_w_adjs_log = True

    # run dcdi
    return main(args, x, interv_mask)

