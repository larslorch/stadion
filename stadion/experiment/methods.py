import tensorflow as tf
tf.config.set_visible_devices([], 'GPU') # hide gpus to tf to avoid OOM and cuda errors by conflicting with jax

import copy
import argparse
import re
import platform
from pathlib import Path
import json
import subprocess
import numpy as onp
import random as pyrandom
import torch
from jax import random
import jax.numpy as jnp
import math

from stadion.utils.parse import load_methods_config, timer, get_git_hash_long, load_data, save_json
from stadion.core import index_data, get_intv_stats
from stadion.definitions import BASELINE_GIES, BASELINE_DCDI, BASELINE_IGSP, BASELINE_LLC, BASELINE_NODAGS, \
    BASELINE_OURS, FOLDER_TRAIN, FOLDER_TEST

from baselines import run_igsp, run_gies, run_dcdi, run_llc, run_nodags

from stadion.run import run_algo as run_ours
from stadion.intervention import search_intv_theta_shift


if __name__ == "__main__":
    """
    Runs methods on a data instance and creates predictions 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--descr", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--path_results", type=Path, required=True)
    parser.add_argument("--path_data_root", type=Path)
    parser.add_argument("--data_id", type=int)
    parser.add_argument("--path_methods_config", type=Path, required=True)
    parser.add_argument("--train_validation", action="store_true")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--n_train_validation", type=int, default=1)
    kwargs = parser.parse_args()

    # generate directory if it doesn't exist
    kwargs.path_results.mkdir(exist_ok=True, parents=True)
    (kwargs.path_results / "logs").mkdir(exist_ok=True, parents=True)

    # load data
    assert kwargs.data_id is not None
    methods_config = load_methods_config(kwargs.path_methods_config, abspath=True)
    assert kwargs.method in methods_config,  f"{kwargs.method} not in config with keys {list(methods_config.keys())}"
    config = methods_config[kwargs.method]

    train_targets, train_means, train_intv = load_data(kwargs.path_data_root / f"{kwargs.data_id}"/ FOLDER_TRAIN)
    test_means, test_intv = load_data(kwargs.path_data_root / f"{kwargs.data_id}"/ FOLDER_TEST, eval_mode=True)

    """
    Leave-one-out train data validation
    """
    # if `train_validation`, ignore test set and evaluate on a held-out train env for hyperparameter tuning
    if kwargs.train_validation:
        # checks
        del test_intv
        del test_means

        n_train_envs = len(train_targets.data)

        assert n_train_envs > 1
        assert onp.isclose(train_targets.intv[0].sum(-1), 0), "first train env should be observational"
        assert not onp.isclose(train_targets.intv[1:].sum(-1), 0).any(), "all but first train env should be interventional"

        # select one interventional environment of training distribution at random and make it the new test set
        rng = onp.random.default_rng(kwargs.seed)
        test_idx = rng.choice(onp.where(~onp.isclose(train_targets.intv.sum(-1), 0))[0],
                              size=kwargs.n_train_validation, replace=False)

        # assemble new data sets (first test, to avoid modification of train)
        test_targets =  index_data(train_targets, test_idx)
        train_targets = index_data(train_targets, onp.setdiff1d(onp.arange(n_train_envs), test_idx))

        assert len(train_targets) == n_train_envs - kwargs.n_train_validation
        assert len(test_targets) == kwargs.n_train_validation

        # recompute known test-distribution statistics
        train_means, train_intv = get_intv_stats(train_targets)
        test_means, test_intv = get_intv_stats(test_targets)


    """
    Run algorithm
    """
    base_run_name = f"{kwargs.method}_{kwargs.data_id}.json"
    base_run_name_validation = f"{kwargs.method}_train_validation_{kwargs.data_id}.json"
    base_method = kwargs.method.split("__")[0] # catch hyperparameter calibration case where name differs

    if base_method == BASELINE_GIES:

        # run gies
        with timer() as walltime:
            rng = onp.random.default_rng(kwargs.seed)
            sampler_gies, pred = run_gies(kwargs.seed, train_targets, config)

        # sample train predictions
        intv_theta_train = dict(shift=train_means)
        pred_samples_train = sampler_gies(rng, intv_theta_train, train_intv, n_samples=kwargs.n_samples)

        # sample test predictions
        intv_theta_test = dict(shift=test_means)
        pred_samples_test = sampler_gies(rng, intv_theta_test, test_intv, n_samples=kwargs.n_samples)

        # store in dictionary with results
        pred["samples_train"] = pred_samples_train
        pred["samples_test"] = pred_samples_test


    elif base_method == BASELINE_IGSP:

        # run igsp
        with timer() as walltime:
            rng = onp.random.default_rng(kwargs.seed)
            sampler_igsp, pred = run_igsp(kwargs.seed, train_targets, config)

        # sample train predictions
        intv_theta_train = dict(shift=train_means)
        pred_samples_train = sampler_igsp(rng, intv_theta_train, train_intv, n_samples=kwargs.n_samples)

        # sample test predictions
        intv_theta_test = dict(shift=test_means)
        pred_samples_test = sampler_igsp(rng, intv_theta_test, test_intv, n_samples=kwargs.n_samples)

        # store in dictionary with results
        pred["samples_train"] = pred_samples_train
        pred["samples_test"] = pred_samples_test


    elif base_method == BASELINE_DCDI:

        # run dcdi
        with timer() as walltime:
            rng = onp.random.default_rng(kwargs.seed)
            sampler_dcdi, pred = run_dcdi(kwargs.seed, train_targets, config)

        # set random seed
        pyrandom.seed(kwargs.seed)
        onp.random.seed(kwargs.seed)
        torch.manual_seed(kwargs.seed)
        torch.backends.cudnn.deterministic = True

        # sample train predictions
        intv_theta_train = dict(shift=train_means)
        pred_samples_train = sampler_dcdi(intv_theta_train, train_intv, n_samples=kwargs.n_samples)

        # sample test predictions
        intv_theta_test = dict(shift=test_means)
        pred_samples_test = sampler_dcdi(intv_theta_test, test_intv, n_samples=kwargs.n_samples)

        # store in dictionary with results
        pred["samples_train"] = pred_samples_train
        pred["samples_test"] = pred_samples_test


    elif base_method == BASELINE_LLC:

        # run llc
        with timer() as walltime:
            rng = onp.random.default_rng(kwargs.seed)
            sampler_llc, pred = run_llc(kwargs.seed, train_targets, config)

        # infer most likely parameters for train interventions via search
        key, subk = random.split(random.PRNGKey(kwargs.seed))
        intv_theta_train_init = dict(shift=jnp.zeros(train_intv.shape))
        intv_theta_train, search_logs_train = search_intv_theta_shift(subk, theta=None,
                                                                            intv_theta=intv_theta_train_init,
                                                                            target_means=train_means,
                                                                            target_intv=train_intv,
                                                                            sampler=sampler_llc,
                                                                            n_samples=kwargs.n_samples)

        key, subk = random.split(key)
        pred_samples_train = sampler_llc(subk, None, intv_theta_train, train_targets.intv, n_samples=kwargs.n_samples)

        # sample test predictions
        intv_theta_test_init = dict(shift=jnp.zeros(test_intv.shape))
        intv_theta_test, search_logs_test = search_intv_theta_shift(subk, theta=None,
                                                                          intv_theta=intv_theta_test_init,
                                                                          target_means=test_means,
                                                                          target_intv=test_intv,
                                                                          sampler=sampler_llc,
                                                                          n_samples=kwargs.n_samples)

        key, subk = random.split(key)
        pred_samples_test = sampler_llc(subk, None, intv_theta_test, test_intv, n_samples=kwargs.n_samples)


        # store in dictionary with results
        pred["samples_train"] = pred_samples_train
        pred["samples_test"] = pred_samples_test

        # add sanity check metrics from test intv_theta search
        pred["search_intv_theta_train_shift"] = {k.split("/")[-1]: v for k, v in search_logs_train.items()}
        pred["search_intv_theta_test_shift"] = {k.split("/")[-1]: v for k, v in search_logs_test.items()}

    elif base_method == BASELINE_NODAGS:

        # run nodags
        with timer() as walltime:
            rng = onp.random.default_rng(kwargs.seed)
            sampler_nodags, pred = run_nodags(kwargs.seed, train_targets, config)

        # infer most likely parameters for train interventions via search
        key, subk = random.split(random.PRNGKey(kwargs.seed))
        intv_theta_train_init = dict(shift=jnp.zeros(train_intv.shape))
        intv_theta_train, search_logs_train = search_intv_theta_shift(subk, theta=None,
                                                                      intv_theta=intv_theta_train_init,
                                                                      target_means=train_means,
                                                                      target_intv=train_intv,
                                                                      sampler=sampler_nodags,
                                                                      n_samples=kwargs.n_samples,
                                                                      jit_sampler=False)

        key, subk = random.split(key)
        pred_samples_train = sampler_nodags(subk, None, intv_theta_train, train_targets.intv, n_samples=kwargs.n_samples)

        # sample test predictions
        intv_theta_test_init = dict(shift=jnp.zeros(test_intv.shape))
        intv_theta_test, search_logs_test = search_intv_theta_shift(subk, theta=None,
                                                                    intv_theta=intv_theta_test_init,
                                                                    target_means=test_means,
                                                                    target_intv=test_intv,
                                                                    sampler=sampler_nodags,
                                                                    n_samples=kwargs.n_samples,
                                                                    jit_sampler=False)

        key, subk = random.split(key)
        pred_samples_test = sampler_nodags(subk, None, intv_theta_test, test_intv, n_samples=kwargs.n_samples)

        # store in dictionary with results
        pred["samples_train"] = pred_samples_train
        pred["samples_test"] = pred_samples_test

        # add sanity check metrics from test intv_theta search
        pred["search_intv_theta_train_shift"] = {k.split("/")[-1]: v for k, v in search_logs_train.items()}
        pred["search_intv_theta_test_shift"] = {k.split("/")[-1]: v for k, v in search_logs_test.items()}


    elif BASELINE_OURS in base_method:

        # run ours
        with timer() as walltime:
            config_namespace = argparse.Namespace(seed=kwargs.seed, **config)
            sampler, intv_theta_initializer, theta, intv_theta_train = \
                run_ours(train_targets, None, config=config_namespace, eval_mode=True)

        # infer most likely parameters for test interventions via search
        key, subk = random.split(random.PRNGKey(kwargs.seed))
        intv_theta_test_init = intv_theta_initializer(subk, *test_intv.shape, scale=0.0)
        key, subk = random.split(key)
        intv_theta_test, search_logs = search_intv_theta_shift(subk, theta=theta,
                                                               intv_theta=intv_theta_test_init,
                                                               target_means=test_means,
                                                               target_intv=test_intv,
                                                               sampler=sampler,
                                                               n_samples=kwargs.n_samples)
        # sample train predictions
        key, subk = random.split(key)
        pred_samples_train = sampler(subk, theta, intv_theta_train, train_targets.intv, n_samples=kwargs.n_samples)

        # sample test predictions
        key, subk = random.split(key)
        pred_samples_test = sampler(subk, theta, intv_theta_test, test_intv, n_samples=kwargs.n_samples)

        # store dictionary with results
        pred = dict(
            samples_train=pred_samples_train,
            samples_test=pred_samples_test,
            theta=theta,
            intv_theta_train=intv_theta_train,
            intv_theta_test=intv_theta_test,
        )

        # add sanity check metrics from test intv_theta search
        pred["search_intv_theta_shift"] = {k.split("/")[-1]: v for k, v in search_logs.items()}

        # remember eigenvalue information for analysis
        if "w1" in theta and theta["w1"].ndim == 2 and theta["w1"].shape[0] == theta["w1"].shape[1]:
            pred["theta_w1_eigenvals"] = onp.sort(onp.real(onp.linalg.eigvals(theta["w1"])))

    else:
        raise KeyError(f"Unknown method `{kwargs.method}`")

    t_finish = walltime() / 60.0 # mins

    """Save all predictions"""
    pred["method_base"] = base_method
    pred["method"] = kwargs.method
    pred["config"] = config
    pred["walltime"] = t_finish
    pred["git_version"] = get_git_hash_long()
    pred["train_validation"] = kwargs.train_validation
    pred["n_samples"] = kwargs.n_samples

    if kwargs.train_validation:
        pred["test_idx"] = onp.array(test_idx)

    # save
    filename = base_run_name_validation if kwargs.train_validation else base_run_name
    save_json(pred, kwargs.path_results / filename)

    print(f"{kwargs.descr}:  {kwargs.method} seed {kwargs.seed} data_id {kwargs.data_id} finished successfully.")