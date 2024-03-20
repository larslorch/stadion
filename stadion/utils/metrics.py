import functools
from collections import defaultdict
from functools import partial
import math

import jax
import jax.random as random
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as onp
from jax import vmap

from ott.geometry import pointcloud
from ott.tools.sinkhorn_divergence import sinkhorn_divergence


def squared_norm(x, y):
    # handle singular dims
    x_single = x.ndim == 1
    y_single = y.ndim == 1
    if x_single:
        x = x[..., None, :]
    if y_single:
        y = y[..., None, :]

    # kernel
    k = jnp.power(x[..., None, :] - y[..., None, :, :], 2).sum(-1)

    # handle singular dims
    if x_single and y_single:
        k = k.squeeze((-1, -2))
    elif x_single:
        k = k.squeeze(-2)
    elif y_single:
        k = k.squeeze(-1)
    return k


def rbf_kernel(x, y, ls):
    assert type(ls) == float or ls.ndim == 0
    return jnp.exp(- squared_norm(x, y) / (2.0 * (ls ** 2)))


def matern_12_kernel(x, y, ls):
    assert type(ls) == float or ls.ndim == 0
    r = jnp.sqrt(squared_norm(x, y))
    return jnp.exp(- r / ls)


def matern_32_kernel(x, y, ls):
    assert type(ls) == float or ls.ndim == 0
    r = jnp.sqrt(squared_norm(x, y))
    return (1.0 + jnp.sqrt(3) * r / ls) * jnp.exp(- jnp.sqrt(3) * r / ls)


def matern_52_kernel(x, y, ls):
    assert type(ls) == float or ls.ndim == 0
    r2 = squared_norm(x, y)
    r = jnp.sqrt(r2)
    return (1.0 + jnp.sqrt(5) * r / ls + 5 * r2 / (3 * (ls ** 2))) * jnp.exp(- jnp.sqrt(5) * r / ls)


@jax.jit
def mmd_fun(target_x, target_y, ls):
    n_x, n_y = target_x.shape[-2], target_y.shape[-2]

    k_xx = rbf_kernel(target_x, target_x, ls=ls)
    k_xy = rbf_kernel(target_x, target_y, ls=ls)
    k_yy = rbf_kernel(target_y, target_y, ls=ls)

    dis =  (1. / (n_x * (n_x - 1))) * (k_xx.sum((-1, -2)) - jnp.einsum("...ii->...", k_xx))
    dis -= (2. / (n_x * n_y)) * k_xy.sum((-1, -2))
    dis += (1. / (n_y * (n_y - 1))) * (k_yy.sum((-1, -2)) - jnp.einsum("...ii->...", k_yy))
    return dis


@jax.jit
def mse_fun(samples_pred, samples_target):
    mu_pred = samples_pred.mean(-2)
    mu_tar = samples_target.mean(-2)
    return jnp.square(mu_pred - mu_tar).mean(-1)


@jax.jit
def relmse_fun(samples_pred, samples_target):
    mu_pred = samples_pred.mean(-2)
    mu_tar = samples_target.mean(-2)
    return (jnp.square(mu_pred - mu_tar).sum(-1) ** 0.5) / (jnp.square(mu_tar).sum(-1) ** 0.5)


@jax.jit
def vse_fun(samples_pred, samples_target):
    std_pred = samples_pred.std(-2)
    std_tar = samples_target.std(-2)
    return jnp.square(std_pred - std_tar).mean(-1)


@jax.jit
def kde_nll_fun(pred, target):
    n_pred, d = pred.shape
    bw_scotts = float(n_pred) ** float(-1. / (d + 4))
    sig_scotts = math.sqrt(bw_scotts / 2.0)

    # [n_target, n_pred] diagonal Gaussian assumption summing over last axis
    logprobs = vmap(jsp.stats.norm.logpdf, (0, None, None), 0)(target, pred, sig_scotts).sum(-1)

    # [n_target]
    ll = jsp.special.logsumexp(logprobs, axis=-1) - jnp.log(n_pred)

    # return average negative log likelihood across target datapoints
    return - ll.mean(0)


@jax.jit
def wasserstein_fun(target_x, target_y, epsilon):
    assert target_x.ndim == target_y.ndim == 2 and target_x.shape[-1] == target_y.shape[-1]
    a = jnp.ones(len(target_x)) / len(target_x)
    b = jnp.ones(len(target_y)) / len(target_y)

    out_xy = sinkhorn_divergence(
        pointcloud.PointCloud,
        target_x,
        target_y,
        a=a,
        b=b,
        epsilon=epsilon,
        symmetric_sinkhorn=False,
    )
    return out_xy.divergence


def is_nan(samples):
    nan = onp.isnan(samples).any(axis=-2)
    return onp.any(nan, axis=-1).astype(float)


def is_inf(samples, inf_bound=1e10, std_bound=1e2):
    inf = (onp.abs(samples) > inf_bound).any(axis=-2)
    inf_std = onp.std(samples, axis=-2) > std_bound
    return onp.any(inf | inf_std, axis=-1).astype(float)


def is_unstable(samples, inf_bound=1e10, std_bound=1e2):
    return (is_nan(samples).astype(bool) |
            is_inf(samples, inf_bound=inf_bound, std_bound=std_bound).astype(bool)).astype(float)


def mat_accuracy(true_params, theta):
    if true_params is None:
        return None
    else:
        true = true_params[0]
        pred = theta["w1"]
        return jnp.abs(true - pred).sum()


def balanced_mat_accuracy(true_params, theta):
    if true_params is None:
        return None
    else:
        true = true_params[0]
        pred = theta["w1"]

        # get zero/nonzero masks
        nonzero_mask = (~onp.isclose(true, 0.0)).astype(float)
        zero_mask = (onp.isclose(true, 0.0)).astype(float)

        # accuracies in for both zero/nonzero entries
        nonzero_acc = onp.abs(nonzero_mask * (true - pred)).sum()
        zero_acc = onp.abs(zero_mask * (true - pred)).sum()

        nonzeros = nonzero_mask.sum()
        zeros = zero_mask.sum()

        acc = (nonzero_acc / onp.where(onp.isclose(nonzeros, 0.0), 1.0, nonzeros)) + \
              (zero_acc / onp.where(onp.isclose(zeros, 0.0), 1.0, zeros))
        return acc


def metric_fun_envs(fun, x, y, intv, *args):
    """
    Helper function for evaluating a metric in parallel over multiple environments
    For now done using a for loop because x_env/y_env may not have the same shapes for real datasets
    """
    assert len(x) == len(y) == len(intv), f"x: {x.shape}\ny: {y.shape}\nintv: {intv.shape}"

    fun = jax.jit(fun)

    metric = []
    metric_intv = []

    for pred_env, tar_env, intv_env, *sargs in zip(x, y, intv, *args):
        # full
        metric.append(fun(pred_env, tar_env, *sargs))

        # non intervened dims
        # metric_nonintv.append(fun(x_env * (1. - intv_env), y_env * (1. - intv_env), *sargs))

        # intervened dims
        metric_intv.append(fun(pred_env * intv_env, tar_env * intv_env, *sargs))

    metric = jnp.array(metric)
    metric_intv = jnp.array(metric_intv)

    assert (x.shape[0],) == metric.shape == metric_intv.shape, \
        f"x: {x.shape}\nmetric: {metric.shape}\nmetric_intv: {metric_intv.shape}"

    return metric, metric_intv


mmd_fun_envs = partial(metric_fun_envs, mmd_fun)
wasserstein_fun_envs = partial(metric_fun_envs, wasserstein_fun)
mse_fun_envs = partial(metric_fun_envs, mse_fun)
relmse_fun_envs = partial(metric_fun_envs, relmse_fun)
vse_fun_envs = partial(metric_fun_envs, vse_fun)
kde_nll_fun_envs = partial(metric_fun_envs, kde_nll_fun)


def make_metric(metr_fun_envs, *args, sampler, n=256):
    """
    Helper function for evaluating a metric evaluating in parallel over multiple environments
    using a sampler and target data
    """
    @jax.jit
    def _metric_fun(key, tars, theta, intv_theta):
        # simulate model rollouts for each env
        key, subk = random.split(key)
        pred_samples = sampler(subk, theta, intv_theta, tars.intv, n_samples=n)

        # [*envs, n_test_samples, d]
        assert pred_samples.shape[-2] >= n
        pred_samples = pred_samples[..., :n, :]

        # # sample batch of test data for each env
        # # list of [<=n_test_samples, d]
        # key, subk = random.split(key)
        # tars_batched = sample_subset(key, tars, n)
        tars_batched = tars

        # iterate over envs individually and compute metric
        # (python for loop because target env shapes may not be the same)
        metrics, imetrics = metr_fun_envs(pred_samples, tars_batched.data, tars_batched.intv, *args)

        # only count intv metric if we had an intervention
        msk = jnp.isclose(tars_batched.intv, 1).any(-1).astype(jnp.float32)
        assert metrics.shape == imetrics.shape == msk.shape == (len(tars.data),)

        metrics_mean = metrics.mean(0)
        imetrics_mean = (imetrics * msk).sum() / msk.sum()

        return metrics_mean, imetrics_mean

    def metric_fun(*args):
        return tuple([out.item() for out in _metric_fun(*args)])  # call .item() for wandb

    return metric_fun


make_mmd = partial(make_metric, mmd_fun_envs)
make_wasserstein = partial(make_metric, wasserstein_fun_envs)
make_mse = partial(make_metric, mse_fun_envs)
make_relmse = partial(make_metric, relmse_fun_envs)
make_vse = partial(make_metric, vse_fun_envs)


def sortability(data, mat, tol=1e-8):
    """
    Based on https://github.com/Scriddie/Varsortability/blob/main/src/varsortability.py
    Args:
        data: [n, d] data matrix
        mat: [d, d] adjacency matrix, where mat[i,j] != 0 if i is a causal parent of j
        tol: tolerance
    """
    assert data.ndim == 2 and mat.ndim == 2
    assert mat.shape[0] == mat.shape[1] and mat.shape[0] == data.shape[1]
    d = mat.shape[0]
    mat = 1 - onp.isclose(mat, 0.0).astype(int)
    mat_k = onp.eye(d)

    # ignore diagonal
    mat[onp.diag_indices(d)] = 0

    var = onp.var(data, axis=0, keepdims=True)
    corr = onp.abs(onp.corrcoef(data.T))
    corrsum = corr.sum(axis=0, keepdims=True)
    corrmax = (corr - onp.eye(d)).max(axis=0, keepdims=True)

    n_paths = 0
    n_ordered_paths = defaultdict(float)

    for _ in range(d - 1):

        mat_k = mat_k @ mat
        n_paths += mat_k.sum()

        for k, vals in [
            ("varsort", var),
            ("corrsumsort", corrsum),
            ("corrmaxsort", corrmax),
        ]:

            assert vals.shape == (1, d)
            assert not onp.isclose(vals, 0).any(), f"{k} \n{vals} \n{mat}"

            ratio = mat_k * vals / vals.T

            n_ordered_paths[k] += (ratio > 1 + tol).sum()
            n_ordered_paths[k] += 0.5 * ((ratio <= 1 + tol) * (ratio > 1 - tol)).sum()

    stats = {k: v / n_paths for k, v in n_ordered_paths.items()}
    return stats