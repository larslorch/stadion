import functools

import jax
from jax import vmap, random
import jax.numpy as jnp

from stadion.utils.tree import tree_init_normal, tree_global_norm, tree_variance_initialization


def f_j(x, theta, *, activation):
    """[..., d], {param tree} -> [...] """

    if activation == "tanh":
        nonlin = jnp.tanh
    elif activation == "relu":
        nonlin = jax.nn.relu
    elif activation == "sigmoid":
        nonlin = jax.nn.sigmoid
    elif activation == "rbf":
        nonlin = lambda arr: jnp.exp(- jnp.power(arr, 2))
    else:
        raise KeyError(f"Unknown activation {activation}")

    linear = jnp.einsum("...d,d->...", x, theta["v1"]) + theta["b1"]
    hid = nonlin(jnp.einsum("...d,dh->...h", x, theta["x1"]) + theta["f1"])
    out = linear + jnp.einsum("...h,h->...", hid, theta["x2"])
    return out


def sigma_j(x, theta, is_x_j):
    """[..., d], {param tree}, [d,] -> [..., d] """
    assert is_x_j.shape[0] == x.shape[-1] and is_x_j.ndim == 1 and theta["c1"].ndim == 0
    c = jnp.exp(theta["c1"])
    return c * jnp.ones(x.shape) * is_x_j


def _group_lasso_j(theta, j):
    d = theta["v1"].shape[-1]

    # group all first layer parameters by which x_j they touch (i.e. linear and nonlinear parts)
    theta_grouped = jnp.concatenate([theta["x1"], theta["v1"][..., None]], axis=-1)

    # [d,] compute L2 norm for each group (each causal parent)
    group_lasso_terms_j = vmap(functools.partial(tree_global_norm, p=2.0, eps=1e-16), 0, 0)(theta_grouped)

    # mask self-influence
    group_lasso_terms_j = jnp.where(jnp.eye(d)[j], 1e-16, group_lasso_terms_j)

    # [] compute Lp group lasso (in classical group lasso, p=1)
    lasso = tree_global_norm(group_lasso_terms_j, p=1.0)
    return lasso


def group_lasso(theta):
    d = theta["v1"].shape[0]

    # group lasso: for each causal mechansim
    # [d,] for each first layer parameter touching x
    reg = vmap(_group_lasso_j, (0, 0), 0)(theta, jnp.arange(d)).mean(0)
    return reg


def init_theta(key, d, hidden_size, scale=1.0, mode='fan_in', distribution='uniform', force_linear_diag=False):
    shape = {
        "x1": jnp.zeros((d, d, hidden_size)),
        "f1": jnp.zeros((d, hidden_size)),
        "x2": jnp.zeros((d, hidden_size)),
        "v1": jnp.zeros((d, d)),
        "b1": jnp.zeros((d,)),
        "c1": jnp.zeros((d,)),
    }
    _initializer = functools.partial(tree_variance_initialization, scale=scale, mode=mode, distribution=distribution)
    theta = vmap(_initializer, (0, 0), 0)(jnp.array(random.split(key, d)), shape)
    if force_linear_diag:
        theta["v1"] = theta["v1"].at[...].set(0.0)
        theta["v1"] = theta["v1"].at[jnp.diag_indices(d)].set(-1.0)
        theta["x1"] = theta["x1"].at[jnp.diag_indices(d)].set(0.0)
    return theta



""" 
Shift interventions 
"""

def f(x, theta, intv_theta, intv, *, activation):
    """[..., d], {param tree}, {param tree}, [d,] -> [..., d] """
    d = x.shape[-1]

    # intv params
    # [d,]
    shift = jnp.where(intv, intv_theta["shift"], jnp.zeros(d))
    assert shift.shape == (d,)

    # compute drift scalar f(x)_j for each dx_j using the input x
    f_j_parameterized = functools.partial(f_j, activation=activation)
    f_vec = vmap(f_j_parameterized, in_axes=(None, 0), out_axes=-1)(x, theta)

    # apply shift
    f_vec += shift

    assert x.shape[-1] == d
    assert intv.shape[0] == d and intv.ndim == 1
    assert x.shape == f_vec.shape

    return f_vec


def sigma(x, theta, intv_theta, intv):
    """[..., d], {param tree}, {param tree}, [d,] ->  [..., d, d]"""
    d = x.shape[-1]

    # intv params
    # [d,]
    scale = jnp.exp(intv_theta["scale"]) if "scale" in intv_theta else jnp.ones(d)
    assert scale.shape == (d,)

    # compute diffusion vector sigma(x)_j for each dx_j using the input x
    sig_mat = vmap(sigma_j, in_axes=(None, 0, 0), out_axes=-2)(x, theta, jnp.eye(d))

    # scale rows of sigma by scale
    sig_mat = jnp.einsum("...ab,a->...ab", sig_mat, jnp.where(intv, scale, jnp.ones(d)))

    assert intv.shape == (d,)
    assert x.shape[-1] == d
    assert sig_mat.shape == (*x.shape, d)

    return sig_mat


def init_intv_theta(key, n_envs, d, scale_param=False, scale=1.0):
    # pytree of [n_envs, d, ...]
    # learned intervention effect parameters
    shape = {
        "shift": jnp.zeros((n_envs, d)),
    }
    if scale_param:
        shape["scale"] = jnp.zeros((n_envs, d))

    intv_theta = tree_init_normal(key, shape, scale=scale)
    return intv_theta

