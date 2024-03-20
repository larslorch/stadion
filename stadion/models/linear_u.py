import functools

import jax
from jax import vmap
import jax.numpy as jnp

from stadion.utils.tree import tree_init_normal, tree_global_norm

"""
Linear model
"""
def f_j(x, theta):
    """[..., d], {param tree} -> [...] """
    w = theta["w1"]
    b = theta["b1"]
    assert w.shape[0] == x.shape[-1] and w.ndim == 1 and b.ndim == 0
    return x @ w + b


def sigma_j(x, theta, is_x_j):
    """[..., d], {param tree}, [d,] -> [..., d] """
    assert is_x_j.shape[0] == x.shape[-1] and is_x_j.ndim == 1
    return jnp.ones(x.shape) * is_x_j


def _group_lasso_j(theta, j):
    d = theta["w1"].shape[0]

    # [d,] compute L2 norm for each group (each causal parent)
    group_lasso_terms_j = vmap(functools.partial(tree_global_norm, p=2.0, eps=1e-16), 0, 0)(theta["w1"])

    # mask self-influence
    group_lasso_terms_j = jnp.where(jnp.eye(d)[j], 1e-16, group_lasso_terms_j)

    # [] compute Lp group lasso (in classical group lasso, p=1)
    lasso = tree_global_norm(group_lasso_terms_j, p=1.0)
    return lasso


def group_lasso(theta):
    d = theta["w1"].shape[0]

    # group lasso: for each causal mechansim
    # [d,] for each first layer parameter with w1 tag and bias
    reg_w1 = vmap(_group_lasso_j, (0, 0), 0)(theta, jnp.arange(d)).mean(0)
    return reg_w1


def _weighted_group_lasso_j(theta, j, eps):
    d = theta["w1"].shape[0]

    # [d,] compute L2 norm for each group (each causal parent)
    group_lasso_terms_j = vmap(functools.partial(tree_global_norm, p=2.0, eps=1e-16), 0, 0)(theta["w1"])

    # mask self-influence
    group_lasso_terms_j = jnp.where(jnp.eye(d)[j], 1e-16, group_lasso_terms_j)

    # weight terms as in Eq. 7 of Candes et al: Enhancing Sparsity by Reweighted L1 Minimization
    # https://link.springer.com/content/pdf/10.1007/s00041-008-9045-x.pdf
    weighted_group_lasso_terms_j = group_lasso_terms_j / jax.lax.stop_gradient(group_lasso_terms_j + eps)

    # [] compute Lp group lasso (in classical group lasso, p=1)
    weight_group_lasso = tree_global_norm(weighted_group_lasso_terms_j, p=1.0)
    return weight_group_lasso


def weighted_group_lasso(theta, eps):
    d = theta["w1"].shape[0]

    # group lasso: for each causal mechansim
    # [d,] for each first layer parameter with w1 tag and bias
    reg_w1 = vmap(_weighted_group_lasso_j, (0, 0, None), 0)(theta, jnp.arange(d), eps).mean(0)

    return reg_w1


def init_theta(key, d, scale=1.0, init_diag=None, force_linear_diag=False):
    shape = {
        "w1": jnp.zeros((d, d)),
        "b1": jnp.zeros((d,)),
    }
    theta = tree_init_normal(key, shape, scale=scale)
    if init_diag:
        theta["w1"] += init_diag * jnp.eye(d)
    if force_linear_diag:
        theta["w1"] = theta["w1"].at[jnp.diag_indices(d)].set(-1.0)
    return theta

""" 
Shift-scale interventions 

For each env, there are two times [d,] intv parameters, as each env may intervene on up to `d` variables.
An intervention on dx_j is modeled by shift on only the variable x_j (not the other inputs to dx_j) 
and a scaling of the diffusion matrix component sigma(x)_j for dx_j.
Hence, we have 2 parameters per dx_j per env.
"""

def f(x, theta, intv_theta, intv):
    """[..., d], {param tree}, {param tree}, [d,] -> [..., d] """
    d = x.shape[-1]

    # intv params
    # [d,]
    shift = jnp.where(intv, intv_theta["shift"], jnp.zeros(d))
    assert shift.shape == (d,)

    # compute drift scalar f(x)_j for each dx_j using the input x
    f_vec = vmap(f_j, in_axes=(None, 0), out_axes=-1)(x, theta)

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

