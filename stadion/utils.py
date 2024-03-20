import math
import jax
import numpy as onp
from jax import numpy as jnp, random


def to_diag(x):
    """
    Args:
        x: [..., d]

    Returns:
        [..., d, d] where last two dimensions are ``jnp.diag()`` of last dimension in ``x``
    """
    assert x.ndim >= 1
    return jnp.einsum("...i,...ij->...ij", x, jnp.eye(x.shape[-1]))


def update_ave(ave_d, d):
    # online mean and variance with Welfords algorithm
    for k, v in d.items():
        ave_d[("__ctr__", k)] += 1
        delta = v - ave_d[("__mean__", k)]
        ave_d[("__mean__", k)] += delta / ave_d[("__ctr__", k)]
        delta2 = v - ave_d[("__mean__", k)]
        ave_d[("__welford_m2__", k)] += delta * delta2
    return ave_d


def retrieve_ave(ave_d, mean_only=True):
    out = dict(mean={}, std={})
    for k, v in ave_d.items():
        assert isinstance(k, tuple)
        # check if `k` is a ctr element
        if k[0] == "__ctr__":
            continue
        # process value `v`
        try:
            v_val = v.item()
        # array case
        except TypeError:
            v_val = onp.array(v)
        # not an array
        except AttributeError:
            v_val = v
        assert ("__ctr__", k[1]) in ave_d.keys()
        if k[0] == "__mean__":
            out["mean"][k[1]] = v_val
        else:
            if ave_d[("__ctr__", k[1])] == 1:
                out["std"][k[1]] = 0.0
            else:
                out["std"][k[1]] = math.sqrt((v_val + 1e-18) / (ave_d[("__ctr__", k[1])] - 1))
    if mean_only:
        return out["mean"]
    else:
        return out


def tree_global_norm(tree, p=2.0):
    norms = jax.tree_util.tree_map(lambda l: jnp.sum(jnp.abs(l) ** p), tree)
    return jax.tree_util.tree_reduce(jnp.sum, norms) ** (1.0 / p)


def tree_isnan(tree):
    return jnp.any(jnp.stack(jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda l: jnp.isnan(l).any(), tree)), axis=-1), axis=-1)


def random_split_like_tree(rng_key, target=None, treedef=None):
    if treedef is None:
        treedef = jax.tree_util.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)


def tree_init_normal(rng_key, target, scale=1.0):
    keys_tree = random_split_like_tree(rng_key, target)
    return jax.tree_util.tree_map(
        lambda l, k: l + scale * jax.random.normal(k, l.shape, l.dtype),
        target,
        keys_tree,
    )


def tree_variance_initialization(key, target, scale=1.0, mode='fan_in', distribution='uniform'):
    keys_tree = random_split_like_tree(key, target)
    return jax.tree_util.tree_map(
        lambda l, k: l + variance_initialization(k, l.shape, scale=scale, mode=mode, distribution=distribution),
        target,
        keys_tree,
    )


def variance_initialization(key, shape, scale=1.0, mode='fan_in', distribution='uniform'):
    # Taken from dm-haiku initializers.py

    # compute fans
    if len(shape) < 1:
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in, fan_out = shape
    else:
        raise ValueError(shape)

    # compute scale
    if mode == 'fan_in':
      scale /= max(1.0, fan_in)
    elif mode == 'fan_out':
      scale /= max(1.0, fan_out)
    else:
      scale /= max(1.0, (fan_in + fan_out) / 2.0)

    if distribution == 'truncated_normal':
      stddev = onp.sqrt(scale)
      # Adjust stddev for truncation.
      # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
      distribution_stddev = onp.asarray(.87962566103423978)
      stddev = stddev / distribution_stddev
      return stddev * random.truncated_normal(key, -2., 2., shape)

    elif distribution == 'normal':
      stddev = onp.sqrt(scale)
      return stddev * random.normal(key, shape)

    elif distribution == 'uniform':
      limit = onp.sqrt(3.0 * scale)
      return random.uniform(key, shape, minval=-limit, maxval=limit)

    else:
        raise KeyError(f"Unknown distribution initialization: {distribution}")
