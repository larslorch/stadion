import jax
import numpy as onp
from jax import numpy as jnp, random


def tree_sum_reduce(tree):
    # tree_reduce would be more efficient (not instantiating the whole array) but doesn't work with this jax version
    return jnp.sum(jnp.stack(jax.tree_util.tree_leaves(tree), axis=-1), axis=-1)


def tree_global_norm(tree, p=2.0, eps=0.0):
    norms = jax.tree_util.tree_map(lambda l: jnp.sum(jnp.abs(l) ** p), tree)
    return (tree_sum_reduce(norms) + eps) ** (1.0 / p)


def tree_grad_clip(tree, norm=1.0):
    tree_nrm = tree_global_norm(tree)
    return jax.tree_util.tree_map(lambda l: l / jnp.maximum(tree_nrm, norm), tree)


def tree_isnan(tree):
    return jnp.any(jnp.stack(jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda l: jnp.isnan(l).any(), tree)), axis=-1), axis=-1)


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


def iter_tree(tree):
    flat, treedef = jax.tree_util.tree_flatten(tree)
    for vals in zip(*flat):
        yield jax.tree_util.tree_unflatten(treedef, vals)


def variance_initialization(key, shape, scale=1.0, mode='fan_in', distribution='uniform'):
    """
    Taken from dm-haiku initializers.py
    """
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


def tree_variance_initialization(key, target, scale=1.0, mode='fan_in', distribution='uniform'):
    keys_tree = random_split_like_tree(key, target)
    return jax.tree_util.tree_map(
        lambda l, k: l + variance_initialization(k, l.shape, scale=scale, mode=mode, distribution=distribution),
        target,
        keys_tree,
    )