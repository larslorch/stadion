import functools

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU') # hide gpus to tf to avoid OOM and cuda errors by conflicting with jax

import tensorflow_datasets as tfds
import numpy as onp

import jax
import jax.random as random
from jax import jacrev, vmap, jit
import jax.numpy as jnp

from stadion.utils.tree import iter_tree

from typing import NamedTuple, Any

def _device_put_sharded(sharded_tree, devices):
    """Taken from  https://github.com/deepmind/dm-haiku/blob/main/examples/imagenet/dataset.py """
    leaves, treedef = jax.tree_util.tree_flatten(sharded_tree)
    n = leaves[0].shape[0]
    return jax.device_put_sharded([jax.tree_util.tree_unflatten(treedef, [l[i] for l in leaves]) for i in range(n)],
                                  devices)

def _double_cache(ds):
    """Taken from https://github.com/deepmind/dm-haiku/blob/main/examples/imagenet/dataset.py

    Keeps at least two batches on the accelerator.
    The current GPU allocator design reuses previous allocations. For a training
    loop this means batches will (typically) occupy the same region of memory as
    the previous batch. An issue with this is that it means we cannot overlap a
    host->device copy for the next batch until the previous step has finished and
    the previous batch has been freed.
    By double buffering we ensure that there are always two batches on the device.
    This means that a given batch waits on the N-2'th step to finish and free,
    meaning that it can allocate and copy the next batch to the accelerator in
    parallel with the N-1'th step being executed.
    Args:
    ds: Iterable of batches of numpy arrays.
    Yields:
    Batches of sharded device arrays.
    """
    batch = None
    devices = jax.local_devices()
    for next_batch in ds:
        assert next_batch is not None
        next_batch = Batch(**next_batch)
        next_batch = _device_put_sharded(next_batch, devices)
        if batch is not None:
            yield batch
        batch = next_batch
    if batch is not None:
        yield batch


def _single_cache(ds):
    devices = jax.local_devices()
    for next_batch in ds:
        assert next_batch is not None
        next_batch = Batch(**next_batch)
        next_batch = _device_put_sharded(next_batch, devices)
        yield next_batch


def _default_yielder(ds):
    for batch_dict in ds:
        yield Batch(**batch_dict)


def structured_py_function(func, inp, Tout, name=None):
    """
    Workaround for allowing `tf.py_function` that returns structured elements like dict or NamedTuple
    Taken from
    https://github.com/tensorflow/tensorflow/issues/27679#issuecomment-522578000
    """
    def wrapped_func(*flat_inp):
        reconstructed_inp = tf.nest.pack_sequence_as(inp, flat_inp,
                                                     expand_composites=True)
        result = func(*reconstructed_inp)
        return tf.nest.flatten(result, expand_composites=True)
    flat_Tout = tf.nest.flatten(Tout, expand_composites=True)
    flat_out = tf.py_function(
        func=wrapped_func,
        inp=tf.nest.flatten(inp, expand_composites=True),
        Tout=[_tensor_spec_to_dtype(v) for v in flat_Tout],
        name=name)
    spec_out = tf.nest.map_structure(_dtype_to_tensor_spec, Tout,
                                   expand_composites=True)
    out = tf.nest.pack_sequence_as(spec_out, flat_out, expand_composites=True)
    return out

def _dtype_to_tensor_spec(v):
    return tf.TensorSpec(None, v) if isinstance(v, tf.dtypes.DType) else v

def _tensor_spec_to_dtype(v):
    return v.dtype if isinstance(v, tf.TensorSpec) else v


class Batch(NamedTuple):
    """
    Data vector [..., d] and a copy to simplify vmap over pairs
    """
    x: Any = None
    y: Any = None

    """
    Four derivative terms of the generator objective, shapes depending on joint or interventional marginal batch are
    [..., d, d], [..., d], [..., d], [...] or [..., 1, 1, d], [..., 1, d], [..., 1, d], [..., d]
    """
    k0: Any = None
    k1: Any = None
    k2: Any = None
    k3: Any = None

    """
    One-hot indicator vector [..., d] encoding which environment this batch belongs to
    """
    env_indicator: Any = None

    """
    Multi-hot indicator vector [..., d] encoding which variables were intervened upon in this sample
    """
    intv: Any = None

    dtype = dict(
        x=onp.float32,
        y=onp.float32,
        k0=onp.float32,
        k1=onp.float32,
        k2=onp.float32,
        k3=onp.float32,
        env_indicator=onp.float32,
        intv=onp.int32,
    )

    def __len__(self):
        return len(self.x)


def laplacian(func, argnum):
    return lambda *args: jnp.trace(jax.hessian(func, argnum)(*args))


@functools.partial(jit, static_argnums=(0,), backend="cpu") # cpu to avoid OOM
@functools.partial(vmap, in_axes=(None, 0, None), out_axes=0)
@functools.partial(vmap, in_axes=(None, None, 0), out_axes=0)
def _compute_four_objective_terms(kernel, x, y):

    term0 = jacrev(jacrev(kernel, 0), 1)(x, y)
    term1 = jacrev(laplacian(kernel, 1), 0)(x, y)
    term2 = jacrev(laplacian(kernel, 0), 1)(x, y)
    term3 = laplacian(laplacian(kernel, 0), 1)(x, y)

    return term0, term1, term2, term3


def compute_four_objective_terms(kernel, data):
    k0, k1, k2, k3 = _compute_four_objective_terms(kernel, data, data)
    return onp.array(k0), onp.array(k1), onp.array(k2), onp.array(k3)


def sample_batch(_, *, rng, batch_size, batch_size_env, targets, k):
    n_envs = len(targets.data)

    # sample random envs
    envs = rng.choice(n_envs, size=(batch_size_env,), replace=True)

    # assemble submatrices from cache
    tups = []
    for env in envs:
        n = int(targets.data[env].shape[0])

        # batch corresponding to full generator objective on joint distribution
        idx = rng.choice(n, size=batch_size, replace=True)
        x = targets.data[env][idx]
        assert x.ndim == 2

        tup = dict(
            x=x,
            y=x,
            env_indicator=onp.eye(n_envs)[env],
            intv=targets.intv[env],
        )

        if k is None:
            # auto diff case (dummy values for cached kernel terms)
            nan = onp.array(onp.nan).astype(onp.float32)
            tup.update(dict(
                k0=nan,
                k1=nan,
                k2=nan,
                k3=nan,
            ))

        else:
            # caching case
            tup.update(dict(
                k0=k[0][env][onp.ix_(idx, idx)],
                k1=k[1][env][onp.ix_(idx, idx)],
                k2=k[2][env][onp.ix_(idx, idx)],
                k3=k[3][env][onp.ix_(idx, idx)],
            ))

        tups.append(tup)

    # stack submatrices now that shapes are equal
    tup_stack = dict()
    for key in tups[0].keys():
        tup_stack[key] = onp.stack([tp[key] for tp in tups], axis=0)

    return tup_stack


def sample_batch_jax(*, targets, batch_size, batch_size_env, key):

    # sample random envs
    n_envs = len(targets.data)
    key, subk = random.split(key)
    envs = random.choice(subk, n_envs, shape=(batch_size_env,), replace=True)

    # assemble submatrices from cache
    tups = []
    for env in envs:
        n = targets.data[env].shape[0]

        # batch corresponding to full generator objective on joint distribution
        key, subk = random.split(key)
        idx = random.choice(subk, n, shape=(batch_size,), replace=True)
        x = targets.data[env][idx]
        assert x.ndim == 2

        nan = jnp.array(jnp.nan).astype(jnp.float32)
        tup = dict(
            x=x,
            y=x,
            env_indicator=jnp.eye(n_envs)[env],
            intv=targets.intv[env],
            k0=nan,
            k1=nan,
            k2=nan,
            k3=nan,
        )
        tups.append(tup)

    # stack submatrices now that shapes are equal
    tup_stack = dict()
    for kk in tups[0].keys():
        tup_stack[kk] = jnp.stack([tp[kk] for tp in tups], axis=0)

    # yield batch
    return Batch(**tup_stack)


def make_dataset(*, seed, kernel, targets, auto_diff, batch_size, batch_size_env, deterministic=True):

    rng = onp.random.default_rng(seed)

    # if caching, precompute kernel terms in objective for full dataset
    if not auto_diff:
        print(f"Caching objective terms...", flush=True)
        # k: list of shape [4, n_envs, np.ndarray]
        k = [[] for _ in range(4)]
        for data in targets.data:
            ks = compute_four_objective_terms(kernel, data)
            for j in range(4):
                k[j].append(ks[j])
    else:
        k = None

    # initialize dataset
    f = functools.partial(sample_batch, rng=rng, batch_size=batch_size, batch_size_env=batch_size_env,
                          targets=targets, k=k)

    ds = tf.data.Dataset.from_tensor_slices([0]).repeat(None)  # loop indefinitely
    ds = ds.map(lambda *args: structured_py_function(func=f, inp=[*args], Tout=Batch.dtype),
                # deterministic=True not sufficient as rng state is shared and may be called out of order
                # even if arranged correctly
                deterministic=deterministic, num_parallel_calls=1 if deterministic else tf.data.experimental.AUTOTUNE)
    ds = ds.prefetch(1)
    ds = tfds.as_numpy(ds)
    # ds = _default_yielder(ds)
    ds = _double_cache(ds)
    return ds
