import math
import functools
from frozendict import frozendict
from typing import NamedTuple, Any

import jax
import numpy as onp
from jax import numpy as jnp, lax, random

from stadion.utils.newton import implicit_euler_step as implicit_step, log_init as implicit_log_init


class SimulationConfig(NamedTuple):
    method: Any = None
    method_config: Any = dict()
    dt: Any = None
    thinning: Any = None
    rollouts_shape: Any = None
    n_samples_per_rollout: Any = None


# vmap over intervention environments
@functools.partial(jax.jit, static_argnums=(1, 2, 6))
@functools.partial(jax.vmap, in_axes=(0, None, None, None, 0, 0, None), out_axes=0)
def _simulate_dynamical_system(main_key, f, sigma, theta, intv_theta, intv_mask, config):

    # initialize drift and diffusion functions based on arguments
    d = intv_mask.shape[-1]

    f_theta = lambda x : f(x, theta, intv_theta, intv_mask)
    f_dx_theta = jnp.vectorize(jax.jacrev(f_theta), signature='(n)->(m,n)')
    f_dx2_theta = jnp.vectorize(jax.hessian(f_theta), signature='(n)->(m,n,n)')

    sigma_theta = lambda x : sigma(x, theta, intv_theta, intv_mask)

    # forward Euler-Maruyama step
    def _explicit_euler_step(carry, _):
        x_t, key, log = carry

        # sample driving noise
        # [..., d]
        key, subk = random.split(key)
        xi_t = random.normal(subk, shape=(*config.rollouts_shape, d))

        # compute next state
        # [..., d]
        assert x_t.shape[-1:] == (d,)
        drift = f_theta(x_t)
        assert drift.shape == x_t.shape == xi_t.shape, f"{drift.shape} {x_t.shape} {xi_t.shape}"

        # [..., d, d]
        eps_mat = sigma_theta(x_t)
        assert eps_mat.shape[-2:] == (d, d)
        assert drift.shape[:-1] == eps_mat.shape[:-2], f"{drift.shape} {eps_mat.shape}"

        # [..., d]
        diffusion = jnp.einsum("...dm,...m->...d", eps_mat, xi_t)
        assert diffusion.shape == x_t.shape, f"{diffusion.shape} {x_t.shape}"

        x_t_next = x_t + drift * config.dt + diffusion * jnp.sqrt(config.dt)

        return (x_t_next, key, log), _

    # backward/implicit Euler-Maruyama step
    def _implicit_euler_step(carry, _):
        x_t, key, log = carry

        # sample driving noise
        # [..., d]
        key, subk = random.split(key)
        xi_t = random.normal(subk, shape=(*config.rollouts_shape, d))

        # compute next state
        # [..., d]
        assert x_t.shape[-1:] == (d,)
        x_t_next, log_next = implicit_step(f_theta, sigma_theta, x_t, xi_t, config.dt, **config.method_config)

        # update log state average
        for k, v in log.items():
            if v.dtype == bool:
                log_next[k] = log[k] | log_next[k]
            else:
                log_next[k] += log[k] / config.thinning

        return (x_t_next, key, log_next), _

    # Ito-Taylor step (Algorithm 8.4, p. 136, Särkkä, Applied SDEs)
    # Assumes Brownian motion is diagonal and standard (Q = I)
    eye = jnp.eye(d)
    t_mat = jnp.concatenate([
        jnp.concatenate([(config.dt ** 3)/3. * eye, (config.dt ** 2)/2. * eye]),
        jnp.concatenate([(config.dt ** 2)/2. * eye,  config.dt          * eye]),
    ], axis=-1).T
    t_mat_sqrt = jnp.linalg.cholesky(t_mat)
    assert t_mat_sqrt.shape == (2 * d, 2 * d)

    def _taylor_step(carry, _):
        x_t, key, log = carry

        # sample driving noise
        # [..., d]
        key, subk = random.split(key)
        noise_t = jnp.einsum("ij,...j->...i", t_mat_sqrt, random.normal(subk, shape=(*config.rollouts_shape, 2 * d)))
        xi_t = noise_t[..., :d]
        beta_t = noise_t[..., d:]
        assert x_t.shape == xi_t.shape == beta_t.shape

        # compute next state
        # [..., d, d]
        l_mat = sigma_theta(x_t)
        ll_mat = jnp.einsum("...mn,...on->...mo", l_mat, l_mat)
        assert l_mat.shape[-2:] == (d, d)
        assert x_t.shape[:-1] == l_mat.shape[:-2]
        assert l_mat.shape == ll_mat.shape

        # a term
        # [..., d]
        f_x = f_theta(x_t)
        f_dx = f_dx_theta(x_t)
        f_dx2 = f_dx2_theta(x_t)
        assert x_t.shape + (d, d) == f_x.shape + (d, d) == f_dx.shape + (d,) == f_dx2.shape

        a_t = jnp.einsum("...mn,...n->...m", f_dx, f_x) + 0.5 * jnp.einsum("...mnN,...nN->...m", f_dx2, ll_mat)
        assert a_t.shape == x_t.shape

        # b term
        # [..., d, d]
        b_mat_t = jnp.einsum("...ni,...ij->...nj", f_dx, l_mat)
        assert b_mat_t.shape == l_mat.shape

        # combine
        # [..., d]
        x_t_next = x_t + f_x * config.dt \
                       + jnp.einsum("...ij,...j->...i", l_mat, beta_t)\
                       + a_t * (config.dt ** 2)/2. \
                       + jnp.einsum("...nj,...j->...n", b_mat_t, xi_t)

        assert x_t.shape == x_t_next.shape

        return (x_t_next, key, log), _

    # iterate in chunks of `thinning` steps to keep memory footprint of trajectory returned by lax.scan low
    # determine simulation method
    if config.method == "explicit":
        step = _explicit_euler_step
        log_init = dict()

    elif config.method == "implicit":
        step = _implicit_euler_step
        log_init = implicit_log_init(config.rollouts_shape)

    elif config.method == "taylor":
        step = _taylor_step
        log_init = dict()

    else:
        raise KeyError(f"Unknown SDE simulation method `{config.method}`")

    def _euler_chunk(carry, _):
        carry = lax.scan(step, carry, None, length=config.thinning)[0]
        return carry, carry

    # sample x_init
    # [*rollouts_shape, d]
    main_key, main_subk = random.split(main_key)
    x_init = random.normal(main_subk, shape=(*config.rollouts_shape, d))
    carry_init = (x_init, main_key, log_init)

    # run approximation in `n_samples_per_rollout` chunks of `thinning` steps
    # traj: [n_samples_per_rollout, *rollouts_shape, d]
    _, thinned_traj = lax.scan(_euler_chunk, carry_init, None, length=config.n_samples_per_rollout)
    assert thinned_traj[0].shape == (config.n_samples_per_rollout, *config.rollouts_shape, d)

    # discard random key from trajectory
    thinned_traj = thinned_traj[0], thinned_traj[2]

    # move rollout axes to front, s.t. x trajectory has shape [*rollouts_shape, n_samples_per_rollout, d]
    thinned_traj = jax.tree_util.tree_map(lambda e: jnp.moveaxis(e, 0, len(config.rollouts_shape)), thinned_traj)

    return jax.lax.stop_gradient(thinned_traj)


def sample_dynamical_system(key, theta, intv_theta, intv_mask, *, n_samples, config, f, sigma, return_traj=False):
    """Wrapper around _sample_dynamical_system to handle thinning and burnin"""

    assert intv_mask.ndim == 2
    n_envs, d = intv_mask.shape
    assert all([e.shape[0] == n_envs for e in jax.tree_util.tree_leaves(intv_theta)]), \
        f"`intv_theta` has to have same leading dimensions as number of environments specified by `intv_mask`.\n" \
        f"Got `intv_mask` {intv_mask.shape} and {[e.shape for e in jax.tree_util.tree_leaves(intv_theta)]}"

    # compute horizon
    n_samples_per_rollout = math.ceil((n_samples / config.rollout_restarts) + config.n_samples_burnin)

    sim_config = SimulationConfig(
        method=config.method,
        method_config=frozendict(config.method_config) if "method_config" in config else frozendict(),
        dt=config.dt,
        thinning=config.thinning,
        rollouts_shape=(config.rollout_restarts,),
        n_samples_per_rollout=n_samples_per_rollout,
    )

    # simulate trajectories, vmapping over different envs specified by intv_theta, intv_mask
    # traj["x"]: [n_envs, n_samples_per_rollout, config.rollout_restarts, d]
    key, *subkeys = random.split(key, n_envs + 1)
    subkeys = jnp.array(subkeys)
    traj, log = _simulate_dynamical_system(subkeys, f, sigma, theta, intv_theta, intv_mask, sim_config)

    assert traj.shape == (n_envs, config.rollout_restarts, n_samples_per_rollout, d)

    # discard burnin
    samples = traj[:, :, config.n_samples_burnin:, :]

    # fold random rollouts into the samples axis
    # [n_envs, n_samples, d]
    samples = samples.reshape(n_envs, -1, d)

    # permute samples and ensure we have exactly `n_samples` samples when `n_samples % rollout_restarts` is nonzero
    key, subk = random.split(key)
    samples = samples[:, random.permutation(subk, samples.shape[1])[:n_samples], :]

    if return_traj:
        # [n_envs, n_samples, d], [n_envs, rollout_restarts, n_samples_per_rollout, d], pytree
        return samples, traj, log
    else:
        # [n_envs, n_samples, d]
        return samples


class Data(NamedTuple):
    """
    data: list of length `n_envs` of unequally shaped [n_i, d]
    intv: [n_envs, d]
    true_param: [d, d]
    traj: [n_envs, t_max, d]
    """
    data: Any
    intv: Any
    true_param: Any = None
    traj: Any = None

    def __len__(self):
        return len(self.data)


def iter_data(tars):
    """
    Yields `Data` objects with single leaves of a multi-env `Data` object
    """
    n_envs = len(tars.data)
    dct = tars._asdict()
    for i in range(n_envs):
        yield Data(**{k: v[i] if v is not None else None for k, v in dct.items()})


def index_data(tars, idx):
    """
    Indexes into first axis of a multi-env `Data` object
    """
    dct = tars._asdict()
    new_dct = {}
    for k, v in dct.items():
        if type(v) == list:
            new_dct[k] = [v[ii] for ii in idx]
        elif v is None:
            new_dct[k] = v
        else:
            new_dct[k] = v[idx]

    return Data(**new_dct)


def get_stats(tars):
    """
    Helper for extracting summary statistics from targets
    """
    intv = tars.intv
    means = onp.array([data.mean(-2) for data in tars.data])
    return means, intv


def get_intv_stats(tars):
    """
    Helper for extracting intervention summary statistics from targets
    """
    intv = tars.intv
    means = onp.array([data.mean(-2) for data in tars.data])
    return means * intv, intv


@functools.partial(jax.jit, static_argnums=(2,))
def sample_subset_strict(key, tars, n):
    """
    Strict means: throws Error if there are no `n` observations in all of the data sets.
    This function returns `data` and `n` as jnp.arrays
    """
    tars_data, tars_n = tars.data, tars.n
    dct = tars._asdict()
    assert type(tars_data) == list or tars_data.ndim == 3, "`Data.data` leaves must be lists or of ndim == 3"
    assert all([x.shape[0] >= n for x in tars_data]), f"Each matrix of tars.data must have at least `{n}` rows, but" \
                                                      f"got: {[x.shape for x in tars_data]}"
    keys = jnp.array(random.split(key, len(tars_data)))
    dct["data"] = jnp.stack([x[random.permutation(subk, x.shape[-2])[:n]] for subk, x in zip(keys, tars_data)])
    return Data(**dct)


@functools.partial(jax.jit, static_argnums=(2,))
def sample_subset(key, tars, n):
    """
    This function returns `data` and `n` as lists
    """
    tars_data = tars.data
    dct = tars._asdict()
    assert type(tars_data) == list or tars_data.ndim == 3, "`Data.data` leaves must be lists or of ndim == 3"
    keys = jnp.array(random.split(key, len(tars_data)))
    dct["data"] = [x[random.permutation(subk, x.shape[-2])[:min(n, x.shape[-2])]] for subk, x in zip(keys, tars_data)]
    return Data(**dct)


class NoStandardizer:

    def __init__(self, shift=None, scale=None):
        self.shift = shift
        self.scale = scale

    def __call__(self, tars):
        return tars


class Standardizer:

    def __init__(self, shift, scale):
        self.shift = shift
        self.scale = scale

    def __call__(self, tars):
        if self.shift is None and self.scale is None:
            return tars
        dct = tars._asdict()
        dct["data"] = [(x - self.shift) / self.scale for x in dct["data"]]
        return Data(**dct)


class ScaleStandardizer:

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, tars):
        if self.scale is None:
            return tars
        dct = tars._asdict()
        dct["data"] = [x / self.scale for x in dct["data"]]
        return Data(**dct)


def make_target_standardizer(data, ignore=False, std_eps=0.0):
    assert data.ndim == 2
    shift = data.mean(-2)
    scale = data.std(-2) + std_eps
    scale = onp.where(onp.isclose(scale, 0.0), 1.0, scale)

    standardizer = NoStandardizer if ignore else Standardizer

    return standardizer(shift=shift, scale=scale)


def make_sergio_standardizer(data, ignore=False, std_eps=0.0):
    assert data.ndim == 2
    scale = data.std(-2) + std_eps

    standardizer = NoStandardizer if ignore else ScaleStandardizer

    return standardizer(scale=scale)