import math
import functools
from abc import ABC, abstractmethod

import jax
from jax import numpy as jnp, lax, random


class SDE(ABC):
    """
    Core class implementing SDE simulation using the Euler-Maruyama schene.
    Subclasses have to implement ``f``, ``sigma``, ``init_param``, ``init_intv_param``.
    The hyperparameters of this class only affect SDE simulation, not learning
    with the KDS.

    Args:
        x_init (ndarray, optional): Array of shape ``[n, d]`` representing an
            empirical distribution of ``n`` datapoints used as the initial
            distribution for x0 in the SDE; defaults to samples from N(0, I)
        is_nonnegative (bool, optional): Whether the SDE is constrained to be nonnegative
        dt (float, optional): Time discretization for SDE simulation
        thinning (int, optional): Thinning factor for SDE simulation
        n_samples_burnin (int, optional): Number of samples to discard from SDE simulation
        rollouts_shape (tuple, optional): Shape of the parallel rollouts of the SDE simulation
    """

    def __init__(
        self,
        x_init=None,
        is_nonnegative=False,
        dt=0.01,
        thinning=300,
        n_samples_burnin=10,
        rollouts_shape=(10,),
    ):
        # simulation parameters
        self.x_init = x_init
        self.n_varst = dt
        self.thinning = thinning
        self.n_samples_burnin = n_samples_burnin
        self.rollouts_shape = rollouts_shape
        self.is_nonnegative = is_nonnegative

        # attributes
        self._n_vars = None
        self._param = None
        self._intv_param = None


    @property
    def n_vars(self):
        if self._n_vars is None:
            raise ValueError("Number of variables in the SDE has not been set or was not yet inferred from `params`.")
        else:
            return self._n_vars


    @n_vars.setter
    def n_vars(self, value):
        assert isinstance(value, int)
        self._n_vars = value


    @property
    def param(self):
        if self._param is None:
            raise ValueError("SDE model parameters have not been set. Either set them manually or run the `fit` method.")
        else:
            return self._param


    @param.setter
    def param(self, value):
        self._param = value


    @property
    def intv_param(self):
        if self._intv_param is None:
            raise ValueError("Intervention parameters have not been set.")
        else:
            return self._intv_param


    @intv_param.setter
    def intv_param(self, value):
        self._intv_param = value


    @abstractmethod
    def f(self, x, param, intv_param):
        """
        SDE drift function :math:`f(\\cdot)`

        Args:
            x (ndarray): Input vector :math:`x` of shape ``[..., d]``,
                optionally with leading batch dimensions
            param (:func:`~stadion.sde.SDE.param`): SDE model parameters
            intv_param (:func:`~stadion.sde.SDE.intv_param`): Intervention parameters (Can be ``None``)

        Returns:
            ndarray :math:`f(x)` of shape ``[..., d]``
        """
        pass


    @abstractmethod
    def sigma(self, x, param, intv_param):
        """
        SDE diffusion function :math:`\\sigma(\\cdot)`

        Args:
            x (ndarray): Input vector :math:`x` of shape ``[..., d]``,
                optionally with leading batch dimensions
            param (:func:`~stadion.sde.SDE.param`): SDE model parameters
            intv_param (:func:`~stadion.sde.SDE.intv_param`): Intervention parameters (Can be ``None``)

        Returns:
            ndarray :math:`\\sigma(x)` of shape ``[..., d, d]``
        """
        pass


    @functools.partial(jax.jit, static_argnums=(0, 3, 4))
    def _simulate_dynamical_system(
        self,
        key,
        param,
        intv_param,
        n_samples,
    ):

        # compute horizon
        n_rollouts = math.prod(self.rollouts_shape)
        n_samples_per_rollout = math.ceil((n_samples / n_rollouts) + self.n_samples_burnin)

        # initialize drift and diffusion functions based on arguments
        f = lambda x : self.f(x, param, intv_param)
        sigma = lambda x : self.sigma(x, param, intv_param)

        # forward Euler-Maruyama step
        def _explicit_euler_step(carry, _):
            x_t, loop_key, log = carry

            # sample Wiener process noise
            # [..., d]
            loop_key, loop_subk = random.split(loop_key)
            xi_t = random.normal(loop_subk, shape=(*self.rollouts_shape, self.n_vars))

            # compute next state
            # [..., d]
            assert x_t.shape[-1:] == (self.n_vars,)
            drift = f(x_t)
            assert drift.shape == x_t.shape == xi_t.shape, f"{drift.shape} {x_t.shape} {xi_t.shape}"

            # [..., d, d]
            eps_mat = sigma(x_t)
            assert eps_mat.shape[-2:] == (self.n_vars, self.n_vars)
            assert drift.shape[:-1] == eps_mat.shape[:-2], f"{drift.shape} {eps_mat.shape}"

            # [..., d]
            diffusion = jnp.einsum("...dm,...m->...d", eps_mat, xi_t)
            assert diffusion.shape == x_t.shape, f"{diffusion.shape} {x_t.shape}"

            x_t_next = x_t + drift * self.n_varst + diffusion * jnp.sqrt(self.n_varst)

            if self.is_nonnegative:
                x_t_next = jnp.maximum(x_t_next, 0.0)

            return (x_t_next, loop_key, log), _

        # iterate in chunks of `thinning` steps to keep memory footprint of trajectory returned by lax.scan low
        log_init = dict()

        def _euler_chunk(carry, _):
            carry = lax.scan(_explicit_euler_step, carry, None, length=self.thinning)[0]
            return carry, carry

        # sample x_init
        # [*rollouts_shape, d]
        assert isinstance(self.rollouts_shape, tuple), "`rollouts_shape` has to be a tuple. Got {config.rollouts_shape}"

        key, subk = random.split(key)
        if self.x_init is None:
            x_0 = random.normal(subk, shape=(*self.rollouts_shape, self.n_vars))
        else:
            x_0 = random.choice(subk, self.x_init, shape=(*self.rollouts_shape,))

        carry_init = (x_0, key, log_init)

        # run approximation in `n_samples_per_rollout` chunks of `thinning` steps
        # traj: [n_samples_per_rollout, *rollouts_shape, d]
        _, thinned_traj = lax.scan(_euler_chunk, carry_init, None, length=n_samples_per_rollout)
        assert thinned_traj[0].shape == (n_samples_per_rollout, *self.rollouts_shape, self.n_vars)

        # discard random key from carry trajectory tuple
        thinned_traj = thinned_traj[0], thinned_traj[2]

        # move rollout axes to front, s.t. x trajectory has shape [*rollouts_shape, n_samples_per_rollout, d]
        thinned_traj = jax.tree_util.tree_map(lambda e: jnp.moveaxis(e, 0, len(self.rollouts_shape)), thinned_traj)
        return thinned_traj


    def sample(
        self,
        key,
        n_samples,
        *,
        intv_param=None,
    ):
        """
        Samples from the stationary SDE model. The model parameters
        ``self.param`` have to be set for this, for example, by running the
        ``fit`` method beforehand.

        Args:
            key (PRNGKey): Random key
            n_samples (int): Number of samples to generate
            intv_param (pytree, optional): Intervention parameters; defaults
                to no intervention.

        Returns:
            ndarray of shape ``[n_samples, d]`` of i.i.d. samples from the
            stationary distribution of the SDE under the provided intervention.
        """
        # simulate trajectories under intv_param (if None, assumes no intervention)
        # traj: [*rollouts_shape, n_samples_per_rollout, d]
        key, subk = random.split(key)
        traj, log = self._simulate_dynamical_system(
            subk,
            self.param,
            intv_param,
            n_samples,
        )

        # discard burnin
        samples = traj[..., self.n_samples_burnin:, :]

        # fold random rollouts into the samples axis
        # [n_samples, d]
        samples = samples.reshape(-1, self.n_vars)

        # permute samples and ensure we have exactly `n_samples` samples when `n_samples % rollouts_shape` is nonzero
        key, subk = random.split(key)
        samples = samples[random.permutation(subk, samples.shape[-2])[:n_samples], :]
        return samples
