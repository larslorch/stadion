from abc import ABC, abstractmethod
from collections import defaultdict
import time
from functools import partial

import numpy as onp
import math

import jax
from jax import numpy as jnp, random, tree_map
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
from optax._src import linear_algebra
import optax

from stadion.sde import SDE
from stadion.kds import kds_loss
from stadion.data import make_dataloader
from stadion.utils import update_ave, retrieve_ave, tree_isnan


def wrapped_mean(func, axis=None):
    def wrapped(*args):
        return tree_map(partial(jnp.mean, axis=axis), func(*args))
    return wrapped


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


def rbf_kernel(x, y, *, bandwidth):
    assert type(bandwidth) == float or bandwidth.ndim == 0
    return jnp.exp(- squared_norm(x, y) / (2.0 * (bandwidth ** 2)))



class KDSMixin(SDE, ABC):
    """
    Mixin class handling stationary SDE inference via the KDS.
    """

    @abstractmethod
    def init_param(self, key, d, *, scale=0.001, fix_speed_scaling=True):
        """
        Samples random initialization of the SDE model parameters

        Args:
            key (PRNGKey)
            d (int, optional): number of variables
            scale (float, optional): scale of random values sampled
            fix_speed_scaling (bool, optional): Whether to fix the speed
                scaling of the SDE model parameters. ``True`` is recommended and the
                default, as ``False`` can lead to instabilities when learning
                the SDE model parameters with the KDS.
        Returns:
            :class:`~stadion.parameters.ModelParameters`
        """
        pass


    @abstractmethod
    def init_intv_param(self, key, d, *, n_envs=None, scale=0.001, targets=None, x=None):
        """
        Samples random initialization of the intervention parameters.
        If intervened variables are known and provided via ``intv``, the modeled interventions are restricted
        to variables where ``intv[j] == 1``.
        If both ``intv`` and ``x`` are provided, the intervention shifts on the known target variables
        are warm-startedbased on the mean shift from the first dataset (which is assumed to be observational data).

        Args:
            key (PRNGKey)
            d (optional): number of variables
            n_envs (int, optional): number of environments. When provided, PyTree has additional leading axis of `n_envs`.
            scale (float, optional): scale of random values sampled
            x (Dataset, optional): Datasets to warm-start parameters. Both ``x`` and ``intv`` have to be provided.
            targets (ndarray, optional): Intervention mask(s) to warm-start parameters. Both ``x`` and ``intv`` have to be provided.

        Returns:
            :class:`~stadion.parameters.InterventionParameters`
        """
        pass


    @abstractmethod
    def regularize_sparsity(self, param):
        """
        Sparsity regularization term to penalize causal dependencies of the variables in the system of SDEs.

        Args:
            param (pytree): SDE model parameters

        Returns:
            scalar value of the regularization penalty
        """
        pass



    @staticmethod
    def _format_input_data(x, intv):
        if isinstance(x, (onp.ndarray, jnp.ndarray)) and x.ndim == 2:
            x = [x]

        x = [onp.array(x_) for x_ in x]
        n_vars = x[0].shape[-1]
        n_envs = len(x)

        assert all([x_.ndim == 2 for x_ in x]), "All datasets have to be 2D arrays."
        assert all([x_.shape[-1] == n_vars for x_ in x]), "All datasets have to have same number of variables."

        if intv is not None:
            if isinstance(intv, (onp.ndarray, jnp.ndarray)) and intv.ndim == 1:
                intv = [intv]

            intv = [onp.array(intv_) for intv_ in intv]
            assert len(intv) == n_envs, "Number of intervention masks has to match number of datasets."
            assert all([intv_.ndim == 1 for intv_ in intv]), "All intervention masks have to be 1D arrays."
            assert all([intv_.shape[0] == n_vars for intv_ in intv]), "Intervention masks have to have same number "\
                                                                      "of variables and match datasets."

        else:
            intv = [onp.zeros(n_vars, dtype=onp.int32) if j == 0 else onp.ones(n_vars, dtype=onp.int32)
                    for j in range(len(x))]

        return x, intv, n_envs, n_vars


    def fit(
        self,
        key,
        x,
        targets=None,
        bandwidth=5.0,
        estimator="linear",
        learning_rate=0.003,
        steps=10000,
        batch_size=128,
        reg=0.001,
        warm_start_intv=True,
        verbose=10,
    ):
        """
        Args:
            key (PRNGKey)
            x (Dataset): This can be any of:
                ndarray of shape ``[n, d]`` for a single dataset,
                ndarray of shape ``[m, n, d]`` for multiple datasets, or
                list of length ``m`` of ndarrays of shape ``[:, d]`` for
                multiple datasets
                with different numbers of samples.
            targets (ndarray, optional): Known intervention targets provided
                as a multi-hot indicator vector and aligned with input ``x``
                of the form: ndarray of shape ``[d,]`` encoding which
                variables were intervened upon, ndarray of shape ``[m, d]``
                encoding which variables were intervened upon in each
                environment, or list of length ``m` of ndarrays of shape
                ``[d]``. When ``None`` and providing a single dataset in
                ``x``, defaults to no variables being intervened upon,
                so ``intv[...] == 0`` (in causality terms, the observational
                setting), When ``None`` and providing multiple datasets
                in ``x``, defaults to the first environment being
                observational and all other environments having unknonw
                targets, so ``intv[0, :] == 0`` and ``intv[1:, :] == 1``.
            bandwidth (float, optional): Bandwidth of the RBF kernel.
            estimator (str, optional): Estimator for the KDS loss. Options:
                ``u-statistic``, ``v-statistic``, ``linear`` (Default:
                ``linear``). The u-statistic/v-statistic estimates scale
                quadratically in the dataset size ``n``, the only difference
                being that the v-statistic does not drop the diagonals in
                the double sum. The linear estimator has higher variance
                but can be advantageous in large-scale settings.
            learning_rate (float, optional): Learning rate for Adam.
            steps (int, optional): Number of optimization steps.
            batch_size (int, optional): Batch size for data loader.
            reg (float, optional): Regularization strength for sparsity
                penalty as specified by function implementing abstract
                method :func:`~stadion.inference.KDSMixin.regularize_sparsity`.
            warm_start_intv (bool, optional): Whether to warm-start the
                intervention parameters based on provided data ``x`` and
                targets ``intv``
            verbose (int, optional): Print log ``verbose`` number of times
                during gradient descent.

        Returns:
            ``self``
        """

        # convert x and intv into the same format
        x, targets, n_envs, self.n_vars = KDSMixin._format_input_data(x, targets)

        # set up device sharding
        device_count = jax.device_count()
        devices = jax.devices()
        mesh = mesh_utils.create_device_mesh((device_count,), devices)
        sharding = PositionalSharding(mesh)

        # initialize parameters and load to device (replicate across devices)
        key, subk = random.split(key)
        param = self.init_param(subk, self.n_vars)

        key, subk = random.split(key)
        intv_param = self.init_intv_param(subk, self.n_vars, n_envs=n_envs, targets=targets,
                            x=x if warm_start_intv else None)

        param = jax.device_put(param, sharding.replicate())
        intv_param = jax.device_put(intv_param, sharding.replicate())

        # init dataloader
        key, subk = random.split(key)
        train_loader = make_dataloader(seed=subk[0].item(), sharding=sharding, x=x, batch_size=batch_size)

        # init kernel
        kernel = partial(rbf_kernel, bandwidth=bandwidth)

        # init KDS loss
        loss_fun = kds_loss(self.f, self.sigma, kernel, estimator=estimator)

        def objective_fun(param_tup, _, batch_):
            param_, intv_param_ = param_tup

            # select interventional parameters of the batch
            # by taking dot-product with environment one-hot indicator vector
            select = lambda leaf: jnp.einsum("e,e...", batch_.env_indicator, leaf)
            intv_param_ = tree_map(select, intv_param_)
            intv_param_.targets = tree_map(select, intv_param_.targets)

            # compute mean KDS loss over environments
            loss = loss_fun(batch_.x, param_, intv_param_)
            assert loss.ndim == 0

            # compute any regularizers
            # scale by variables to be less-tuning sensitive w.r.t. to dimension
            reg_penalty = reg * self.regularize_sparsity(param_) / self.n_vars
            assert reg_penalty.ndim == 0

            # return loss, aux info dict
            l = loss + reg_penalty
            return l, dict(kds_loss=loss)

        value_and_grad =  jax.value_and_grad(objective_fun, 0, has_aux=True)

        # init optimizer and update step
        optimizer = optax.chain(optax.adam(learning_rate))
        opt_state = optimizer.init((param, intv_param))


        @jax.jit
        def update_step(key_, batch_, param_, intv_param_, opt_state_):
            """
            A single update step of the optimizer
            """
            # compute gradient of objective
            (l, l_aux), (dparam, dintv_param) = value_and_grad((param_, intv_param_), key_, batch_)

            # apply any gradient masks specified by the model
            dparam = dparam.masked(grad=True)
            dintv_param = dintv_param.masked(grad=True)

            # compute parameter update given gradient and apply masks there too
            (param_update, intv_param_update), opt_state_ = optimizer.update((dparam, dintv_param),
                                                                             opt_state_,
                                                                             (param_, intv_param_))
            param_update = param_update.masked(grad=True)
            intv_param_update = intv_param_update.masked(grad=True)

            # update step
            param_, intv_param_ = optax.apply_updates((param_, intv_param_),
                                                      (param_update, intv_param_update))

            # logging
            grad_norm = linear_algebra.global_norm(dparam)
            intv_grad_norm = linear_algebra.global_norm(dintv_param)
            nan_occurred_param = tree_isnan(dparam) | tree_isnan(param)
            nan_occurred_intv_param = tree_isnan(dintv_param) | tree_isnan(intv_param)
            aux = dict(loss=l,
                       **l_aux,
                       grad_norm=grad_norm,
                       intv_grad_norm=intv_grad_norm,
                       nan_occurred_param=nan_occurred_param,
                       nan_occurred_intv_param=nan_occurred_intv_param)

            return (param_, intv_param_, opt_state_), aux


        # optimization loop
        logs = defaultdict(float)
        t_loop = time.time()
        log_every = math.ceil(steps / verbose if verbose else 0)

        for t in range(steps + 1):

            # sample data batch
            batch = next(train_loader)

            # update step
            key, subk = random.split(key)
            (param, intv_param, opt_state), logs_t = \
                update_step(subk, batch, param, intv_param, opt_state)

            # update average of training metrics
            logs = update_ave(logs, logs_t)

            if verbose and ((not t % log_every and t != 0) or t == steps):
                t_elapsed = time.time() - t_loop
                t_loop = time.time()
                ave_logs = retrieve_ave(logs)
                logs = defaultdict(float)
                print_str = f"step: {t: >5d} " \
                            f"kds: {ave_logs['loss']: >12.6f}  | " \
                            f"min remain: {(steps - t) * t_elapsed / log_every / 60.0: >4.1f}  " \
                            f"sec/step: {t_elapsed / log_every: >5.3f}"
                print(print_str, flush=True)

        if targets is not None and intv_param.targets is not None:
            assert jnp.array_equal(jnp.array(targets), intv_param.targets)

        # save parameters
        self.param = param
        self.intv_param = intv_param

        return self
