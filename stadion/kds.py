from functools import partial

import jax
from jax import vmap, tree_map
import jax.numpy as jnp


def _wrapped_mean(func, axis=None):
    def wrapped(*args):
        return tree_map(partial(jnp.mean, axis=axis), func(*args))
    return wrapped


def kds_loss(f, sigma, kernel, estimator="linear"):
    """
    KDS loss function for arbitrary SDE functions :math:`f(x, \\dots)`
    and :math:`\\sigma(x, \\dots)`.

    The KDS loss uses a kernel function to quantify the discrepancy between
    the empirical distribution of a dataset and the stationary distribution of
    the SDE defined by :math:`f(x, \\dots)` and :math:`\\sigma(x, \\dots)`.
    Under some regularity conditions, the KDS is zero iff the distribution of
    the data is equal to the stationary distribution induced by the SDE, and
    it is positive otherwise, measuring the degree of the discrepancy.
    See [Lorch et al. (2024)](https://arxiv.org/abs/2310.17405) for details.

    Args:
        f (func): SDE drift function satisfying ``[d], *args -> [d]``
        sigma (func): SDE diffusion function satisfying ``[d], *args -> [d, d]``
        kernel (func): Kernel function satisfying ``[d], [d] -> []``
        estimator (str, optional): Options: ``u-statistic``, ``v-statistic``,
            ``linear`` (Default: ``linear``). The u-statistic/v-statistic
            estimates scale quadratically in the dataset size ``n``, the only
            difference being that the v-statistic does not drop the diagonals
            in the double sum. The linear estimator has higher variance but can
            be advantageous in large-scale settings.

    Returns:
        Loss function with signature
        Args:
            data (ndarray): dataset of shape ``[n, d]``
            *args (tuple, optional): additional arguments passed to ``f``, ``sigma`
        Returns:
            scalar loss value

        Typically, the additional arguments ``args`` are parameters of the
        SDE functions ``f`` and ``sigma``, which can then be learned by
        computing the gradient of the returned loss function with respect
        to ``args``.
    """

    def _kernel(x, y, *args):
        return kernel(x, y)

    def generator(h, argnum):
        def h_out(x, y, *args):
            assert x.ndim == y.ndim == 1
            assert x.shape == y.shape
            z = x if argnum == 0 else y
            f_x = f(z, *args)
            sigma_x = sigma(z, *args)
            return f_x @ jax.grad(h, argnum)(x, y, *args) \
                   + 0.5 * jnp.trace(sigma_x @ sigma_x.T @ jax.hessian(h, argnum)(x, y, *args))
        return h_out

    def loss_term(x, y, *args):
        return generator(generator(_kernel, 0), 1)(x, y, *args)

    if estimator == "v-statistic":
        # run check to make sure kernel is differentiable at x = x'
        x0, x1 = jnp.array([0.0]), jnp.array([1.0])
        x0_check = jnp.isnan(jax.grad(kernel)(x0, x0)).any()
        x1_check = jnp.isnan(jax.grad(kernel)(x1, x1)).any()
        assert not x0_check and not x1_check, \
            ("Kernel is not differentiable at x = x', "
             "which is required for the v-statistic. "
             "Try another estimator or re-writing the kernel. "
             "For example, for kernels involving L2 norms, "
             "`jnp.linalg.norm(x - y) ** 2` is not differentiable at x = y, "
             "but `jnp.square(x - y).sum(-1) is.")

        @partial(_wrapped_mean, axis=(0, 1))
        @partial(vmap, in_axes=(0, None, None))
        @partial(vmap, in_axes=(None, 0, None))
        def _loss(x, y, args):
            return loss_term(x, y, *args)

        def loss(x, *args):
            assert x.ndim == 2
            return _loss(x, x, args)

    elif estimator == "u-statistic":
        @partial(_wrapped_mean, axis=(0, 1))
        @partial(vmap, in_axes=(0, None, None, 0), out_axes=0)
        @partial(vmap, in_axes=(None, 0, None, 0), out_axes=0)
        def _loss(x, y, args, mask):
            return jnp.where(mask, 0.0, loss_term(x, y, *args))

        def loss(x, *args):
            assert x.ndim == 2
            n, _ = x.shape
            return _loss(x, x, args, jnp.eye(n)) * n / (n - 1)

    elif estimator == "linear":
        @partial(_wrapped_mean, axis=(0,))
        @partial(vmap, in_axes=(0, 0, None))
        def _loss(x, y, args):
            return loss_term(x, y, *args)

        def loss(x, *args):
            assert x.ndim == 2
            n, d = x.shape
            x = (x[:-1] if n % 2 else x).reshape(2, -1, d)
            return _loss(x[0], x[1], args)

    else:
        raise ValueError(f"Unknown estimator `{estimator}`. "
                         f"Options: `linear`, `u-statistic`, `v-statistic`.")

    return loss




