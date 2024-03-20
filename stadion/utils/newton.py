import functools
from frozendict import frozendict

import jax
from jax import numpy as jnp


def newton(f, x0, *, steps):

    def _newton_step(carry, _):
        x, was_nan = carry
        inv = jnp.linalg.inv(jax.jacrev(f)(x))
        was_nan |= jnp.isnan(inv).any()
        x_new = x - inv @ f(x)
        return (x_new, was_nan), _

    carry_init = (x0, False)
    (sol, inv_nan), _ = jax.lax.scan(_newton_step, carry_init, None, length=steps)
    error = jnp.linalg.norm(f(sol))
    return sol, dict(inv_nan=inv_nan, error=error)


def log_init(shape):
    return dict(error=jnp.zeros(shape), inv_nan=jnp.zeros(shape).astype(bool))


@functools.partial(jax.jit, static_argnums=(0, 1, 5))
@functools.partial(jnp.vectorize, excluded=(0, 1, 5), signature='(k),(k),()->(k),()')
def _implicit_euler_step(f, sigma, xt, noise, dt, newton_kwargs):
    """
    One iteration of backward/semi-implicit Euler-Maruyama using stepsize `dt`
    and Newton's method for finding the implicit x(t+1) given x(t)

    Args:
        f: [..., d] -> [..., d] SDE drift function
        sigma: [..., d] -> [..., d, d] SDE diffusion function
        xt: [..., d]
        noise: [..., d] Gaussian driving noise
        dt: float
        newton_kwargs: dict of keyword arguments passed to `newton`

    Returns:
        [..., d] x(t+1)

    """
    assert xt.shape == noise.shape, f"{xt.shape} {noise.shape}"

    # function for which we find the root in the backward Euler-Maruyama method
    def h(x_next):

        drift = f(x_next)
        diffusion = sigma(xt)

        assert drift.shape == xt.shape == x_next.shape, f"{drift.shape} {xt.shape} {x_next.shape}"
        assert drift.shape[:-1] == diffusion.shape[:-2], f"{drift.shape} {diffusion.shape}"

        return xt + drift * dt + sigma(xt) @ noise * jnp.sqrt(dt) - x_next

    # run newtons method to find the root x(t+1), initializing at the current x(t)
    return newton(h, xt, **newton_kwargs)


def implicit_euler_step(f, sigma, x, xi, dt, **newton_kwargs):
    """Wrapper around _implicit_euler_step to allow for kwargs in combination with jax.numpy.vectorize"""
    return _implicit_euler_step(f, sigma, x, xi, dt, frozendict(newton_kwargs))
