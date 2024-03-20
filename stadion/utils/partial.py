import jax.numpy as jnp

def wrapped_mean(func, axis=None):
    def wrapped(*args):
        return jnp.mean(func(*args), axis=axis)

    return wrapped
