from jax import random, numpy as jnp, value_and_grad
from stadion import kds_loss
from pprint import pprint


if __name__ == "__main__":

    key = random.PRNGKey(0)
    n, d = 1000, 5

    # generate a dataset
    key, subk = random.split(key)
    w = random.normal(subk, shape=(d, d))

    key, subk = random.split(key)
    data = random.normal(subk, shape=(n, d)) @ w

    # define SDE functions
    f = lambda x, param: param["w"] @ x + param["b"]
    sigma = lambda x, param: jnp.exp(param["c"]) * jnp.eye(d)

    # define kernel
    kernel = lambda x, y: jnp.exp(- jnp.square(x - y).sum(-1) / 100)

    # create KDS loss function
    loss_fun = kds_loss(f, sigma, kernel, estimator="linear")

    # compute loss and parameter gradient for dataset at a parameter setting
    key, *subk = random.split(key, 4)
    p = {
        "w": random.normal(subk[0], shape=(d, d)),
        "b": random.normal(subk[1], shape=(d,)),
        "c": random.normal(subk[2], shape=(d,)),
    }

    loss, dp = value_and_grad(loss_fun, argnums=1)(data, p)

    print("Loss")
    print(loss)
    print("Parameter gradient")
    pprint(dp)
