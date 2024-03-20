from jax import random
from stadion.models import LinearSDE, MLPSDE
from pprint import pprint


if __name__ == "__main__":

    key = random.PRNGKey(0)
    n, d = 1000, 5

    # generate a dataset
    key, subk = random.split(key)
    w = random.normal(subk, shape=(d, d))

    key, subk = random.split(key)
    data = random.normal(subk, shape=(n, d)) @ w

    # fit stationary diffusion model
    model = LinearSDE()
    # model = MLPSDE()

    key, subk = random.split(key)
    model.fit(subk, data)

    # sample from model or get parameters
    key, subk = random.split(key)
    x_pred = model.sample(subk, 1000)
    param = model.param

    pprint(param)
