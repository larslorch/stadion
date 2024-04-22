from jax import random, numpy as jnp
from stadion.models import LinearSDE
from pprint import pprint


if __name__ == "__main__":

    key = random.PRNGKey(0)
    n, d = 1000, 5

    # generate a dataset
    key, subk = random.split(key)
    w = random.normal(subk, shape=(d, d))

    key, subk = random.split(key)
    data = random.normal(subk, shape=(n, d)) @ w

    # sample two more datasets with shift interventions
    a, targets_a =  3, jnp.array([0, 1, 0, 0, 0])
    b, targets_b = -5, jnp.array([0, 0, 0, 1, 0])

    key, subk_0, subk_1 = random.split(key, 3)
    data_a = (random.normal(subk_0, shape=(n, d)) + a * targets_a) @ w
    data_b = (random.normal(subk_1, shape=(n, d)) + b * targets_b) @ w

    # fit stationary diffusion model
    model = LinearSDE()
    key, subk = random.split(key)
    model.fit(
        subk,
        [data, data_a, data_b],
        targets=[jnp.zeros(d), targets_a, targets_b],
    )

    # get inferred model and intervention parameters
    param = model.param
    intv_param = model.intv_param

    pprint(param)
    pprint(intv_param)

    # sample from model under intervention parameters learned for 1st environment
    intv_param_a = intv_param.index_at(1)
    x_pred_a = model.sample(subk, 1000, intv_param=intv_param_a)

    print("Means of observed and generated data under 1st intervention: ")
    print(data_a.mean(0))
    print(x_pred_a.mean(0))
