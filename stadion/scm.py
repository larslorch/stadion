from argparse import Namespace
from collections import defaultdict
import math

import igraph as ig
import numpy as onp
from jax import numpy as jnp, random

from stadion.core import Data

from stadion.synthetic import make_linear_model_parameters, make_interventions
from stadion.utils.tree import iter_tree


def mat_to_toporder(mat):
    return onp.array(ig.Graph.Weighted_Adjacency(mat.tolist()).topological_sorting()).astype(int)


def sample_linear_scm(key, theta, intv_theta, intv_msks, n_samples):

    assert "scale" not in intv_theta, "scale interventions not supported for linear SCM"

    weights = theta["w1"]
    bias = theta["b1"]
    scales = theta["c1"]
    shifts = intv_theta["shift"] * intv_msks

    n_envs = intv_msks.shape[0]
    d = weights.shape[0]

    key, subk = random.split(key)
    eps = random.normal(subk, shape=(n_envs, n_samples, d)) * scales

    eps_shifted = eps + bias[None, None, :] + shifts[:, None, :]
    x = jnp.einsum("pd,end->enp", jnp.linalg.inv(jnp.eye(d) - weights), eps_shifted)
    return x


def synthetic_scm_data(seed, config):

    key = random.PRNGKey(seed)

    # sample ground truth parameters
    key, subk = random.split(key)
    true_theta = make_linear_model_parameters(subk, config)

    # make diagonal zero by default in SCM
    if "mask_self_loops" in config and config["mask_self_loops"]:
        true_theta["w1"] = true_theta["w1"].at[jnp.diag_indices(true_theta["w1"].shape[0])].set(0)

    # set up interventions
    key, subk = random.split(key)
    envs = make_interventions(key, config)

    # sample envs
    dataset_fields = []
    for env_idx, (intv_msks, intv_theta) in enumerate(envs):

        # sample data
        key, subk = random.split(key)
        if "synth-scm-linear" == config["id"] or "synth-scm-linear-raw" == config["id"]:
            samples = sample_linear_scm(subk, true_theta, intv_theta, intv_msks, config["n_samples"])

        else:
            raise ValueError(f"Unknown SCM type: {config['id']}")

        dataset_fields.append(dict(
            data=onp.array(samples),
            intv=intv_msks,
            true_param=jnp.tile(true_theta["w1"], (intv_msks.shape[0], 1, 1)),
        ).copy())

    return Data(**dataset_fields[0]), Data(**dataset_fields[1])