import warnings
warnings.filterwarnings("ignore", message="No GPU automatically detected")
warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"


import numpy as onp
import pandas as pd

from cdt.causality.graph import GIES as rGIES

from stadion.utils.graph import nx_adjacency, random_consistent_expansion, is_acyclic
from baselines.mle import compute_linear_gaussian_mle_params, make_linear_gaussian_sampler

def run_gies(seed, targets_input, config):
    """
    Greedy interventional equivalence search
    The output is a Partially Directed Acyclic Graph (PDAG)
    (A markov equivalence class). The available scores assume linearity of
    mechanisms and gaussianity of the data.

    https://rdrr.io/cran/pcalg/man/gies.html
    https://cran.r-project.org/web/packages/pcalg/vignettes/vignette2018.pdf

    Uses the score 'int' : `GaussL0penIntScore`.
    This is analogous to 'obs' used by GES but adapted for a mix of observational and interventional data

    """

    rng = onp.random.default_rng(seed)

    # create interv mask for all observations
    target_mask = []
    for data, intv in zip(targets_input.data, targets_input.intv):
        assert intv.shape[-1] == data.shape[-1]
        target_mask.append(onp.ones_like(data) * intv)

    # concatenate all observations
    x = onp.concatenate([data for data in targets_input.data], axis=0)
    target_mask = onp.concatenate(target_mask, axis=0).astype(onp.int32)
    assert x.shape == target_mask.shape

    # convert mask [N, d] notation to the two expected target and target_indices variables (1-indexed for R)
    # `targets`: list of lists, where each inner list represents one "environment" by its intervened upon variables
    # `target_indices`: [N,] list containing environment index for each data sample
    unique, indices = onp.unique(target_mask, axis=0, return_inverse=True)
    targets = [((onp.where(v)[0] + 1).tolist() if v.sum() else [-1]) for v in unique]
    target_indices = (indices + 1).tolist()

    # flatten targets (nested list) into list; remember length of inner lists
    assert len(set(tuple(v) for v in targets)) == len(targets), \
        f"Each element in `targets` needs to be unique. \nGot {targets}"
    assert min(target_indices) == 1, "R is 1-indexed"
    assert max(target_indices) == len(targets), \
        "Provided more environments in `targets` than used in `target_indices`"
    flat_targets = []
    flat_target_lengths = []
    for idx, v in enumerate(targets):
        assert type(v) in [list, tuple]
        assert len(v) > 0 and len(set(v)) == len(v), "element of `targets` contains duplicates or is empty"
        assert not (any([w is None for w in v]) and len(v) > 1), \
            f"`None`, i.e. observational data, can only be an env by itself.\nGot: {v} in targets {targets}"
        flat_target_lengths.append(len(v))
        flat_targets += v

    # infer cpdag using GIES
    pred_cpdag = rGIES(seed, score="int", verbose=False).predict(
        pd.DataFrame(data=x),
        flat_targets=pd.DataFrame(data=flat_targets),
        flat_target_lengths=pd.DataFrame(data=flat_target_lengths),
        target_indices=pd.DataFrame(data=target_indices))

    cpdag = onp.array(nx_adjacency(pred_cpdag), dtype=onp.int32)

    # random consistent extension (DAG in MEC)
    dag = random_consistent_expansion(rng=rng, cpdag=cpdag)
    assert is_acyclic(dag)

    # closed-form mle parameters
    theta, noise_vars = compute_linear_gaussian_mle_params(dag, x, target_mask)

    # make sampler from parameters
    sampler = make_linear_gaussian_sampler(dag, theta, noise_vars)

    # build prediction
    pred = dict(dag=dag, g_edges=dag, cpdag_edges=cpdag, theta=theta, noise_vars=noise_vars, is_acyclic=is_acyclic(dag))

    return sampler, pred

