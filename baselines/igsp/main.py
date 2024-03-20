import warnings
warnings.filterwarnings("ignore", message="No GPU automatically detected")
warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"


import random
import numpy as np
import pandas as pd

# try:
#     from .igsp import prepare_igsp, format_to_igsp, igsp
# except ImportError:
#     from igsp import prepare_igsp, format_to_igsp, igsp

from baselines.igsp.igsp import prepare_igsp, format_to_igsp, igsp
from baselines.mle import compute_linear_gaussian_mle_params, make_linear_gaussian_sampler
from stadion.utils.graph import is_acyclic


def run_igsp(seed, targets, config):

    np.random.seed(seed)
    random.seed(seed)

    # create interv mask for all observations
    interv_mask = []
    for data, intv in zip(targets.data, targets.intv):
        assert intv.shape[-1] == data.shape[-1]
        interv_mask.append(np.ones_like(data) * intv)

    # concatenate all observations
    x = np.concatenate([data for data in targets.data], axis=0)
    interv_mask = np.concatenate(interv_mask, axis=0)
    assert x.shape == interv_mask.shape

    if "alpha_inv" not in config:
        config["alpha_inv"] = config["alpha"]

    # add noise to avoid inversion problems for sparse data
    if "rank_fail_add" in config and "rank_check_tol" in config:

        # check full rank of covariance matrix
        full_rank = np.linalg.matrix_rank((x - x.mean(-2, keepdims=True)).T @
                                          (x - x.mean(-2, keepdims=True)),
                                          tol=config["rank_check_tol"]) == x.shape[-1]

        if not full_rank:
            zero_cols = np.where((x == 0.0).all(-2))[0]
            zero_rows = np.where((x == 0.0).all(-1))[0]
            warnings.warn(f"covariance matrix not full rank; "
                          f"we have {len(zero_rows)} zero rows and {len(zero_cols)} zero cols "
                          f"(can occur in gene expression data). "
                          f"Adding infinitesimal noise to observations.")

            x += np.random.normal(loc=0, scale=config["rank_fail_add"], size=x.shape)


    # regimes: np.ndarray [n_observations,] indicating which regime/environment row i originated from
    # masks: list of length [n_obs], where masks[i] is list of intervened nodes; empty list for observational
    unique, regimes = np.unique(interv_mask, axis=0, return_inverse=True)
    interv_targets = [(np.where(v)[0].tolist() if v.sum() else []) for v in unique]
    masks = [interv_targets[jj] for jj in regimes]

    # existing code
    nodes, obs_samples, iv_samples_list, targets_list = format_to_igsp(pd.DataFrame(x), pd.DataFrame(masks), regimes)

    ci_tester, invariance_tester = prepare_igsp(obs_samples,
                                                iv_samples_list, targets_list,
                                                config["alpha"], config["alpha_inv"], config["ci_test"])

    # Run IGSP
    setting_list = [dict(interventions=targets) for targets in targets_list]
    est_dag = igsp(setting_list, nodes, ci_tester, invariance_tester, nruns=5)
    dag = est_dag.to_amat()[0]

    # closed-form mle parameters
    theta, noise_vars = compute_linear_gaussian_mle_params(dag, x, interv_mask)

    # make sampler from parameters
    sampler = make_linear_gaussian_sampler(dag, theta, noise_vars)

    # build prediction
    pred = dict(dag=dag, g_edges=dag, theta=theta, noise_vars=noise_vars, is_acyclic=is_acyclic(dag))

    return sampler, pred

