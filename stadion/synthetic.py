from argparse import Namespace
from collections import defaultdict

import numpy as onp
from jax import numpy as jnp, random
import igraph as ig
import random as pyrandom

from stadion.core import sample_dynamical_system, Data
from stadion.utils.stable import project_closest_stable_matrix


def erdos_renyi(key, *, d, edges_per_var, acyclic):
    key, subk = random.split(key)
    p = min((2.0 if acyclic else 1.0) * d * edges_per_var / (d * (d - 1)), 0.99)
    mask = random.choice(subk, jnp.array([0, 1]), p=jnp.array([1 - p, p]), shape=(d, d))
    mask = mask.at[jnp.diag_indices(d)].set(0)

    if acyclic:
        mask = jnp.tril(mask, k=-1)

    # randomly permute
    key, subk = random.split(key)
    p = random.permutation(subk, onp.eye(d).astype(int))
    mask = p.T @ mask @ p

    return mask


def scale_free(key, *, d, power, edges_per_var, acyclic):

    key, subk = random.split(key)
    pyrandom.seed(subk[1].item()) # seed pyrandom based on state of jax rng

    key, subk = random.split(key)
    perm = random.permutation(subk, d).tolist()

    # sample scale free graph
    g = ig.Graph.Barabasi(n=d, m=edges_per_var, directed=True, power=power).permute_vertices(perm)
    mask = jnp.array(g.get_adjacency().data).astype(jnp.int32).T

    if not acyclic:
        # randomly orient each edge to potentially create cycles
        key, subk = random.split(key)
        flip = random.bernoulli(subk, p=0.5, shape=(d, d))
        mask = flip * mask + (1 - flip) * mask.T
        assert mask.max() <= 1

    return mask


def sbm(key, *, d, intra_edges_per_var, n_blocks, damp, acyclic):
    """
    Stochastic Block Model

    Args:
        key (PRNGKey): jax random key
        d (int): number of nodes
        intra_edges_per_var (int): expected number of edges per node inside a block
        n_blocks (int): number of blocks in model
        damp (float): if p is probability of intra block edges, damp * p is probability of inter block edges
            p is determined based on `edges_per_var`. For damp = 1.0, this is equivalent to erdos renyi
        acyclic (bool): whether to sample acyclic graph
    """

    # sample blocks
    key, subk = random.split(key)
    splits = random.choice(subk, d, shape=(n_blocks - 1,), replace=False)
    key, subk = random.split(key)
    blocks = onp.split(random.permutation(subk, d), splits)
    block_sizes = onp.array([b.shape[0] for b in blocks])

    # select p s.t. we get requested intra_edges_per_var in expectation (including self loops)
    intra_block_edges_possible = block_sizes * (block_sizes - 1)
    p_intra_block = jnp.minimum(0.99, (2.0 if acyclic else 1.0) * intra_edges_per_var * block_sizes.sum() / intra_block_edges_possible.sum())

    # sample graph
    key, subk = random.split(key)
    mat_intra = random.choice(subk, jnp.array([0, 1]), p=jnp.array([1 - p_intra_block, p_intra_block]), shape=(d, d))
    key, subk = random.split(key)
    mat_inter = random.choice(subk, jnp.array([0, 1]), p=jnp.array([1 - damp * p_intra_block, damp * p_intra_block]), shape=(d, d))

    mat = onp.array(mat_inter)
    for i, bi in enumerate(blocks):
        mat[onp.ix_(bi, bi)] = mat_intra[onp.ix_(bi, bi)]

    mat[onp.diag_indices(d)] = 0

    if acyclic:
        mat = onp.tril(mat, k=-1)

    # randomly permute
    key, subk = random.split(key)
    p = random.permutation(subk, onp.eye(d).astype(int))
    mat = p.T @ mat @ p
    return jnp.array(mat)



def make_linear_model_parameters(key, config):
    
    d = config["n_vars"]

    # sample biases
    key, subk = random.split(key)
    biases = random.uniform(subk, shape=(d,), minval=-config["maxbias"],
                                              maxval=config["maxbias"])

    # sample scales
    key, subk = random.split(key)
    scales = random.uniform(subk, shape=(d,), minval=config["minscale_log"],
                                              maxval=config["maxscale_log"])

    # sample values
    assert config["minval"] < config["maxval"]
    key, subk = random.split(key)
    vals = random.uniform(subk, shape=(d, d), minval=config["minval"] - config["maxval"],
                                              maxval=config["maxval"] - config["minval"])
    vals += config["minval"] * jnp.sign(vals)

    # sample sparse mask with `edges_per_var` edges
    key, subk = random.split(key)

    if config["graph"] == "erdos_renyi":
        mask = erdos_renyi(subk, d=d, edges_per_var=config["edges_per_var"], acyclic=False)

    elif config["graph"] == "erdos_renyi_acyclic":
        mask = erdos_renyi(subk, d=d, edges_per_var=config["edges_per_var"], acyclic=True)

    elif config["graph"] == "scale_free":
        mask = scale_free(subk, d=d, power=1.0, edges_per_var=config["edges_per_var"], acyclic=False)

    elif config["graph"] == "scale_free_acyclic":
        mask = scale_free(subk, d=d, power=1.0, edges_per_var=config["edges_per_var"], acyclic=True)

    elif config["graph"] == "sbm":
        mask = sbm(subk, d=d, intra_edges_per_var=config["edges_per_var"], n_blocks=5, damp=0.1, acyclic=False)

    elif config["graph"] == "sbm_acyclic":
        mask = sbm(subk, d=d, intra_edges_per_var=config["edges_per_var"], n_blocks=5, damp=0.1, acyclic=True)

    else:
        raise ValueError(f"Unknown random graph structure model: {config['graph']}")

    assert onp.all(onp.isclose(onp.diag(mask), 0)), "Diagonal of mask must be 0 always."

    # fill weight matrix with sampled values according to mask, and set diagonal 1
    mask = mask.at[jnp.diag_indices(d)].set(1)
    w = vals * mask

    # fill diagonal with negative sampled values
    if config["adjust"] == "eigen":
        if config["adjust_eps"] is not None:
            offset_raw_eigval = onp.real(onp.linalg.eigvals(onp.array(w))).max()
            w = w.at[onp.diag_indices(d)].add(- offset_raw_eigval - config["adjust_eps"])
        else:
            raise KeyError(f"Must specify `adjust_eps` when `adjust` is set to `eigen`.")

    elif config["adjust"] == "circle":
        w = w.at[jnp.diag_indices(d)].set(- jnp.abs(w).sum(-1))
        if config["adjust_eps"] is not None:
            offset_raw_eigval = onp.real(onp.linalg.eigvals(onp.array(w))).max()
            w = w.at[jnp.diag_indices(d)].add( - offset_raw_eigval - config["adjust_eps"])
        else:
            raise KeyError(f"Must specify `adjust_eps` when `adjust` is set to `circle`.")

    elif config["adjust"] == "project":
        if config["adjust_eps"] is not None:
            w = jnp.array(project_closest_stable_matrix(onp.array(w), eps=config["adjust_eps"]))
        else:
            raise KeyError(f"Must specify `adjust_eps` when `adjust` is set to `project`.")

    elif config["adjust"] is None:
        pass
    else:
        raise KeyError(f"Unknown adjustment mode: {config['adjustment']}")

    # final stability check
    eigenvals_check = onp.real(onp.linalg.eigvals(onp.array(w)))
    assert onp.all(eigenvals_check <= - (config["dynamic_range_eps"] if "dynamic_range_eps" in config else 0)) + 1e-3, \
        f"Eigenvalues positive:\nmat\n{w}\neigenvalues:\n{eigenvals_check}"

    return dict(w1=w, b1=biases, c1=scales)


def make_interventions(key, config):

    envs = []

    # stack permutations to ensure intervention targets are unseen when less than `d` interventions
    key, *subkeys = random.split(key, 3)
    interv_nodes_ordering = jnp.concatenate([random.permutation(subk, config["n_vars"]) for subk in subkeys])
    n_intervened = 0

    assert config["intv_shift_min"] < config["intv_shift_max"]

    for n_interv, with_observ in [(config["n_intv_train"], True),
                                  (config["n_intv_test"], False)]:
        if with_observ:
            interventions = [[]] # empty list means init with observational data setting
        else:
            interventions = []

        key, subk = random.split(key)
        intv_shift_scalars = random.uniform(subk,
                                            shape=(config["n_vars"], config["n_vars"]),
                                            minval=config["intv_shift_min"] - config["intv_shift_max"],
                                            maxval=config["intv_shift_max"] - config["intv_shift_min"])
        intv_shift_scalars += config["intv_shift_min"] * onp.sign(intv_shift_scalars)

        # log scale
        if "intv_scale" in config and config["intv_scale"]:
            assert 0.0 <= config["intv_scale_min"] <= config["intv_scale_max"]
            intv_scale_scalars = random.uniform(subk, shape=(config["n_vars"], config["n_vars"]),
                                                      minval=jnp.log(config["intv_scale_min"]),
                                                      maxval=jnp.log(config["intv_scale_max"]))
        else:
            intv_scale_scalars = None

        if n_interv:
            nodes = interv_nodes_ordering[n_intervened:(n_interv + n_intervened)]
            key, subk = random.split(key)
            ordering = random.permutation(subk, config["n_vars"])

            # add `intv_per_env` interventions per env, where each env contains one topk_node
            for node in nodes:
                # ensure `node` is first
                ordering_node = (ordering + (node - ordering[0])) % config["n_vars"]
                interventions.append([(u,
                                       intv_shift_scalars[node, u],
                                       intv_scale_scalars[node, u] if intv_scale_scalars is not None else 0.0)
                                      for u in ordering_node[:config["intvs_per_env"]]])

            n_intervened += n_interv

        # encode interventions
        intv_msks, intv_theta = [], defaultdict(list)
        for env in interventions:
            msk = onp.zeros(config["n_vars"])
            shift = onp.zeros(config["n_vars"])
            scale = onp.zeros(config["n_vars"])
            for node, shift_val, scale_val in env:
                assert msk[node] == 0, "Can only intervene once on node in one experiment"
                msk[node] = 1
                shift[node] = shift_val
                scale[node] = scale_val
            intv_msks.append(msk)
            intv_theta["shift"].append(shift)
            if intv_scale_scalars is not None:
                intv_theta["scale"].append(scale)

        intv_msks = jnp.array(intv_msks)
        intv_theta = {k: jnp.array(v) for k, v in intv_theta.items()}

        envs.append((intv_msks, intv_theta))

    return envs


def synthetic_sde_data(seed, config):

    key = random.PRNGKey(seed)

    # initialize model
    if "intv_shift_u" in config and config["intv_shift_u"]:
        from stadion.models.linear_u_diag import f, sigma
    else:
        from stadion.models.linear_diag import f, sigma

    # sample ground truth parameters
    key, subk = random.split(key)
    true_theta = make_linear_model_parameters(subk, config)

    # set up interventions
    key, subk = random.split(key)
    envs = make_interventions(key, config)

    # sample envs
    dataset_fields = []
    for env_idx, (intv_msks, intv_theta) in enumerate(envs):

        # simulate SDE
        key, subk = random.split(key)
        samples, traj, log = sample_dynamical_system(
            subk, true_theta, intv_theta, intv_msks,
            n_samples=config["n_samples"],
            config=Namespace(**config["sde"]),
            f=f, sigma=sigma,
            return_traj=True,
        )
        samples = onp.array(samples)
        log = {k: onp.array(v) for k, v in log.items()}

        # check if traj should be saved
        # if yes, store one random rollout (prioritizing nans and large values for debugging) and store 2x burnin samples
        priority = sorted(range(traj.shape[-3]), key=lambda i: (-onp.isnan(traj[:, i]).sum(), -onp.abs(traj[:, i]).max()))
        traj_idx = priority[0]

        # discard burnin (note: traj is already thinned, and traj are by factor `n_rollouts` shorter than `n_samples`)
        if "save_traj" in config and config["save_traj"]:
            traj = onp.array(traj[:, traj_idx, config["sde"]["n_samples_burnin"]:, :])
        else:
            traj = None

        # select traj_idx also in log
        if log:
            for k in log.keys():
                log[k] = log[k][:, traj_idx, :]

        dataset_fields.append((
            dict(
                data=samples,
                intv=intv_msks,
                true_param=jnp.tile(true_theta["w1"], (intv_msks.shape[0], 1, 1)),
                traj=traj,
            ).copy(),
            log,
        ))

    return (Data(**dataset_fields[0][0]), dataset_fields[0][1]), \
           (Data(**dataset_fields[1][0]), dataset_fields[1][1])


# if __name__ == "__main__":
#
#     from jax.scipy.linalg import expm
#
#     key = random.PRNGKey(0)
#
#     acyclic = 0
#     d = 20
#     n = 100
#     for _ in range(n):
#
#         key, subk = random.split(key)
#         # mask = erdos_renyi(subk, d=d, edges_per_var=3, acyclic=False).astype(jnp.float32)
#         # mask = sbm(subk, d=d, intra_edges_per_var=3, n_blocks=5, damp=0.1, acyclic=True)
#         mask = scale_free(subk, d=d, power=1.0, edges_per_var=3, acyclic=False)
#         s = 0
#         for i in range(5):
#             s += jnp.trace(jnp.linalg.matrix_power(mask, i))
#         acyclic += s - d
#
#     print(acyclic / n)
