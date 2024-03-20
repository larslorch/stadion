import jax.numpy as jnp
from jax import random
import numpy as np
from jax.scipy.optimize import minimize

from baselines.nodags.datagen.graph import DirectedGraphGenerator
from baselines.nodags.datagen.structuralModels import linearSEM
from baselines.nodags.datagen.generateDataset import Dataset


def get_coefficients(cov, i, u, intervention_set, observed_set):
    coefs = np.zeros(len(intervention_set) + len(observed_set)-1)
    
    get_index = lambda x: x if x < u else x-1
    for node in observed_set:
        if node != u:
            coefs[get_index(node)] = cov[i, node]
    
    coefs[get_index(i)] = 1
    return coefs


def parse_experiment(dataset, intervention_set, T, t, curr_row=0):
    n_nodes = dataset.shape[1]
    observed_set = np.setdiff1d(np.arange(n_nodes), intervention_set)

    # get the covariance matrix
    dataset_cent = dataset - dataset.mean(axis=0)
    Cov_x = (1/dataset.shape[0]) * dataset_cent.T @ dataset_cent

    st_row = curr_row

    # construct T and t
    for int_node in intervention_set:
        for obs_node in observed_set:
            coefs = get_coefficients(Cov_x, int_node, obs_node, intervention_set, observed_set)
            st_col = obs_node * (n_nodes - 1)
            T[st_row, st_col : st_col + n_nodes - 1] = coefs
            t[st_row] = Cov_x[int_node, obs_node]
            st_row += 1
            
    return T, t, st_row


def compute_n_rows(n_nodes, intervention_sets):
    n_rows = 0
    for intervention_set in intervention_sets:
        n_rows += len(intervention_set) * (n_nodes - len(intervention_set))
    return n_rows


def predict_adj_llc(datasets, intervention_sets, config):
    n_nodes = datasets[0].shape[1]
    n_rows = compute_n_rows(n_nodes, intervention_sets)
    n_cols = n_nodes * (n_nodes - 1)

    T = np.zeros((n_rows, n_cols))
    t = np.zeros((n_rows, 1))
    st_row = 0
    
    lat_var = np.zeros(n_nodes)
    lat_count = np.zeros(n_nodes)

    for dataset, intervention_set in zip(datasets, intervention_sets):
        T, t, st_row = parse_experiment(dataset, intervention_set, T, t, st_row)

    # Solve linear system
    if "reg_coeff" not in config or config["reg_coeff"] == 0:
        # closed-form when not using regularization
        b_est = np.linalg.pinv(T) @ t
    else:
        # BFGS when using regularization
        obj = lambda b: jnp.sum((jnp.array(T) @ b - jnp.array(t.squeeze())) ** 2) + config["reg_coeff"] * jnp.sum(jnp.abs(b))
        sol = minimize(obj, jnp.zeros(n_cols), method="BFGS", tol=1e-20)
        b_est = sol.x[..., None]

    # Estimating the adjacency matrix
    B_est = np.zeros((n_nodes, n_nodes))
    for n in range(n_nodes):
        exc_n_set = np.setdiff1d(np.arange(n_nodes), n)
        B_est[exc_n_set, n] = b_est[n * (n_nodes-1) : (n + 1) * (n_nodes - 1)].squeeze()

    # Estimating noise variances
    for dataset, intervention_set in zip(datasets, intervention_sets):
        observed_set = np.setdiff1d(np.arange(n_nodes), intervention_set)
        U = np.zeros((n_nodes, n_nodes))
        U[observed_set, observed_set] = 1
        dataset_cent = dataset - dataset.mean(axis=0)
        Cov_x = (1/dataset.shape[0]) * dataset_cent.T @ dataset_cent
        for obs_node in observed_set:
            lat_obs_cov = (np.eye(n_nodes) - U @ B_est.T) @ Cov_x @ (np.eye(n_nodes) - B_est @ U)
            lat_var[obs_node] += lat_obs_cov[obs_node, obs_node] 
            lat_count[obs_node] += 1
        
    lat_var /= lat_count

    return T, t, B_est, np.sqrt(lat_var)



def run_llc(seed, train_targets, config):
    """Wrapper for LLC for our data and sampler interfact"""

    # format data
    datasets = [data for data in train_targets.data]
    intervention_sets = [(np.where(intv)[0] if intv.sum() else np.array([])) for intv in train_targets.intv]

    # run LLC
    _, _, weights, noise_scale_est  = predict_adj_llc(datasets, intervention_sets, config)
    weights = weights.T # LLC code has transposed convention
    g = 1 - np.isclose(weights, 0).astype(np.int32)

    # make sampler from fitted parameters
    def sampler(key, _, intv_theta, intv, n_samples):
        # shift intervention
        # [n_envs, N, d]
        n_envs, d = intv.shape
        shift = intv_theta["shift"] * intv
        eps = random.normal(key, shape=(n_envs, n_samples, d)) * noise_scale_est
        eps_shifted = eps + shift[:, None, :]
        x = jnp.einsum("pd,end->enp", jnp.linalg.inv(np.eye(d) - weights), eps_shifted)
        return x

    pred = dict(g_edges=g, theta=weights, noise_vars=noise_scale_est ** 2)
    return sampler, pred


def debug_llc():

    np.set_printoptions(precision=2, suppress=True)
    rng = np.random.default_rng(123)

    n_nodes = 5
    graph_generator = DirectedGraphGenerator(nodes=n_nodes, expected_density=2)
    graph = graph_generator()

    gen_model_true = linearSEM(graph, abs_weight_low=0.2, abs_weight_high=0.9, noise_scale=0.5, contractive=True)
    B_true = gen_model_true.weights

    dataset_gen = Dataset(n_nodes=n_nodes, expected_density=2, n_samples=1000, n_experiments=n_nodes,
                          mode='indiv-node', graph_provided=True, graph=graph, gen_model_provided=True,
                          gen_model=gen_model_true, min_targets=1, max_targets=3)
    datasets = dataset_gen.generate(rng=rng, interventions=True, fixed_interventions=False)

    # normalize datasets by observational mean and std
    # datasets = [(data - datasets[0].mean(axis=0)) / datasets[0].std(axis=0) for data in datasets]

    # run LLC
    cconfig = {"reg_coeff": 0.0}
    _, _, B_est, noise_scale_est  = predict_adj_llc(datasets, dataset_gen.targets, cconfig)

    print("||W - W_gt|| (from samples): {}".format(np.linalg.norm(B_true - B_est, 'fro')))

    print(B_true)
    print(np.where(np.abs(B_est) > 0.1, B_est, 0))
    print()
    print(noise_scale_est)
    print()

    """Simulate some interventional data"""
    observ_data = gen_model_true.generateData(rng=rng, n_samples=5000)

    intervention_set = [0]
    intervention = np.array([5.0])
    interv_data = gen_model_true.generateData(rng=rng, n_samples=5000, intervention_set=intervention_set,
                                              fixed_intervention=True, intervention=intervention)

    print(f"Observational data:  {observ_data.mean(axis=0)}")
    print(f"Interventional data: {interv_data.mean(axis=0)}")

    gen_model_est = linearSEM(graph, weights=B_est, noise_scale=noise_scale_est)

    # true weights, true noise
    print(gen_model_true.compute_log_likelihood(interv_data, intervention_set).sum())

    # estimated weights, estimated noise
    print(gen_model_est.compute_log_likelihood(interv_data, intervention_set).sum())

    exit()


if __name__ == '__main__':

    debug_llc()