import random as pyrandom
import jax.numpy as jnp
from jax import random
import numpy as onp

from baselines.nodags.datagen.graph import DirectedGraphGenerator
from baselines.nodags.datagen.structuralModels import linearSEM
from baselines.nodags.datagen.generateDataset import Dataset
from baselines.nodags.utils import *

from baselines.nodags.models.resblock_trainer import resflow_train_test_wrapper


def run_nodags(seed, train_targets, config):
    """Wrapper for LLC for our data and sampler interfact"""

    pyrandom.seed(seed)
    onp.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # format data
    datasets = [data for data in train_targets.data]
    intervention_sets = [(np.where(intv)[0] if intv.sum() else np.array([])) for intv in train_targets.intv]

    # run LLC
    nodags_wrapper = resflow_train_test_wrapper(
        n_nodes=datasets[0].shape[-1],
        batch_size=64,
        l1_reg=True,
        lambda_c=config["lambda_c"],
        n_lip_iter=config["n_lip_iter"],
        fun_type='gst-mlp',
        act_fun='relu',
        epochs=100,
        lr=config["lr"],
        lip_const=0.9,
        optim='adam',
        inline=True,
        n_hidden=config["n_hidden"],
        lin_logdet=False,
        dag_input=False,
        thresh_val=0.05,
        upd_lip=True,
        centered=True)

    nodags_wrapper.train(datasets, intervention_sets, batch_size=256)
    g = np.where(nodags_wrapper.get_adjacency() > 0.5, 1, 0)
    g_prob = nodags_wrapper.f.gumbel_soft_layer.get_proba().detach().cpu().numpy()
    noise_scale = np.diag(np.exp(nodags_wrapper.model.var.detach().cpu().numpy()))

    # make sampler from fitted parameters
    def sampler(key, _, intv_theta, intv, n_samples):
        # shift intervention
        # [n_envs, N, d]
        n_envs, d = intv.shape
        shift = intv_theta["shift"] * intv
        x = []
        for shft, intv_set in zip(shift, intv):
            key, subk = random.split(key)
            eps = onp.array(random.normal(subk, shape=(n_samples, d)) @ noise_scale)
            intv_set = onp.where(intv_set.astype(bool))[0]
            intervention = onp.array(shft[intv_set])
            xj = nodags_wrapper.predictSamples(intervention, intv_set, n_samples=n_samples,
                                               noise_vec=eps, imperfect=True)
            x.append(xj.detach().numpy())

        return onp.stack(x, axis=0)

    pred = dict(g_edges=g, g_edge_probs=g_prob, noise_vars=jnp.diag(noise_scale ** 2))
    return sampler, pred


def debug_nodags():

    # search params
    lambda_c = 0.001
    n_lip_iter = 5

    seed = 123

    np.set_printoptions(precision=2, suppress=True)
    rng = np.random.default_rng(seed)

    # # set random seed
    pyrandom.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    n_nodes = 5
    graph_generator = DirectedGraphGenerator(nodes=n_nodes, expected_density=2)
    graph = graph_generator()

    gen_model_true = linearSEM(graph, abs_weight_low=0.2, abs_weight_high=0.9, noise_scale=0.5, contractive=True)
    B_true = gen_model_true.weights

    dataset_gen = Dataset(n_nodes=n_nodes, expected_density=2, n_samples=1000, n_experiments=n_nodes,
                          mode='indiv-node', graph_provided=True, graph=graph, gen_model_provided=True,
                          gen_model=gen_model_true, min_targets=1, max_targets=3)
    datasets = dataset_gen.generate(rng=rng, interventions=True, fixed_interventions=False)

    # center data
    observ_mean = datasets[0].mean(axis=0)
    observ_std = datasets[0].std(axis=0)
    datasets_centered = [(dataset - observ_mean)/observ_std for dataset in datasets]

    # run LLC
    nodags_wrapper = resflow_train_test_wrapper(
        n_nodes=n_nodes,
        batch_size=64,
        l1_reg=True,
        lambda_c=lambda_c,
        n_lip_iter=n_lip_iter,
        fun_type='gst-mlp',
        act_fun='relu',
        epochs=100,
        lr=1e-2,
        lip_const=0.99,
        optim='adam',
        inline=True,
        n_hidden=1,
        lin_logdet=False,
        dag_input=False,
        thresh_val=0.05,
        upd_lip=True,
        centered=True)

    nodags_wrapper.train(datasets_centered, dataset_gen.targets, batch_size=256)

    # extract results
    print("B_true")
    print(B_true)

    print("graph est")
    g_est = np.where(nodags_wrapper.get_adjacency() > 0.5, 1, 0)
    print(g_est)

    # intervention_set =
    n_samples = 2000
    intervention_set = [0, 2]

    intervention = np.array([-10, 10])
    intervention = rng.normal(loc=intervention, scale=1, size=(n_samples, len(intervention)))

    samples = nodags_wrapper.predictSamples(intervention, intervention_set, n_samples=n_samples, imperfect=True)
    print(samples.mean(0))

    exit()


if __name__ == '__main__':
    debug_nodags()