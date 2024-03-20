import os 
import argparse

from baselines.nodags.utils import *
from baselines.nodags.datagen.generateDataset import Dataset

CONFIGS = {
    'linear': {'inter': True, 'sem_type': 'lin', 'act_fun': 'none', 'enforce_dag': False, 'contractive': True},
    'nonlinear_relu': {'inter': True, 'sem_type': 'nnl', 'act_fun': 'relu', 'enforce_dag': False, 'contractive': True},
    'nonlinear_selu': {'inter': True, 'sem_type': 'nnl', 'act_fun': 'selu', 'enforce_dag': False, 'contractive': True},
    'dags_linear_n_contract': {'inter': True, 'sem_type': 'lin', 'act_fun': 'none', 'enforce_dag': True, 'contractive': False},
    'dags_nonlinear_selu_n_contract': {'inter': True, 'sem_type': 'nnl', 'act_fun': 'selu', 'enforce_dag': True, 'contractive': False}
}

def generate_config_data(n_nodes, 
    exp_dens, 
    n_samples, 
    n_exp, 
    mode, 
    n_hidden, 
    lip_const, 
    data_output_path, 
    g_id,
    config):

    dataset_gen = Dataset(n_nodes=n_nodes,
                                  expected_density=exp_dens,
                                  n_samples=n_samples, 
                                  n_experiments=n_exp,
                                  mode=mode,
                                  sem_type=config['sem_type'], 
                                  n_hidden=n_hidden, 
                                  act_fun=config['act_fun'], 
                                  lip_constant=lip_const,
                                  enforce_dag=config['enforce_dag'],
                                  contractive=config['contractive'])
    
    datasets = dataset_gen.generate(interventions=config['inter'])
    data_path = os.path.join(data_output_path, "training_data/nodes_{}/graph_{}".format(n_nodes, g_id))
    dataset_gen.store_data(data_path, datasets, interventions=config['inter'])

    for n_targets in [2, 3]:
        val_data_gen = Dataset(n_nodes=n_nodes,
                                        expected_density=1,
                                        mode='no-constraint',
                                        graph_provided=True, 
                                        graph=dataset_gen.graph,
                                        gen_model_provided=True, 
                                        gen_model=dataset_gen.gen_model,
                                        min_targets=n_targets, 
                                        max_targets=n_targets,
                                        n_samples=n_samples,
                                        n_experiments=10,
                                        sem_type=config['sem_type'])
        val_datasets = val_data_gen.generate(fixed_interventions=True)
        data_path = os.path.join(data_output_path, 'validation_data/nodes_{}/graph_{}/n_inter_{}'.format(n_nodes, g_id, n_targets))
        dataset_gen.store_data(data_path, datasets=val_datasets, interventions=True)

def generate_synth_dataset(args):

    for config in CONFIGS:
        for g_id in range(args.n_graphs):
            print("Generating {}: graph: {}".format(config, g_id))
            data_output_path = os.path.join(args.dop, config)

            generate_config_data(
                n_nodes=args.n_nodes,
                exp_dens=args.exp_dens,
                n_samples=args.n_samples,
                n_exp=args.n_exp,
                mode=args.mode,
                n_hidden=args.n_hidden,
                lip_const=args.lip_const,
                data_output_path=data_output_path,
                g_id=g_id,
                config=CONFIGS[config]
            )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--n-graphs', type=int, default=5)
    parser.add_argument('--n-nodes', type=int, default=10)
    parser.add_argument('--exp-dens', type=int, default=2)
    parser.add_argument('--n-samples', type=int, default=5000)
    parser.add_argument('--n-exp', type=int, default=10)
    parser.add_argument('--mode', type=str, default='indiv-node')
    parser.add_argument('--n-hidden', type=int, default=0)
    parser.add_argument('--lip-const', type=float, default=0.8)
    parser.add_argument('--dop', type=str, default='~/projects/synth_benchmark_data')

    args = parser.parse_args()

    generate_synth_dataset(args)
            

    


