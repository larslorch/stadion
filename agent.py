import os
import argparse
from pprint import pprint

from stadion.run import run_algo_wandb
from stadion.utils.parse import load_methods_config, dict_tree_to_ordered
from stadion.definitions import PROJECT_DIR, SUBDIR_SWEEPS, CLUSTER_SCRATCH_DIR, IS_CLUSTER

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True, type=str)
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--cluster_t_max", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--descr") # only used by slurm launcher to create job name
    kwargs = parser.parse_args()

    # load and select config
    sweep_path = PROJECT_DIR / SUBDIR_SWEEPS / f"{kwargs.sweep}.yaml"
    configs = load_methods_config(sweep_path)
    configs = list(configs.values())
    config = configs[kwargs.seed]
    config["cluster_t_max"] = kwargs.cluster_t_max

    # set wandb config config
    jobid = os.environ.get('SLURM_ARRAY_JOB_ID')
    taskid = os.environ.get('SLURM_ARRAY_TASK_ID')
    jobid = "" if jobid is None else f"{jobid}-"
    taskid = "" if taskid is None else f"id={taskid}"

    init_wandb_dir = CLUSTER_SCRATCH_DIR if IS_CLUSTER else PROJECT_DIR
    init_wandb_dir.mkdir(exist_ok=True, parents=True)

    project = f"{kwargs.sweep}.{kwargs.name}"
    name = f"{kwargs.name}.{jobid}{taskid}"

    init_wandb_dir = CLUSTER_SCRATCH_DIR if IS_CLUSTER else PROJECT_DIR
    init_wandb_dir.mkdir(exist_ok=True, parents=True)

    wandb_config = dict(
        config=config,
        entity="anonymous",
        project=project,
        name=name,
        dir=init_wandb_dir,
        mode="offline",
        save_code=True,
    )

    print("\nLaunching wandb run with config:")
    pprint(dict_tree_to_ordered(wandb_config, to_dict=True))

    # run algo
    run_algo_wandb(wandb_config=wandb_config)
