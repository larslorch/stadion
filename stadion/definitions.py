from pathlib import Path
import psutil

try:
    cpu_count = len(psutil.Process().cpu_affinity())
except AttributeError:
    cpu_count = psutil.cpu_count(logical=True)

# subdirectories
SUBDIR_EXPERIMENTS = "config"
SUBDIR_SWEEPS = "sweeps"
SUBDIR_ASSETS = "assets"
SUBDIR_SLURM_LOGS = "slurm_logs"
SUBDIR_RESULTS = "results"

# directories
ROOT_DIR = Path(__file__).parents[0]
PROJECT_DIR = Path(__file__).parents[1]

LOCAL_STORE_DIR = Path("/cluster/path/to/anonymous/directory")
CLUSTER_GROUP_DIR = Path("/cluster/path/to/anonymous/directory")
CLUSTER_SCRATCH_DIR = Path("/cluster/path/to/anonymous/directory")

IS_CLUSTER = Path("/cluster").is_dir()
STORE_DIR = CLUSTER_GROUP_DIR if IS_CLUSTER else LOCAL_STORE_DIR
SCRATCH_STORE_DIR = CLUSTER_SCRATCH_DIR if IS_CLUSTER else LOCAL_STORE_DIR

CONFIG_DIR = PROJECT_DIR / SUBDIR_EXPERIMENTS

# experiments
EXPERIMENT_DATA = "data"
EXPERIMENT_PREDS = "predictions"
EXPERIMENT_SUMMARY = "summary"
EXPERIMENT_DATA_SUMMARY = "data_summary"

EXPERIMENT_CONFIG_DATA = "data.yaml"
EXPERIMENT_CONFIG_DATA_GRID = "data_grid.yaml"
EXPERIMENT_CONFIG_METHODS = "methods.yaml"
EXPERIMENT_CONFIG_METHODS_VALIDATION = "methods_validation.yaml"

FOLDER_TRAIN = "train"
FOLDER_TEST = "test"
FILE_X = "x.csv"
FILE_INTV = "intv.csv"
FILE_TRUE_PARAM = "param.json"
FILE_TRAJ = "traj.csv"
FILE_META_DATA = "meta_data.json"
FILE_DATA_SANITY_CHECKS = "checks"

BASELINE_GIES = "gies"
BASELINE_DCDI = "dcdi"
BASELINE_IGSP = "igsp"
BASELINE_LLC = "llc"
BASELINE_NODAGS = "nodags"
BASELINE_OURS = "ours"

NAN_MIN = -1e4
NAN_MAX = 1e4

# cluster
YAML_RUN = "__run__"
DEFAULT_RUN_KWARGS = {"n_cpus": 1, "n_gpus": 0, "length": "short"}
