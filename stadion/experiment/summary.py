import warnings
warnings.filterwarnings("ignore", message="No GPU automatically detected")
warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"

# in t-testing when running t-test between best method and itself
warnings.filterwarnings("ignore", message=
    "Precision loss occurred in moment calculation due to catastrophic cancellation. "
    "This occurs when the data are nearly identical. ")

import argparse
from pathlib import Path

import math
import numpy as onp
import copy
from collections import OrderedDict

from pprint import pprint

from ott.geometry import pointcloud

from collections import defaultdict
from stadion.experiment.plot import benchmark_summary, METRICS_CHECKS
from stadion.experiment.plot_config import TRAIN_VALIDATION_METRICS, METRICS_TABLE_ORDERING, TRUE
from stadion.utils.version_control import str2bool

from stadion.utils.parse import load_methods_config, get_id, load_data, load_json, _ddicts_to_dicts
from stadion.core import index_data

from stadion.definitions import FOLDER_TRAIN, FOLDER_TEST, cpu_count, \
    BASELINE_OURS, BASELINE_LLC, BASELINE_NODAGS, BASELINE_IGSP, BASELINE_DCDI, BASELINE_GIES
from stadion.utils.metrics import wasserstein_fun_envs
from stadion.utils.metrics import mse_fun_envs
from stadion.utils.metrics import relmse_fun_envs
from stadion.utils.metrics import vse_fun_envs
from stadion.utils.metrics import is_inf, is_nan


def _load_targets(pred, data_train, data_test, train_validation):
    """
    Load train and test targets
    For train validation, adjust (as in methods.py) the correct train and test set from the train data only
    """

    if train_validation:
        n_train_envs = len(data_train.data)

        # only use data_train
        train_targets = index_data(data_train, onp.setdiff1d(onp.arange(n_train_envs), pred["test_idx"]))
        test_targets = index_data(data_train, pred["test_idx"])

    else:
        # target data
        train_targets = data_train
        test_targets = data_test

    return train_targets, test_targets


class MetricsComputer(object):
    """
    Class to compute metrics in parallel
    """
    def __init__(self, method, data_train, data_test, hparam_wasser, hparam_methods, kwargs):

        self.method = method
        self.data_train = data_train
        self.data_test = data_test

        self.hparam_wasser = hparam_wasser
        self.hparam_methods = hparam_methods
        self.kwargs = kwargs

    def __call__(self, path):
        
        results = dict()

        """Load data"""
        task_id = get_id(path)
        pred = load_json(path)

        # load ground truth data
        train_targets, test_targets = _load_targets(pred, self.data_train[task_id], self.data_test[task_id], self.kwargs.train_validation)

        x_train, intv_train = train_targets.data, train_targets.intv
        x_test, intv_test = test_targets.data, test_targets.intv

        # load predictions
        samples_train = pred["samples_train"]
        samples_test = pred["samples_test"]


        """Compute metrics"""
        # standard metrics
        metric_list = []
        for tup in [
            ("mse", mse_fun_envs, False),
            ("relmse", relmse_fun_envs, False),
            ("vse", vse_fun_envs, False),
        ]:
            if not self.kwargs.train_validation or any([(tup[0] in metr) for metr in TRAIN_VALIDATION_METRICS]):
                metric_list.append(tup)

        for prefix, metr_fun_envs, ave in metric_list:
            metr_train, imetr_train = metr_fun_envs(samples_train, x_train, intv_train)
            metr_test, imetr_test = metr_fun_envs(samples_test, x_test, intv_test)

            results[f"{prefix}_train"] = [metr_train[1:].mean().item()] if ave else metr_train[1:].tolist()
            results[f"i{prefix}_train"] = [imetr_train[1:].mean().item()] if ave else imetr_train[1:].tolist()

            results[f"{prefix}_test"] = [metr_test.mean().item()] if ave else metr_test.tolist()
            results[f"i{prefix}_test"] = [imetr_test.mean().item()] if ave else imetr_test.tolist()

        # Wasserstein metric
        if self.kwargs.wasser and (not self.kwargs.train_validation or any([("wasser" in metr) for metr in TRAIN_VALIDATION_METRICS])):

            # take minimum epsilon among epsilons
            # for train_validation: only variants of the considered method
            # for evaluation: all benchmarked methods
            if self.kwargs.wasser_eps is not None:
                eps_train = self.kwargs.wasser_eps * onp.ones(samples_train.shape[0])
                eps_test = self.kwargs.wasser_eps * onp.ones(samples_test.shape[0])
            else:
                if self.kwargs.train_validation:
                    mask = onp.array([self.method.split("__")[0] == v.split("__")[0] for v in self.hparam_methods[task_id]])
                    eps_train = self.hparam_wasser[task_id]["train"][mask].min(0)
                    eps_test = self.hparam_wasser[task_id]["test"][mask].min(0)
                else:
                    eps_train = self.hparam_wasser[task_id]["train"].min(0)
                    eps_test = self.hparam_wasser[task_id]["test"].min(0)

            wasser_train, iwasser_train = wasserstein_fun_envs(samples_train, x_train, intv_train, eps_train)
            wasser_test, iwasser_test = wasserstein_fun_envs(samples_test, x_test, intv_test, eps_test)

            results["wasser_train"] = wasser_train.astype(onp.float32).tolist()
            results["iwasser_train"] = iwasser_train[1:].astype(onp.float32).tolist()
            # results["wasser_env_train"] = wasser_train.astype(onp.float32).mean().item()
            # results["iwasser_env_train"] = iwasser_train[1:].astype(onp.float32).mean().item()

            results["wasser_test"] = wasser_test.astype(onp.float32).tolist()
            results["iwasser_test"] = iwasser_test.astype(onp.float32).tolist()
            # results["wasser_env_test"] = wasser_test.astype(onp.float32).mean().item()
            # results["iwasser_env_test"] = iwasser_test.astype(onp.float32).mean().item()

        # search_intv_theta_shift
        if "search_intv_theta_shift" in pred:
            for kk, vv in pred["search_intv_theta_shift"].items():
                results[f"search-{kk}"] = vv

        # debugging metrics
        results[f"nan_train"] = is_nan(samples_train).tolist()
        results[f"nan_test"] = is_nan(samples_test).tolist()
        results[f"inf_train_{100}"] = is_inf(samples_train, std_bound=100).tolist()
        results[f"inf_test_{100}"] = is_inf(samples_test, std_bound=100).tolist()

        # walltime
        if "walltime" in pred:
            results["walltime"] = pred["walltime"]

        # num_params
        if "num_params" in pred:
            results["num_params"] = pred["num_params"]

        """Save parameters and samples for visualization"""
        samples = (samples_train, intv_train), (samples_test, intv_test)
        
        param = None
        if "theta" in pred:
            if self.method in BASELINE_GIES:
                param = pred["theta"]

            elif self.method in BASELINE_IGSP:
                param = pred["theta"]

            elif self.method in BASELINE_DCDI:
                # dcdi has first layer: [num_vars, out_dim, in_dim]
                # so same convention as ours MLP
                param = pred["theta"]
                assert param.ndim == 3
                param = onp.mean(param, axis=1)

                # transpose s.t. [i, j] indicates causal effect i->j
                # mask by g_edge_probs (which already has correct convention)
                param = param.T * pred["g_edge_probs"]

            elif self.method in BASELINE_LLC:
                param = pred["theta"]

            elif self.method in BASELINE_NODAGS:
                # nodags does not factorize parameters into independent mechansims
                pass 
            
            elif BASELINE_OURS in self.method:
                if "w1" in pred["theta"]:
                    param = pred["theta"]["w1"]

                    # transpose s.t. [i, j] indicates causal effect i->j
                    param = param.T

                elif "x1" in pred["theta"]:
                    param = pred["theta"]["x1"]
                    param = onp.mean(param, axis=-1) # mean over last dimension in neural net case

                    # transpose s.t. [i, j] indicates causal effect i->j
                    param = param.T

                elif "y1" in pred["theta"]:
                    param = pred["theta"]["y1"]
                    param = onp.mean(param, axis=-1) # mean over last dimension in neural net case

                    # transpose s.t. [i, j] indicates causal effect i->j
                    param = param.T


        """Sanity checks"""
        for metr in results.keys():
            if metr not in METRICS_TABLE_ORDERING:
                warnings.warn(f"Metric `{metr}` not configured in plot_config.py")

        # if nan in prediction, print to stdout to facilitate debugging
        for key, val in pred.items():
            try:
                if onp.isnan(val).all():
                    print(f"NaN: {self.method} {task_id} {key}", flush=True)
            except TypeError:
                pass  # nan not well-defined
        
        return results, samples, param


def make_summary(summary_path, data_paths, result_paths, kwargs):
    """
    Args:
        summary_path
        data_paths (list): [paths to data csv's]
        result_paths (dict): {method: [paths to result csv's]}
        kwargs: if true, follows train validation procedure from methods.py and considers a train env as test
    """

    # method: dict of metrics
    results = defaultdict(lambda: defaultdict(list))
    samples_all = defaultdict(lambda: defaultdict(dict))
    params_all = defaultdict(dict)

    """
    Load true data
    """
    # id: data dict
    data_train = {get_id(p): load_data(p / FOLDER_TRAIN)[0] for p in data_paths}
    data_test =  {get_id(p): load_data(p / FOLDER_TEST)[0] for p in data_paths}

    """
    Compute optimal hyperparameters for metrics
    """
    print(f"Computing optimal hyperparameters for metrics...", flush=True)
    hparam_methods = defaultdict(list)
    hparam_wasser = defaultdict(lambda: defaultdict(list))

    for j, (method, method_paths) in enumerate(result_paths.items()):
        for jj, path in enumerate(method_paths):
            print(f"loading {method} {j + 1}/{len(result_paths)}: {jj + 1}/{len(method_paths)} {path}", flush=True)

            # load ground truth data
            task_id = get_id(path)
            pred = load_json(path)
            train_targets, test_targets = _load_targets(pred, data_train[task_id], data_test[task_id], kwargs.train_validation)
            hparam_methods[task_id].append(method)

            # wasserstein: epsilon heuristic (jax-ott)
            if kwargs.wasser and kwargs.wasser_eps is None and \
                    (not kwargs.train_validation or any([("wasser" in metr) for metr in TRAIN_VALIDATION_METRICS])):
                hparam_wasser[task_id]["train"].append(onp.array([
                    pointcloud.PointCloud(x, s).epsilon for x, s in zip(train_targets.data, pred["samples_train"])]))
                hparam_wasser[task_id]["test"].append(onp.array([
                    pointcloud.PointCloud(x, s).epsilon for x, s in zip(test_targets.data, pred["samples_test"])]))

            if j == 0:
                # remember true data
                samples_all["train"][TRUE][task_id] = train_targets.data, train_targets.intv
                samples_all["test"][TRUE][task_id] = test_targets.data, test_targets.intv

                # remember true parameters
                # transpose s.t. [i, j] indicates causal effect i->j
                if train_targets.true_param is not None:
                    params_all[TRUE][task_id] = train_targets.true_param[0].T


    hparam_wasser = _ddicts_to_dicts(hparam_wasser)
    hparam_methods = _ddicts_to_dicts(hparam_methods)

    pprint(hparam_wasser)
    print(f"Done.", flush=True)

    """
    Compute metrics
    """
    for ctr, (method, method_paths) in enumerate(result_paths.items()):

        print(f"summarizing {ctr + 1}/{len(result_paths)}: {method}", flush=True)

        # compute metrics individually for every test case
        compute_metrics = MetricsComputer(method, data_train, data_test, hparam_wasser, hparam_methods, kwargs)

        # pool = multiprocessing.Pool(processes=math.floor(cpu_count / 2))
        # results_method, samples_method, param_method = zip(*pool.map(compute_metrics, method_paths))
        # pool.close()
        # pool.terminate()
        # pool.join()

        results_method, samples_method, param_method = zip(*[compute_metrics(inpt) for inpt in method_paths])

        # aggregate results
        for result in results_method:
            for metr, val in result.items():
                if type(val) == list:
                    results[method][metr].extend(val)
                else:
                    results[method][metr].append(val)

        for path, sample, param in zip(method_paths, samples_method, param_method):
            task_id = get_id(path)
            samples_all["train"][method][task_id], samples_all["test"][method][task_id] = sample
            if param is not None:
                params_all[method][task_id] = param

    samples_all = OrderedDict([(kk, samples_all[kk]) for kk in ["test", "train"]])
    params_all = dict(**params_all)

    # sanity checks
    # modes = [(True, "median")]
    # modes = [(True, "median"), (False, "mean")]
    modes = [(False, "mean"), (True, "median")]
    # modes = [(False, "mean")]
    for j, (median_mode, subfolder) in enumerate(modes):

        summary_subpath = summary_path / subfolder

        benchmark_summary(kwargs, summary_subpath / "sanity_checks",
                          copy.deepcopy(results), copy.deepcopy(samples_all), copy.deepcopy(params_all),
                          median_mode=median_mode,
                          only_metrics=METRICS_CHECKS,
                          skip_plot=True)

        # dump all metrics for debugging
        benchmark_summary(kwargs, summary_subpath / "dump",
                          copy.deepcopy(results), copy.deepcopy(samples_all), copy.deepcopy(params_all),
                          median_mode=median_mode,
                          dump_main=True,
                          skip_plot=True)

        if kwargs.train_validation:
            # dump method-wise summary for hparam calibration on training data
            benchmark_summary(kwargs, summary_subpath / "train_validation",
                              copy.deepcopy(results), copy.deepcopy(samples_all), copy.deepcopy(params_all),
                              median_mode=median_mode,
                              only_metrics=TRAIN_VALIDATION_METRICS,
                              method_summaries=True,
                              skip_plot=True)

        else:
            # benchmark results
            benchmark_summary(kwargs, summary_subpath / "benchmark",
                              copy.deepcopy(results), copy.deepcopy(samples_all), copy.deepcopy(params_all),
                              median_mode=median_mode,
                              only_metrics=[
                                "wasser_train",
                                "wasser_test",
                                "mse_train",
                                "mse_test",
                                "relmse_train",
                                "relmse_test",
                                "vse_train",
                                "vse_test",
                              ], skip_plot=True)
    
            benchmark_summary(kwargs, summary_subpath / "test_benchmark",
                              copy.deepcopy(results), copy.deepcopy(samples_all), copy.deepcopy(params_all),
                              median_mode=median_mode,
                              only_metrics=[
                                "wasser_test",
                                "mse_test",
                                "relmse_test",
                              ], skip_plot=j != (len(modes) - 1))

    print("Finished successfully.", flush=True)




if __name__ == "__main__":
    """
    Runs plot call
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods_config_path", type=Path, required=True)
    parser.add_argument("--path_data", type=Path, required=True)
    parser.add_argument("--path_plots", type=Path, required=True)
    parser.add_argument("--path_results", type=Path, required=True)
    parser.add_argument("--train_validation", action="store_true")
    parser.add_argument("--only_methods", nargs="+", type=str)
    parser.add_argument("--descr")
    parser.add_argument("--max_data_plots_train", type=int, default=2) # maximum data plot
    parser.add_argument("--max_data_plots_test", type=int, default=4) # maximum data plot
    parser.add_argument("--max_data_plots_env", type=int, default=2) # maximum data plot
    parser.add_argument("--wasser", type=str2bool, default=True)
    parser.add_argument("--wasser_eps", type=float, default=0.1)

    kkwargs = parser.parse_args()

    # adjust configs based on only_methods
    methods_config = load_methods_config(kkwargs.methods_config_path, abspath=True,
                                         warn_if_grid=not kkwargs.train_validation,
                                         warn_if_not_grid=kkwargs.train_validation)

    if kkwargs.only_methods is not None:
        for k in list(methods_config.keys()):
            if not any([m in k for m in kkwargs.only_methods]):
                del methods_config[k]

    # check data and results found
    data_found = sorted([p for p in kkwargs.path_data.iterdir() if p.is_dir()])
    results_p = sorted([p for p in kkwargs.path_results.iterdir()])
    results_found = {}
    for meth, _ in methods_config.items():
        if kkwargs.train_validation:
            results_found[meth] = list(filter(lambda p: p.name.rsplit("_", 1)[0] == meth + "_train_validation", results_p))
        else:
            results_found[meth] = list(filter(lambda p: p.name.rsplit("_", 1)[0] == meth, results_p))
        if not results_found[meth]:
            del results_found[meth]

    any_results = any([len(v) for v in results_found.values()])
    if not any_results:
        print(f"No results found. (requested train validation: {kkwargs.train_validation})")
        exit()

    # train validation summary
    make_summary(kkwargs.path_plots, data_found, results_found, kkwargs)

    print("Done.")

