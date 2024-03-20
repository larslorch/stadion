import warnings
warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"

import importlib.util
import os
import json
import yaml
import subprocess
import sys
import time
import itertools
import functools
from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import math
import torch
from pprint import pprint

import numpy as onp
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt

from stadion.definitions import PROJECT_DIR, FILE_X, FILE_INTV, FILE_TRUE_PARAM, FILE_DATA_SANITY_CHECKS, FILE_TRAJ, \
    FILE_META_DATA
from stadion.core import Data, get_intv_stats
from stadion.experiment.plot_config import *
from stadion.utils.plot_data import scatter_matrix

types_list = [list]
types_dict = [dict, defaultdict, OrderedDict]


def option_to_descr(d):
    if type(d) in types_list:
        raise ValueError("No lists should occur in option_to_descr")
    elif type(d) in types_dict:
        l = []
        for k, v in d.items():
            s = option_to_descr(v)
            if s[0] == "=":
                l.append(f"{k}{s}")
            else:
                l.append(f"{k}_{s}")
        return '-'.join(l)
    else:
        return f"={d}"


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, jnp.ndarray):
            obj = onp.array(obj)
        if isinstance(obj, onp.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        return json.JSONEncoder.default(self, obj)


class NumpyJSONDecoder(json.JSONDecoder):
    def _postprocess(self, obj):
        if isinstance(obj, dict):
            return {k: self._postprocess(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            out = [self._postprocess(v) for v in obj]
            # accounts for the fact that lists of non-equal shaped arrays don't get nicely formatted as np.arrays
            if all([type(v) == onp.ndarray for v in out]) and not all([v.shape == out[0].shape for v in out]):
                return out
            else:
                return onp.array(out)
        else:
            return obj

    def decode(self, obj, recurse=False):
        decoded = json.JSONDecoder.decode(self, obj)
        return self._postprocess(decoded)


def save_json(tree, path):
    with open(path, "w") as file:
        json.dump(tree, file, indent=4, sort_keys=True, cls=NumpyJSONEncoder)


def load_json(path):
    try:
        with open(path, "r") as file:
            pred = json.load(file, cls=NumpyJSONDecoder)
            return pred
    except (json.JSONDecodeError, json.decoder.JSONDecodeError) as err:
        print(f"\n\nJSONDecodeError for path: {path}", flush=True)
        raise err

def save_csv(arr, path, columns=None, dtype=onp.float32):
    if arr.size > 0:
        pd.DataFrame(arr.astype(dtype), columns=columns).to_csv(
            path,
            index=False,
            header=columns is not None,
            # float_format="%.8f",
        )


def load_csv(path, header=None, dtype=onp.float32):
    return onp.array(pd.read_csv(path, index_col=False, header=header), dtype=dtype)


@contextmanager
def timer() -> float:
    start = time.time()
    yield lambda: time.time() - start


def get_id(path):
    return int(path.name.split("_")[-1].split('.')[0])


def get_git_hash_long():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode(sys.stdout.encoding)


def cartesian_dict(d):
    """
    Cartesian product of nested dict/defaultdicts of lists
    Example:

    d = {'s1': {'a': 0,
                'b': [0, 1, 2]},
         's2': {'c': [0, 1],
                'd': [0, 1]}}

    yields
        {'s1': {'a': 0, 'b': 0}, 's2': {'c': 0, 'd': 0}}
        {'s1': {'a': 0, 'b': 0}, 's2': {'c': 0, 'd': 1}}
        {'s1': {'a': 0, 'b': 0}, 's2': {'c': 1, 'd': 0}}
        {'s1': {'a': 0, 'b': 0}, 's2': {'c': 1, 'd': 1}}
        {'s1': {'a': 0, 'b': 1}, 's2': {'c': 0, 'd': 0}}
        ...
    """
    if type(d) in types_dict:
        keys, values = d.keys(), d.values()
        for c in itertools.product(*(cartesian_dict(v) for v in values)):
            yield dict(zip(keys, c))
    elif type(d) in types_list:
        for c in d:
            yield from cartesian_dict(c)
    else:
        yield d


def _get_list_leaves(d):
    if type(d) in types_list:
        out = d
        grid = True
    elif type(d) in types_dict:
        out = {}
        grid = False
        for k, v in d.items():
            out_k, grid_k = _get_list_leaves(v)
            grid = grid or grid_k
            if grid_k:
                out[k] = out_k
    else:
        out = d
        grid = False
    return out, grid


def get_list_leaves(d):
    """
    Filter nested dict of lists/dicts for leaves with lists
    and return cartesian product
    """
    leaf_dict, _ = _get_list_leaves(d)
    return cartesian_dict(leaf_dict)


def match_option(option, config):
    if type(option) in types_list:
        raise ValueError("No lists should occur when matching configs")
    elif type(option) in types_dict:
        return all([match_option(option[k], config[k]) for k in option.keys()])
    else:
        return option == config

def _comparer(x):
    if x is None:
        return -math.inf
    elif type(x) in types_dict:
        return tuple(zip(*x.items()))
    else:
        return x

def dict_tree_to_ordered(d, to_dict=False):
    if type(d) in types_dict:
        out = sorted({k: dict_tree_to_ordered(v, to_dict=to_dict) for k, v in d.items()}.items(), key=_comparer)
        if to_dict:
            return dict(out)
        else:
            return OrderedDict(out)

    elif type(d) == list:
        l = [dict_tree_to_ordered(k, to_dict=to_dict) for k in d]
        try:
            return sorted(l, key=_comparer)
        except TypeError as e:
            print(f"Sorting error. Did all leaves have the same type? (list or nonlist). "
                  f"Try using a singleton list when there is only one option.")
            raise e
    else:
        return d


def load_config(path, abspath=False, warn=True):
    """Load plain yaml config"""

    load_path = path if abspath else (PROJECT_DIR / path)

    try:
        with open(load_path, "r") as stream:
            try:
                config = yaml.safe_load(stream)
                return config

            except yaml.YAMLError as exc:
                warnings.warn(f"YAML parsing error. Returning `None` for config.\n")
                print(exc, flush=True)
                return None

    except FileNotFoundError as exc_outside:
        if warn:
            warnings.warn(f"Returning `None` for config. FileNotFoundError error for path: {load_path}")
            print(exc_outside, flush=True)
        return None


def load_data_config(path, abspath=False, warn_not_found=True, warn_if_grid=False, warn_if_not_grid=False):
    """Load yaml config for data specification"""

    config = load_config(path, abspath=abspath, warn=warn_not_found)
    if config is None:
        return None

    # sanity checks
    if warn_if_grid:
        if "__grid__" in config:
            warnings.warn(f"__grid__ found in config at:\n{path}")

    if warn_if_not_grid:
        if "__grid__" not in config:
            warnings.warn(f"__grid__ not found in config at:\n{path}")


    # make dict tree ordered
    config = dict_tree_to_ordered(config)

    # if grid validation, expand options and update name of resulting methods
    if "__grid__" in config:
        del config["__grid__"]
        expanded_config = {}
        all_data_settings = list(cartesian_dict(config))
        all_options = list(get_list_leaves(config))
        assert all_options, f"__grid__ config did not have options. Do not specify __grid__ if only one setting is used."

        # match expanded set of full settings with differences in leaves for unique naming
        if not all_options:
            # only one setting
            assert len(all_data_settings) == 1
            expanded_config = next(iter(all_data_settings))

        else:

            for option in all_options:
                match = list(filter(functools.partial(match_option, option), all_data_settings))
                assert len(match) == 1, f"{match}"
                option_descr = option_to_descr(option)
                expanded_config[f"{option_descr}"] = next(iter(match))

        return dict_tree_to_ordered(expanded_config)
    else:
        return config



def load_methods_config(path, abspath=False, warn_not_found=True, warn_if_grid=False, warn_if_not_grid=False):
    """Load yaml config for method specification"""

    config = load_config(path, abspath=abspath, warn=warn_not_found)
    if config is None:
        return None

    # sanity checks
    if warn_if_grid:
        if "__grid__" in config:
            warnings.warn(f"__grid__ found in config at:\n{path}")

    if warn_if_not_grid:
        if "__grid__" not in config:
            warnings.warn(f"__grid__ not found in config at:\n{path}")

    # make dict tree ordered
    config = dict_tree_to_ordered(config)

    # if grid validation, expand options and update name of resulting methods
    if "__grid__" in config:
        del config["__grid__"]
        expanded_config = {}
        for method, hparams in config.items():
            all_full_settings = list(cartesian_dict(hparams))
            all_options = list(get_list_leaves(hparams))

            # match expanded set of full settings with differences in leaves for unique naming
            if not all_options:
                # only one setting
                assert len(all_full_settings) == 1
                expanded_config[method] = next(iter(all_full_settings))

            else:

                for option in all_options:
                    match = list(filter(lambda setting: all([setting[k] == v for k, v in option.items()]),
                                        all_full_settings))
                    assert len(match) == 1, f"{match}"
                    option_descr = option_to_descr(option)
                    expanded_config[f"{method}__{option_descr}"] = next(iter(match))

        return dict_tree_to_ordered(expanded_config)
    else:
        return config


def _save_data_checks(x, intv, idx, path, ref_x=None):
    plot_path = path.parent
    # plot_path = path.parent.parent
    d = intv.shape[-1]
    var_names = [("$x_{" + str(jj) + "}$") for jj in range(1, d + 1)]

    if onp.isnan(x).all(0).any():
        # at least on dimension only has nan, skip scatter plot
        print(f"_save_data_checks: nan in idx {idx} and intv {intv} ", flush=True)
        return

    data = pd.DataFrame(x, columns=var_names)
    color_mask = onp.zeros((len(data.columns), len(data.columns)), dtype=onp.int32)
    for intv_idx in onp.where(intv)[0]:
        color_mask[:, intv_idx] = 1
        color_mask[intv_idx, :] = 1

    if ref_x is not None and not onp.allclose(x, ref_x):
        data_target = pd.DataFrame(ref_x, columns=var_names)
    else:
        data_target = None

    figax = scatter_matrix(
        data,
        data_target,
        size_per_var=FIGSIZE["grid"]["size_per_var"],
        color_mask=color_mask,
        binwidth=None,
        n_bins_default=20,
        percentile_cutoff_limits=0.05,
        range_padding=0.5,
        contours=GRID_CONTOURS,
        contours_alphas=GRID_CONTOURS_ALPHAS,
        cmain=CMAIN,
        ctarget=CTARGET,
        cref=CREF,
        ref_alpha=GRID_REF_ALPHA,
        ref_data=None,
        diagonal="hist",
        scotts_bw_scaling=SCOTTS_BW_SCALING,
        offdiagonal="scatter",
        # offdiagonal="kde",
        scat_kwds=dict(
            marker=".",
            alpha=0.10,
            s=30,
        ),
        scat_kwds_target=SCAT_KWARGS_TAR,
        hist_kwds=HIST_KWARGS,
        hist_kwds_target=HIST_KWARGS_TAR,
        density1d_kwds=DENSITY_1D_KWARGS,
        density1d_kwds_target=DENSITY_1D_KWARGS_TAR,
        density2d_kwds=DENSITY_2D_KWARGS,
        density2d_kwds_target=DENSITY_2D_KWARGS_TAR,
        minmin=MINMIN,
        maxmax=MAXMAX,
    )
    if figax is None:
        print("Warning. `figax` was None in _save_data_checks", flush=True)
        return

    plt.tight_layout()
    plt.gcf().subplots_adjust(wspace=0, hspace=0)
    filepath = plot_path / f"{FILE_DATA_SANITY_CHECKS}_grid_{idx}"
    plt.savefig(filepath.with_suffix(".pdf"), format="pdf", facecolor=None,
                dpi=DPI, bbox_inches='tight')
    plt.close()
    # plt.show()

    print(f"saved sanity check at: {filepath}", flush=True)

    return


def save_data(targets, path, meta_data=None, sanity_check_plots=False):

    n_envs = len(targets.data)
    assert len(targets.data) == len(targets.intv)

    # subfolder for each environment
    for n in range(n_envs):
        path_env = path / f"{n}"
        path_env.mkdir(exist_ok=True, parents=True)

        # save x
        save_csv(targets.data[n], path_env / FILE_X, dtype=onp.float32)

        # save intv
        save_csv(targets.intv[n][None], path_env / FILE_INTV, dtype=onp.int32)

        # if exist, save true parameters
        if targets.true_param is not None:
            save_json(targets.true_param[n], path_env / FILE_TRUE_PARAM)

        # if exist, save trajectory
        if targets.traj is not None:
            save_csv(targets.traj[n], path_env / FILE_TRAJ, dtype=onp.float32)

        # sanity check plots
        if sanity_check_plots:
            _save_data_checks(targets.data[n], targets.intv[n], n, path_env, ref_x=targets.data[0])

    # if exists, save meta data for later analysis
    if meta_data is not None:
        save_json(meta_data, path / FILE_META_DATA)

    return


def load_data(path, eval_mode=False):

    loaded_data = defaultdict(list)

    # sort paths to ensure that observational environment is first (in case there is one, it is always folder 0)
    paths = sorted(list(filter(lambda p: p.is_dir(), path.iterdir())),
                   key=lambda p: int(p.name.rsplit("-", 1)[0]))

    # collect all environments
    for path_env in paths:

        # load x
        data = load_csv(path_env / FILE_X, dtype=onp.float32)
        loaded_data["data"].append(data)

        # load intv
        intv = load_csv(path_env / FILE_INTV, dtype=onp.int32)
        loaded_data["intv"].append(intv.squeeze(0))

        # if exist, load true parameters
        if (path_true_param := path_env / FILE_TRUE_PARAM).is_file():
            true_param = load_json(path_true_param)
            loaded_data["true_param"].append(true_param)

        # if exist, load trajectory
        if (path_traj := path_env / FILE_TRAJ).is_file():
            traj = load_csv(path_traj, dtype=onp.float32)
            loaded_data["traj"].append(traj)

    # stack if all have same shape
    for k in loaded_data.keys():
        if all([x.shape == loaded_data[k][0].shape for x in loaded_data[k]]):
            loaded_data[k] = onp.stack(loaded_data[k])

    # make Data object
    targets = Data(**loaded_data)
    assert len(targets.data) == len(targets.intv)
    assert "true_param" not in loaded_data or len(targets.data) == len(targets.true_param)
    assert "traj" not in loaded_data or len(targets.data) == len(targets.traj)

    # eval_mode only returns intervention statistics
    intv_stats = get_intv_stats(targets)

    if eval_mode:
        return intv_stats
    else:
        return targets, *intv_stats


def load_meta_data(path):

    # if exists, load meta data for later analysis
    if (path_meta := path / FILE_META_DATA).is_file():
        meta_dict = load_json(path_meta)
        return meta_dict

    else:
        return dict()


def _ddicts_to_dicts(ddict):
    if type(ddict) == defaultdict:
        return {key: _ddicts_to_dicts(val) for key, val in ddict.items()}
    elif type(ddict) == list:
        return onp.array(ddict)
    else:
        return ddict