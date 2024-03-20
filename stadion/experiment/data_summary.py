import warnings
warnings.filterwarnings("ignore", message="No GPU automatically detected")
warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"

import argparse
from pathlib import Path
import shutil
import math

import numpy as onp
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from collections import defaultdict

from stadion.utils.parse import load_data, load_meta_data, dict_tree_to_ordered

from stadion.utils.plot_data import scatter_matrix
from stadion.experiment.plot_config import *
from stadion.utils.metrics import sortability

from stadion.definitions import FOLDER_TRAIN, IS_CLUSTER, FILE_META_DATA
from pprint import pprint

N_OBS = 99999 if IS_CLUSTER else 300
N_SEEDS = 5 if IS_CLUSTER else 1
N_ENVS = 3 if IS_CLUSTER else 3
N_ENVS_TRAJ = 3 if IS_CLUSTER else 3


q0 = lambda arr, **kwargs: onp.min(arr, **kwargs)
q0.__name__ = "q_min"

q10 = lambda arr, **kwargs: onp.quantile(arr, 0.10, **kwargs)
q10.__name__ = "q10"

q25 = lambda arr, **kwargs: onp.quantile(arr, 0.25, **kwargs)
q25.__name__ = "q25"

q50 = lambda arr, **kwargs: onp.quantile(arr, 0.50, **kwargs)
q50.__name__ = "q50" # median -- leave as q50 for sorting columns

fmedian = lambda arr, **kwargs: onp.quantile(arr, 0.50, **kwargs)
fmedian.__name__ = "median"

fmean = lambda arr, **kwargs: onp.mean(arr, **kwargs)
fmean.__name__ = "mean"

q75 = lambda arr, **kwargs: onp.quantile(arr, 0.75, **kwargs)
q75.__name__ = "q75"

q90 = lambda arr, **kwargs: onp.quantile(arr, 0.90, **kwargs)
q90.__name__ = "q90"

q95 = lambda arr, **kwargs: onp.quantile(arr, 0.95, **kwargs)
q95.__name__ = "q95"

q100 = lambda arr, **kwargs: onp.max(arr, **kwargs)
q100.__name__ = "q_max"

fmax = lambda arr, **kwargs: onp.max(arr, **kwargs)
fmax.__name__ = "max"


def autocorrelation_plot(series, ax, xlabel=None, maxlags=None, lagstep=None, **kwargs):
    """
    Adapted from
    pandas/pandas/plotting/_matplotlib/misc.autocorrelation_plot
    """

    n = len(series)
    data = onp.asarray(series)
    if ax is None:
        ax = plt.gca()
        ax.set_xlim(1, n)
        ax.set_ylim(-1.0, 1.0)
    mean = onp.mean(data)
    c0 = onp.sum((data - mean) ** 2) / n

    def r(h):
        return ((data[: n - h] - mean) * (data[h:] - mean)).sum() / n / c0

    x = onp.arange(0, min(n, maxlags) if maxlags is not None else n, lagstep or 1) + 1
    y = [r(loc) for loc in x]
    z95 = 1.959963984540054
    z99 = 2.5758293035489004
    ax.axhline(y=z99 / onp.sqrt(n), linestyle="--", color="grey")
    ax.axhline(y=z95 / onp.sqrt(n), color="grey")
    ax.axhline(y=0.0, color="black")
    ax.axhline(y=-z95 / onp.sqrt(n), color="grey")
    ax.axhline(y=-z99 / onp.sqrt(n), linestyle="--", color="grey")
    ax.set_xlabel(xlabel or "Lag")
    ax.set_ylabel("Autocorrelation")
    ax.plot(x, y, **kwargs)
    if "label" in kwargs:
        ax.legend()
    ax.grid()
    return ax


def grid_plot(path, data):

    # for each data setting
    for descr_nr, (descr, data_dict) in enumerate(data.items()):

        # order seeds by nans first (for debugging purposes)
        if list(data_dict.values())[0].traj is not None:
            data_dict_ordered = sorted(data_dict.items(), key=lambda kv: (-onp.isnan(kv[1].traj).sum(), -onp.abs(kv[1].traj).max()))
        else:
            data_dict_ordered = sorted(data_dict.items())

        for ctr, (seed, target) in enumerate(data_dict_ordered):
            if ctr >= N_SEEDS:
                continue

            # iterate over each env in the dataset
            x_obs = target.data[0]
            d = x_obs.shape[-1]
            for ii, (x_intv, intv) in enumerate(zip(target.data[1:], target.intv[1:])):
                if ii >= N_ENVS:
                    break

                assert x_obs.ndim == x_intv.ndim == 2

                # visualize in scatter matrix
                data_obs = pd.DataFrame(x_obs[:N_OBS], columns=list(range(d)))
                data_int = pd.DataFrame(x_intv[:N_OBS], columns=list(range(d)))

                # color mask
                color_mask = onp.zeros((len(data_obs.columns), len(data_obs.columns)), dtype=onp.int32)
                for intv_idx in onp.where(intv)[0]:
                    color_mask[:, intv_idx] = 1
                    color_mask[intv_idx, :] = 1

                # grid_type = PAIRWISE_GRID_TYPE
                grid_type = ("hist", "scatter")

                figax = scatter_matrix(
                    data_int,
                    data_obs,
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
                    ref_data=x_obs[:N_OBS],
                    diagonal=grid_type[0],
                    scotts_bw_scaling=SCOTTS_BW_SCALING,
                    offdiagonal=grid_type[1],
                    # offdiagonal="scatter",
                    scat_kwds=SCAT_KWARGS,
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
                    plt.close()
                    assert False
                    # break

                fig, axes = figax

                subfolder = path / "grid"
                subfolder.mkdir(exist_ok=True, parents=True)
                # filepath = subfolder / f"{descr_nr}-{descr}-seed={seed}-env={ii}.pdf"
                filepath = subfolder / f"{descr}-seed={seed}-env={ii}.pdf"

                fig.suptitle(filepath.name)
                plt.tight_layout()
                plt.gcf().subplots_adjust(wspace=0, hspace=0)

                plt.savefig(filepath, format="pdf", facecolor=None,
                            dpi=DPI, bbox_inches='tight')
                plt.close('all')

    return


def traj_plot(path, data, cols=3, x_size=3., y_size=1.5, maxlags=1000, lagstep=1):

    # for each data setting
    for descr_nr, (descr, data_dict) in enumerate(data.items()):

        if list(data_dict.values())[0].traj is None:
            continue

        # order seeds by nans first and max deviation second (for debugging purposes)
        data_dict_ordered = sorted(data_dict.items(), key=lambda kv: (-onp.isnan(kv[1].traj).sum(), -onp.abs(kv[1].traj).max()))

        for ctr, (seed, target) in enumerate(data_dict_ordered):
            if ctr >= N_SEEDS:
                continue

            # iterate over each env in the dataset
            for ii, (traj, intv) in enumerate(zip(target.traj, target.intv)):
                if ii >= N_ENVS_TRAJ:
                    break

                assert traj.ndim == 2
                time = onp.arange(traj.shape[0])
                d = traj.shape[-1]
                n_rows = ((d - 1) // cols) + 1
                n_cols = min(d, cols)

                for mode in ["traj", "autocorr"]:

                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(x_size * n_cols, y_size * n_rows))
                    if d == 1:
                        axes = [[axes]]
                    elif n_rows == 1:
                        axes = [axes]

                    for k in range(d):
                        i, j = (k // n_cols), (k % n_cols)
                        tr = traj[..., k]

                        if mode == "traj":
                            if onp.any(onp.abs(tr) > 1e18):
                                axes[i][j].plot(time, onp.abs(tr))
                                axes[i][j].set_yscale('log')
                            else:
                                axes[i][j].plot(time, tr)

                        elif mode == "autocorr":
                            autocorrelation_plot(tr, axes[i][j], maxlags=maxlags, lagstep=lagstep, xlabel=r"thinning $\times$ lag")

                        else:
                            raise KeyError(mode)

                        axes[i][j].set_title(f"x{k}")
                        is_left, is_bottom = j == 0, i == n_rows - 1
                        if not is_left:
                            axes[i][j].set_ylabel("")
                        if not is_bottom:
                            axes[i][j].set_xlabel("")
                        axes[i][j].tick_params(
                            axis='both',
                            which='both',
                            bottom=False,
                            left=False,
                            right=False,
                            top=False,
                            labelbottom=is_bottom,
                            labelleft=is_left,
                        )
                    for kk in range(d, n_cols * n_rows):
                        i, j = (kk // n_cols), (kk % n_cols)
                        fig.delaxes(axes[i][j])

                    plt.gcf().subplots_adjust(wspace=0, hspace=0)
                    plt.draw()
                    plt.tight_layout()

                    subfolder = path / mode
                    subfolder.mkdir(exist_ok=True, parents=True)
                    filepath = subfolder / f"{descr_nr}-{descr}-seed={seed}-env={ii}.pdf"

                    fig.suptitle(filepath.name)
                    plt.tight_layout()
                    plt.gcf().subplots_adjust(wspace=0, hspace=0)

                    plt.savefig(filepath, format="pdf", facecolor=None, dpi=DPI, bbox_inches='tight')
                    plt.close('all')

    return



def sim_error_plot(path, meta_data, cols=3, x_size=3., y_size=1.5, maxlags=1000, lagstep=1):

    # for each data setting
    for descr_nr, (descr, meta_dict) in enumerate(meta_data.items()):

        meta_keys = meta_dict[list(meta_dict.keys())[0]].keys()
        if not ("sde_sim_error" in meta_keys and "sde_sim_inv_nan" in meta_keys):
            continue

        # order seeds by nans first and max deviation second (for debugging purposes)
        meta_dict_ordered = sorted(meta_dict.items(), key=lambda kv: (-onp.isnan(kv[1]["sde_sim_inv_nan"]).sum(),
                                                                      -onp.abs(kv[1]["sde_sim_error"]).max()))

        for ctr, (seed, meta) in enumerate(meta_dict_ordered):
            if ctr >= N_SEEDS:
                continue

            sim_errors = meta["sde_sim_error"]
            inv_nans = meta["sde_sim_inv_nan"]

            assert sim_errors.shape == inv_nans.shape

            assert inv_nans.ndim == 2
            time = onp.arange(sim_errors.shape[-1])
            n_envs = len(sim_errors)
            n_rows = ((n_envs - 1) // cols) + 1
            n_cols = min(n_envs, cols)

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(x_size * n_cols, y_size * n_rows))
            if n_envs == 1:
                axes = [[axes]]
            elif n_rows == 1:
                axes = [axes]

            for k in range(len(sim_errors)):
                i, j = (k // n_cols), (k % n_cols)
                sim_error = sim_errors[k]
                inv_nan = inv_nans[k]

                axes[i][j].plot(time, sim_error)
                axes[i][j].set_yscale('log')

                # reference line
                axes[i][j].plot(time, onp.ones_like(sim_error) * 1e-6, color="black")
                axes[i][j].plot(time, onp.ones_like(sim_error) * 1e-12, color="black", linestyle="dashed")
                axes[i][j].set_ylim((1e-24, 1e-1))

                # nan counter
                ax2 = axes[i][j].twinx()  # instantiate a second axes that shares the same x-axis
                ax2.plot(time, onp.cumsum(inv_nan), color="red")
                ax2.set_ylim((0, math.ceil(onp.cumsum(inv_nan).max() + 1)))
                ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

                axes[i][j].set_title(f"env {k}")
                is_left, is_bottom = j == 0, i == n_rows - 1
                if not is_left:
                    axes[i][j].set_ylabel("")
                if not is_bottom:
                    axes[i][j].set_xlabel("")
                axes[i][j].tick_params(
                    axis='both',
                    which='both',
                    bottom=False,
                    left=False,
                    right=False,
                    top=False,
                    labelbottom=is_bottom,
                    labelleft=is_left,
                )
            for kk in range(n_envs, n_cols * n_rows):
                i, j = (kk // n_cols), (kk % n_cols)
                fig.delaxes(axes[i][j])

            plt.gcf().subplots_adjust(wspace=0, hspace=0)
            plt.draw()
            plt.tight_layout()

            subfolder = path / "sim_error"
            subfolder.mkdir(exist_ok=True, parents=True)
            filepath = subfolder / f"{descr_nr}-{descr}-seed={seed}.pdf"

            fig.suptitle(filepath.name)
            plt.tight_layout()
            plt.gcf().subplots_adjust(wspace=0, hspace=0)

            plt.savefig(filepath, format="pdf", facecolor=None, dpi=DPI, bbox_inches='tight')
            plt.close('all')

    return


def _append(lst, descr, setting, metric, array):
    try:
        array = array.flatten()
    except AttributeError:
        array = onp.array([array])

    for val in array:
        lst.append((descr, setting, metric, val))


def store_data_summary(path, data, meta_data):

    path.mkdir(exist_ok=True, parents=True)

    # compute metrics
    # store as list of (descr, metric, value) for all seeds
    df = []

    for descr in data.keys():
        for seed in data[descr].keys():
            target = data[descr][seed]
            meta = meta_data[descr][seed]

            # preproc
            d = target.data.shape[-1]
            x_obs = target.data[0]
            x_int = target.data[1:]

            abs_corr_obs = onp.abs(onp.array([onp.corrcoef(arr) for arr in x_obs]) if x_obs.ndim == 3 else onp.corrcoef(x_obs))
            abs_corr_int = onp.abs(onp.array([onp.corrcoef(arr) for arr in x_int]) if x_int.ndim == 3 else onp.corrcoef(x_int))

            mean_obs = onp.abs(meta["mean"][0])
            mean_int = onp.abs(meta["mean"][1:])
            std_obs = meta["std"][0]
            std_int = meta["std"][1:]

            sort_stats_obs = sortability(x_obs, target.true_param[0].T)
            sort_stats_int = [sortability(x_int_j, target.true_param[0].T) for x_int_j in x_int]

            # distribution statistics
            _append(df, descr, "mean_max", "observ", q100(onp.abs(x_obs.mean(-2)), axis=-1))
            _append(df, descr, "mean_max", "interv", q100(onp.abs(x_int.mean(-2)), axis=-1))

            _append(df, descr, "nan_perc", "observ", onp.any(onp.isnan(x_obs), axis=-2).mean(-1))
            _append(df, descr, "nan_perc", "interv", onp.any(onp.isnan(x_int), axis=-2).mean(-1))


            _append(df, descr, "raw_mean_max", "observ", q100(mean_obs, axis=-1))
            _append(df, descr, "raw_mean_max", "interv", q100(mean_int, axis=-1))

            _append(df, descr, "raw_std_min", "observ", q0(std_obs, axis=-1))
            _append(df, descr, "raw_std_min", "interv", q0(std_int, axis=-1))
            _append(df, descr, "raw_std_max", "observ", q100(std_obs, axis=-1))
            _append(df, descr, "raw_std_max", "interv", q100(std_int, axis=-1))

            # correlation statistics
            i1, i2 = onp.triu_indices(d, k=1)
            _append(df, descr, "corr_q10", "observ", q10(abs_corr_obs[..., i1, i2], axis=-1))
            _append(df, descr, "corr_q50", "observ", q50(abs_corr_obs[..., i1, i2], axis=-1))
            _append(df, descr, "corr_q75", "observ", q75(abs_corr_obs[..., i1, i2], axis=-1))
            _append(df, descr, "corr_q90", "observ", q90(abs_corr_obs[..., i1, i2], axis=-1))
            _append(df, descr, "corr_max", "observ", q100(abs_corr_obs[..., i1, i2], axis=-1))

            # sortability statistics
            for key in sort_stats_obs.keys():
                _append(df, descr, key, "observ", sort_stats_obs[key])
                _append(df, descr, key, "interv", onp.array([s[key] for s in sort_stats_int]))

            # simulation statistics
            if target.traj is not None:
                autocorr_n = target.traj.shape[-2]
                autocorr_mean = onp.mean(target.traj, axis=-2, keepdims=True)
                autocorr_c0 = onp.mean((target.traj - autocorr_mean) ** 2, axis=-2, keepdims=True)
                step_autocorr = ((target.traj[..., :autocorr_n - 1, :] - autocorr_mean) *
                                 (target.traj[..., 1:, :] - autocorr_mean)).sum(-2) / autocorr_n/ autocorr_c0.squeeze(-2)
                abs_step_autocorr = onp.abs(step_autocorr)
                _append(df, descr, "autocorr_max", "observ", q100(abs_step_autocorr[0], axis=-1))
                _append(df, descr, "autocorr_max", "interv", q100(abs_step_autocorr[1:], axis=-1))

            # parameter statistics
            if "mat_eigenvals" in meta:
                _append(df, descr, "mat_eigenvals", "observ", meta["mat_eigenvals"])

            if "walltime" in meta:
                _append(df, descr, "walltime", "observ", meta["walltime"])


    # aggregate as pivot table
    df = pd.DataFrame(df, columns=["descr", "data", "metric", "val"])
    df.to_csv(float_format="%.3f", path_or_buf=(path / "df").with_suffix(".csv"))

    # table_agg_metrics = [fmean, fmax]
    table_agg_metrics = [fmean]

    table = df.pivot_table(
        index=["descr"],
        columns=["metric", "data"],
        values="val",
        aggfunc=table_agg_metrics,
        dropna=False)\
        .swaplevel(0, -1, axis=1)

    table = table.sort_index(axis="columns", level=[0, 1, 2], ascending=[True, False, True])

    table.to_csv(float_format="%.3f", path_or_buf=(path / "table").with_suffix(".csv"))

    # plots
    sim_error_plot(path, meta_data)
    traj_plot(path, data)
    grid_plot(path, data)

    return


if __name__ == "__main__":
    """
    Runs plot call
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_data", type=Path, required=True)
    parser.add_argument("--path_plots", type=Path, required=True)
    parser.add_argument("--descr")
    kkwargs = parser.parse_args()

    # init folder
    onp.set_printoptions(precision=4, suppress=True)
    kkwargs.path_plots.mkdir(exist_ok=True, parents=True)

    # load data
    paths = sorted([p for p in kkwargs.path_data.iterdir() if p.is_dir()])
    print(f"Found {len(paths)} data folders.", flush=True)

    dat = defaultdict(dict)
    meta_dat = defaultdict(dict)
    for p in paths:
        desc, sd = p.name.rsplit("-", 1)
        dat[desc][int(sd)] = load_data(p / FOLDER_TRAIN)[0]
        meta_dat[desc][int(sd)] = load_meta_data(p / FOLDER_TRAIN)

    dat = dict_tree_to_ordered(dat)
    meta_dat = dict_tree_to_ordered(meta_dat)
    print("Loaded.", flush=True)

    # copy meta info to summary directory
    for p in paths:
        if (path_meta := p / FOLDER_TRAIN / FILE_META_DATA).is_file():
            name, suff = FILE_META_DATA.rsplit(".")
            dest = kkwargs.path_plots / name
            dest.mkdir(exist_ok=True, parents=True)
            shutil.copy(path_meta, dest / (p.name +  "." + suff))

    # make summary
    store_data_summary(kkwargs.path_plots, dat, meta_dat)
    print("Done.", flush=True)

