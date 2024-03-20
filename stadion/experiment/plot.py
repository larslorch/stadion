import warnings
warnings.filterwarnings("ignore", message="No GPU automatically detected")
warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"

import itertools
import matplotlib
import matplotlib.transforms
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf
from matplotlib.colors import ListedColormap
import matplotlib.font_manager # avoids loading error of fontfamily["serif"]
import matplotlib.ticker as ticker

import seaborn as sns
import numpy as onp
import scipy
import pandas as pd
from collections import defaultdict
from pprint import pprint
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable

import umap
from sklearn.decomposition import PCA

from pandas.plotting import autocorrelation_plot

from stadion.definitions import IS_CLUSTER
from stadion.experiment.plot_config import *
from stadion.utils.plot_data import scatter_matrix


mean_func = onp.nanmean
mean_func.__name__ = "__mean"

median_func = onp.nanmedian
median_func.__name__ = "__median"

sem_func = lambda a: scipy.stats.sem(a, nan_policy="omit", ddof=0)
sem_func.__name__ = "_sem"

mad_func = lambda a: scipy.stats.median_abs_deviation(a, nan_policy="omit")
mad_func.__name__ = "_mad"


def dict_to_ordered_list(d, key_order):
    return sorted(list(d.items()), key=lambda tup: ((key_order.index(tup[0]) if tup[0] in key_order else (len(key_order) + 1)), tup[0]))


def _add_best_markers(df, table, best_aggmetr, test_threshold=0.05, t_test=False, mark_only_best=False):

    for metr in df.metric.unique():
        lo = table.loc[:, (metr, best_aggmetr)] == table.loc[:, (metr, best_aggmetr)].min()
        hi = table.loc[:, (metr, best_aggmetr)] == table.loc[:, (metr, best_aggmetr)].max()

        assert (metr in METRICS_LOWER_BETTER) != (metr in METRICS_HIGHER_BETTER), f"{metr}"
        table[(metr, "best")] = lo if metr in METRICS_LOWER_BETTER else hi

        if not (table[(metr, "best")].sum() == 1 or metr in METRICS_CHECKS):
            warnings.warn(f'Best method not unique. Should only happen for sanity check metrics.' \
                          f'\nmetric: {metr}\n{table[(metr, "best")]}\nTable:\n{table}')

    table = table.sort_index(axis="columns", level="metric")

    # do overlap test or t-test (unequal variances, two-sided) with best method (per metric and per category)
    for metr in df.metric.unique():
        best_method = table.loc[table[(metr, "best")] == True].index.values
        if best_method.size:
            best_method = best_method[0]
        else:
            print(f"\nNo best method found.\n"
                  f"Metric: {metr}\n"
                  f"df:\n{df}\n"
                  f"table:\n{table}\n", flush=True)
            exit(1)

        best_values = df.loc[(df.method == best_method) & (df.metric == metr)].val.values

        if t_test:
            ttest_func = lambda arr: scipy.stats.ttest_ind(arr, best_values, equal_var=False, alternative="two-sided")[1]

            table[(metr, "_pval_best_")] = df.pivot_table(
                index=["method"],
                columns="metric",
                values="val",
                aggfunc=ttest_func,
                dropna=False) \
                .loc[:, metr]

        else:
            # MAD overlap
            # def overlap_func(arr):
            #     arr_median = onp.median(arr)
            #     arr_mad = scipy.stats.median_abs_deviation(arr)
            #     arr_lo = arr_median - arr_mad
            #     arr_hi = arr_median + arr_mad
            #
            #     best_median = onp.median(best_values)
            #     best_mad = scipy.stats.median_abs_deviation(best_values)
            #     best_lo = best_median - best_mad
            #     best_hi = best_median + best_mad
            #
            #     return best_lo < arr_hi and arr_lo < best_hi

            # IQR overlap
            def overlap_func(arr):
                arr_lo, arr_hi = onp.percentile(arr, [25, 75])
                best_lo, best_hi = onp.percentile(arr, [25, 75])
                return best_lo < arr_hi and arr_lo < best_hi


            table[(metr, "_has_overlap_")] = df.pivot_table(
                index=["method"],
                columns="metric",
                values="val",
                aggfunc=overlap_func,
                dropna=False)\
                .loc[:, metr]


    # add an indicator for methods to be marked
    for metr in df.metric.unique():
        table[(metr, "_marked_")] = onp.zeros_like(table.index).astype(bool)

        # mark best
        table.loc[(table[(metr, "best")] == True), (metr, "_marked_")] = True

        # mark for overlaps or p-val > test_threshold (inside (1-test_threshold)% confidence interval of t-distribution)
        if t_test:
            if not mark_only_best:
                table.loc[(table[(metr, "_pval_best_")] >= test_threshold), (metr, "_marked_")] = True

            # drop helpers
            table = table.drop([(metr, "best"), (metr, "_pval_best_")], axis=1, errors="ignore")

        else:
            if not mark_only_best:
                table.loc[table[(metr, "_has_overlap_")], (metr, "_marked_")] = True

            # drop helpers
            table = table.drop([(metr, "best"), (metr, "_has_overlap_")], axis=1, errors="ignore")

    table = table.sort_index(axis="columns", level="metric")

    return table


def _format_table_str(full_table, metr, metrname):

    # format str
    mean_str_formatter = (lambda val: f"{val:.3f}") if metr not in METRICS_LOWER_PREC else \
                         (lambda val: f"{val:.1f}")

    rounding = 3 if metr not in METRICS_LOWER_PREC else 1
    mean_col = full_table.loc[:, (metr, metrname[0])].round(rounding).apply(mean_str_formatter)
    full_table.loc[:, (metr, "str")] = mean_col
    full_table.drop(columns=(metr, metrname[0]), inplace=True)
    full_table.drop(columns=(metr, metrname[1]), inplace=True)

    # add marker str
    marked = "\\highlight{" + full_table.loc[:, (metr, "str")] + "}"
    is_marked = full_table.loc[:, (metr, "_marked_")]
    full_table.loc[is_marked, (metr, "str")] = marked.loc[is_marked]
    full_table.drop(columns=(metr, "_marked_"), inplace=True)
    return full_table


def benchmark_summary(kwargs, save_path, method_results_input, samples_all, params_all, median_mode,
                      only_metrics=None, without_metrics=None,
                      method_summaries=False, dump_main=False, skip_plot=False):

    assert not (dump_main and only_metrics is not None), "should only dump the full df"
    assert not (only_metrics is not None and without_metrics is not None)

    plot_path = save_path.parent / f"plots-{save_path.stem}"

    # preprocess results
    metric_results_dict = defaultdict(dict)
    for method, d in method_results_input.items():
        for metr, l in d.items():
            if only_metrics is not None and metr not in only_metrics:
                continue
            if without_metrics is not None and metr in without_metrics:
                continue
            metric_results_dict[metr][method] = l

    # impose metric ordering
    metric_results = dict_to_ordered_list(metric_results_dict, METRICS_TABLE_ORDERING)

    # print overview of metrics selected
    if only_metrics is None and without_metrics is None:
        metrics_descr = "<all metrics>"
    else:
        metrics_descr = [metr for metr, _ in metric_results]

    print("\n" + "=" * 80)
    print(f"{'dumping ' if dump_main else ''}benchmark_summary:  {metrics_descr}")
    print("\tonly_metrics:    ", only_metrics)
    print("\twithout_metrics: ", without_metrics)

    # create long list of (metric, method, value) tuples which are then aggregated into a df
    df = []
    for metric, res in metric_results:
        for m, l in res.items():
            for v in l:
                df.append((metric, m, v))
    df = pd.DataFrame(df, columns=["metric", "method", "val"])

    if df.empty:
        warn_str = f"\nNo results reported for metrics: {metrics_descr} "\
                   f"Methods: `{list(method_results_input.keys())}` \n"\
                   f"Skipping this benchmark_summary call.\n"
        print(warn_str, flush=True)
        return

    # dump summary table of metrics
    save_path.parent.mkdir(exist_ok=True, parents=True)
    if dump_main:
        informative_name = f"df-{'-'.join(save_path.parents[1].parts[-2:])}"
        df.to_csv(path_or_buf=(save_path.parent / informative_name).with_suffix(".csv"))
        print("dumped summary df")
        return

    # check whether we need to aggregate at all (not for real data, for which we only measure 1 metric score per method)
    is_singular = all([df.loc[(df.metric == metr) & (df.method == meth)].val.size == 1
        for metr, meth in itertools.product(df.metric.unique(), df.method.unique())])
    if is_singular:
        # print("*" * 30 + "\nWarning: singular measurements; ignore std dev, std err or t-tests.")
        df = pd.concat([df, df]).reset_index(drop=True)

    # create table
    if median_mode:
        table_agg_metrics = [median_func, mad_func]
    else:
        table_agg_metrics = [mean_func, sem_func]

    best_aggmetr = table_agg_metrics[0].__name__
    table = df.pivot_table(
        index=["method"],
        columns=["metric"],
        values="val",
        aggfunc=table_agg_metrics,
        dropna=False) \
        .swaplevel(-1, -2, axis=1)\
        .sort_index(axis="columns", level="metric")\
        .sort_index()

    # add marker for best
    table = _add_best_markers(df, table, best_aggmetr, t_test=not median_mode, mark_only_best=median_mode)

    # reorder and rename rows
    method_order = [method for method, *_ in METHODS_CONFIG]
    for method in sorted(table.index):
        if method not in method_order:
            method_order.append(method)
    all_methods_sorted = [method for method in method_order if method in table.index]
    full_table = table.reindex(all_methods_sorted)

    # convert full table to string in correct float format
    # mean (sem)
    # print(full_table)
    # print(full_table.columns.values)
    # print(full_table.columns.names)
    # print(full_table.columns)
    # print()
    # print(full_table.loc[:, ("f1", "mean")])
    # print(full_table.loc[:, ("f1", "mean")].round(3).apply(lambda val: f"{val:.3f}"))
    # print(full_table.loc[:, ("f1", "mean")].round(3).apply(lambda val: f"{val:.3f}")
    #       + full_table.loc[:, ("shd", "mean")].round(3).apply(lambda val: f" ({val:.3f})"))

    all_metrics = set(m for m, _ in full_table.columns.values)
    all_metrics_ordered = [metric for metric in METRICS_TABLE_ORDERING if metric in all_metrics]
    assert len(all_metrics_ordered) == len(set(all_metrics_ordered)), \
        "METRICS_TABLE_ORDERING not unique"

    for metr in all_metrics_ordered:
        full_table = _format_table_str(full_table, metr, [metr.__name__ for metr in table_agg_metrics])

    if method_summaries:
        # save individual table for each method
        save_path.mkdir(exist_ok=True, parents=True)
        base_methods = set(meth.split("__")[0] for meth in table.index)
        sort_crit = only_metrics[0]
        print("method_summaries sort criterium:", sort_crit)
        for base_method in base_methods:
            method_table = table[[base_method in s for s in table.index]].copy(deep=True)
            method_table = method_table.reindex(method_table[sort_crit].sort_values(by=best_aggmetr, ascending=True).index)

            # drop _marked_ column for method_summaries
            method_table = method_table.filter(regex='^(?!.*_marked_).*')

            # order table
            method_table = method_table[TRAIN_VALIDATION_METRICS]

            method_table.to_csv(path_or_buf=(save_path / base_method).with_suffix(".csv"))
        return

    else:
        full_table = full_table.droplevel(1, axis=1) # drop the dummy string level

    print(full_table)

    full_table.to_csv(path_or_buf=save_path.with_suffix(".csv"))
    # full_table.to_latex(buf=save_path.with_suffix(".tex"), escape=False)
    full_table.style.to_latex(buf=save_path.with_suffix(".tex"))

    """ Skip plotting for sanity check metrics"""

    if only_metrics is not None and only_metrics == METRICS_CHECKS:
        return


    """ ------------ Metric figures ------------ """
    print("\nStarting plots...", flush=True)

    metric_plot_config = [("", False), ("_log", True)]
    # metric_plot_config = [("", False)]
    for log_suffix, log_scale in metric_plot_config:

        # set plot configuration
        # sns.set_theme(style="ticks", rc=NEURIPS_RCPARAMS)
        fig, axs = plt.subplots(1, len(metric_results), figsize=(FIGSIZE["metric"]["ax_width"] * len(metric_results),
                                                                 FIGSIZE["metric"]["ax_height"]))
        if len(metric_results) == 1:
            axs = [axs]

        handles, labels = [], []

        for ii, (metric, _) in enumerate(metric_results):
            df_metric = df.loc[df["metric"] == metric].drop("metric", axis=1)

            # add placeholder if method not measured for this metric
            for m in method_results_input.keys():
                if m not in df_metric["method"].unique():
                    df_metric = pd.concat([df_metric, pd.DataFrame({"method": [m], "val": [onp.nan]})], ignore_index=True)

            df_metric = df_metric.reset_index(drop=True)

            # formatting of methods
            unique = df_metric["method"].unique()
            config = []
            seen = set()
            for k, *tup in METHODS_CONFIG:
                matches = list(filter(lambda w: k == w.split("__")[0], unique))
                if matches:
                    assert not any([w in seen for w in matches]), f"matches `{matches}` already seen"
                    seen.update(matches)
                    for w in matches:
                        config.append((w, tup[0]))

            for w in filter(lambda w: w not in seen, unique):
                # unknown method substring
                config.append((w, METHODS_DEFAULT_CONFIG[0]))

            # plot
            plot_kwargs = dict(
                ax=axs[ii],
                x="method",
                y="val",
                hue="method",
                dodge=False,  # violin positions do not change position or width according to hue
                data=df_metric,
                order=list(w for w, *_ in config),
                hue_order=list(w for w, *_ in config),
                palette=dict(config),
                saturation=COLOR_SATURATION,
                showfliers=False,
            )

            # sns_plot = sns.violinplot(**plot_kwargs)
            # sns_plot = sns.boxenplot(**plot_kwargs)
            sns_plot = sns.boxplot(**plot_kwargs)

            if log_scale:
                sns_plot.set_yscale("log")

                # formatting
                formatter = ticker.ScalarFormatter(useMathText=True)
                formatter.set_scientific(True)
                # formatter.set_powerlimits((-1, 1))
                formatter.set_powerlimits((0,0))
                axs[ii].yaxis.set_major_formatter(formatter)

            else:
                # formatting
                axs[ii].yaxis.grid()
                axs[ii].yaxis.set_major_locator(plt.MaxNLocator(8))


            axs[ii].set_xticklabels([])
            axs[ii].set_xlabel("")
            axs[ii].set_ylabel("")
            axs[ii].set_title(metric)

            # def format_tick(value, tick_number, decimal=BENCHMARK_Y_DECIMALS):
            #     return ("{:." + str(decimal) + "f}").format(round(value.item(), decimal))
            #
            #
            # for ax in onp.asarray(axs).ravel():
            #     ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_tick))
            #     # if xlabelsize is not None:
            #     #     plt.setp(ax.get_xticklabels(), fontsize=xlabelsize)
            #     # if xrot is not None:
            #     #     plt.setp(ax.get_xticklabels(), rotation=xrot)
            #     # if ylabelsize is not None:
            #     #     plt.setp(ax.get_yticklabels(), fontsize=ylabelsize)
            #     # if yrot is not None:
            #     #     plt.setp(ax.get_yticklabels(), rotation=yrot)

            # legend
            axs[ii].legend([], [], frameon=False)
            handles_, labels_ = axs[ii].get_legend_handles_labels()
            for h_, l_ in zip(handles_, labels_):
                if l_ not in labels:
                    handles.append(h_)
                    labels.append(l_)

        # multi figure legend
        axs[-1].legend(handles, labels, loc='upper right', bbox_to_anchor=(1.07, -0.15), borderaxespad=1.0)

        # tight layout
        fig.tight_layout()

        filepath = save_path.parent / f"{save_path}{log_suffix}"

        plt.savefig(filepath.with_suffix(".pdf"), format="pdf", facecolor=None,
                    dpi=DPI, bbox_inches='tight')
        plt.close()


    """ ------------ param vs true ------------ """
    if skip_plot:
        return

    # get observational reference dists
    reference_data = {}
    for task_id, (true_samples, intv) in samples_all["train"][TRUE].items():
        assert onp.allclose(intv[0], 0)
        reference_data[task_id] = true_samples[0]

    d = next(iter(reference_data.values())).shape[-1]
    var_names = [("$x_{" + str(jj) + "}$") for jj in range(1, d + 1)]

    # init folder for the plot
    plot_path.mkdir(exist_ok=True, parents=True)


    # for each method
    for ctr, (task_id, true_param) in enumerate(params_all[TRUE].items()):
        if ctr >= kwargs.max_data_plots_test:
            break
        for method, method_dict in params_all.items():
            if task_id not in method_dict:
                continue

            pred_param = method_dict[task_id]

            fig, ax = plt.subplots(1, 1, figsize=(FIGSIZE["param"]["ax_width"],
                                                  FIGSIZE["param"]["ax_height"]))

            cbar_pad = 0.1
            vmin, vmax = pred_param.min() - cbar_pad, pred_param.max() + cbar_pad
            vmin = min(-vmax, vmin)
            vmax = max(-vmin, vmax)

            # assert that parameters are matrix with [i, j] measuring causal effect from variable i to j
            assert pred_param.ndim == 2
            matim = ax.matshow(pred_param, vmin=vmin, vmax=vmax, cmap=CMAP_PARAM)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.15)
            plt.colorbar(matim, cax=cax)

            ax.set_title(METHODS_NAMES.get(method, method))
            # axes[1].set_title("Predicted")

            ax.tick_params(axis='both', which='both', length=0)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            # ax.axis('off')
            ax.grid(False)

            # fig.suptitle(f"{param_name} true $t={t_current}$")
            plt.tight_layout()

            # colorbar
            # fig.subplots_adjust(right=0.8)
            # divider = make_axes_locatable(ax)
            # cbar_ax = divider.append_axes("right", size="5%", pad=0.15)
            # fig.colorbar(matim, cax=cbar_ax)

            filepath = plot_path / f"param_{task_id}_{method}"

            if PLOT_SHOW and not IS_CLUSTER:
                plt.show()
            else:
                plt.savefig(filepath.with_suffix(".pdf"), format="pdf", facecolor=None,
                            dpi=DPI, bbox_inches='tight')

                plt.close()

    """ ------------ grid plots ------------ """

    # train or test
    for train_or_test, samples_train_or_test in samples_all.items():

        # for each method
        for ctr, (task_id, true_samples_intv) in enumerate(samples_train_or_test[TRUE].items()):
            if ctr >= (kwargs.max_data_plots_train if train_or_test == "train" else kwargs.max_data_plots_test):
                break
            for method, method_dict in samples_train_or_test.items():
                if method == TRUE or task_id not in method_dict:
                    continue

                pred_samples_intv = method_dict[task_id]

                # iterate over each env in the dataset
                for ii, (pred_samples, true_samples, intv) in enumerate(zip(pred_samples_intv[0], true_samples_intv[0], true_samples_intv[1])):
                    if ii >= kwargs.max_data_plots_env:
                        break
                    if onp.isnan(pred_samples).all(0).any():
                        # at least on dimension only has nan, skip scatter plot
                        continue

                    # visualize in scatter matrix
                    data = pd.DataFrame(pred_samples, columns=var_names)
                    data_target = pd.DataFrame(true_samples, columns=var_names)

                    # color mask
                    color_mask = onp.zeros((len(data.columns), len(data.columns)), dtype=onp.int32)
                    for intv_idx in onp.where(intv)[0]:
                        color_mask[:, intv_idx] = 1
                        color_mask[intv_idx, :] = 1

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
                        ref_data=reference_data[task_id],
                        diagonal=PAIRWISE_GRID_TYPE[0],
                        scotts_bw_scaling=SCOTTS_BW_SCALING,
                        offdiagonal=PAIRWISE_GRID_TYPE[1],
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
                        break

                    fig, axes = figax

                    # format title
                    intv_vals = true_samples.mean(-2)
                    intv_vals_std = true_samples.std(-2)
                    title = ""
                    if onp.where(intv)[0].size:
                        title += "interv"
                        for j in onp.where(intv)[0]:
                            if onp.allclose(intv_vals_std[j], 0.0):
                                title += f" {var_names[j]}$=${intv_vals[j].item():.1f}"
                            else:
                                # title += f" {var_names[j]}$\\approx${intv_vals[j].item():.1f} ($\\pm${intv_vals_std[j].item():.1f})"
                                title += f" {var_names[j]}$\\approx${intv_vals[j].item():.1f}"
                    else:
                        title += "observ"

                    fig.suptitle(title)
                    plt.tight_layout()
                    plt.gcf().subplots_adjust(wspace=0, hspace=0)

                    filepath = plot_path / f"grid_{train_or_test}_{task_id}-{ii}_{method}"

                    if PLOT_SHOW and not IS_CLUSTER:
                        plt.show()
                    else:
                        plt.savefig(filepath.with_suffix(".pdf"), format="pdf", facecolor=None,
                                    dpi=DPI, bbox_inches='tight')

                        plt.close()


    """ ------------ projection plots ------------ """
    projection = False
    if projection:
        # train or test
        for train_or_test, samples_train_or_test in samples_all.items():

            # for each method
            for ctr, (task_id, true_samples_intv) in enumerate(samples_train_or_test[TRUE].items()):
                if ctr >= (kwargs.max_data_plots_train if train_or_test == "train" else kwargs.max_data_plots_test):
                    break

                # compute projections based on the whole true data
                all_target_data = onp.concatenate(true_samples_intv[0], axis=0)
                assert all_target_data.ndim == 2

                # proj_umap = umap.UMAP(
                #     n_components=2,
                #     random_state=0,
                #     n_neighbors=15,
                #     min_dist=0.1,
                #     # n_neighbors=100,
                #     # min_dist=0.9,
                # )
                # _ = proj_umap.fit_transform(all_target_data)

                # pca
                proj_pca = PCA(n_components=2, random_state=0)
                _ = proj_pca.fit(all_target_data)

                # proj_list = [("umap", proj_umap), ("pca", proj_pca)]
                proj_list = [("pca", proj_pca)]

                for method, method_dict in samples_train_or_test.items():
                    if method == TRUE or task_id not in method_dict:
                        continue

                    pred_samples_intv = method_dict[task_id]

                    # iterate over each env in the dataset
                    for ii, (pred_samples, true_samples, intv) in enumerate(
                            zip(pred_samples_intv[0], true_samples_intv[0], true_samples_intv[1])):
                        if ii >= kwargs.max_data_plots_env:
                            break
                        if onp.isnan(pred_samples).all(0).any():
                            # at least on dimension only has nan, skip scatter plot
                            continue

                        # filter nan and inf
                        target_mask = ~onp.isnan(true_samples) & (true_samples < MAXMAX) & (true_samples > MINMIN)
                        s_mask = ~onp.isnan(pred_samples) & (pred_samples < MAXMAX) & (pred_samples > MINMIN)

                        true_samples = true_samples[onp.where(target_mask.all(-1))[0]]
                        pred_samples = pred_samples[onp.where(s_mask.all(-1))[0]]

                        for name, projection in proj_list:

                            fig, ax = plt.subplots(1, 1, figsize=(FIGSIZE["lowdim"]["ax_width"], FIGSIZE["lowdim"]["ax_height"]))

                            target_proj = projection.transform(true_samples)
                            s_proj = projection.transform(pred_samples)
                            ax.scatter(target_proj[..., 0], target_proj[..., 1], c=CTARGET, **SCAT_KWARGS_PROJ)
                            ax.scatter(s_proj[..., 0], s_proj[..., 1], c=CMAIN, **SCAT_KWARGS_PROJ)

                            # format title
                            intv_vals = true_samples.mean(-2)
                            intv_vals_std = true_samples.std(-2)
                            title = ""
                            if onp.where(intv)[0].size:
                                title += "interv"
                                for j in onp.where(intv)[0]:
                                    if onp.allclose(intv_vals_std[j], 0.0):
                                        title += f" {var_names[j]}$=${intv_vals[j].item():.1f}"
                                    else:
                                        # title += f" {var_names[j]}$\\approx${intv_vals[j].item():.1f} ($\\pm${intv_vals_std[j].item():.1f})"
                                        title += f" {var_names[j]}$\\approx${intv_vals[j].item():.1f}"
                            else:
                                title += "observ"

                            # remove gridlines
                            ax.tick_params(
                                axis='both',
                                which='both',
                                bottom=False,
                                left=False,
                                right=False,
                                top=False,
                                labelbottom=False,
                                labelleft=False,
                            )
                            ax.grid(visible=False, axis='both')

                            plt.gcf().subplots_adjust(wspace=0, hspace=0)
                            plt.draw()
                            plt.tight_layout()

                            filepath = plot_path / f"proj_{name}_{train_or_test}_{task_id}-{ii}_{method}"

                            if PLOT_SHOW and not IS_CLUSTER:
                                plt.show()
                            else:
                                plt.savefig(filepath.with_suffix(".pdf"), format="pdf", facecolor=None,
                                            dpi=DPI, bbox_inches='tight')

                                plt.close()


    print("\nFinished plots.\n", flush=True)
    return








