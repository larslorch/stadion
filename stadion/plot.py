import math

import wandb

import numpy as onp
import pandas as pd
from jax import numpy as jnp
from matplotlib import pyplot as plt

from stadion.utils.plot_data import scatter_matrix, get_2d_percentile_contours
from pandas.plotting import autocorrelation_plot
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot(samples, sampled_traj_full, batched_tars,
         title_prefix=None, theta=None, intv_theta=None, true_param=None,
         x_size=14., y_size=3.0,
         binwidth=0.25,
         t_current=None, title=None,
         max_target_samples=1000, margin=1,
         max_cols=20, max_rows=20,
         max_scatter=500,
         n_bins_default=20,
         size_per_var=1.25,
         max_proj_cols=6,
         grid_type="both",
         contours=(0.68, 0.95),
         contours_alphas=(0.33, 1.0),
         scotts_bw_scaling=1.0,
         minmin=-1000,
         maxmax=1000,
         cbar_pad=0.1,
         plot_max=None,
         proj=None,
         cmain="black",
         cfit="blue",
         cref="black",
         ref_data=None,
         ref_alpha=0.9,
         plot_mat=True,
         plot_params=True,
         plot_pairwise_grid=False,
         plot_paths=False,
         plot_intv_marginals=False,
         plot_acorr=True,
         to_wandb=True,
         ):

    scat_kwds = dict(
        marker=".",
        alpha=0.15,
        s=40,
    )
    scat_kwds_target = dict(
        marker=".",
        alpha=0.20,
        s=40,
    )

    hist_kwds = dict(
        alpha=0.5,
        histtype="stepfilled",
        ec=cfit,
    )
    hist_kwds_target = dict(
        alpha=0.6,
        histtype="stepfilled",
        ec=cmain,
    )

    density1d_kwds = dict(linestyle="dashed")
    # density1d_kwds = dict(linestyle="dotted")
    density1d_kwds_target = dict(linestyle="solid")

    density2d_kwds = dict(linestyles="dashed")
    # density2d_kwds = dict(linestyles="dotted")
    density2d_kwds_target = dict(linestyles="solid")

    # preprocess data
    samples = onp.array(samples)
    assert samples.ndim == 3
    d = samples.shape[-1]

    def get_bins(arr):
        # return onp.arange(max(minmin, arr.min()), min(maxmax, arr.max()) + binwidth, binwidth)
        return onp.arange(minmin, maxmax + binwidth, binwidth)

    wandb_images = {}

    var_names = [r"$x_{" + f"{j}" + r"}$" for j in range(d)]

    cmap = "seismic"
    # cmap = "bwr"

    # parameter plot
    if plot_mat or plot_params:

        # assemble available params
        params = []
        if theta is not None and plot_mat:
            for k, v in theta.items():
                params.append((f"{k}", v, True))

        if intv_theta is not None and plot_params:
            for k, v in intv_theta.items():
                params.append((f"intv-{k}", v, False))

        # plot each individually
        for param_name, arr_raw, is_model in params:
            if "scale" in param_name or "c" in param_name:
                arr = jnp.exp(arr_raw)
            else:
                arr = arr_raw

            ##### special case: also plot the true mat
            if plot_mat and param_name == "w1" and true_param is not None and true_param[0].shape == arr.shape:

                fig, axes = plt.subplots(1, 2, figsize=(2 * y_size, y_size))
                plot_true_param = true_param[0]

                vmin, vmax = arr.min() - cbar_pad, arr.max() + cbar_pad
                vmin = min(-vmax, vmin)
                vmax = max(-vmin, vmax)
                matim_arr = axes[0].matshow(arr, vmin=vmin, vmax=vmax, cmap=cmap)

                divider = make_axes_locatable(axes[0])
                cax = divider.append_axes("right", size="5%", pad=0.15)
                plt.colorbar(matim_arr, cax=cax)

                vmin, vmax = plot_true_param.min() - cbar_pad, plot_true_param.max() + cbar_pad
                vmin = min(-vmax, vmin)
                vmax = max(-vmin, vmax)
                matim_true = axes[1].matshow(plot_true_param, vmin=vmin, vmax=vmax, cmap=cmap)

                divider = make_axes_locatable(axes[1])
                cax = divider.append_axes("right", size="5%", pad=0.15)
                plt.colorbar(matim_true, cax=cax)

                axes[1].set_title("true")

                for ax in axes.ravel():
                    ax.tick_params(axis='both', which='both', length=0)
                    plt.setp(ax.get_xticklabels(), visible=False)
                    plt.setp(ax.get_yticklabels(), visible=False)
                    # ax.axis('off')
                    ax.grid(False)

                fig.suptitle(f"{param_name} true $t={t_current}$")
                plt.tight_layout()

                if to_wandb:
                    wandb_images[
                        f"plots/{'' if title_prefix is None else f'{title_prefix}/'}param-true-{param_name}"] = wandb.Image(plt)
                    plt.close()
                else:
                    plt.show()

            ##### regular param plot
            if plot_params:
                if arr.ndim == 1:
                    arr = arr[None, None]
                    n_params = 1
                else:
                    n_params = arr.shape[0]
                    if is_model:
                        arr = arr.reshape(arr.shape[0], arr.shape[1], -1)
                    else:
                        arr = arr.reshape(arr.shape[0], -1, arr.shape[1])

                # rotate arrays that are upright column vectors
                rotated = arr.shape[-1] == 1
                if rotated:
                    arr = onp.swapaxes(arr, -2, -1)

                fig, axes = plt.subplots(n_params, 1, figsize = (y_size, y_size * math.sqrt(n_params)))
                if n_params == 1:
                    axes = onp.array([[axes]])
                else:
                    axes = onp.array(axes)[..., None]

                vmin, vmax = arr.min() - cbar_pad, arr.max() + cbar_pad
                vmin = min(-vmax, vmin)
                vmax = max(-vmin, vmax)

                assert len(axes) == len(arr) and len(axes[0]) == 1 and arr.ndim == 3

                # hack to make scale color center at 1
                if "scale" in param_name or "c1" in param_name:
                    vmax += 2

                for jj, (ax, subarr) in enumerate(zip(axes, arr)):
                    matim_arr = ax[0].matshow(subarr, vmin=vmin, vmax=vmax, cmap=cmap)
                    # fig.colorbar(matim_arr, ax=ax[0], location='right', shrink=0.7)
                    if is_model:
                        ylabel = f"$d${var_names[jj]}" if not subarr.shape[-2] == 1 or rotated else ""
                        ax[0].set_ylabel(ylabel, rotation=0, fontsize=12, labelpad=20, y=0.4)

                for ax in axes.ravel():
                    ax.tick_params(axis='both', which='both', length=0)
                    plt.setp(ax.get_xticklabels(), visible=False)
                    plt.setp(ax.get_yticklabels(), visible=False)
                    # ax.axis('off')
                    ax.grid(False)

                fig.suptitle(f"{param_name} $t={t_current}$")
                plt.tight_layout()

                # colorbar
                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                fig.colorbar(matim_arr, cax=cbar_ax)

                if to_wandb:
                    wandb_images[f"plots/{'' if title_prefix is None else f'{title_prefix}/'}param-{param_name}"] = wandb.Image(plt)
                    plt.close()
                else:
                    plt.show()


    # ============  d = 2  ============
    if d == 2 and plot_paths:
        t = onp.arange(sampled_traj_full.shape[-2])
        n_rows = samples.shape[0]

        ax_dyn = 0
        ax_scatter = 1
        ax_hist0 = 2
        ax_hist1 = 3
        colors = ["b", "g", "r", "c", "m", "y"]

        fig, axes = plt.subplots(n_rows, 4,
                                 figsize=(x_size, y_size * n_rows),
                                 gridspec_kw={'width_ratios': [2, 1, 1, 1]})
        if n_rows == 1:
            axes = [axes]

        xlims = [(math.inf, -math.inf) for _ in range(samples.shape[0])]
        ylims = [(math.inf, -math.inf) for _ in range(samples.shape[0])]


        for ii, (s, traj) in enumerate(zip(samples, sampled_traj_full)):

            # filter nan/inf
            finite = onp.isfinite(traj).all(1)
            traj = traj[finite, :]
            if onp.any(~finite):
                print("Warning: nan or inf in simulated diffusion.")
                continue

            # trajectory
            for jj, var_series in enumerate(onp.swapaxes(traj, -2, -1)):
                alph =  (jj + 1) / traj.shape[-1]
                lab = f"{ii} {var_names[jj]}"
                axes[ii][ax_dyn].plot(t, var_series, label=lab, c=colors[ii], alpha=alph)

            axes[ii][ax_dyn].set_xlim((0, axes[ii][ax_dyn].get_xlim()[1]))
            intvs_str = {idx: batched_tars.data[ii][..., idx].mean().item()
                         for idx in jnp.where(batched_tars.intv[ii])[0].tolist()}
            intvs_str_formatted = " ".join([f"{var_names[k]} = {v:.1f}" for k, v in intvs_str.items()])
            axes[ii][ax_dyn].set_title("obs" if batched_tars.intv[ii].sum() == 0 else
                                       f"intv {intvs_str_formatted}")

            # density (only use snapshots)
            axes[ii][ax_scatter].scatter(s[..., 0], s[..., 1], c=colors[ii], **scat_kwds)
            axes[ii][ax_hist0].hist(s[..., 0], color=colors[ii], alpha=0.5, density=True, bins=get_bins(s[..., 0]))
            axes[ii][ax_hist1].hist(s[..., 1], color=colors[ii], alpha=0.5, density=True, bins=get_bins(s[..., 1]))

            xlims[ii] = min(xlims[ii][0], s[..., 0].min()), max(xlims[ii][1], s[..., 0].max())
            ylims[ii] = min(ylims[ii][0], s[..., 1].min()), max(ylims[ii][1], s[..., 1].max())

            if t_current is not None:
                axes[ii][ax_scatter].set_title(f"t: {t_current}")

            # target
            x0, x1 = onp.array(batched_tars.data[ii][:max_target_samples, 0]), \
                     onp.array(batched_tars.data[ii][:max_target_samples, 1])
            axes[ii][ax_scatter].scatter(x0, x1, color=cmain, zorder=0, **scat_kwds_target)
            axes[ii][ax_hist0].hist(x0, color=cmain, alpha=0.5, density=True, bins=get_bins(s[..., 0]))
            axes[ii][ax_hist1].hist(x1, color=cmain, alpha=0.5, density=True, bins=get_bins(s[..., 1]))

            axes[ii][ax_hist0].set_title("x0")
            axes[ii][ax_hist1].set_title("x1")
            xlims[ii] = min(xlims[ii][0], x0.min()), max(xlims[ii][1], x0.max())
            ylims[ii] = min(ylims[ii][0], x1.min()), max(ylims[ii][1], x1.max())

        # limits consistent in each row
        for ii, (xl, yl, axarr) in enumerate(zip(xlims, ylims, axes)):
            xl = (xl[0] - margin, xl[1] + margin)
            yl = (yl[0] - margin, yl[1] + margin)
            axarr[ax_dyn].set_ylim((min(xl[0], yl[0]),
                                    max(xl[1], yl[1])))
            axarr[ax_scatter].set_xlim(xl)
            axarr[ax_scatter].set_ylim(yl)
            axarr[ax_hist0].set_xlim(xl)
            axarr[ax_hist1].set_xlim(yl)

        fig.legend(loc="upper left")
        fig.tight_layout()
        if title is not None:
            fig.suptitle(title)

        if to_wandb:
            wandb_images[f"plots/{'' if title_prefix is None else f'{title_prefix}/'}trajectory"] = wandb.Image(plt)
            plt.close()
        else:
            plt.show()

    # ============  d > 2  ============
    if plot_pairwise_grid:

        for ii, s in enumerate(samples):
            if plot_max is not None and ii >= plot_max:
                break

            if onp.isnan(s).all(0).any():
                # at least on dimension only has nan, skip scatter plot
                continue

            # visualize in scatter matrix
            data = pd.DataFrame(s, columns=var_names)
            data_target = pd.DataFrame(onp.array(batched_tars.data[ii]), columns=var_names)

            # color mask
            color_mask = onp.zeros((len(data.columns), len(data.columns)), dtype=onp.int32)
            for intv_idx in onp.where(batched_tars.intv[ii])[0]:
                color_mask[:, intv_idx] = 1
                color_mask[intv_idx, :] = 1

            if grid_type == "hist-kde":
                grids = [("hist", "kde")]
            elif grid_type == "kde-kde":
                grids = [("kde", "kde")]
            elif grid_type == "hist-scatter":
                grids = [("hist", "scatter")]
            elif grid_type == "kde-scatter":
                grids = [("kde", "scatter")]
            else:
                raise KeyError(f"Unknown grid type {grid_type}")

            for diagonal, offdiagonal in grids:
                figax = scatter_matrix(
                    data[:max_scatter],
                    data_target[:max_scatter],
                    size_per_var=size_per_var,
                    max_cols=max_cols, max_rows=max_rows,
                    color_mask=color_mask,
                    binwidth=None,
                    n_bins_default=n_bins_default,
                    percentile_cutoff_limits=0.05,
                    range_padding=0.5,
                    contours=contours,
                    contours_alphas=contours_alphas,
                    cmain=cfit,
                    ctarget=cmain,
                    cref=cref,
                    ref_alpha=ref_alpha,
                    ref_data=ref_data,
                    diagonal=diagonal,
                    scotts_bw_scaling=scotts_bw_scaling,
                    offdiagonal=offdiagonal,
                    scat_kwds=scat_kwds,
                    scat_kwds_target=scat_kwds_target,
                    hist_kwds=hist_kwds,
                    hist_kwds_target=hist_kwds_target,
                    density1d_kwds=density1d_kwds,
                    density1d_kwds_target=density1d_kwds_target,
                    density2d_kwds=density2d_kwds,
                    density2d_kwds_target=density2d_kwds_target,
                    minmin=minmin,
                    maxmax=maxmax,
                )
                if figax is None:
                    plt.close()
                    break

                fig, axes = figax

                # format title
                intv_vals = batched_tars.data[ii].mean(-2)
                intv_vals_std = batched_tars.data[ii].std(-2)
                title = title_prefix or ""
                if onp.where(batched_tars.intv[ii])[0].size:
                    title += " interv"
                    for j in onp.where(batched_tars.intv[ii])[0]:
                        if onp.allclose(intv_vals_std[j], 0.0):
                            title += f" {var_names[j]}$=${intv_vals[j].item():.1f}"
                        else:
                            # title += f" {var_names[j]}$\\approx${intv_vals[j].item():.1f} ($\\pm${intv_vals_std[j].item():.1f})"
                            title += f" {var_names[j]}$\\approx${intv_vals[j].item():.1f}"
                else:
                    title += " observ"

                title += f"  ($t={t_current}$)"
                fig.suptitle(title)
                plt.tight_layout()
                plt.gcf().subplots_adjust(wspace=0, hspace=0)

                if to_wandb:
                    wandb_images[f"plots/{'' if title_prefix is None else f'{title_prefix}/'}pairwise-{offdiagonal}={ii}"] = wandb.Image(plt)
                    plt.close()
                else:
                    plt.show()


    # ============  interventional marginals  ============
    max_n_intvs = int(batched_tars.intv.sum(1).max().item())

    # 1: intv target marginals
    if plot_intv_marginals and max_n_intvs > 0:
        envs = batched_tars.intv.sum(1).nonzero()[0]
        n_envs = len(envs)
        n_rows_marginals = max_n_intvs
        n_cols_marginals = n_envs
        fig, axes = plt.subplots(n_rows_marginals, n_cols_marginals, figsize=(y_size * n_cols_marginals, y_size * n_rows_marginals))
        try:
            if axes.ndim == 1:
                if axes.shape == (n_rows_marginals,):
                    axes = axes[..., None]
                elif axes.shape == (n_cols_marginals,):
                    axes = axes[None]
        except AttributeError:
            axes = onp.array([[axes]])

        # for each env, get intervention indices
        assert batched_tars.intv.ndim == 2
        intv_indices = []
        for env in envs:
            intv_indices.append(batched_tars.intv[env].nonzero()[0].tolist())

        n_deleted_axes = 0
        for j, (env, env_intv_indices) in enumerate(zip(envs, intv_indices)):
            # delete unnecessary axes for envs with less than `max_n_intvs` intervened nodes
            if not env_intv_indices:
                for i in range(n_rows_marginals):
                    fig.delaxes(axes[i, j])
                    n_deleted_axes += 1
                continue

            for i in range(len(env_intv_indices), n_rows_marginals):
                fig.delaxes(axes[i, j])
                n_deleted_axes += 1

            for i, idx_j in enumerate(env_intv_indices):
                ax = axes[i, j]

                marginal_j = samples[env][:, idx_j].copy()
                marginal_target_j = batched_tars.data[env][:, idx_j].copy()
                if onp.sum(~(onp.isnan(marginal_j) | (marginal_j < minmin) | (marginal_j > maxmax))) <= 10:
                    fig.delaxes(axes[i, j])
                    n_deleted_axes += 1
                    continue

                marginal_j = marginal_j[~onp.isnan(marginal_j)]
                marginal_target_j = marginal_target_j[~onp.isnan(marginal_target_j)]

                degenerate = onp.isclose(onp.std(marginal_j), 0.0, atol=1e-3)
                degenerate_target = onp.isclose(onp.std(marginal_target_j), 0.0, atol=1e-3)

                binmin = min(marginal_j.min(), marginal_target_j.min())
                binmax = max(marginal_j.max(), marginal_target_j.max())
                if ref_data is not None:
                    assert ref_data.shape[-1] == d
                    ref_data_j = onp.array(ref_data[:, idx_j])
                    binmin = min(binmin, ref_data_j.min())
                    binmax = max(binmax, ref_data_j.max())
                if degenerate and degenerate_target:
                    binmin, binmax = binmax - 1.0, binmax + 1.0

                binwidth_i = (binmax - binmin) / int(n_bins_default * 1.5)
                bins = onp.arange(binmin, binmax + binwidth_i, binwidth_i)
                ax.hist(marginal_target_j, color=cmain, bins=bins, density=True, **hist_kwds_target)
                ax.hist(marginal_j, color=cfit, bins=bins, density=True, **hist_kwds)
                ax.set_xlabel(var_names[idx_j])
                # ax.set_title(var_names[idx_j])

                # plot reference data
                if ref_data is not None:
                    assert ref_data.shape[-1] == d
                    ref_data_j = onp.array(ref_data[:, idx_j])
                    ax.hist(ref_data_j, histtype="step", color=cref, bins=bins, density=True,
                            alpha=ref_alpha, zorder=-1)

                # format title
                if i == 0:
                    title = "intv on " + ", ".join([var_names[jj] for jj in env_intv_indices])
                    ax.set_title(title)

        # remove gridlines
        for ax in onp.asarray(axes).ravel():
            ax.grid(visible=False, axis='both')

        suptitle = title_prefix or ""
        fig.suptitle(suptitle + " intv-marginals")

        plt.tight_layout()

        if n_deleted_axes < n_rows_marginals * n_cols_marginals:
            if to_wandb:
                wandb_images[f"plots/{'' if title_prefix is None else f'{title_prefix}/'}" \
                             f"intv-marginals"] = wandb.Image(plt)
                plt.close()
            else:
                plt.show()
        else:
            plt.close()

    # 2: intv offtarget marginals
    if plot_intv_marginals and max_n_intvs > 0:
        envs = batched_tars.intv.sum(1).nonzero()[0]
        n_envs = len(envs)
        n_cols_marginals = n_envs
        fig, axes = plt.subplots(1, n_cols_marginals, figsize=(y_size * n_cols_marginals, y_size))
        if n_envs == 1:
            axes = [axes]

        # for each env, get intervention indices
        assert batched_tars.intv.ndim == 2
        intv_indices = []
        for env in envs:
            intv_indices.append(batched_tars.intv[env].nonzero()[0].tolist())

        for j, (env, env_intv_indices) in enumerate(zip(envs, intv_indices)):
            # find variable with largest intv effect that was not intervened on directly
            ax = axes[j]

            # effect measured in sigmas
            effect = onp.abs(samples[env].mean(0) - batched_tars.data[env].mean(0)) / batched_tars.data[env].std(0)
            effect_ranking = onp.argsort(-effect)

            is_off_target = ~batched_tars.intv[env].astype(bool)
            if not is_off_target.any():
                continue

            off_target_effect_ranking = onp.array([e for e in effect_ranking if is_off_target[e]])
            max_idx = off_target_effect_ranking[0]

            marginal_j = samples[env][:, max_idx].copy()
            marginal_target_j = batched_tars.data[env][:, max_idx].copy()
            if onp.sum(~(onp.isnan(marginal_j) | (marginal_j < minmin) | (marginal_j > maxmax))) <= 10:
                fig.delaxes(axes[j])
                continue

            marginal_j = marginal_j[~onp.isnan(marginal_j)]
            marginal_target_j = marginal_target_j[~onp.isnan(marginal_target_j)]

            degenerate = onp.isclose(onp.std(marginal_j), 0.0, atol=1e-3)
            degenerate_target = onp.isclose(onp.std(marginal_target_j), 0.0, atol=1e-3)

            binmin = min(marginal_j.min(), marginal_target_j.min())
            binmax = max(marginal_j.max(), marginal_target_j.max())
            if ref_data is not None:
                assert ref_data.shape[-1] == d
                ref_data_j = onp.array(ref_data[:, max_idx])
                binmin = min(binmin, ref_data_j.min())
                binmax = max(binmax, ref_data_j.max())
            if degenerate and degenerate_target:
                binmin, binmax = binmax - 1.0, binmax + 1.0

            binwidth_i = (binmax - binmin) / int(n_bins_default * 1.5)
            bins = onp.arange(binmin, binmax + binwidth_i, binwidth_i)
            ax.hist(marginal_target_j, color=cmain, bins=bins, density=True, **hist_kwds_target)
            ax.hist(marginal_j, color=cfit, bins=bins, density=True, **hist_kwds)
            ax.set_xlabel(var_names[max_idx])
            # ax.set_title(var_names[idx_j])

            # plot reference data
            if ref_data is not None:
                assert ref_data.shape[-1] == d
                ref_data_j = onp.array(ref_data[:, max_idx])
                ax.hist(ref_data_j, histtype="step", color=cref, bins=bins, density=True,
                        alpha=ref_alpha, zorder=-1)

            # format title
            if i == 0:
                title = "intv on " + ", ".join([var_names[jj] for jj in env_intv_indices])
                ax.set_title(title)

        # remove gridlines
        for ax in onp.asarray(axes).ravel():
            ax.grid(visible=False, axis='both')

        suptitle = title_prefix or ""
        fig.suptitle(suptitle + " max-off-intv-marginals")

        plt.tight_layout()

        if to_wandb:
            wandb_images[f"plots/{'' if title_prefix is None else f'{title_prefix}/'}" \
                         f"max-off-intv-marginals"] = wandb.Image(plt)
            plt.close()
        else:
            plt.show()


    # ============  autocorrelation  ============
    if plot_acorr:
        assert sampled_traj_full.ndim == 3 and sampled_traj_full.shape[-1] == d, \
            f"Sampled trajectory must be shape [n_envs, t, d], got `{sampled_traj_full.shape}`"

        sampled_traj_obs = sampled_traj_full[0] # select observational dist
        n_rows_acorr = ((d - 1) // max_proj_cols) + 1
        n_cols_acorr = min(d, max_proj_cols)
        fig, axes = plt.subplots(n_rows_acorr, n_cols_acorr, figsize=(y_size * n_cols_acorr, y_size * n_rows_acorr))
        if d == 1:
            axes = [[axes]]
        elif n_rows_acorr == 1:
            axes = [axes]
        for k in range(d):
            ii, jj = (k // n_cols_acorr), (k % n_cols_acorr)
            autocorrelation_plot(sampled_traj_obs[..., k], axes[ii][jj])
            axes[ii][jj].set_title(var_names[k])
            axes[ii][jj].set_ylim((-0.4, 1.0))
            is_left, is_bottom = jj != 0, ii != n_rows_acorr - 1
            if is_left:
                axes[ii][jj].set_ylabel("")
            if is_bottom:
                axes[ii][jj].set_xlabel("")
            axes[ii][jj].tick_params(
                axis='both',
                which='both',
                bottom=False,
                left=False,
                right=False,
                top=False,
                labelbottom=not is_bottom,
                labelleft=not is_left,
            )
        for kk in range(d, n_cols_acorr * n_rows_acorr):
            ii, jj = (kk // n_cols_acorr), (kk % n_cols_acorr)
            fig.delaxes(axes[ii][jj])

        plt.gcf().subplots_adjust(wspace=0, hspace=0)
        plt.draw()
        plt.tight_layout()
        if to_wandb:
            wandb_images[f"plots/{'' if title_prefix is None else f'{title_prefix}/'}acorr-obs"] = wandb.Image(plt)
            plt.close()
        else:
            plt.show()


    # ============  low-dim projection  ============
    if proj is not None:
        n_rows_proj = ((samples.shape[0] - 1) // max_proj_cols) + 1
        n_cols_proj = min(samples.shape[0], max_proj_cols)
        fig, axes = plt.subplots(n_rows_proj, n_cols_proj, figsize=(y_size * n_cols_proj, y_size * n_rows_proj))
        if samples.shape[0] == 1:
            axes = [[axes]]
        elif n_rows_proj == 1:
            axes = [axes]
        for k, s in enumerate(samples):
            ii, jj = (k // n_cols_proj),  (k % n_cols_proj)

            # filter nan and inf
            target_data = batched_tars.data[k]
            s_data = s

            target_mask = ~onp.isnan(target_data) & (target_data < maxmax) & (target_data > minmin)
            s_mask = ~onp.isnan(s_data) & (s_data < maxmax) & (s_data > minmin)

            target_data = target_data[onp.where(target_mask.all(-1))[0]]
            s_data = s_data[onp.where(s_mask.all(-1))[0]]

            if not target_data.size or not s_data.size:
                fig.delaxes(axes[ii][jj])
                continue

            target_proj = proj.transform(target_data)
            s_proj = proj.transform(s_data)

            # scatter
            # axes[ii][jj].scatter(target_proj[..., 0], target_proj[..., 1], c=cmain, **scat_kwds_target)
            # axes[ii][jj].scatter(s_proj[..., 0], s_proj[..., 1], c=cfit, **scat_kwds)

            # kde
            try:
                target_proj = target_proj.T
                s_proj = s_proj.T

                contours_alphas_ = onp.sort(onp.array(contours_alphas))

                bw_scotts_target = float(target_proj.shape[-1]) ** float(-1. / (2 + 4))
                kde_target = gaussian_kde(target_proj, bw_method=scotts_bw_scaling * bw_scotts_target)
                kde_target_fun = lambda x: kde_target(x.reshape(-1, 2).T).reshape(*x.shape[:-1])

                xlim_target = (target_proj[0].min(), target_proj[0].max())
                ylim_target = (target_proj[1].min(), target_proj[1].max())

                z_target, z_target_levels = get_2d_percentile_contours(kde_target_fun, levels=contours,
                                                                       xlim=xlim_target, ylim=ylim_target)

                for z_target_level, z_alpha in zip(z_target_levels, contours_alphas_):
                    axes[ii][jj].contour(z_target, levels=onp.array([z_target_level]),
                                         extent=[*xlim_target, *ylim_target],
                                         colors=cmain, alpha=z_alpha,
                                         **density2d_kwds_target)

                bw_scotts = float(s_proj.shape[-1]) ** float(-1. / (2 + 4))
                kde = gaussian_kde(s_proj, bw_method=scotts_bw_scaling * bw_scotts)
                kde_fun = lambda x: kde(x.reshape(-1, 2).T).reshape(*x.shape[:-1])

                xlim_s = (s_proj[0].min(), s_proj[0].max())
                ylim_s = (s_proj[1].min(), s_proj[1].max())

                z, z_levels = get_2d_percentile_contours(kde_fun, levels=contours,
                                                         xlim=xlim_s, ylim=ylim_s)

                for z_level, z_alpha in zip(z_levels, contours_alphas_):
                    axes[ii][jj].contour(z, levels=onp.array([z_level]),
                                         extent=[*xlim_s, *ylim_s],
                                         colors=cfit, alpha=z_alpha,
                                         **density2d_kwds)

            except ValueError:
                # sometimes happens that error like this occurs if system is not stable yet:
                # "A value ... in x_new is below the interpolation range's minimum value."
                pass

            # format title
            intv_vals = batched_tars.data[k].mean(-2)
            intv_vals_std = batched_tars.data[k].std(-2)
            title = title_prefix or ""
            if onp.where(batched_tars.intv[k])[0].size:
                title += " interv"
                for j in onp.where(batched_tars.intv[k])[0]:
                    if onp.allclose(intv_vals_std[j], 0.0):
                        title += f" {var_names[j]}$=${intv_vals[j].item():.1f}"
                    else:
                        # title += f" {var_names[j]}$\\approx${intv_vals[j].item():.1f} ($\\pm${intv_vals_std[j].item():.1f})"
                        title += f" {var_names[j]}$\\approx${intv_vals[j].item():.1f}"
            else:
                title += " observ"

            axes[ii][jj].set_title(title)

        # remove gridlines
        for ax in onp.asarray(axes).ravel():
            # ax.tick_params(
            #     axis='both',
            #     which='both',
            #     bottom=False,
            #     left=False,
            #     right=False,
            #     top=False,
            #     labelbottom=False,
            #     labelleft=False,
            # )
            ax.grid(visible=False, axis='both')

        # unify axes
        x0, x1, y0, y1 = math.inf, -math.inf, math.inf, -math.inf
        for kk in range(n_cols_proj * n_rows_proj):
            ii, jj = (kk // n_cols_proj), (kk % n_cols_proj)
            if kk <= d:
                x0 = min(x0, axes[ii][jj].get_xlim()[0])
                x1 = max(x1, axes[ii][jj].get_xlim()[1])
                y0 = min(y0, axes[ii][jj].get_ylim()[0])
                y1 = max(y1, axes[ii][jj].get_ylim()[1])
            else:
                fig.delaxes(axes[ii][jj])

        for kk in range(n_cols_proj * n_rows_proj):
            ii, jj = (kk // n_cols_proj), (kk % n_cols_proj)
            if kk <= d:
                axes[ii][jj].set_xlim((x0, x1))
                axes[ii][jj].set_ylim((y0, y1))

        plt.gcf().subplots_adjust(wspace=0, hspace=0)
        plt.draw()
        plt.tight_layout()
        if to_wandb:
            wandb_images[f"plots/{'' if title_prefix is None else f'{title_prefix}/'}proj"] = wandb.Image(plt)
            plt.close()
        else:
            plt.show()

    plt.close('all')


    return wandb_images if wandb_images else None