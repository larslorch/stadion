import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import warnings

import pandas as pd
from pandas.plotting._matplotlib.tools import create_subplots
from scipy.stats import gaussian_kde
from scipy import interpolate


def get_2d_percentile_contours(f, levels=(0.683, 0.955, 0.997), n_grid=100, xlim=(-5, 5), ylim=(-5, 5)):
    """
    Based on https://stackoverflow.com/questions/37890550/python-plotting-percentile-contour-lines-of-a-probability-distribution

    Args:
        f: function mapping input points to function (pdf) values, i.e. [..., 2] -> [...]
        levels: array of levels
        n_grid: interpolation points
        xlim: x range of interpolation grid
        ylim: y range of interpolation grid

    Returns:
        Array of shape `levels.shape` corresponding to function values of `f` such that `levels[i]` percentage of
        the integral lies above `levels[i]`. Can be used to plot e.g. 95% percentile line of an arbitrary 2d pdf
    """
    x, y = np.meshgrid(np.linspace(*xlim, n_grid), np.linspace(*ylim, n_grid))
    # points: [Nx, Ny, 2]
    points = np.stack([x, y], axis=-1)

    # z: [Nx, Ny]
    f_vals = f(points)
    f_vals = f_vals / f_vals.sum()

    # t: range of possible function values:
    t = np.linspace(0, f_vals.max(), n_grid)

    # integral: proportion of integral above `t`
    integral = ((f_vals >= t[:, None, None]) * f_vals).sum(axis=(1, 2))
    interpolated_f = interpolate.interp1d(integral, t)
    return f_vals, interpolated_f(np.sort(np.array(levels))[::-1])


def scatter_matrix(
    frame,
    target=None,
    size_per_var=1.0,
    ax=None,
    diagonal="hist",
    offdiagonal="scatter",
    range_padding=0.01,
    percentile_cutoff_limits=0.05,
    contours=(0.683, 0.955, 0.997),
    contours_alphas=(0.33, 0.66, 1.0),
    max_rows=None,
    max_cols=None,
    color_mask=None,
    cmain="black",
    ctarget="blue",
    cref="black",
    ref_alpha=0.8,
    ref_data=None,
    color_mask_color="#CCCCCC",
    minmin=-1000,
    maxmax=1000,
    binwidth=0.5,
    scotts_bw_scaling=1.0,
    n_bins_default=20,
    ylabel_rot=90,
    xlabelsize=8,
    xrot=0,
    ylabelsize=8,
    yrot=0,
    hist_kwds=None,
    hist_kwds_target=None,
    scat_kwds=None,
    scat_kwds_target=None,
    density1d_kwds=None,
    density1d_kwds_target=None,
    density2d_kwds=None,
    density2d_kwds_target=None,
    skip=None,
):
    """Adapted from pandas version"""

    df = frame._get_numeric_data()
    if target is not None:
        df_target = target._get_numeric_data()
        assert len(df.columns) == len(df_target.columns)

    # filter by max/min values
    for col in df.columns:
        df = df[(df[col] > minmin) & (df[col] < maxmax)]
        if target is not None:
            df_target = df_target[(df_target[col] > minmin) & (df_target[col] < maxmax)]

    # exit in case no reasonable values available; may occur at the beginning of training
    if df.empty:
        print(f"Skipped `scatter_matrix` because no values available in provided range ({minmin}, {maxmax})")
        return

    n = len(df.columns)
    n_rows = min(max_rows, n) if max_rows is not None else n
    n_cols = min(max_cols, n) if max_cols is not None else n
    naxes = n_rows * n_cols
    layout = (n_rows, n_cols)

    figsize = (n_cols * size_per_var, n_rows * size_per_var)

    # plt.rcParams['figure.figsize'] = figsize
    fig, axes = create_subplots(naxes=naxes, figsize=figsize, ax=ax, layout=layout, squeeze=False)

    # no gaps between subplots
    fig.subplots_adjust(wspace=0, hspace=0)

    mask = pd.core.dtypes.missing.notna(df)
    if target is not None:
        mask_target = pd.core.dtypes.missing.notna(df_target)

    hist_kwds = hist_kwds or {}
    hist_kwds_target = hist_kwds_target or {}
    scat_kwds = scat_kwds or {}
    scat_kwds_target = scat_kwds_target or {}
    density1d_kwds = density1d_kwds or {}
    density1d_kwds_target = density1d_kwds_target or {}
    density2d_kwds = density2d_kwds or {}
    density2d_kwds_target = density2d_kwds_target or {}

    assert len(contours) == len(contours_alphas)

    # GH 14855
    boundaries_list = []
    for idx in range(n):
        values = df.iloc[:, idx].values[mask[df.columns[idx]].values]
        if not values.size:
            continue
        rmin_, rmax_ = np.quantile(values, percentile_cutoff_limits / 2), \
                       np.quantile(values, 1.0 - (percentile_cutoff_limits / 2))
        rdelta_ext = (rmax_ - rmin_) * range_padding / 2
        if rmin_ == rmax_:
            rmin_ -= range_padding
            rmax_ += range_padding

        boundaries_list.append((rmin_ - rdelta_ext, rmax_ + rdelta_ext))

        if target is not None:
            values_target = df_target.iloc[:, idx].values[mask_target[df_target.columns[idx]].values]
            rmin_target_, rmax_target_ = np.quantile(values_target, percentile_cutoff_limits / 2), \
                                         np.quantile(values_target, 1.0 - (percentile_cutoff_limits / 2))
            if rmin_target_ == rmax_target_:
                rmin_target_ -= range_padding
                rmax_target_ += range_padding
            rdelta_ext_target = (rmax_target_ - rmin_target_) * range_padding / 2

            boundaries_list[-1] = (min(boundaries_list[-1][0], rmin_target_ - rdelta_ext_target), max(boundaries_list[-1][1], rmax_target_ + rdelta_ext_target))

        if ref_data is not None:
            assert ref_data.shape[-1] == df.shape[-1]
            values_ref = ref_data[:, idx]
            rmin_ref_, rmax_ref_ = np.quantile(values_ref, percentile_cutoff_limits / 2), \
                                         np.quantile(values_ref, 1.0 - (percentile_cutoff_limits / 2))
            if rmin_ref_ == rmax_ref_:
                rmin_ref_ -= range_padding
                rmax_ref_ += range_padding
            rdelta_ext_ref = (rmax_ref_ - rmin_ref_) * range_padding / 2

            boundaries_list[-1] = (min(boundaries_list[-1][0], rmin_ref_ - rdelta_ext_ref),
                                   max(boundaries_list[-1][1], rmax_ref_ + rdelta_ext_ref))


    for i, idx_i in enumerate(range(n)): # here could enumerate only over non-dropped cols
        if i >= n_rows:
            break
        for j, idx_j in enumerate(range(n)):
            if j >= n_cols:
                break
            ax = axes[i][j]

            if color_mask is not None and np.array(color_mask)[idx_i, idx_j] == 1:
                ax.set_facecolor(color_mask_color)

            scat_kwds_ = scat_kwds
            scat_kwds_target_ = scat_kwds_target
            hist_kwds_ = hist_kwds
            hist_kwds_target_ = hist_kwds_target
            density1d_kwds_ = density1d_kwds
            density1d_kwds_target_ = density1d_kwds_target
            density2d_kwds_ = density2d_kwds
            density2d_kwds_target_ = density2d_kwds_target

            if skip is None or (i, j) not in skip:
                # diagonal
                if idx_i == idx_j:
                    values = df.iloc[:, idx_i].values[mask[df.columns[idx_i]].values]
                    degenerate = np.isclose(np.std(values), 0.0, atol=1e-3)

                    # get min and max bin point
                    if target is None:
                        binmin = values.min()
                        binmax = values.max()
                        target_degenerate = False
                    else:
                        values_target = df_target.iloc[:, idx_i].values[mask_target[df_target.columns[idx_i]].values]
                        binmin = min(values.min(), values_target.min())
                        binmax = max(values.max(), values_target.max())
                        target_degenerate = np.isclose(np.std(values_target), 0.0, atol=1e-3)

                    if ref_data is not None:
                        assert ref_data.shape[-1] == df.shape[-1]
                        ref_data_i = np.array(ref_data[:, idx_i])
                        binmin = min(binmin, ref_data_i.min())
                        binmax = max(binmax, ref_data_i.max())

                    # if binwidth not specified, select binwidth such that exactly `n_bins_default` are in visible range
                    if binwidth is None:
                        binwidth_i = (boundaries_list[idx_i][1] - boundaries_list[idx_i][0]) / n_bins_default
                    else:
                        binwidth_i = binwidth

                    bins = np.arange(binmin, binmax + binwidth_i, binwidth_i)

                    # Deal with the diagonal by drawing a histogram there.
                    if diagonal == "hist" or degenerate or target_degenerate:
                        if target is not None:
                            if not target_degenerate:
                                ax.hist(values_target, color=ctarget, zorder=1,bins=bins, density=True, **hist_kwds_target_)

                        if not degenerate:
                            ax.hist(values, color=cmain, bins=bins, zorder=2, density=True, **hist_kwds_)

                    elif diagonal == "kde":
                        if target is not None:
                            bw_scotts_target = float(values_target.shape[0]) ** float(-1. / (1 + 4))
                            gkde = gaussian_kde(values_target, bw_method=scotts_bw_scaling * bw_scotts_target)
                            ind = np.linspace(values_target.min(), values_target.max(), 1000)
                            ax.plot(ind, gkde.evaluate(ind), color=ctarget, zorder=1, **density1d_kwds_target_)

                        bw_scotts = float(values.shape[0]) ** float(-1./(1 + 4))
                        gkde = gaussian_kde(values, bw_method=scotts_bw_scaling * bw_scotts)
                        ind = np.linspace(values.min(), values.max(), 1000)
                        ax.plot(ind, gkde.evaluate(ind), color=cmain, zorder=2, **density1d_kwds_)

                    else:
                        raise KeyError(f"diagonal {diagonal}")

                    # plot reference data
                    if ref_data is not None:
                        assert ref_data.shape[-1] == df.shape[-1]
                        ref_data_i = np.array(ref_data[:, idx_i])

                        # mean line
                        # ylim = ax.get_ylim()
                        # ax.plot(ref_data_i.mean(0) * np.ones(5), np.linspace(ylim[0] - 1, ylim[1] + 1, 5), color=cref,
                        #         linestyle="solid", alpha=ref_alpha, zorder=10)
                        # ax.set_ylim(ylim)

                        if diagonal == "hist":
                            ax.hist(ref_data_i, histtype="step", color=cref, bins=bins, density=True,
                                    alpha=ref_alpha, zorder=0)

                        elif diagonal == "kde":
                            bw_scotts_ref = float(ref_data_i.shape[0]) ** float(-1. / (1 + 4))
                            gkde = gaussian_kde(ref_data_i, bw_method=scotts_bw_scaling * bw_scotts_ref)
                            ind = np.linspace(ref_data_i.min(), ref_data_i.max(), 1000)
                            ax.plot(ind, gkde.evaluate(ind), color=cref, zorder=0)

                    ax.set_xlim(boundaries_list[i])

                # off diagonal
                else:
                    if target is not None:
                        common_target = (mask_target[df_target.columns[idx_i]] & mask_target[df_target.columns[idx_j]]).values
                        x_tar_values, y_tar_values = df_target.iloc[:, idx_j][common_target], df_target.iloc[:, idx_i][common_target]
                        target_pair_degenerate = np.isclose(np.std(x_tar_values), 0.0, atol=1e-3) or np.isclose(np.std(y_tar_values), 0.0, atol=1e-3)
                    else:
                        target_pair_degenerate = False

                    common = (mask[df.columns[idx_i]] & mask[df.columns[idx_j]]).values
                    x_values, y_values = df.iloc[:, idx_j][common], df.iloc[:, idx_i][common]
                    pair_degenerate = np.isclose(np.std(x_values), 0.0, atol=1e-3) or np.isclose(np.std(y_values), 0.0, atol=1e-3)

                    if offdiagonal == "scatter" or pair_degenerate or target_pair_degenerate:
                        if target is not None:
                            ax.scatter(
                                x_tar_values, y_tar_values,
                                edgecolors="none",
                                color=ctarget,
                                **scat_kwds_target_,
                            )
                        ax.scatter(
                            x_values, y_values,
                            edgecolors="none",
                            color=cmain,
                            **scat_kwds_,
                        )

                    elif offdiagonal == "kde":
                        try:
                            contours_alphas_ = np.sort(np.array(contours_alphas))

                            if target is not None:
                                vals_target = np.vstack([df_target.iloc[:, idx_j][common_target],
                                                         df_target.iloc[:, idx_i][common_target]])
                                bw_scotts_target = float(vals_target.shape[-1]) ** float(-1./(2 + 4))
                                kde_target = gaussian_kde(vals_target, bw_method=scotts_bw_scaling * bw_scotts_target)
                                kde_target_fun = lambda x: kde_target(x.reshape(-1, 2).T).reshape(*x.shape[:-1])

                                z_target, z_target_levels = get_2d_percentile_contours(kde_target_fun, levels=contours,
                                                                                       xlim=boundaries_list[j], ylim=boundaries_list[i])
                                for z_target_level, z_alpha in zip(z_target_levels, contours_alphas_):
                                    ax.contour(z_target, levels=np.array([z_target_level]),
                                               extent=[*boundaries_list[j], *boundaries_list[i]],
                                               colors=ctarget, alpha=z_alpha,
                                               **density2d_kwds_target_)

                            vals = np.vstack([df.iloc[:, idx_j][common],
                                              df.iloc[:, idx_i][common]])
                            bw_scotts = float(vals.shape[-1]) ** float(-1./(2 + 4))
                            kde = gaussian_kde(vals, bw_method=scotts_bw_scaling * bw_scotts)
                            kde_fun = lambda x: kde(x.reshape(-1, 2).T).reshape(*x.shape[:-1])

                            z, z_levels = get_2d_percentile_contours(kde_fun, levels=contours,
                                                                     xlim=boundaries_list[j], ylim=boundaries_list[i])

                            for z_level, z_alpha in zip(z_levels, contours_alphas_):
                                ax.contour(z, levels=np.array([z_level]), extent=[*boundaries_list[j], *boundaries_list[i]],
                                           colors=cmain, alpha=z_alpha,
                                           **density2d_kwds_)

                        except ValueError:
                            # sometimes happens that error like this occurs if system is not stable yet:
                            # "A value ... in x_new is below the interpolation range's minimum value."
                            pass
                    else:
                        raise KeyError(f"offdiagonal {offdiagonal}")


                    ax.set_xlim(boundaries_list[j])
                    ax.set_ylim(boundaries_list[i])

            ax.set_xlabel(df.columns[idx_j])
            ax.set_ylabel(df.columns[idx_i], rotation=ylabel_rot, labelpad=10)# fontsize=20, labelpad=20

            if j != 0:
                ax.yaxis.set_visible(False)
            if i != n - 1:
                ax.xaxis.set_visible(False)

    def format_tick(value, tick_number):
        return "{:.1f}".format(round(value.item(), 1))
        # return "{:.2f}".format(round(value.item(), 2))

    for ax in np.asarray(axes).ravel():
        # ax.grid() # this somehow hides the grid lines
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_tick))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_tick))
        if xlabelsize is not None:
            plt.setp(ax.get_xticklabels(), fontsize=xlabelsize)
        if xrot is not None:
            plt.setp(ax.get_xticklabels(), rotation=xrot)
        if ylabelsize is not None:
            plt.setp(ax.get_yticklabels(), fontsize=ylabelsize)
        if yrot is not None:
            plt.setp(ax.get_yticklabels(), rotation=yrot)

    # adjust plot at [0,0] to have correct axis of the nondiagonal
    if len(df.columns) > 1:
        lim1 = boundaries_list[0]
        locs = axes[0][1].yaxis.get_majorticklocs()
        locs = locs[(lim1[0] <= locs) & (locs <= lim1[1])]
        adj = (locs - lim1[0]) / (lim1[1] - lim1[0])

        lim0 = axes[0][0].get_ylim()
        adj = adj * (lim0[1] - lim0[0]) + lim0[0]
        axes[0][0].yaxis.set_ticks(adj)

        if np.all(locs == locs.astype(int)):
            # if all ticks are int
            locs = locs.astype(int)
        axes[0][0].yaxis.set_ticklabels([format_tick(loc, None) for loc in locs])

    # remove gridlines
    for ax in np.asarray(axes).ravel():
        ax.grid(visible=False, axis='both')

    return fig, axes
