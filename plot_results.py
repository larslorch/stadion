import warnings
warnings.filterwarnings("ignore", message="No GPU automatically detected")
warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"

import itertools
import matplotlib.pyplot as plt
import numpy as onp
import scipy
import pandas as pd
from pprint import pprint
import math

from stadion.definitions import IS_CLUSTER
from stadion.experiment.plot_config import *
from stadion.definitions import *
import matplotlib as mpl

import matplotlib.collections as mcol
from stadion.utils.legend import HandlerDashedLines
from stadion.definitions import PROJECT_DIR
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch

from matplotlib.markers import MarkerStyle

NEURIPS_RCPARAMS = {
    "text.usetex": True,                # use LaTeX to write all text
    # "font.family": "serif",             # use serif rather than sans-serif
    'font.family': 'lmodern',
    # "font.serif": "Times New Roman",    # use "Times New Roman" as the standard font
    "font.size": 12,
    "pgf.texsystem": "xelatex",         # use Xelatex which is TTF font aware
    "pgf.rcfonts": False,               # Use pgf.preamble, ignore standard Matplotlib RC
    "pgf.preamble": (
        r'\usepackage{fontspec}'
        r'\usepackage{unicode-math}'
        r'\setmainfont{Times New Roman}'
    ),
    "text.latex.preamble": (
        r'\usepackage{amsmath}'
        r'\usepackage{amssymb}'
        r'\usepackage{amsfonts}'
        r'\usepackage{wasysym}'
        r'\usepackage{graphicx}'
    ),
}


PLOT_METRICS = [
    "wasser_test",
    "mse_test",
]

FILTER_METHODS = []

mpl.rcParams.update(NEURIPS_RCPARAMS)


if __name__ == "__main__":

    # show = True
    show = False

    # log_scales =  [False, True]
    # log_scales =  [True]
    log_scales = [False]

    # fliers = [False, True]
    # fliers = [True]
    fliers = [False]

    show_names = [False, True]
    # show_names = [True]
    names_width_offset = 0.91

    # final_save = False
    final_save = True


    folder = "final/"

    for j, (name, plot_adjustment) in enumerate([
        ("df-linear-er-summary_00_01_00.csv", dict()),
        ("df-linear-sf-summary_00_01_00.csv", dict()),
        ("df-scm-er-summary_00_01_00.csv", dict()),
        ("df-scm-sf-summary_00_01_00.csv", dict()),
        ("df-sergio-er-summary_00_01_00.csv", dict(mse_test=9.5 , wasser_test=250)),
        ("df-sergio-sf-summary_00_01_00.csv", dict(mse_test=9.5, wasser_test=250)),
    ]):
        print("\nPlotting", name, flush=True)
        name = Path(name)

        # load results
        df = pd.read_csv((PROJECT_DIR / SUBDIR_RESULTS / folder / name).with_suffix(".csv"), header=0, index_col=0)


        """ ------------ Metric figures ------------ """

        metrics = sorted(set(df["metric"].unique()) & set(PLOT_METRICS))

        combos = itertools.product(metrics, log_scales, fliers, show_names)

        for j, (metric, log_scale, showfliers, showname) in enumerate(combos):

            print(metric, f"(log_scale = {log_scale})")

            if final_save:
                plot_name = f"{name.stem.rsplit('-', 1)[0]}-{metric}"
            else:
                plot_name = f"{name.stem}-{metric}"

            if showname:
                plot_name += "-named"

            plot_path = PROJECT_DIR / SUBDIR_RESULTS / folder / "plots" / plot_name

            df_metric = df.loc[df["metric"] == metric].drop("metric", axis=1)
            df_metric = df_metric.reset_index(drop=True)

            # filter methods
            df_metric = df_metric.loc[~df_metric["method"].isin(FILTER_METHODS)]

            # set plot configuration
            width = 3.7
            height = 2.15

            if not showname:
                width -= names_width_offset
            fig, ax = plt.subplots(1, 1, figsize=(width, height))

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
                        config.append((w, tup[0], tup[1], tup[2]))

            for w in filter(lambda w: w not in seen, unique):
                # unknown method substring
                config.append((w, METHODS_DEFAULT_CONFIG[0]))

            # when plotting horizontally, invert order
            config = list(reversed(config))

            print(config)

            methods = [tup[0] for tup in config]
            colors = [tup[1] for tup in config]
            cyclic = [tup[2] for tup in config]
            labels = [tup[3] for tup in config]

            data = [df_metric.loc[df_metric["method"] == method]["val"].to_numpy() for method in methods]

            # sanity checks
            for method, dat in zip(methods, data):
                if onp.any(onp.isnan(dat)):
                    print(f"WARNING: method {method} has {onp.isnan(dat).sum()} NaN values for `{metric}`")

            data = [dat[~onp.isnan(dat)] for dat in data]

            # convert to log scale
            if log_scale:
                data = [onp.log10(dat) for dat in data]

            # horizontal marker
            diamond_marker = MarkerStyle("d")
            diamond_marker._transform.rotate_deg(90)

            # plot
            plot_kwargs = dict(
                x=data,
                labels=labels,
                widths=0.6,
                vert=False,
                # notch=True,
                notch=False,
                whis=1.5,
                patch_artist=True,
                bootstrap=None,
                flierprops=dict(marker=diamond_marker, markersize=3, markeredgecolor='black', markerfacecolor='black',  alpha=0.2),
                # flierprops=dict(marker="|", markersize=3, markeredgecolor='black', markerfacecolor='black',  alpha=0.2),
                boxprops=dict(alpha=0.80),
                showfliers=showfliers,
            )

            bp = ax.boxplot(**plot_kwargs)

            for i, box in enumerate(bp['boxes']):
                box.set_facecolor(colors[i])  # Set box color
                if not cyclic[i]:
                    box.set_hatch('///')

            for median in bp['medians']:
                median.set_color('black')  # Set median line color
                median.set_linewidth(2.0)

            title = METRICS_NAMING.get(metric, metric)
            if log_scale:
                title = f"$\log$ " + title
            ax.set_title(title, fontsize=12)

            # axes
            mindata = min([onp.min(dat) for dat in data])
            maxdata = max([onp.max(dat) for dat in data])

            # exlude fliers beyond 3x interquartile range
            # iqrs = [onp.nanpercentile(d, 75) - onp.nanpercentile(d, 25) for d in data]
            # xmax = max([onp.nanpercentile(d, 50) + 2.5 * iqr for d, iqr in zip(data, iqrs)])
            # _, xmax_current = ax.get_xlim()
            # xmax = min(xmax, xmax_current)
            # ax.set_xlim((-0.05 * abs(xmax), xmax))

            # set right axis limits based on whiskers instead of fliers
            whisker_ends = onp.array([whisker.get_xdata() for whisker in bp['caps']]).flatten()
            xmin = onp.min(whisker_ends)
            xmax = onp.max(whisker_ends)

            if metric in plot_adjustment:
                xmax = plot_adjustment[metric]

            if not log_scale:
                ax.set_xlim((- 0.05 * xmax, 1.025 * xmax))

            if log_scale:
                yticks = onp.arange(math.floor(mindata), math.ceil(maxdata))
                ax.set_xticks(yticks)
                ax.set_xticklabels(10.0 ** yticks)
                ax.xaxis.set_major_formatter(lambda ll, pos: rf"$10^{ll}$")


            if not showname:
                # ax.set_yticklabels(["" for _ in labels])
                ax.set_yticks([])

            # finalize plot
            ax.xaxis.grid(linestyle="dashed")
            plt.tight_layout()

            if show:
                plt.show()
            else:
                plot_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(plot_path.with_suffix(f".pdf"), format="pdf", bbox_inches=None, facecolor=None, dpi=300)

            plt.close()

    print("\nFinished plots.\n", flush=True)

    """
    Legend
    """
    fig, legend_ax = plt.subplots(1, 1, figsize=(12.0, 0.5))
    legend_ax.axis("off")
    fig.subplots_adjust(bottom=0.12)

    leg_description = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, alpha=0.0)
    custom_lines = [
        leg_description,
        Patch(facecolor=COLORS[0], edgecolor="black"),
        Patch(facecolor=COLORS[0], edgecolor="black"),
        Patch(facecolor=COLORS[1], edgecolor="black"),
        Patch(facecolor=COLORS[2], edgecolor="black"),
    ]

    custom_lines[1].set_hatch('///')

    leg = legend_ax.legend(
        custom_lines, [
            r"--------------------",
            r"Nonlinear $f_j$",
            r"Acyclic SCM",
            r"Cyclic SCM",
            r"Stationary Diffusion",
        ],
        ncol=5,
        fontsize=14,
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
    )
    leg.get_texts()[0].set_visible(False)

    # shift down misaligned legend text
    for text, item in zip(leg.get_texts(), leg.legendHandles):
        text.set_y(-2.0)
        item.set_y(-2.0)

    legend_ax.text(-0.045, 0.70, r"Method assumptions:",
                   transform=legend_ax.transAxes, fontsize=14,
                   verticalalignment='top', bbox=dict(facecolor=None, alpha=0.0), zorder=1000)

    plot_path = PROJECT_DIR / SUBDIR_RESULTS / folder / "plots" / "legend"

    if show:
        plt.show()
    else:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path.with_suffix(f".pdf"), format="pdf", bbox_inches="tight", facecolor=None, dpi=300)

    plt.close()












