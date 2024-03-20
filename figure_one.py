import warnings
from matplotlib import MatplotlibDeprecationWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=MatplotlibDeprecationWarning)
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

from mpl_toolkits.mplot3d import art3d
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap

import numpy as np
from jax import numpy as jnp, random

from stadion.definitions import PROJECT_DIR

"""
Code in parts from 
https://github.com/probml/pyprobml/blob/master/deprecated/scripts/pyprobml_utils.py
https://github.com/probml/pyprobml/blob/master/notebooks/book2/12/mcmc_gmm_demo.ipynb
"""

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

mpl.rcParams.update(NEURIPS_RCPARAMS)

def univariate_kde(x, x_test, h=None):
    """
    Univariate KDE under a Gaussian kernel
    Args:
        x: [..., n]
        x_test: [..., n_eval]
        h: float
    Returns:
        [..., n_eval] KDE around the observed values
    """
    *b, n = x.shape
    *b, n_test = x_test.shape
    if h is None:
        # scott's rule
        h = float(n) ** float(-0.5)

    x_hat = x.reshape(*b, 1, n)
    x_test_hat = x_test.reshape(*b, n_test, 1)
    u = (x_test_hat - x_hat) ** 2 / (2 * h ** 2)
    density_test = np.exp(- u).sum(axis=-1) / (n * h * np.sqrt(2 * np.pi))
    return density_test


def scale_3d(ax, x_scale, y_scale, z_scale, factor):
    # careful: `factor` can move plotted regions outside of bounding box,
    # resulting in sides being cropped off despite axis being plotted there
    # add `axs.set_facecolor(None)` for debugging the bounding box
    scale_vec = np.diag([x_scale, y_scale, z_scale, 1.0])
    scale_vec = scale_vec / scale_vec.max()
    scale_vec[3, 3] = factor

    def short_proj():
        return np.dot(Axes3D.get_proj(ax), scale_vec)

    return short_proj


def custom_colormap(colors, segments=None):
    segments = segments or [0, 1]
    assert len(colors) == len(segments)
    return LinearSegmentedColormap.from_list('custom_cmap', list(zip(segments, colors)))


def custom_alpha_colormap(color, from_alpha, to_alpha):
    color_cmap = custom_colormap(color, color)
    my_cmap = color_cmap(np.arange(color_cmap.N))

    # set alpha
    my_cmap[:, -1] = np.linspace(from_alpha, to_alpha, color_cmap.N)
    return ListedColormap(my_cmap)


def color_line(ax, x, y, color_values, cmap=None, linewidth=1.0, alpha=1.0):
    """
    https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/multicolored_line.html
    """
    assert x.ndim == y.ndim == color_values.ndim ==1
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(color_values.min(), color_values.max())
    lc = LineCollection(segments, cmap=cmap or "viridis", norm=norm)

    # Set the values used for colormapping
    lc.set_array(color_values)
    lc.set_linewidth(linewidth)
    lc.set_alpha(alpha)
    lc.set_zorder(-1)

    # lc.set_array(color_values)
    # lc.set_linewidth(1.5)
    # lc.set_zorder(-1)
    # lc.set_linestyle([(0.0, [1.0, 10.0])]) # dash: offset, [(]length, space, ...]
    #

    # line_object = ax.add_collection(lc)
    line_object = ax.add_collection3d(lc, zs=0, zdir='z')

    # fig.colorbar(line_object, ax=axs[0])
    return line_object


def sample_figure_data(sd, n_samples):
    from stadion.core import sample_dynamical_system
    from argparse import Namespace
    from stadion.models.linear import f, sigma

    config = Namespace()
    config.method = "explicit"
    config.rollout_restarts = 1
    config.n_samples_burnin = n_samples // 2
    config.thinning = 1
    config.dt = 0.01

    # shift scale
    theta = dict(
        w1=np.array([
            [-2, -1],
            [ 2, -1],
        ]),
        b1=np.array([ 6, -6]),
    )

    intv_theta = dict(
        shift=np.array([
            [0, 0],
            [-2, 0]
        ]),
        scale=np.array([
            [0, 0],
            [np.log(0.5), 0],
        ]),
    )

    intv_msks = np.array([
        [0, 0],
        [1, 0],
    ])

    # simulate SDE
    _, trajectory, _ = sample_dynamical_system(
        random.PRNGKey(sd), theta, intv_theta, intv_msks,
        n_samples=n_samples,
        config=config,
        f=f, sigma=sigma,
        return_traj=True,
    )
    trajectory = trajectory[:, 0, :, :] # select first rollout
    return np.array(trajectory)


if __name__ == "__main__":

    seed = 7
    t_max_sampling = 100000
    t_max = 1000

    bandwidth = 0.2
    xmin, xmax = -4.0, 4.5

    plot_path = PROJECT_DIR / "fig1"
    n_eval_points = 300

    # FIG_SIZE = (5.0, 4.0)
    FIG_SIZE = (6.0, 4.0)
    FONT_SIZE_SMALL = 16
    FONT_SIZE_MEDIUM = 14
    FONT_SIZE_LARGE = 24

    # colors = [
    #     # "#4daf4a", # green
    #     "#377eb8", # blue
    #     "#e41a1c", # red
    # ]

    colors_dark = [
        "#377eb8", # blue
        "#e41a1c", # red
    ]

    # corresponds to dark colors with alpha=0.2, measured back from screen with color picker
    colors_light = [
        "#B7CCE2", # blue
        "#F4ACAC", # red
    ]

    dashed_line_bottom_thres = 0.001

    # sample data
    traj = sample_figure_data(seed, t_max_sampling)
    x_samples = traj[..., ::1, :]
    traj = traj[..., -t_max:, :]

    # traj *= np.linspace(0.0, 1.0, num=t_max)[None, :, None] # trajectory direction debugging

    n_vars = traj.shape[-1]

    print("x_samples: ", x_samples.shape)
    print("traj:      ", traj.shape)

    np.set_printoptions(precision=3, suppress=True)
    print(x_samples.mean(-2))
    print(x_samples.std(-2))

    """
    ========= Plot ==========
    """
    # plot
    fig = plt.figure(figsize=FIG_SIZE)
    # fig.set_panecolor(None)

    axs = plt.axes(projection="3d", computed_zorder=False)
    axs.view_init(elev=25, azim=-25, roll=-1.5)

    zmax = -np.inf
    assert len(colors_dark) == n_vars
    for j in range(2):
        # compute KDE
        eps = 0.15
        x_eval = np.tile(np.linspace(xmin - eps, xmax + eps, n_eval_points), (n_vars, 1)).T
        kde_eval = univariate_kde(x_samples[j].T, x_eval.T, h=bandwidth).T
        assert x_eval.shape == kde_eval.shape
        zmax = np.maximum(zmax, kde_eval.max())
        time = np.arange(t_max)

        # trajectories
        # only observational
        for line, col_dark, col_light in zip(traj[j].T, colors_dark, colors_light):
            if j == 0:
                color_line(axs, time - t_max, line, time, linewidth=1.0,
                           cmap=custom_colormap(["white", col_light, col_light], [0, 0.5, 1]))

            else:
                color_line(axs, time - t_max, line, time, linewidth=1.4,
                           cmap=custom_colormap(["white", col_dark, col_dark], [0, 0.5, 1]))

        # KDEs
        for i, (x_line, f_line, col_dark, col_light) in enumerate(zip(x_eval.T, kde_eval.T, colors_dark, colors_light)):
            if j == 0:
                # axs.add_collection3d(
                #     plt.fill_between(x_line, 0, f_line, color=col_light, linewidth=0, zorder=-11), 1, zdir="x")
                axs.add_collection3d(
                    plt.fill_between(x_line, 0, f_line, color=col_dark, alpha=0.20, linewidth=0), 1, zdir="x")
            else:
                idx_above_thres = jnp.where(f_line >= dashed_line_bottom_thres)

                axs.plot(x_line[idx_above_thres], f_line[idx_above_thres], "--", c=col_dark, linewidth=1.7, zdir="x", zorder=100)

    # axs.set_box_aspect([1.0, 1.5, 0.8], zoom=1.0)
    axs.set_box_aspect([1.0, 1.85, 0.85], zoom=1.0)

    # style 3D
    plt.gca().patch.set_facecolor('white')
    axs.xaxis.set_pane_color((0, 0, 0, 0))
    axs.yaxis.set_pane_color((0, 0, 0, 0))
    axs.zaxis.set_pane_color((0, 0, 0, 0))


    # axes and labels
    axs.xaxis.set_rotate_label(False) # to be able to rotate the label
    axs.yaxis.set_rotate_label(False) # to be able to rotate the label
    axs.zaxis.set_rotate_label(False) # to be able to rotate the label
    axs.set_xlabel(r"$t$", fontsize=None, rotation=10, labelpad=-13)
    axs.set_ylabel(r"$x_j$", fontsize=None, labelpad=0, rotation=0)
    axs.set_zlabel(r"$p(x_j)$", fontsize=None, rotation=0, labelpad=-8)
    # axs.tick_params(labelsize=None)

    axs.set_xticks([])  # remove time ticks
    axs.set_xticklabels([]) # hide time ticks


    axs.set_yticks([-3, 0, 3], [-3, 0, 3]) # x space ticks customized


    axs.set_zticks([0.2 * zmax, 0.4 * zmax, 0.6 * zmax, 0.8 * zmax])  # custom p(x) ticks
    axs.set_zticklabels([]) # hide p(x) tick labels

    axs.tick_params(axis='x', pad=0, length=0)
    axs.tick_params(axis='y', pad=-3, direction='out')
    axs.tick_params(axis='z', pad=0, direction='out')

    # hide time ticks
    for line in axs.xaxis.get_ticklines():
        line.set_visible(False)

    # position of ticks
    axs.zaxis.set_ticks_position('both')  # Set ticks to appear on both sides

    # change positions of z-axis (right to left)
    pp = axs.zaxis._PLANES
    axs.zaxis._PLANES = (pp[1], pp[0], pp[4], pp[5], pp[2], pp[3])

    axs.set_xlim((-t_max, 0))
    axs.set_ylim((xmin, xmax))
    axs.set_zlim((0, zmax * 1.01))

    # grid lines
    axs.xaxis.gridlines.set_zorder(100)
    axs.yaxis.gridlines.set_zorder(100)
    axs.zaxis.gridlines.set_zorder(100)
    axs.xaxis.gridlines.set_alpha(0.0)
    axs.zaxis.gridlines.set_alpha(0.0)
    # axs.grid(False)
    axs.invert_xaxis()

    axs.xaxis.set_zorder(100)
    axs.yaxis.set_zorder(100)
    axs.zaxis.set_zorder(100)

    # legend
    legend_lines = [Line2D([0], [0], color=colors_dark[0], lw=2),
                    Line2D([0], [0], color=colors_dark[1], lw=2)]

    show = False
    # show = True

    fig.legend(legend_lines,
               # ["$x_{1}$", r"$x_{2}$"],
               ["$x_{1}\leftarrow$", r"$x_{2}$"],
               # loc=(0.7, 0.58) if show else (0.83, 0.69),
               loc=(0.7, 0.58) if show else (0.80, 0.69), # with arrow
               # loc=(0.83, 0.69), # save
               handlelength=1.0,
               columnspacing=1.5,
               handletextpad=0.5,
               frameon=False)

    # axs.text(2.5, 10.5, 0.85, r"\lightning", (0, 0, 0), color='red', zorder=10, fontsize=30)

    # to align grid lines with axes
    # https://stackoverflow.com/questions/53906234/merge-grid-lines-and-axis-in-matplotlib-3d-plot

    # font sizes
    plt.rc('font', size=FONT_SIZE_SMALL)  # controls default text sizes
    plt.rc('axes', titlesize=FONT_SIZE_SMALL)  # fontsize of the axes title
    plt.rc('axes', labelsize=FONT_SIZE_SMALL)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=FONT_SIZE_SMALL)   # fontsize of the tick labels
    plt.rc('ytick', labelsize=FONT_SIZE_SMALL)  # fontsize of the tick labels
    plt.rc('legend', fontsize=FONT_SIZE_SMALL)  # legend fontsize
    plt.rc('figure', titlesize=FONT_SIZE_LARGE)  # fontsize of the figure title

    if show:
        plt.show()
    else:
        # crop white space when saving pdf
        # x0, y0, width, height
        # bbox = fig.bbox_inches.from_bounds(0.65, 0.65, FIG_SIZE[0] - 1.48, FIG_SIZE[1] - 1.60) # with arrow
        bbox = fig.bbox_inches.from_bounds(1.10, 0.77, FIG_SIZE[0] - 1.90, FIG_SIZE[1] - 1.80) # with arrow

        plt.savefig(plot_path.with_suffix(f".pdf"), format="pdf", bbox_inches=bbox, facecolor=None, dpi=200)
        plt.close()

