import warnings
from matplotlib import MatplotlibDeprecationWarning

import math
import copy

from jax import random, vmap, grad, hessian, jit
import jax.numpy as jnp

from stadion.utils.metrics import rbf_kernel as rbf_kernel

import matplotlib.pyplot as plt
import scipy.stats as st

warnings.simplefilter(action='ignore', category=MatplotlibDeprecationWarning)

from matplotlib.lines import Line2D
import matplotlib.collections as mcol
from stadion.utils.legend import HandlerDashedLines
from stadion.definitions import PROJECT_DIR
from matplotlib.patches import Rectangle
import numpy as onp

import matplotlib as mpl

NEURIPS_RCPARAMS = {
    "text.usetex": True,                # use LaTeX to write all text
    # "font.family": "serif",             # use serif rather than sans-serif
    'font.family': 'lmodern',
    # "font.serif": "Times New Roman",    # use "Times New Roman" as the standard font
    "font.size": 13,
    "pgf.texsystem": "xelatex",         # use Xelatex which is TTF font aware
    "pgf.rcfonts": False,               # Use pgf.preamble, ignore standard Matplotlib RC
    "pgf.preamble": (
        r'\usepackage{fontspec}'
        r'\usepackage{unicode-math}'
        r'\setmainfont{Times New Roman}'
    ),
    "text.latex.preamble": (
        r'\usepackage{amsfonts}'
        r'\usepackage{amsmath}'
        r'\usepackage{amssymb}'
        r'\usepackage{wasysym}'
        r'\usepackage{graphicx}'
        r'\usepackage{nicefrac}'
        r'\usepackage{xfrac}'
    ),
}

mpl.rcParams.update(NEURIPS_RCPARAMS)


colors = ["#e41a1c", "#377eb8"]

double_vmap = lambda h: vmap(vmap(h, (None, 0, None)), (0, None, None))


def get_param(pa, pb, pc):
    assert pb < 0
    return (- pa / pb), math.sqrt(- (pc ** 2) / (2 * pb))


def f(x, theta):
    return theta["a"] + x * theta["b"]


def sigma(x, theta):
    if logc_mode:
        return jnp.exp(theta["c"]) * jnp.ones_like(x)
    else:
        return theta["c"] * jnp.ones_like(x)


def kernel(x, y, _, ls=0.666):
    return rbf_kernel(x[..., None], y[..., None], ls)


def op(h, arg):
    def op_h(*z):
        return f(z[arg], z[2]) * grad(h, arg)(*z) + 0.5 * (sigma(z[arg], z[2]) ** 2) * hessian(h, arg)(*z)
    return op_h


op_kernel = double_vmap(op(kernel, 0))
op_op_kernel = double_vmap(op(op(kernel, 0), 1))


@jit
def kds(theta, data):
    kds_value = jnp.mean(op_op_kernel(data, data, theta), axis=(0, 1))
    return kds_value


dtheta_kds = jit(vmap(grad(kds, 0), (0, None)))


@jit
def witness(x, theta, data):
    witness_values = jnp.mean(op_kernel(x, data, theta), axis=1)
    rkhs_norm = jnp.sqrt(kds(theta, data))
    return witness_values / rkhs_norm


@jit
def integrand_wo_mu(x, theta, data):
    return jnp.mean(op_op_kernel(x, data, theta), axis=1)


if __name__ == "__main__":

    jnp.set_printoptions(precision=4, suppress=True)
    key = random.PRNGKey(0)

    plot_path = PROJECT_DIR / "witness_main"

    a = 4
    b = -4
    c = 2

    logc_mode = True
    # logc_mode = False

    # show = True
    show = False

    xmin, xmax = -3, 5
    xrange = jnp.linspace(xmin, xmax, 200)

    """
    Closed-form Gaussian stationary density
    """
    a_delta = -3
    c_delta = -1

    theta_true = dict(a=jnp.array(a, dtype=jnp.float32),
                      b=jnp.array(b, dtype=jnp.float32),
                      c=jnp.array(c, dtype=jnp.float32))

    theta_a = dict(a=jnp.array(a + a_delta, dtype=jnp.float32),
                   b=jnp.array(b, dtype=jnp.float32),
                   c=jnp.array(c, dtype=jnp.float32))

    theta_c = dict(a=jnp.array(a, dtype=jnp.float32),
                   b=jnp.array(b, dtype=jnp.float32),
                   c=jnp.array(c + c_delta, dtype=jnp.float32))


    mu = st.norm.pdf(xrange, *get_param(a, b, c))
    mu_mean, mu_scale = get_param(a, b, c)

    key, subk = random.split(key)
    mu_data = random.normal(subk, shape=(1000,)) * mu_scale + mu_mean

    if logc_mode:
        theta_true["c"] = jnp.log(theta_true["c"])
        theta_a["c"] = jnp.log(theta_a["c"])
        theta_c["c"] = jnp.log(theta_c["c"])

    """
    ##################################################################
    Witness function plots for wrong thetas
    """

    # plot
    width_size = 2.35
    height_size = 3.1
    fig, axs = plt.subplots(1, 5, figsize=(5 * width_size, 1 * height_size))

    axs[0].plot(xrange, mu, color="black", ls="dashed")
    axs[0].plot(xrange, st.norm.pdf(xrange, *get_param(a + a_delta, b, c)), c=colors[0])
    axs[0].plot(xrange, st.norm.pdf(xrange, *get_param(a, b, c + c_delta)), c=colors[1])
    axs[0].set_title("pdf$(x)$")
    axs[0].set_xlabel("$x$")
    axs[0].set_ylim((0, axs[0].get_ylim()[1]))

    print(f"kds true: {kds(theta_true, mu_data)}")
    print(f"kds theta_a: {kds(theta_a, mu_data)}")
    print(f"kds theta_c: {kds(theta_c, mu_data)}")

    # axs[1].plot(xrange, witness(xrange, theta_true, mu_data), c="black", ls="dashed")
    axs[1].plot(xrange, witness(xrange, theta_a, mu_data), c=colors[0])
    axs[1].plot(xrange, witness(xrange, theta_c, mu_data), c=colors[1])
    axs[1].set_title(r"Witness $w_{\mu,\mathcal{L}}(x)$")

    integrand_a = mu * integrand_wo_mu(xrange, theta_a, mu_data)
    integrand_c = mu * integrand_wo_mu(xrange, theta_c, mu_data)

    # axs[2].plot(xrange, mu * integrand_wo_mu(xrange, theta_true, mu_data), c="black", ls="dashed")
    axs[2].plot(xrange, integrand_a, c=colors[0])
    axs[2].plot(xrange, integrand_c, c=colors[1])
    axs[2].fill_between(xrange, integrand_a, color=colors[0], alpha=0.25)
    axs[2].fill_between(xrange, integrand_c, color=colors[1], alpha=0.25)

    # axs[2].set_title(r"Integrand $\mu(x) \mathbb{E}_{x' \sim \mu}[\mathcal{L}_x \mathcal{L}_{x'} k(x, x')] $")
    # axs[2].set_title(r"$\mu(x) \mathcal{L} g_{\mu,\mathcal{L}}(x) / \lVert g_{\mu,\mathcal{L}} \rVert_\mathcal{H}  $")
    axs[2].set_title(r"$\mu(x) (\mathcal{L} w_{\mu,\mathcal{L}})(x)  $")

    axs[0].set_xlim((xmin, xmax))
    axs[1].set_xlim((xmin, xmax))
    axs[2].set_xlim((xmin, xmax))

    axs[1].set_xlabel("$x$")
    axs[2].set_xlabel("$x$")
    axs[1].axhline(0, c="black", linewidth=0.8)
    axs[2].axhline(0, c="black", linewidth=0.8)

    axs[0].set_xticks([-2, 0, 2, 4])
    axs[1].set_xticks([-2, 0, 2, 4])
    axs[2].set_xticks([-2, 0, 2, 4])

    """
    KDS gradients
    """
    n = 50
    grid_a = a + jnp.linspace(-5, 5, num=n)
    if logc_mode:
        grid_c = jnp.log(jnp.linspace(0.55, 2.55, num=n))
    else:
        grid_c = jnp.linspace(0.00, 2.65, num=n)

    theta_grid = dict(a=jnp.ones(n, dtype=jnp.float32) * a,
                      b=jnp.ones(n, dtype=jnp.float32) * b,
                      c=jnp.ones(n, dtype=jnp.float32) * c)

    thetas_a = copy.deepcopy(theta_grid)
    thetas_a["a"] = grid_a
    axs[3].plot(thetas_a["a"], dtheta_kds(thetas_a, mu_data)["a"], c="black")
    axs[3].axvline(a, ls="dashed", c="black")
    axs[3].axvline(a + a_delta, ls="dashed", c=colors[0])
    axs[3].axhline(0, c="black", linewidth=0.8)

    thetas_c = copy.deepcopy(theta_grid)
    thetas_c["c"] = grid_c
    axs[4].axhline(0, c="black", linewidth=0.8)
    if logc_mode:
        axs[4].plot(thetas_c["c"], dtheta_kds(thetas_c, mu_data)["c"], c="black")
        axs[4].axvline(onp.log(c), ls="dashed", c="black")
        axs[4].axvline(onp.log(c + c_delta), ls="dashed", c=colors[1])
    else:
        axs[4].plot(thetas_c["c"], dtheta_kds(thetas_c, mu_data)["c"], c="black")
        axs[4].axvline(c, ls="dashed", c="black")
        axs[4].axvline(c + c_delta, ls="dashed", c=colors[1])

    axs[3].set_title(r"$\partial\hspace{-1pt} / \hspace{-1pt}\partial a \, \mathrm{KDS}(\mathcal{L}, \mu; \mathcal{F})$")
    if logc_mode:
        title = axs[4].set_title(r"$\partial \hspace{-1pt} / \hspace{-1pt} \partial \hspace{1pt} \mathrm{ln} (c) \, \mathrm{KDS}(\mathcal{L}, \mu; \mathcal{F})$")
        title.set_position([0.48, 1.0])
    else:
        axs[4].set_title(r"$\partial\hspace{-1pt} /\hspace{-1pt} \partial c \, \mathrm{KDS}(\mathcal{L}, \mu; \mathcal{F})$")

    axs[3].set_xlabel(r"$a$")
    if logc_mode:
        axs[4].set_xlabel(r"$\mathrm{ln} (c)$")
    else:
        axs[4].set_xlabel(r"$c$")

    axs[3].set_xlim((thetas_a["a"].min(), thetas_a["a"].max()))
    axs[4].set_xlim((thetas_c["c"].min(), thetas_c["c"].max()))

    axs[3].set_xticks([0, 2, 4, 6, 8])
    if logc_mode:
        axs[4].set_xticks([-0.5, 0.0, 0.5])
    else:
        axs[4].set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])

    """
    Legend
    """

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.32)

    legend_ax = fig.add_axes([0.1, 0.0, 0.5, 0.1])
    legend_ax.axis("off")

    leg_description = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, alpha=0.0)
    lcmu = mcol.LineCollection([[(0, 0)]], linestyles=["dashed"], colors=["black"], lw=1.22)
    lc0 = mcol.LineCollection([[(0, 0)], [(0, 0)]], linestyles=["-", "dashed"], colors=[colors[0], colors[0]], lw=1.22)
    lc1 = mcol.LineCollection([[(0, 0)], [(0, 0)]], linestyles=["-", "dashed"], colors=[colors[1], colors[1]], lw=1.22)

    custom_lines = [
        leg_description,
        lcmu,
        lc0,
        lc1,
    ]

    leg = legend_ax.legend(
        custom_lines, [
            r"----------------------------",
            r"$\mu(x)$: $a = 4, c = 2$",
            "Wrong shift: $\mathbf{a = 1}, c = 2$",
            "Wrong scale: $a = 4, \mathbf{c = 1}$",
        ],
        loc=(-0.12, 0.2),
        handler_map={type(lc0): HandlerDashedLines()},
        fontsize=14,
        labelspacing=0.8,
        handlelength=2.1,
        ncol=4,
        handleheight=2.0,
        handletextpad=0.8,
        borderpad=0.2,
    )

    leg.get_texts()[0].set_visible(False)

    legend_ax.text(-0.08, 1.07, r"$dx_t = (a - 4 x_t)dt + c \, d\mathbb{W}_t:$",
                   transform=legend_ax.transAxes, fontsize=14,
                   verticalalignment='top', bbox=dict(facecolor=None, alpha=0.0), zorder=1000)


    if show:
        plt.show()
    else:
        plt.savefig(((plot_path.parent / (plot_path.name + "_logc")) if logc_mode else plot_path).with_suffix(f".pdf"),
                    format="pdf", bbox_inches="tight", facecolor=None, dpi=200)
        plt.close()
