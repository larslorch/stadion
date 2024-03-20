COLOR_SATURATION = 0.8

DPI = 100

LINE_WIDTH = 7.0
COL_WIDTH = 3.333

FIG_SIZE_TRIPLE = (COL_WIDTH / 3, COL_WIDTH / 3 * 4/6)
FIG_SIZE_TRIPLE_TALL = (COL_WIDTH / 3, COL_WIDTH / 3 * 5/6)

FIG_SIZE_DOUBLE = (COL_WIDTH / 2, COL_WIDTH / 2 * 4/6)
FIG_SIZE_DOUBLE_TALL = (COL_WIDTH / 2, COL_WIDTH / 2 * 5/6)

CUSTOM_FIG_SIZE_FULL_PAGE_TRIPLE = (LINE_WIDTH / 3, COL_WIDTH / 2 * 5/6)

FIG_SIZE_FULL_PAGE_TRIPLE = (LINE_WIDTH / 3, LINE_WIDTH / 3 * 4/6)
FIG_SIZE_FULL_PAGE_TRIPLE_TALL = (LINE_WIDTH / 3, LINE_WIDTH / 3 * 5/6)

CUSTOM_FIG_SIZE_FULL_PAGE_QUAD = (LINE_WIDTH / 4, COL_WIDTH / 2 * 5/6)


NEURIPS_LINE_WIDTH = 5.5  # Text width: 5.5in (double figure minus spacing 0.2in).
FIG_SIZE_NEURIPS_DOUBLE = (NEURIPS_LINE_WIDTH / 2, NEURIPS_LINE_WIDTH / 2 * 4/6)
FIG_SIZE_NEURIPS_TRIPLE = (NEURIPS_LINE_WIDTH / 3, NEURIPS_LINE_WIDTH / 3 * 4/6)
FIG_SIZE_NEURIPS_DOUBLE_TALL = (NEURIPS_LINE_WIDTH / 2, NEURIPS_LINE_WIDTH / 2 * 5/6)
FIG_SIZE_NEURIPS_TRIPLE_TALL = (NEURIPS_LINE_WIDTH / 3, NEURIPS_LINE_WIDTH / 3 * 5/6)

NEURIPS_RCPARAMS = {
    "figure.autolayout": True,       # `False` makes `fig.tight_layout()` not work
    "figure.figsize": FIG_SIZE_NEURIPS_DOUBLE,
    # "figure.dpi": DPI,             # messes up figisize
    # Axes params
    "axes.linewidth": 0.5,           # Matplotlib's current default is 0.8.
    "xtick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    "ytick.major.width": 0.5,
    "ytick.minor.width": 0.5,

    "hatch.linewidth": 0.3,
    "xtick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    'xtick.major.pad': 3.0,
    'xtick.major.size': 1.75,
    'xtick.minor.pad': 1.0,
    'xtick.minor.size': 1.0,

    'ytick.major.pad': 1.0,
    'ytick.major.size': 1.75,
    'ytick.minor.pad': 1.0,
    'ytick.minor.size': 1.0,

    "axes.labelpad": 0.5,
    # Grid
    "grid.linewidth": 0.3,
    # Plot params
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    'errorbar.capsize': 3.0,
    # Font
    # "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",             # use serif rather than sans-serif
    "font.serif": "Times New Roman",    # use "Times New Roman" as the standard font
    "font.size": 8.5,
    "axes.titlesize": 8.5,                # LaTeX default is 10pt font.
    "axes.labelsize": 8.5,                # LaTeX default is 10pt font.
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    # Legend
    "legend.fontsize": 7,        # Make the legend/label fonts a little smaller
    "legend.frameon": True,              # Remove the black frame around the legend
    "legend.handletextpad": 0.3,
    "legend.borderaxespad": 0.2,
    "legend.labelspacing": 0.1,
    "patch.linewidth": 0.5,
    # PDF
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
    ),
}

COLORS_0 = [
    "#DF000D", # red
    "#23DD20", # green
    "#1ACBF7", # light blue
    "#FB0D89", # pink
    "#7600DA", # purple
    "#FC6608", # orange
    "#001AFD", # dark blue
]

COLORS_BRIGHT = [
    "#001AFD",  # 0 dark blue
    "#FC6608",  # 1 orange
    "#22C32B",  # 2 green
    "#DF000D",  # 3 red
    "#7600DA",  # 4 purple
    "#8C3703",  # 5 brown
    "#EB2BB4",  # 6 pink
    "#929292",  # 7 grey
    "#FEB910",  # 8 dark yellow
    "#1CCEFF",  # 9 light blue

]

COLORS_NEW = [
    "#e41a1c",
    "#4daf4a",
    "#ffff33",
    "#a65628",
    "#984ea3",
    "#ff7f00",
    "#377eb8",
]

COLORS_3 = [
    "white",
    # "#e41a1c",
    "silver",
    # "darkgrey",
    # "#d90d10",
    # "#d90d10",
    # "#ef476f",
    # "#d7263d",
    # "#377eb8",
    # "#3a7abd",
    # "#2b83ba",
    "#2b69ba",
    # "#0a369d",
]

COLORS = COLORS_3

TRUE = "__true__"


METHODS_CONFIG = [
    # interv
    ("gies",                COLORS[0], True, "GIES"),
    ("igsp",                COLORS[0], True, "IGSP"),
    ("dcdi",                COLORS[0], False, "DCDI"),
    ("llc",                 COLORS[1], True, "LLC"),
    ("nodags",              COLORS[1], False, "NODAGS"),
    # ("ours-linear_u_diag",  COLORS[2], True, r"\textbf{KDS {\small (Linear)}}"),
    # ("ours-lnl_u_diag",     COLORS[2], False, r"\textbf{KDS {\small (MLP)}}"),
    ("ours-linear_u_diag", COLORS[2], True, r"\textbf{KDS (Linear)}"),
    ("ours-lnl_u_diag", COLORS[2], False, r"\textbf{KDS (MLP)}"),
    (TRUE,                  None,      True, "True"),
]

METHODS_DEFAULT_CONFIG = (COLORS[-1], True, "default")

METHODS_NAMES = {k: tup[-1] for k, *tup in METHODS_CONFIG}

METHODS_CONFIG_CALIBRATION = METHODS_CONFIG

METRICS_CHECKS = [
    "inf_train_10",
    "inf_test_10",
    "inf_train_30",
    "inf_test_30",
    "inf_train_100",
    "inf_test_100",
    "nan_train",
    "nan_test",
    "search-0-observ-stable",
    "search-0-insignificant-shift",
    "search-1-success",
    "search-1-intv-stable",
    "search-1-reached",
    "search-2-opt-relative-to-boundary",
    "search-2-neg-mean-error",
    "walltime",
]

METRICS_TABLE_ORDERING = [
    "wasser_train",
    "iwasser_train",
    "wasser_test",
    "iwasser_test",
    "wasser_env_train",
    "iwasser_env_train",
    "wasser_env_test",
    "iwasser_env_test",
    "kde_nll_train",
    "ikde_nll_train",
    "kde_nll_test",
    "ikde_nll_test",
    "mse_train",
    "imse_train",
    "mse_test",
    "imse_test",
    "relmse_train",
    "irelmse_train",
    "relmse_test",
    "irelmse_test",
    "vse_train",
    "ivse_train",
    "vse_test",
    "ivse_test",
    # sanity checks
    *METRICS_CHECKS,
    # old
    "heldout_score",
    "cyclic",
    "search-time_step_0",
    "search-time_step_1",
    "search-time_step_2",
]


# rest round to 3 decimal
METRICS_LOWER_PREC = [
    "walltime"
]

METRICS_NAMING = {
    "wasser_test": r"$W_2$",
    "mse_test": r"MSE",
    "relmse_test": r"relMSE",
}


# metrics
METRICS_HIGHER_BETTER = [
    # sanity checks
    "search-0-observ-stable",
    "search-0-insignificant-shift",
    "search-1-success",
    "search-1-intv-stable",
    "search-1-reached",
    "search-2-opt-relative-to-boundary",
    "search-2-neg-mean-error",
]

METRICS_LOWER_BETTER = [
    "wasser_train",
    "iwasser_train",
    "wasser_test",
    "iwasser_test",
    "wasser_env_train",
    "iwasser_env_train",
    "wasser_env_test",
    "iwasser_env_test",
    "kde_nll_train",
    "ikde_nll_train",
    "kde_nll_test",
    "ikde_nll_test",
    "mse_train",
    "imse_train",
    "mse_test",
    "imse_test",
    "relmse_train",
    "irelmse_train",
    "relmse_test",
    "irelmse_test",
    "vse_train",
    "ivse_train",
    "vse_test",
    "ivse_test",
    "walltime",
    "cyclic",
    "inf_train_10",
    "inf_test_10",
    "inf_train_30",
    "inf_test_30",
    "inf_train_100",
    "inf_test_100",
    "nan_train",
    "nan_test",
    "search-time_step_0",
    "search-time_step_1",
    "search-time_step_2",
]

# TRAIN_VALIDATION_METRICS = ["mse_test"]
# TRAIN_VALIDATION_METRICS = ["mse_test", "mse_train"]
TRAIN_VALIDATION_METRICS = ["wasser_test", "mse_test", "wasser_train", "mse_train"]

MINMIN = -1e3
MAXMAX = 1e3

FIGSIZE = {
    "metric": {"ax_width": 2.0, "ax_height": 6.0},
    "param": {"ax_width": 3.0, "ax_height": 4.0, "grid_spec_adjustment": 1.12},
    "grid": {"size_per_var": 0.7},
    "lowdim": {"ax_width": 3.0, "ax_height": 3.0},
}

CMAP_PARAM = "seismic"

CMAIN = "blue"
CTARGET = "grey"
CREF = "grey"

SCAT_KWARGS = dict(
    marker=".",
    alpha=0.15,
    s=40,
)
SCAT_KWARGS_TAR = dict(
    marker=".",
    alpha=0.20,
    s=40,
)

HIST_KWARGS = dict(
    alpha=0.5,
    histtype="stepfilled",
)
HIST_KWARGS_TAR = dict(
    alpha=0.6,
    histtype="stepfilled",
)


BENCHMARK_Y_DECIMALS = 3

PAIRWISE_GRID_TYPE = ("hist", "kde")
# PAIRWISE_GRID_TYPE = ("kde", "kde")
# PAIRWISE_GRID_TYPE = ("hist", "scatter")
# PAIRWISE_GRID_TYPE = ("kde", "scatter")

GRID_REF_ALPHA = 0.9
GRID_CONTOURS = (0.90,)
GRID_CONTOURS_ALPHAS = (1.0,)

SCOTTS_BW_SCALING = 1.0

DENSITY_1D_KWARGS = dict(linestyle="dashed")
DENSITY_1D_KWARGS_TAR = dict(linestyle="solid")
DENSITY_2D_KWARGS = dict(linestyles="dashed")
DENSITY_2D_KWARGS_TAR = dict(linestyles="solid")

SCAT_KWARGS_PROJ = dict(
    marker=".",
    alpha=0.25,
    s=60,
)

PLOT_SHOW = False










