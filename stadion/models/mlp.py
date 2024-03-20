import functools

import jax
from jax import vmap, random
import jax.numpy as jnp

from stadion.parameters import ModelParameters, InterventionParameters
from stadion.sde import SDE
from stadion.inference import KDSMixin
from stadion.utils import to_diag, tree_global_norm, tree_init_normal, tree_variance_initialization


class MLPSDE(KDSMixin, SDE):
    """
    Nonlinear SDE with shift-scale intervention model.

    The drift function is implemented by ``self.n_vars`` separate
    fully-connected neural networks. The diffusion function
    is a constant diagonal matrix. The interventions shift the drift
    and scale the diffusion matrix.

    Args:
        sparsity_regularizer (str, optional): Type of sparsity regularizer to use.
            Implemented are: ``outgoing,``ingoing``,``both`.`
        hidden_size (int, optional): Hidden size of the neural nets
        activation (str, optional): Activation function of the neural nets
        init_distribution (str, optional): Initialization distribution for the neural net parameters
        init_mode (str, optional): Initialization scheme for the neural net parameters
        sde_kwargs (dict, optional): any keyword arguments passed to ``SDE`` superclass.
    """
    
    def __init__(
        self,
        sparsity_regularizer="both",
        hidden_size=8,
        activation="sigmoid",
        init_distribution="uniform",
        init_mode="fan_in",
        sde_kwargs=None,
    ):

        sde_kwargs = sde_kwargs or {}
        SDE.__init__(self, **sde_kwargs)

        self.sparsity_regularizer = sparsity_regularizer

        if activation == "tanh":
            self.nonlin = jnp.tanh
        elif activation == "relu":
            self.nonlin = jax.nn.relu
        elif activation == "sigmoid":
            self.nonlin = jax.nn.sigmoid
        elif activation == "rbf":
            self.nonlin = lambda arr: jnp.exp(- jnp.power(arr, 2))
        else:
            raise KeyError(f"Unknown activation {activation}")

        self.hidden_size = hidden_size
        self.init_distribution = init_distribution
        self.init_mode = init_mode


    def init_param(self, key, d, scale=0.001, fix_speed_scaling=True):
        """
        Samples random initialization of the SDE model parameters.
        See :func:`~stadion.inference.KDSMixin.init_param`.
        """
        # layers should follow convention fan_in -> fan_out for proper initialization (taking into account vmap below)
        shape = {
            "mlp_0": jnp.zeros((d, d, self.hidden_size)),
            "mlp_b_0": jnp.zeros((d, self.hidden_size)),
            "mlp_1": jnp.zeros((d, self.hidden_size)),
            "mlp_b_1": jnp.zeros((d,)),
            "log_reversion": jnp.zeros((d,)),
            "log_noise_scale": jnp.zeros((d,)),
        }

        _initializer = functools.partial(tree_variance_initialization, scale=scale, mode=self.init_mode,
                                         distribution=self.init_distribution)
        param = vmap(_initializer, (0, 0), 0)(jnp.array(random.split(key, d)), shape)

        if fix_speed_scaling:
            return ModelParameters(
                parameters=param,
                fixed={"mlp_0": jnp.diag_indices(d),
                       "log_reversion": ...},
                fixed_values={"mlp_0": 0.0,
                              "log_reversion": 0.0},
            )
        else:
            return ModelParameters(
                parameters=param,
            )



    def init_intv_param(self, key, d, n_envs=None, scale=0.001, targets=None, x=None):
        """
        Samples random initialization of the intervention parameters.
        See :func:`~stadion.inference.KDSMixin.init_intv_param`.
        """
        # pytree of [n_envs, d, ...]
        # intervention effect parameters
        vec_shape = (n_envs, d) if n_envs is not None else (d,)
        shape = {
            "shift": jnp.zeros(vec_shape),
            "log_scale": jnp.zeros(vec_shape),
        }
        intv_param = tree_init_normal(key, shape, scale=scale)

        # if provided, store intervened variables for masking
        if targets is not None:
            targets = jnp.array(targets, dtype=jnp.float32)
            assert targets.shape == vec_shape

        # if provided, warm-start intervention shifts of the target variables
        if x is not None and targets is not None and n_envs is not None:
            assert len(x) == n_envs
            assert all([data.shape[-1] == d] for data in x)
            ref = x[0].mean(-2)
            mean_shift = jnp.array([jnp.where(targets_, (x_.mean(-2) - ref), jnp.array(0.0)) for x_, targets_ in zip(x, targets)])
            intv_param["shift"] += mean_shift

        return InterventionParameters(parameters=intv_param, targets=targets)


    """
    Model
    """

    def f_j(self, x, param, one_hot_j):
        hid = self.nonlin(jnp.einsum("...d,dh->...h", x, param["mlp_0"]) + param["mlp_b_0"])
        out = jnp.einsum("...h,h->...", hid, param["mlp_1"]) + param["mlp_b_1"]
        reversion = jnp.einsum("...h,h->...", x, one_hot_j) * jnp.exp(param["log_reversion"])
        return out - reversion


    def f(self, x, param, intv_param):
        """
        Nonlinear drift :math:`f(\\cdot)` with shift-scale intervention model.
        See :func:`~stadion.sde.SDE.f`.
        """
        d = x.shape[-1]

        # compute drift scalar f(x)_j for each dx_j using the input x
        f_vec = vmap(self.f_j, in_axes=(None, 0, 0), out_axes=-1)(x, param, jnp.eye(d))
        assert x.shape == f_vec.shape

        # intervention: shift f(x) by scalar
        # [d,]
        if intv_param is not None:
            f_vec += intv_param["shift"]

        assert x.shape == f_vec.shape
        return f_vec


    def sigma(self, x, param, intv_param):
        """
        Diagonal constant :math:`\\sigma(\\cdot)` with shift-scale intervention model
        See :func:`~stadion.sde.SDE.sigma`.
        """
        d = x.shape[-1]

        # compute sigma(x)
        c = jnp.exp(param["log_noise_scale"])
        sig_mat = to_diag(jnp.ones_like(x)) * c
        assert sig_mat.shape == (*x.shape, x.shape[-1])

        # intervention: scale sigma by scalar
        # [d,]
        if intv_param is not None:
            scale = jnp.exp(intv_param["log_scale"])
            sig_mat = jnp.einsum("...ab,a->...ab", sig_mat, scale)

        assert sig_mat.shape == (*x.shape, d)
        return sig_mat


    """
    Inference functions
    """

    def modify_dparam(self, dparam):
        """
        Modifies the gradient of the parameters of the SDE model if speed ``self.fix_speed_scaling`` is ``True``.

        Args:
            dparam (pytree): gradient of the parameters of the SDE model

        Returns:
            Modified gradient
        """

        if not self.fix_speed_scaling:
            return dparam

        else:
            dparam["mlp_0"] = dparam["mlp_0"].at[jnp.diag_indices(self.n_vars)].set(0.0)
            dparam["log_reversion"] = dparam["log_reversion"].at[...].set(0.0)
            return dparam


    @staticmethod
    def _regularize_ingoing(param):
        """
        Group LASSO regularization term that L1-penalizes (sparsifies) ingoing causal dependencies
        See :func:`~stadion.inference.KDSMixin.regularize_sparsity`.
        """
        d = param["log_reversion"].shape[0]

        def group_lasso_j(param_j, j):

            # [d,] compute L2 norm for each group (each causal parent)
            group_lasso_terms_j = vmap(functools.partial(tree_global_norm, p=2.0), 0, 0)(param_j["mlp_0"])

            # mask self-influence
            group_lasso_terms_j = jnp.where(jnp.eye(d)[j], 1e-20, group_lasso_terms_j)

            # [] compute Lp group lasso (in classical group lasso, p=1)
            lasso = tree_global_norm(group_lasso_terms_j, p=1.0)
            return lasso

        # group lasso: for each causal mechanism
        # [d,] for each first layer parameter touching x
        reg = vmap(group_lasso_j, (0, 0), 0)(param, jnp.arange(d)).sum(0)
        return reg


    @staticmethod
    def _regularize_outgoing(param):
        """
        Group LASSO regularization term that L1-penalizes (sparsifies) outgoing causal dependencies
        See :func:`~stadion.inference.KDSMixin.regularize_sparsity`.
        """
        d = param["log_reversion"].shape[0]

        # group lasso that groups not by causal mechanism but by outgoing dependencies (masking self-influence)
        # group all first layer parameters by which x_j they touch (i.e. linear and nonlinear parts)
        param_grouped = param["mlp_0"]

        # mask self-influence
        param_grouped = param_grouped.at[jnp.diag_indices(d)].multiply(0.0)

        # [d,] compute L2 norm for each group (axis 1 since we group by outgoing dependencies)
        groupf = vmap(functools.partial(tree_global_norm, p=2.0), 1, 0)
        group_lasso_terms_j = groupf(param_grouped)

        # [] compute Lp group lasso (in classical group lasso, p=1) and scale by number of variables
        lasso = tree_global_norm(group_lasso_terms_j, p=1.0) / d
        return lasso


    def regularize_sparsity(self, param):
        """
        Sparsity regularization.
        See :func:`~stadion.inference.KDSMixin.regularize_sparsity`.
        """
        if self.sparsity_regularizer == "ingoing":
            reg = MLPSDE._regularize_ingoing(param)
        elif self.sparsity_regularizer == "outgoing":
            reg = MLPSDE._regularize_outgoing(param)
        elif self.sparsity_regularizer == "both":
            reg = MLPSDE._regularize_ingoing(param)\
                + MLPSDE._regularize_outgoing(param)
        else:
            raise ValueError(f"Unknown regularizer `{self.regularize_sparsity}`")
        return reg
