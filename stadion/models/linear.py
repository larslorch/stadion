import functools

from jax import vmap
import jax.numpy as jnp

from stadion.parameters import ModelParameters, InterventionParameters
from stadion.sde import SDE
from stadion.inference import KDSMixin
from stadion.utils import to_diag, tree_global_norm, tree_init_normal


class LinearSDE(KDSMixin, SDE):
    """
    Linear SDE with shift-scale intervention model.

    The drift function is a linear function and the diffusion function
    is a constant diagonal matrix. The interventions shift the drift and
    scale the diffusion matrix.

    Args:
        sparsity_regularizer (str, optional): Type of sparsity regularizer to use.
            Implemented are: ``outgoing,``ingoing``,``both`.`
        sde_kwargs (dict, optional): any keyword arguments passed to ``SDE`` superclass.

    """

    def __init__(
        self,
        sparsity_regularizer="both",
        sde_kwargs=None,
    ):

        sde_kwargs = sde_kwargs or {}
        SDE.__init__(self, **sde_kwargs)

        self.sparsity_regularizer = sparsity_regularizer


    def init_param(self, key, d, scale=0.001, fix_speed_scaling=True):
        """
        Samples random initialization of the SDE model parameters.
        See :func:`~stadion.inference.KDSMixin.init_param`.
        """
        shape = {
            "weights": jnp.zeros((d, d)),
            "biases": jnp.zeros((d,)),
            "log_noise_scale": jnp.zeros((d,)),
        }
        param = tree_init_normal(key, shape, scale=scale)

        if fix_speed_scaling:
            return ModelParameters(
                parameters=param,
                fixed={"weights": jnp.diag_indices(d)},
                fixed_values={"weights": -1.0},
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

    def f_j(self, x, param):
        w = param["weights"]
        b = param["biases"]
        assert w.shape[0] == x.shape[-1] and w.ndim == 1 and b.ndim == 0
        return x @ w + b


    def f(self, x, param, intv_param):
        """
        Linear drift :math:`f(\\cdot)` with shift-scale intervention model.
        See :func:`~stadion.sde.SDE.f`.
        """
        # compute drift scalar f(x)_j for each dx_j using the input x
        f_vec = vmap(self.f_j, in_axes=(None, 0), out_axes=-1)(x, param)
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

    @staticmethod
    def _regularize_ingoing(param):
        """
        Group LASSO regularization term that L1-penalizes (sparsifies) ingoing causal dependencies
        See :func:`~stadion.inference.KDSMixin.regularize_sparsity`.
        """
        d = param["weights"].shape[0]

        def group_lasso_j(param_j, j):
            # [d,] compute L2 norm for each group (each causal parent)
            group_lasso_terms_j = vmap(functools.partial(tree_global_norm, p=2.0), 0, 0)(param_j["weights"])

            # mask self-influence
            group_lasso_terms_j = jnp.where(jnp.eye(d)[j], 1e-20, group_lasso_terms_j)

            # [] compute Lp group lasso (in classical group lasso, p=1)
            lasso = tree_global_norm(group_lasso_terms_j, p=1.0)
            return lasso

        # group lasso for each causal mechanism
        reg_w1 = vmap(group_lasso_j, (0, 0), 0)(param, jnp.arange(d)).mean(0)
        return reg_w1


    @staticmethod
    def _regularize_outgoing(param):
        """
        Group LASSO regularization term that L1-penalizes (sparsifies) outgoing causal dependencies
        See :func:`~stadion.inference.KDSMixin.regularize_sparsity`.
        """
        d = param["weights"].shape[0]

        # group lasso that groups not by causal mechanism but by outgoing dependencies (masking self-influence)
        # [d,] compute L2 norm for each group (axis = 1 since we want to group by outgoing dependencies)
        groupf = vmap(functools.partial(tree_global_norm, p=2.0), 1, 0)
        group_lasso_terms_j = groupf(param["weights"] * (1 - jnp.eye(d)))

        # [] compute Lp group lasso (in classical group lasso, p=1) and scale by number of variables
        lasso = tree_global_norm(group_lasso_terms_j, p=1.0) / d
        return lasso


    def regularize_sparsity(self, param):
        """
        Sparsity regularization.
        See :func:`~stadion.inference.KDSMixin.regularize_sparsity`.
        """
        if self.sparsity_regularizer == "ingoing":
            reg = LinearSDE._regularize_ingoing(param)
        elif self.sparsity_regularizer == "outgoing":
            reg = LinearSDE._regularize_outgoing(param)
        elif self.sparsity_regularizer == "both":
            reg = LinearSDE._regularize_ingoing(param)\
                + LinearSDE._regularize_outgoing(param)
        else:
            raise ValueError(f"Unknown regularizer `{self.regularize_sparsity}`")
        return reg



