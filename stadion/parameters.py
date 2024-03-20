from abc import ABC, abstractmethod
import numbers
import pprint
import copy

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


class Parameters(ABC):
    def __init__(self, parameters):
        self._store = parameters

    def __iter__(self):
        return self._store.__iter__()

    def __getitem__(self, item):
        return self._store.__getitem__(item)

    def __setitem__(self, key, value):
        return self._store.__setitem__(key, value)

    @abstractmethod
    def _mask(self, tree, grad=False):
        pass

    def masked(self, grad=False):
        obj = copy.copy(self) # leaves the original object unchanged, no deepcopy because causes errors in tree_map
        obj._store = obj._mask(obj._store, grad=grad)
        return obj

    def tree_flatten(self):
        vals, treedef = jax.tree_util.tree_flatten(self._store)
        aux_data = dict(treedef=treedef)
        if hasattr(self, "_fixed"):
            aux_data["_fixed"] = self._fixed
        if hasattr(self, "_fixed_values"):
            aux_data["_fixed_values"] = self._fixed_values
        if hasattr(self, "targets"):
            aux_data["targets"] = self.targets
        if hasattr(self, "_targets_axis"):
            aux_data["_targets_axis"] = self._targets_axis
        if hasattr(self, "_mask_value"):
            aux_data["_mask_value"] = self._mask_value
        return vals, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, vals):
        obj = object.__new__(cls)
        obj._store = jax.tree_util.tree_unflatten(aux_data["treedef"], vals)
        if "_fixed" in aux_data:
            obj._fixed = aux_data["_fixed"]
        if "_fixed_values" in aux_data:
            obj._fixed_values = aux_data["_fixed_values"]
        if "targets" in aux_data:
            obj.targets = aux_data["targets"]
        if "_targets_axis" in aux_data:
            obj._targets_axis = aux_data["_targets_axis"]
        if "_mask_value" in aux_data:
            obj._mask_value = aux_data["_mask_value"]
        return obj



@register_pytree_node_class
class ModelParameters(Parameters):
    """
    Container for dictionary of SDE model parameters that allows fixing specific learnable parameters,
    for example, to fix the speed scaling in the drift function of an SDE.

    Args:
        parameters: dictionary of parameters to be stored
        fixed (ndarray, optional): indices of parameters that are fixed as
            `parameters = parameters.at[fixed].set(fixed_values)`. Can be a dictionary
            matching the structure of `parameters`, in which case the fixed indices
            are applied to the corresponding leaves. Defaults to ``None``.
        fixed_values (ndarray, optional): values of the fixed parameters. Can be
            a container (pytree) matching the structure of `parameters`, in which case
            the fixed values are applied to the corresponding leaves. Defaults to ``0.0``.

    """
    def __init__(self, parameters, fixed=None, fixed_values=0.0):
        self._store = parameters
        self._fixed = fixed
        self._fixed_values = fixed_values
        super().__init__(self._mask(parameters))


    def __repr__(self):
        descr = f"{self.__class__.__name__}"
        descr += f"\n{pprint.pformat(self._store)}"
        return descr


    def _mask(self, tree, grad=False):
        if self._fixed is None:
            return tree
        else:
            new = copy.copy(tree)
            for k, idx in self._fixed.items():
                if k not in new:
                    raise ValueError(f"Fixed key {k} not found in the parameter tree.")
                if grad:
                    value = 0.0
                elif isinstance(self._fixed_values, numbers.Number):
                    value = self._fixed_values
                elif k in self._fixed_values:
                    value = self._fixed_values[k]
                else:
                    raise ValueError(f"`fixed_values` must be a number or a dictionary containing the keys of `fixed`.")
                new[k] = new[k].at[idx].set(value)
            return new



@register_pytree_node_class
class InterventionParameters(Parameters):
    """
    Container for dictionary of intervention parameters that excludes known intervention
    targets from the learnable parameters and provides functionality for masking the
    parameters based on the known intervention targets.

    Args:
        parameters: dictionary of parameters to be stored. The parameters
            will be masked at initialization based on the targets if provided.
        targets (ndarray, optional): binary array of shape ``[d]`` or ``[n_envs, d]``
            indicating which variables are intervened upon. If provided, intervention
            parameters are automatically masked by `mask_value` on the axis `targets_axis`
            when a variable is not targeted. If an environment axis is provided,
            it is assumed that the intervention parameters have leading axis `n_envs`.
            Defaults to ``None``.
        targets_axis (int, optional): axis along which the targets are broadcasted.
            Defaults to ``0``. Can be a dictionary matching the structure
            of the intervention parameters, in which case the targets are broadcasted
            along the corresponding axes.
        mask_value (float, optional): neutral value to mask the non-targeted parameters
            with. Defaults to ``0.0``. Can be adictionary matching the structure
            of the intervention parameters, in which case the non-targeted parameters
            are masked with the corresponding values.
    """
    def __init__(self, parameters, targets=None, targets_axis=0, mask_value=0.0):
        self.targets = targets
        self._targets_axis = targets_axis
        self._mask_value = mask_value
        super().__init__(self._mask(parameters))


    def __repr__(self):
        descr = f"{self.__class__.__name__}"
        if self.targets is not None:
            descr += f"\nwith targets: {pprint.pformat(self.targets)}"
        descr += f"\n{pprint.pformat(self._store)}"
        return descr


    def _mask(self, tree, grad=False):
        if self.targets is None:
            return tree
        else:
            # mask parameters by `mask_value` wherever `targets == 0` on axis `targets_axis`
            def masker(x, mask, val, ax):
                y = jnp.moveaxis(x, ax, -1)
                y = mask * y + (1 - mask) * val
                y = jnp.moveaxis(y, -1, ax)
                assert x.shape == y.shape, f"Shape changed during masking: {x.shape} {y.shape}"
                return y

            # if there is an environment dimension in the targets, vmap the masker of the parameter tree
            targets = jax.tree_util.tree_map(lambda _: self.targets, tree)
            if not self.targets.ndim == 1:
                masker = jax.vmap(masker, (0, 0, None, None), 0)

            values = 0.0 if grad else self._mask_value
            if not jax.tree_util.tree_structure(values) == jax.tree_util.tree_structure(tree):
                values = jax.tree_util.tree_map(lambda _: values, tree)

            axes = self._targets_axis
            if not jax.tree_util.tree_structure(axes) == jax.tree_util.tree_structure(tree):
                axes = jax.tree_util.tree_map(lambda _: axes, tree)

            return jax.tree_map(masker, tree, targets, values, axes)

