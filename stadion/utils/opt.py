import math
import numpy as onp
import jax
from functools import partial

def update_ave(ave_d, d):
    # online mean and variance with Welfords algorithm
    for k, v in d.items():
        ave_d[("__ctr__", k)] += 1
        delta = v - ave_d[("__mean__", k)]
        ave_d[("__mean__", k)] += delta / ave_d[("__ctr__", k)]
        delta2 = v - ave_d[("__mean__", k)]
        ave_d[("__welford_m2__", k)] += delta * delta2
    return ave_d


def retrieve_ave(ave_d):
    out = dict(mean={}, std={})
    for k, v in ave_d.items():
        assert isinstance(k, tuple)
        # check if `k` is a ctr element
        if k[0] == "__ctr__":
            continue
        # process value `v`
        try:
            v_val = v.item()
        # array case
        except TypeError:
            v_val = onp.array(v)
        # not an array
        except AttributeError:
            v_val = v
        assert ("__ctr__", k[1]) in ave_d.keys()
        if k[0] == "__mean__":
            out["mean"][k[1]] = v_val
        else:
            if ave_d[("__ctr__", k[1])] == 1:
                out["std"][k[1]] = 0.0
            else:
                out["std"][k[1]] = math.sqrt((v_val + 1e-18) / (ave_d[("__ctr__", k[1])] - 1))
    return out


def _aggregate_scan(axis, leaf):
    if leaf.dtype == bool:
        return leaf.any(axis)
    else:
        return leaf.mean(axis)


def aggregate_scan(tree, axis=None):
    return jax.tree_map(partial(_aggregate_scan, axis), tree)