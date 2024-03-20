import math
import jax
from jax import vmap
import jax.numpy as jnp

def chunked_vmap(fun, argnum, chunk_size=1024):
    """
    DOESN'T WORK IN TERMS OF MEMORY YET

    Chunked vmap over axis 0 of argument `argnum`, i.e. `args[argnum]`

    Example:

    @jit
    @functools.partial(chunked_vmap, argnum=0, chunk_size=100)
    @functools.partial(chunked_vmap, argnum=1, chunk_size=100)
    def chunk_batched_objective(*args):
        return generator(generator(kernel, 0), 1)(*args)

    chunk_objective = jit(lambda *args: jnp.mean(batched_objective(*args), axis=(0, 1)))

    """
    def vmapped_fun(*args):
        vmap_arg = args[argnum]
        num_arg = vmap_arg.shape[0]
        scan_length = math.ceil(num_arg / chunk_size)
        in_axes = tuple([(0 if i == argnum else None) for i, _ in enumerate(args)])

        def chunk_scanner(chunk_idx, _):
            chunked_arg = jax.lax.dynamic_slice(vmap_arg,
                start_indices=(chunk_idx, *[0 for _ in range(vmap_arg.ndim - 1)]),
                slice_sizes=(min(chunk_size, num_arg), *vmap_arg.shape[1:]))
            chunked_args = tuple([(chunked_arg if i == argnum else arg) for i, arg in enumerate(args)])
            return chunk_idx + chunk_size, vmap(fun, in_axes, 0)(*chunked_args)

        # [scan_length, query_chunk_size, ...]
        _, res = jax.lax.scan(chunk_scanner, init=0, xs=None, length=scan_length)
        # _, res = hk.scan(chunk_scanner, init=0, xs=None, length=scan_length)

        # [scan_length * query_chunk_size, ...]
        res = jnp.concatenate(res, axis=0)

        # adjustment for num_arg % chunk_size != 0
        # in this case: in last batch of lax.scan, only keep last `num_arg % chunk_size` elements
        # the other ones are repeated (see docs for jax.lax.dynamic_slice)
        if num_arg % chunk_size != 0 and chunk_size < num_arg:
            lo = scan_length * chunk_size - chunk_size
            hi = scan_length * chunk_size - (num_arg % chunk_size)
            res = jnp.concatenate([res[:lo],
                                   res[hi:]], axis=0)

        return res

    return vmapped_fun

