import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal for")
warnings.filterwarnings("ignore", message="entering retry loop")
warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"

import copy
from functools import partial
from jax import random, jit, tree_map
import jax.numpy as jnp
import numpy as onp
from stadion.utils.parse import timer

INFINITY = 1e4

def moving_average(a, n=3) :
    ret = onp.cumsum(a, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def check_if_stable(samples):
    nan = onp.isnan(samples)
    inf = onp.abs(samples) > INFINITY
    return ~onp.any(nan | inf, axis=(-1, -2))


def estimate_stats(sampler, *args):
    # simulate
    samples = onp.array(sampler(*args))

    # check stable
    stable = check_if_stable(samples)

    # stats
    means = onp.mean(samples, axis=-2)
    stds = onp.std(samples, axis=-2)

    return onp.where(stable[..., None], means, onp.nan), \
           onp.where(stable[..., None], stds, onp.nan), \
           stable


def search_intv_theta_shift(key, *, theta, intv_theta, target_means, target_intv, sampler, n_samples,
                            exp_fact_init=0.1,
                            n_powers=20,
                            rel_eps=1.0,
                            observ_eps=0.5,
                            grid=10,
                            ave=3,
                            base_power=2.0,
                            lookback_after_reach=2,
                            n_powers_insignificant=4,
                            jit_sampler=True,
                            verbose=False):
    """
    Assumes that the intervention w(x_j - c) induces a shift proportional to +c on x_j if w < 0
    """
    print("\nStarting test intv_theta shift estimation...", flush=True)

    assert ave % 2 == 1, "Select ood moving average size to select an exact center"
    assert "shift" in intv_theta
    assert target_means.ndim == 2
    assert target_means.shape == intv_theta["shift"].shape[:2]
    assert onp.allclose(intv_theta["shift"], 0.0), "Should init intv_theta test shifts at 0.0"

    # skip observational dimensions
    observ_mask = onp.all(target_intv == 0, axis=1)
    intv_theta_shape_copy = copy.deepcopy(intv_theta)
    target_means = target_means[~observ_mask]
    target_intv = target_intv[~observ_mask]
    for k in intv_theta.keys():
        intv_theta[k] = intv_theta[k][~observ_mask]

    n_envs, d = target_means.shape

    log_dict = dict()
    jit_fun = jit if jit_sampler else lambda f: f
    sampler = jit_fun(partial(sampler, n_samples=n_samples))
    estimator = partial(estimate_stats, sampler)

    if "scale" in intv_theta:
        # force no scaling in the search procedure
        intv_theta["scale"] = intv_theta["scale"].at[...].set(0.0)

    """Step 0: Assert that system is stable in observational state """
    with timer() as time_step_0:

        intv_theta_observ = intv_theta.copy()
        key, subk = random.split(key)
        observ_means, observ_stds, is_observ_stable = estimator(subk, theta, intv_theta_observ,
                                                                onp.zeros_like(target_intv).astype(onp.int32))

        log_dict["sanity/test-search/0-observ-stable"] = int(is_observ_stable.all())

        eigv_check = theta is not None and (
            "w1" in theta and
            theta["w1"].ndim == 2 and
            theta["w1"].shape[0] == theta["w1"].shape[1] and
            onp.real(onp.linalg.eigvals(theta["w1"])).max() >= 0.0
        )
        if not is_observ_stable.all() or eigv_check:
            key, subk = random.split(key)
            example_samples = sampler(subk, theta, intv_theta_observ, onp.zeros_like(target_intv).astype(onp.int32))
            warnings.warn("\nSystem not observationally stable.\n"
                          f"nans: {onp.isnan(example_samples).any(-1).sum(-1)} of {n_samples} samples\n"
                          f"infs: {(onp.abs(example_samples) > INFINITY).any(-1).sum(-1)} of {n_samples} samples\n"
                          "Skipping search_theta_shift")
            intv_theta_nan = tree_map(lambda arr: arr * jnp.nan, intv_theta_shape_copy)
            return intv_theta_nan, log_dict

    log_dict["sanity/test-search/time_step_0"] = time_step_0() / 60.0
    print(f'time_step_0: {log_dict["sanity/test-search/time_step_0"]:.2f} min')

    """
    Step 1: Bidirected exponential search for upper bound on shift in each env, 
    stopping when overshooting relative to observational mean
    """

    if verbose:
        print("=" * 30)
        print("Exponential search:")

    with timer() as time_step_1:

        param = intv_theta.copy()

        intv_sel = list(range(n_envs)), onp.argmax(target_intv, axis=-1)
        if onp.any(target_intv.sum(-1) > 1):
            raise NotImplementedError("double perturbations not implemented yet here; "
                                      "loop assumes we only intervene on 1 node")

        tar_shifted_up =   onp.array(target_means > observ_means - observ_eps * observ_stds)
        tar_shifted_down = onp.array(target_means < observ_means + observ_eps * observ_stds)

        insignificant_shift = (tar_shifted_up & tar_shifted_down)[intv_sel]

        log_dict["sanity/test-search/0-insignificant-shift"] = int(not onp.any(insignificant_shift))

        if onp.any(insignificant_shift):
            warnings.warn(f"\nTarget mean inside `{observ_eps}x` confidence region of simulated observ_mean\n"
                          f"observ_means\n{onp.where(target_intv, observ_means, 0)}\n"
                          f"observ_stds\n{onp.where(target_intv, observ_stds,0)}\n"
                          f"target_means\n{onp.where(target_intv, target_means, 0)}\n"
                          f"target_intv\n{target_intv}\n"
                          f"inside confidence range\n{onp.where(target_intv, (tar_shifted_up & tar_shifted_down), 0)}\n"
                          f"(ignored dims are zeroed here).\n"
                          f"Perhaps the observational distribution is still too unstable\n"
                          f"or the intervention is just ineffective/weak.")

        # initialize exponential search
        reached = onp.ones(n_envs) * onp.nan
        reached_sgn = onp.ones(n_envs) * onp.nan
        stable = onp.ones(n_envs, dtype=bool)
        assert n_powers > 0

        for i in range(n_powers):
            for sgn in [-1., 1.]:

                # setup shift in each env
                # increase window width by factor `base_power` each iteration
                shift = sgn * exp_fact_init * target_intv * (base_power ** i)
                param["shift"] = param["shift"].at[:, :].set(shift)

                if verbose:
                    print()
                    print(f"power: {i} ({sgn}) shift: {shift[shift.nonzero()][0]:.1f} ")

                # simulate rollouts
                key, subk = random.split(key)
                means, _, stable_i = estimator(subk, theta, param, target_intv)
                assert means.shape == target_means.shape
                assert means[intv_sel].shape == stable_i.shape == (n_envs,)

                if verbose:
                    print(f"means \n{target_intv * means}")
                    print(f"target: \n{target_intv * target_means}")
                    print(f"diff: \n{target_intv * (target_means - means)}")

                # update arrays
                # if still stable, warn if got unstable
                if onp.any((~stable) & stable_i):
                    warnings.warn("\nStep 1: an unstable intervention got stable again; "
                                  "consider using more samples or longer burnin")

                stable = onp.where(stable, stable_i, stable)

                # if overshooting relative to observ_means, remember exponent; warn if got unreached after it got reached
                reached_i = ((means > target_means) & tar_shifted_up)  \
                          | ((means < target_means) & tar_shifted_down)
                reached_eps_i = ((means > rel_eps * target_means) & tar_shifted_up)  \
                              | ((means < rel_eps * target_means) & tar_shifted_down)

                if verbose:
                    print()
                    print(f"reached: \nnow prev \n"
                          f"{onp.stack([reached_eps_i[intv_sel], ~onp.isnan(reached)]).T.astype(int)}")

                assert reached_i.shape == reached_eps_i.shape == target_means.shape == means.shape
                assert reached_i[intv_sel].shape == (n_envs,)

                if onp.any(~onp.isnan(reached) & (~reached_i[intv_sel]) & (reached_sgn == sgn) & ~insignificant_shift):
                    warnings.warn(f"\nStep 1: a target with significant shift got unreached after it was reached; "
                                  f"consider adding a larger rel_eps\n"
                                  f"insignificant_shifts:\n{insignificant_shift}\n"
                                  f"means:\n{means}\n"
                                  f"target_means:\n{target_means}\n"
                                  f"reached_i:\n{reached_i}\n"
                                  f"reached_eps_i:\n{reached_eps_i}\n"
                                  f"reached:\n{reached}\n"
                                  )

                if onp.any(~stable & reached_i[intv_sel]):
                    warnings.warn("\nStep 1: a target got reached after it was unstable")

                reached =     onp.where(reached_eps_i[intv_sel] & onp.isnan(reached),     onp.ones(n_envs) * i,   reached)
                reached_sgn = onp.where(reached_eps_i[intv_sel] & onp.isnan(reached_sgn), onp.ones(n_envs) * sgn, reached_sgn)

                # loop maintenance
                if (~onp.isnan(reached) | ~stable).all():
                    break

            # loop maintenance
            if (~onp.isnan(reached) | ~stable).all():
                break

        overall_success = not onp.isnan(reached).any()
        if not overall_success:
            warnings.warn(f"\nDidn't reach all (or unstable stable) after {i} powers (of max {n_powers - 1}):\n"
                          f"reached\n{reached}\n"
                          f"stable\n{stable}")

        # logging
        log_dict["sanity/test-search/1-success"] = int(overall_success)
        log_dict["sanity/test-search/1-intv-stable"] = int(onp.all(stable).item())
        log_dict["sanity/test-search/1-reached"] = onp.max(onp.nan_to_num(reached, nan=-1)).item()

    log_dict["sanity/test-search/time_step_1"] = time_step_1() / 60.0
    print(f'time_step_1: {log_dict["sanity/test-search/time_step_1"]:.2f} min')

    """
    Step 2: Grid search for best match within established search window
    """

    if verbose:
        print("=" * 30)
        print("Grid search:\n")

    with timer() as time_step_2:

        # if unstable, leave at observational
        reached = onp.where(onp.isnan(reached), 0, reached)
        reached_sgn = onp.where(onp.isnan(reached_sgn), 0, reached_sgn)

        lower = reached_sgn * exp_fact_init * (base_power ** (reached - lookback_after_reach))
        upper = reached_sgn * exp_fact_init * (base_power ** reached)
        lower = onp.where(reached >= lookback_after_reach, lower, -upper)

        shift_grid = onp.linspace(lower, upper, num=grid)

        if verbose:
            print(f"lower: \n{lower}")
            print(f"upper: \n{upper}")

        # for insignificant shifts, search in default `n_powers_insignificant` range in log steps around zero
        shift_grid_backup = exp_fact_init * onp.logspace(0, n_powers_insignificant, base=base_power, num=grid)
        shift_grid_backup = onp.concatenate([-shift_grid_backup[::2][::-1], shift_grid_backup[1::2]])[:, None]
        shift_grid = onp.where(~insignificant_shift, shift_grid, shift_grid_backup)

        if verbose:
            print(f"insignificant_shifts: \n{insignificant_shift}")
            print(f"shift_grid: \n{shift_grid}")

        results = onp.zeros((grid, n_envs))
        for j, shift in enumerate(shift_grid):

            # setup shift in each env
            param = intv_theta.copy()
            param["shift"] = param["shift"].at[intv_sel].set(shift)

            # simulate rollouts
            key, subk = random.split(key)
            means, _, stable_i = estimator(subk, theta, param, target_intv)

            if not onp.all(stable_i | ~stable):
                warnings.warn(f"\nStep 2: unstable simulation discovered in presumably stable regime\n"
                              f"{stable_i}\n{lower}\n{upper}")

            results[j] = means[intv_sel]

        # compute moving average
        shift_grid = moving_average(shift_grid, n=ave)
        results = moving_average(results, n=ave)
        assert shift_grid.shape == results.shape == (grid - ave + 1, n_envs)

        # compute distance to target_mean
        dists = onp.abs(results - target_means[intv_sel][None])

        if verbose:
            print(f"dists (ave={ave}): \n{dists}")

        # select optimal shift and return theta
        opt = onp.argmin(dists, axis=0)
        opt_shifts = shift_grid[opt, list(range(n_envs))]
        if onp.any(((opt == 0) | (opt == grid - ave)) & ~onp.isnan(dists).all(0)):
            warnings.warn(f"\nStep 2: extremum near boundary, consider changing lookback_after_reach or rel_eps\n"
                          f"opt:\n{opt}\n"
                          f"reached:\n{reached}\n"
                          f"dists:\n{dists}\n"
                          f"shift_grid:\n{shift_grid}\n"
                          f"insignificant_shift:\n{insignificant_shift}\n")

        opt_theta = intv_theta.copy()
        opt_theta["shift"] = opt_theta["shift"].at[intv_sel].set(opt_shifts)

        # logging
        log_dict["sanity/test-search/2-opt-relative-to-boundary"] = onp.min(opt).item() / (dists.shape[0] - 1)
        log_dict["sanity/test-search/2-neg-mean-error"] = -onp.mean(dists[opt, list(range(n_envs))]).item()

    log_dict["sanity/test-search/time_step_2"] = time_step_2() / 60.0
    print(f'time_step_2: {log_dict["sanity/test-search/time_step_2"]:.2f} min')

    # add observational dimensions back with no intervention
    for k in opt_theta.keys():
        new = onp.zeros(intv_theta_shape_copy[k].shape)
        new[~observ_mask] = opt_theta[k]
        opt_theta[k] = new

    print("done.\n", flush=True)

    return opt_theta, log_dict

