import numpy as onp
from stadion.utils.graph import topological_ordering


def compute_linear_gaussian_mle_params(g_input, x, intv):
    """
    Computes MLE parameters for linear SCM based on observational and interventional data
    See Hauser et al
    https://arxiv.org/pdf/1303.3216.pdf
    Lemma 5, Page 17

    Assumes observational distribution has mean zero, i.e. the linear model has no biases.

    Args:
        g_input [d, d] where g[i, j] == 1 iff i -> j
        x [n, d]
        intv [n, d] where intv[k, j] if k-th datapoint had intervention on node j
    """

    assert g_input.ndim == x.ndim == 2
    assert g_input.shape[0] == g_input.shape[1] == x.shape[1]
    assert x.shape == intv.shape

    d = g_input.shape[-1]

    # this math assumes the adjacancy matrix to be transposed
    # g[i, j] == 1 iff j in pa(i), i.e. j -> i
    g = g_input.T

    # unique interventional settings (\mathcal{I} in paper)
    unique_envs = onp.unique(intv, axis=0)
    n_envs = unique_envs.shape[0]
    assert unique_envs.shape == (n_envs, d)

    # n_i   [n_envs]:       number of samples in each interventional setting
    # cov_i [n_envs, d, d]: empirical covariance in each interventional setting
    n_i = onp.zeros(n_envs)
    cov_i = onp.zeros((n_envs, d, d))
    for i, env in enumerate(unique_envs):
        env_idx = onp.where(onp.isclose(intv, env).all(axis=-1))[0]
        assert env_idx.ndim == 1

        x_i = x[env_idx]
        n_i[i] = len(env_idx)
        cov_i[i] = (x_i.T @ x_i) / n_i[i]

    # n_k   [d]:       number of samples _not_ intervening on node k
    # cov_k [d, d, d]: emirical covariance of samples _not_ intervening on node k
    n_k = onp.zeros(d)
    cov_k = onp.zeros((d, d, d))

    for k in range(d):
        # check which envs did _not_ intervene on k
        envs_not_k = onp.where(~unique_envs[:, k].astype(bool))[0]
        assert onp.allclose(unique_envs[envs_not_k][:, k].sum(0), 0)

        # fill counts n_k
        for env_idx in envs_not_k:
            n_k[k] += n_i[env_idx]

        # fill covariances cov_k using counts
        for env_idx in envs_not_k:
            cov_k[k] += cov_i[env_idx] * n_i[env_idx] / n_k[k]

    # B matrix estimate
    coeffs = onp.zeros(g.shape, dtype=onp.float32)

    for k in range(d):
        parents = onp.where(g[k] == 1)[0]
        assert parents.ndim == 1

        if len(parents) > 0:

            cov_k_pa = cov_k[k][k, parents]
            cov_pa_pa = cov_k[k][onp.ix_(parents, parents)]

            assert cov_k_pa.shape == parents.shape, f"{cov_k_pa.shape} {parents.shape}"
            assert cov_pa_pa.shape == (parents.shape[0], parents.shape[0]), f"{cov_pa_pa.shape}  {parents.shape}"

            if len(parents) > 1:
                inv_cov_pa_pa = onp.linalg.inv(cov_pa_pa)
            else:
                inv_cov_pa_pa = onp.array(1 / cov_pa_pa)

            mle_coeffs_k_pa = cov_k_pa @ inv_cov_pa_pa
            assert mle_coeffs_k_pa.shape == parents.shape

            coeffs[k, parents] = mle_coeffs_k_pa

    # noise estimates
    noise_vars = onp.zeros(d, dtype=onp.float32)
    one_minus_coeffs = onp.eye(d) - coeffs

    for k in range(d):
        noise_var = one_minus_coeffs[k] @ cov_k[k] @ one_minus_coeffs[k].T
        assert noise_var.ndim == 0
        noise_vars[k] = noise_var

    # return transpose of coeffs to match our inserted graph orientation where g[i, j] == 1 iff i -> j
    return coeffs.T, noise_vars


def make_linear_gaussian_sampler(g, coeffs, noise_vars):
    """
    Given parameters of a linear Gaussian SCM, return a function sampling interventional data given multiple envs
    """

    d = g.shape[-1]
    assert g.shape == coeffs.shape
    assert onp.allclose(onp.isclose(g, 0), onp.isclose(coeffs, 0)), \
        f"Sparsity patterns do not match. Got:" \
        f"g:\n{g}\n" \
        f"coeffs:\n{coeffs}\n"

    toporder = topological_ordering(g)
    noise_scales = onp.sqrt(noise_vars + 1e-20)
    assert noise_scales.shape == (d,)

    def sampler(rng, intv_theta, intv, n_samples):

        # extract shift intervention parameters
        shift = intv_theta["shift"] * intv
        assert intv_theta["shift"].shape == intv.shape

        n_envs = intv.shape[0]
        assert intv.shape == (n_envs, d), f"{intv.shape}"

        z = rng.normal(size=(n_envs, n_samples, d))
        x = onp.zeros((n_envs, n_samples, d))

        # regular ancestral sampling
        for j in toporder:

            parents = onp.where(onp.isclose(g[:, j], 1))[0]

            if len(parents > 1):
                theta_j = coeffs[:, j]
                assert  onp.allclose(onp.isclose(theta_j, 0), onp.isclose(g[:, j], 0))

                f_j = onp.einsum("enp,p->en", x[:, :, parents], theta_j[parents]) + shift[:, j, None]

            else:
                f_j = shift[:, j, None]

            # insert observations
            x[:, :, j] = f_j + noise_scales[j] * z[:, :, j]

        return x

    return sampler


if __name__ == "__main__":

    # check constistency of lin gauss estimation
    onp.set_printoptions(suppress=True, precision=3)
    debug_rng = onp.random.default_rng(0)

    debug_g = onp.array([[0, 1, 0, 0],
                         [0, 0, 1, 1],
                         [0, 0, 0, 1],
                         [0, 0, 0, 0]])

    debug_coeffs = onp.array([[0, 2, 0, 0],
                              [0, 0, -1, 3],
                              [0, 0, 0, 2],
                              [0, 0, 0, 0]])

    debug_noise_vars = onp.array([0.5, 2.0, 4.0, 1.0])

    debug_sampler = make_linear_gaussian_sampler(debug_g, debug_coeffs, debug_noise_vars)

    # debug_intv_theta = dict(shift=onp.array([[0, 0, 0, 0]]))
    # debug_intv = onp.array([[0, 0, 0, 0]])


    debug_intv_theta = dict(shift=onp.array([[0, 0, 0, 0],
                                             [0, 0, -3, 0],
                                             [0, 0, 0, 5]]))

    debug_intv = onp.array([[0, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

    debug_n = 100000
    debug_x = debug_sampler(debug_rng, debug_intv_theta, debug_intv, n_samples=debug_n)

    debug_x_stacked = onp.concatenate(debug_x, axis=0)
    debug_intv_stacked = debug_intv[onp.repeat(onp.arange(len(debug_intv)), (debug_n,))]

    # standardize
    # debug_x_stacked = (debug_x_stacked - debug_x[0].mean(axis=0)) / debug_x[0].std(axis=0)

    # compute MLE params
    debug_coeffs_rec, debug_noise_vars_rec = compute_linear_gaussian_mle_params(debug_g, debug_x_stacked, debug_intv_stacked)

    print(debug_coeffs_rec)
    print(debug_noise_vars_rec)
    print()

