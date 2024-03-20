import numpy as onp

"""
Projecting a matrix onto its sparsest stable matrix
Algorithm by https://arxiv.org/abs/1611.00595
"""

def proj_skew(m):
    return 0.5 * (m - m.T)


def proj_psd(m):
    symm = 0.5 * (m + m.T)
    lambd, u = onp.linalg.eig(symm)
    return u @ onp.diag(onp.maximum(0, lambd)) @ u.T


def proj_d(m):
    return proj_skew(m) - proj_psd(- m)


def max_eigval(m):
    lambda_max = onp.max(onp.linalg.eigvals(m))
    assert onp.isclose(onp.imag(lambda_max), 0.0)
    return onp.real(lambda_max)


def project_closest_stable_matrix(m, steps=100, eps=None):
    assert m.ndim == 2 and m.shape[0] == m.shape[1]

    # initialize
    q = onp.eye(m.shape[0])
    d = proj_skew(m) - proj_psd(- 0.5 * (m + m.T))

    for t in range(steps):
        # scale q and d such that max eigenvalues of q @ q.T and d.T @ d are equal and compute optimal stepsize
        lambd_d = max_eigval(d.T @ d)
        lambd_q = max_eigval(q @ q.T)
        q = q * onp.sqrt(lambd_d / lambd_q)
        stepsize = 1. / lambd_d
        assert stepsize > 0

        # compute update direction
        delta_d = (m - d @ q) @ q.T
        delta_q = d.T @ (m - d @ q)

        # perform update and project onto feasible set
        d = proj_d(d + stepsize * delta_d)
        q = proj_psd(q + stepsize * delta_q)

    m_proj = d @ q

    # shift spectrum by `eps` to ensure stable simulation
    return m_proj - (eps or 1e-6) * onp.eye(m.shape[0])