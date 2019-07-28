# stdlib
from time import time
# 3p
import numpy as np
import fbpca


def norm2(X):
    return fbpca.pca(X, k=1, raw=True)[1][0]


def cond1(M, L, S, eps1):
    return np.linalg.norm(M - L - S, ord='fro') <= eps1 * np.linalg.norm(M, ord='fro')


def shrinkage_operator(X, tau):
    return np.sign(X) * np.max(np.abs(X) - tau, 0)


def sv_thresh_operator(X, tau, rank):
    rank = min(rank, np.min(X.shape))
    U, s, Vt = fbpca.pca(X, rank)
    s -= tau
    svp = (s > 0).sum()
    return U[:, :svp] @ np.diag(s[:svp]) @ Vt[:svp], svp


def pcp(M, maxiter=10):
    trans = False
    if M.shape[0] < M.shape[1]:
        M = M.T
        trans = True
    norm0 = np.linalg.norm(M, ord='fro')

    # recommended parameters
    eps1, eps2 = 1e-7, 1e-5
    rho = 1.6
    mu = 1.25 / norm2(M)
    lamda = 1 / np.sqrt(M.shape[0])
    sv = 10
    # initialization
    S, S_1, L = np.zeros_like(M), np.zeros_like(M), np.zeros_like(M)
    Y = M / max(norm2(M), np.max(np.abs(M / lamda)))

    # main loop
    since = time()

    for i in range(maxiter):
        S_1 = S.copy()
        S = shrinkage_operator(M - L + Y / mu, lamda / mu)
        L, svp = sv_thresh_operator(M - S + Y / mu, 1 / mu, sv)
        Y += mu * (M - L - S)

        # check for convergence
        cond2 = (mu * np.linalg.norm(S - S_1, ord='fro') / norm0) < eps2
        if cond1(M, L, S, eps1) and cond2:
            print(f'PCP converged! Took {round(time()-since, 2)} s')
            break

        # update mu and sv
        sv = svp + (1 if svp < sv else round(0.05 * M.shape[1]))
        mu = mu * rho if cond2 else mu
    else:
        print(f'Convergence Not reached! Took {round(time()-since, 2)} s')

    return (L, S) if not trans else (L.T, S.T)
