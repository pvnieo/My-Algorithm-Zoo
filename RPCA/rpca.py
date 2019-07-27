# stdlib
from time import time
# 3p
import numpy as np
import fbpca


def shrinkage_operator(X, tau):
    return np.sign(X) * np.max(np.abs(X) - tau, 0)


def sv_thresh_operator(X, tau, rank):
    rank = min(rank, np.min(X.shape))
    U, s, Vt = fbpca.pcp(X, rank)
    s -= tau
    svp = (s > 0).sum()
    return U[:, :svp] @ np.diag(s[:svp]) @ Vt[:, :svp], svp


def l2_norm(X):
    return fbpca.pca(X, k=1, raw=True)[1][0]


def has_converged(M, L, S, eps):
    return np.linalg.norm(M - L - S, ord='fro') <= eps * np.linalg.norm(M, ord='fro')


def pcp(M, maxiter=10, eps=1e-7):
    trans = False
    if M.shape[0] < M.shape[1]:
        M = M.T
        trans = True
    # initialization
    mu = 0.25 * np.prod(M.shape) / np.sum(np.abs(M))
    lamda = 1 / np.sqrt(M.shape[0])
    sv = 1
    S = np.zeros_like(M)
    Y = M / max(l2_norm(M), np.max(np.abs(M / lamda)))

    # main loop
    since = time()

    for _ in range(maxiter):
        L, svp = sv_thresh_operator(M - S + Y / mu, 1 / mu, sv)
        S = shrinkage_operator(M - L + Y / mu, lamda / mu)
        Y = Y + mu * (M - L - S)

        # update mu and sv
        sv = svp + 1 if svp < sv else min(M.shape[1], svp + round(0.05 * M.shape[1]))
        mu = mu

        # check for convergence
        if has_converged(M, L, S, eps):
            print(f'PCP converged! Took {round(time()-since, 2)} s')
            break
    else:
        print(f'Convergence Not reached! Took {round(time()-since, 2)} s')

    return L, S if not trans else L.T, S.T
