from __future__ import division, print_function, absolute_import
import numpy as np


def zca_whitening_matrix(X, w_epsilon, batch=1000):
    shape = X.shape
    N = shape[0]
    Xflat = X.reshape((N, -1))

    sigma = None
    num_batches = int(np.ceil(N / batch))
    for b in range(num_batches):
        Xb = Xflat[b*batch:(b+1)*batch]
        C = np.dot(Xb.T, Xb)
        if sigma is None:
            sigma = C
        else:
            sigma += C
    sigma /= N

    U, S, _ = np.linalg.svd(sigma)
    shrinker = np.diag(1 / np.sqrt(S + w_epsilon))
    W = np.dot(U, np.dot(shrinker, U.T))
    return W


def apply_whitening_matrix(X, W, batch=1000):
    shape = X.shape
    N = shape[0]
    Xflat = X.reshape((N, -1))
    wX = np.empty(shape)

    num_batches = int(np.ceil(N / batch))
    for b in range(num_batches):
        Xb = Xflat[b*batch:(b+1)*batch]
        wX[b*batch:(b+1)*batch] = np.dot(W, Xb.T).T.reshape((-1,) + shape[1:])
    return wX


def whiten(X, w_epsilon, batch=1000):
    shape = X.shape
    N = shape[0]
    Xflat = X.reshape((N, -1))
    W = zca_whitening_matrix(X, w_epsilon)

    wX = np.empty(shape)

    num_batches = int(np.ceil(N / batch))
    for b in range(num_batches):
        Xb = Xflat[b*batch:(b+1)*batch]
        wX[b*batch:(b+1)*batch] = np.dot(W, Xb.T).T.reshape((-1,) + shape[1:])
    return wX
