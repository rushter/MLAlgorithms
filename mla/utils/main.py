# coding:utf-8
import numpy as np


def one_hot(y):
    n_values = np.max(y) + 1
    return np.eye(n_values)[y]


def batch_iterator(X, batch_size=64):
    """Splits X into equal sized chunks."""
    n_samples = X.shape[0]
    n_batches = n_samples // batch_size
    batch_end = 0

    for b in range(n_batches):
        batch_begin = b * batch_size
        batch_end = batch_begin + batch_size

        X_batch = X[batch_begin:batch_end]

        yield X_batch

    if n_batches * batch_size < n_samples:
        yield X[batch_end:]
