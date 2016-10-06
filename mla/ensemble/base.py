import numpy as np
from scipy import stats


def f_entropy(p):
    ep = stats.entropy(p)
    if ep == -float('inf'):
        return 0.0
    return ep


def information_gain(y, splits):
    splits_entropy = sum([f_entropy(split) * (float(split.shape[0]) / y.shape[0]) for split in splits])
    return f_entropy(y) - splits_entropy


def split_df(X, y, column, value):
    left_mask = (X[:, column] < value)
    right_mask = (X[:, column] >= value)
    return (X[left_mask], y[left_mask]), (X[right_mask], y[right_mask])


def mse_criterion(y, splits):
    y_mean = np.mean(y)
    return -sum([np.sum((split - y_mean) ** 2) * (float(split.shape[0]) / y.shape[0]) for split in splits])
