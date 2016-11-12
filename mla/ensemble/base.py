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


def mse_criterion(y, splits):
    y_mean = np.mean(y)
    return -sum([np.sum((split - y_mean) ** 2) * (float(split.shape[0]) / y.shape[0]) for split in splits])


def xgb_criterion(y, left, right, loss):
    left = loss.gain(left['actual'], left['y_pred'])
    right = loss.gain(right['actual'], right['y_pred'])
    initial = loss.gain(y['actual'], y['y_pred'])
    gain = left + right - initial
    return gain


def get_split_mask(X, column, value):
    left_mask = (X[:, column] < value)
    right_mask = (X[:, column] >= value)
    return left_mask, right_mask


def split(X, y, value):
    left_mask = (X < value)
    right_mask = (X >= value)
    return y[left_mask], y[right_mask]


def split_dataset(X, target, column, value, return_X=True):
    left_mask, right_mask = get_split_mask(X, column, value)

    left, right = {}, {}
    for key in target.keys():
        left[key] = target[key][left_mask]
        right[key] = target[key][right_mask]

    if return_X:
        left_X, right_X = X[left_mask], X[right_mask]
        return left_X, right_X, left, right
    else:
        return left, right
