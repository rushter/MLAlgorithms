import random

import numpy as np
from scipy import stats
from numba import jit

from mla.ensemble.base import split_df

random.seed(111)


class Node:
    def __init__(self, regression=False, criterion=None):
        self.regression = regression
        self.n_samples = None
        self.impurity = None
        self.left_child = None
        self.right_child = None
        self.threshold = None
        self.column_index = None
        self.outcome = None
        self.depth = 0
        self.criterion = criterion

    def size(self):
        if not self.is_empty:
            return 1 + self.left_child.size() + self.right_child.size()
        return 1

    @property
    def is_empty(self):
        return not bool(self.left_child and self.right_child)

    def split(self, X, y, value):
        left_mask = (X < value)
        right_mask = (X >= value)
        return y[left_mask], y[right_mask]

    def find_splits(self, X, y):
        """Find all possible split-values."""

        df = np.rec.fromarrays([X, y], names='x,y')
        df.sort(order='x')

        split_values = set()
        for i in range(1, X.shape[0]):
            if df.y[i - 1] != df.y[i]:
                average = (df.x[i - 1] + df.x[i]) / 2.0
                split_values.add(average)
        return list(split_values)

    def find_best_split(self, X, y, n_features):
        subset = random.sample(list(range(0, X.shape[1])), n_features)
        max_gain = None
        max_col = None
        max_val = None
        for column in subset:
            split_values = self.find_splits(X[:, column], y)
            for value in split_values:
                splits = self.split(X[:, column], y, value)
                gain = self.criterion(y, splits)
                if gain > max_gain or (max_gain is None):
                    max_col = column
                    max_val = value
                    max_gain = gain
        return max_col, max_val, max_gain

    def train(self, X, y, max_features=None, min_samples_split=10, max_depth=None, minimum_gain=0.01,
              parent_gain=0.0):

        if max_features is None:
            max_features = int(np.sqrt(X.shape[1]))

        try:
            assert (X.shape[0] > min_samples_split)
            assert (max_depth > 0)

            column, value, gain = self.find_best_split(X, y, max_features)
            if self.regression:
                assert (gain != 0)
            else:
                assert (gain > minimum_gain)
            assert (gain != parent_gain)

            self.column_index = column
            self.threshold = value
            self.impurity = gain

            (left_X, left_y), (right_X, right_y) = split_df(X, y, column, value)

            assert (left_X.shape[0] > 0)
            assert (right_X.shape[0] > 0)

            self.left_child = Node(self.regression, self.criterion)
            self.left_child.train(left_X, left_y, max_features, min_samples_split, max_depth - 1,
                                  minimum_gain, gain)

            self.right_child = Node(self.regression, self.criterion)
            self.right_child.train(right_X, right_y, max_features, min_samples_split, max_depth - 1,
                                   minimum_gain, gain)

        except AssertionError:
            self._terminal_node(y)

    def _terminal_node(self, y):
        if self.regression:
            self.outcome = np.mean(y)
        else:
            self.outcome = stats.itemfreq(y)[:, 1] / float(y.shape[0])

    def predict_row(self, row):
        if not self.is_empty:
            if row[self.column_index] < self.threshold:
                return self.left_child.predict_row(row)
            else:
                return self.right_child.predict_row(row)
        return self.outcome

    def predict(self, X):
        result = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            result[i] = self.predict_row(X[i, :])
        return result
