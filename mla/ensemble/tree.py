import random

import numpy as np
from scipy import stats

from mla.ensemble.base import split, split_dataset, xgb_criterion

random.seed(111)


class Tree(object):
    """Recursive implementation of decision tree."""

    def __init__(self, regression=False, criterion=None):
        self.regression = regression
        self.impurity = None
        self.threshold = None
        self.column_index = None
        self.outcome = None
        self.criterion = criterion
        self.loss = None

        self.left_child = None
        self.right_child = None

    @property
    def is_terminal(self):
        return not bool(self.left_child and self.right_child)

    def _find_splits(self, X, y):
        """Find all possible split-values."""

        # Sort feature set
        df = np.rec.fromarrays([X, y], names='x,y')
        df.sort(order='x')

        split_values = set()
        for i in range(1, X.shape[0]):
            if df.y[i - 1] != df.y[i]:
                average = (df.x[i - 1] + df.x[i]) / 2.0
                split_values.add(average)
        return list(split_values)

    def _find_best_split(self, X, target, n_features):
        """Find best feature and value for split. Greedy algorithm."""

        # Sample random subset of features
        subset = random.sample(list(range(0, X.shape[1])), n_features)
        max_gain, max_col, max_val = None, None, None

        for column in subset:
            split_values = self._find_splits(X[:, column], target['y'])
            for value in split_values:
                if self.loss is None:
                    # Random forest
                    splits = split(X[:, column], target['y'], value)
                    gain = self.criterion(target['y'], splits)
                else:
                    # Gradient boosting
                    left, right = split_dataset(X, target, column, value, return_X=False)
                    gain = xgb_criterion(target, left, right, self.loss)

                if (max_gain is None) or (gain > max_gain):
                    max_col, max_val, max_gain = column, value, gain
        return max_col, max_val, max_gain

    def train(self, X, target, max_features=None, min_samples_split=10, max_depth=None, minimum_gain=0.01, loss=None):
        """Build a decision tree from training set.

        Parameters
        ----------

        X : array-like
            Feature dataset.
        target : dictionary or array-like
            Target values.
        max_features : int or None
            The number of features to consider when looking for the best split.
        min_samples_split : int
            The minimum number of samples required to split an internal node.
        max_depth : int
            Maximum depth of the tree.
        minimum_gain : float, default 0.01
            Minimum gain required for splitting.
        """

        if not isinstance(target, dict):
            target = {'y': target}

        # Loss for gradient boosting
        if loss is not None:
            self.loss = loss

        try:
            # Exit from recursion using assert syntax
            assert (X.shape[0] > min_samples_split)
            assert (max_depth > 0)

            column, value, gain = self._find_best_split(X, target, max_features)
            assert gain is not None
            if self.regression:
                assert (gain != 0)
            else:
                assert (gain > minimum_gain)

            self.column_index = column
            self.threshold = value
            self.impurity = gain

            # Split dataset
            left_X, right_X, left_target, right_target = split_dataset(X, target, column, value)

            self.left_child = Tree(self.regression, self.criterion)
            self.left_child.train(left_X, left_target, max_features, min_samples_split, max_depth - 1,
                                  minimum_gain, loss)

            self.right_child = Tree(self.regression, self.criterion)
            self.right_child.train(right_X, right_target, max_features, min_samples_split, max_depth - 1,
                                   minimum_gain, loss)
        except AssertionError:
            self._calculate_leaf_value(target)

    def _calculate_leaf_value(self, targets):
        """Find optimal value for leaf."""
        if self.loss is not None:
            # Gradient boosting
            self.outcome = self.loss.approximate(targets['actual'], targets['y_pred'])
        else:
            # Random Forest
            if self.regression:
                # Mean value for regression task
                self.outcome = np.mean(targets['y'])
            else:
                # Probability for classification task
                self.outcome = stats.itemfreq(targets['y'])[:, 1] / float(targets['y'].shape[0])

    def predict_row(self, row):
        if not self.is_terminal:
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
