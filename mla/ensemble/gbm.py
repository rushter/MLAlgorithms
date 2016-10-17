import numpy as np

from mla.base import BaseEstimator
from mla.ensemble.base import mse_criterion
from mla.ensemble.tree import Node


class GradientBoosting(BaseEstimator):
    """Gradient boosting trees."""

    def __init__(self, n_estimators, learning_rate=0.1, max_features=10, max_depth=2, min_samples_split=10):
        self.min_samples_split = min_samples_split
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.trees = []

    def fit(self, X, y=None):
        self._setup_input(X, y)
        self.y_mean = np.mean(y)
        self.weights = np.ones(self.n_estimators)
        self.weights *= self.learning_rate

        self._train()

    def _train(self):
        y_pred = np.full(self.n_samples, self.y_mean)
        for n in range(self.n_estimators):
            y_grad = self.loss(self.y, y_pred)
            tree = Node(regression=True, criterion=mse_criterion)
            tree.train(self.X, y_grad, max_features=self.max_features,
                       min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            y_pred = tree.predict(self.X)
            self.trees.append(tree)

    def _predict(self, X=None):
        y_pred = np.full(X.shape[0], self.y_mean)
        for i, tree in enumerate(self.trees):
            # TODO: implement weight search
            y_pred += self.weights[i] * tree.predict(X)
        return y_pred

    def _update_terminal_regions(self, tree):
        raise NotImplementedError()


class GradientBoostingRegressor(GradientBoosting):
    def loss(self, actual, predicted):
        """Square loss gradient"""
        return actual - predicted


class GradientBoostingClassifier(GradientBoosting):
    def loss(self, actual, predicted):
        """Log loss gradient"""
        q = 1.0 / (1 + np.exp(-actual * predicted))
        return -actual * (q - 1)
