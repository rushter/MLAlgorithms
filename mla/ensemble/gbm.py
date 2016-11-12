import numpy as np
# logistic function
from scipy.special import expit

from mla.base import BaseEstimator
from mla.ensemble.base import mse_criterion
from mla.ensemble.tree import Tree

"""
References:
https://arxiv.org/pdf/1603.02754v3.pdf
http://www.saedsayad.com/docs/xgboost.pdf
https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf
http://stats.stackexchange.com/questions/202858/loss-function-approximation-with-taylor-expansion
"""


class Loss:
    """Base class for loss functions."""

    def __init__(self, regularization=1.0):
        self.regularization = regularization

    def grad(self, actual, predicted):
        """First order gradient."""
        raise NotImplementedError()

    def hess(self, actual, predicted):
        """Second order gradient."""
        raise NotImplementedError()

    def approximate(self, actual, predicted):
        """Approximate leaf value."""
        return self.grad(actual, predicted).sum() / (self.hess(actual, predicted).sum() + self.regularization)

    def transform(self, pred):
        """Transform predictions values."""
        return pred

    def gain(self, actual, predicted):
        """Gain for split finding."""
        nominator = self.grad(actual, predicted).sum() ** 2
        denominator = (self.hess(actual, predicted).sum() + self.regularization)
        return 0.5 * (nominator / denominator)


class LeastSquaresLoss(Loss):
    """Least squares loss"""

    def grad(self, actual, predicted):
        return actual - predicted

    def hess(self, actual, predicted):
        return np.ones_like(actual)


class LogisticLoss(Loss):
    """Logistic loss."""

    def grad(self, actual, predicted):
        return actual * expit(-actual * predicted)

    def hess(self, actual, predicted):
        expits = expit(predicted)
        return expits * (1 - expits)

    def transform(self, output):
        # Apply logistic (sigmoid) function to the output
        return expit(output)


class GradientBoosting(BaseEstimator):
    """Gradient boosting trees with taylor expansion approximation (as in xgboost)."""

    def __init__(self, n_estimators, learning_rate=0.1, max_features=10, max_depth=2, min_samples_split=10):
        self.min_samples_split = min_samples_split
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.trees = []
        self.loss = None

    def fit(self, X, y=None):
        self._setup_input(X, y)
        self.y_mean = np.mean(y)
        self._train()

    def _train(self):
        # Initialize model with zeros
        y_pred = np.zeros(self.n_samples, np.float32)
        # Or mean
        # y_pred = np.full(self.n_samples, self.y_mean)

        for n in range(self.n_estimators):
            residuals = self.loss.grad(self.y, y_pred)
            tree = Tree(regression=True, criterion=mse_criterion)
            # Pass multiple target values to the tree learner
            targets = {
                # Residual values
                'y': residuals,
                # Actual target values
                'actual': self.y,
                # Predictions from previous step
                'y_pred': y_pred
            }
            tree.train(self.X, targets, max_features=self.max_features,
                       min_samples_split=self.min_samples_split, max_depth=self.max_depth, loss=self.loss)
            predictions = tree.predict(self.X)
            y_pred += self.learning_rate * predictions
            self.trees.append(tree)

    def _predict(self, X=None):
        y_pred = np.zeros(X.shape[0], np.float32)

        for i, tree in enumerate(self.trees):
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

    def predict(self, X=None):
        return self.loss.transform(self._predict(X))


class GradientBoostingRegressor(GradientBoosting):
    def fit(self, X, y=None):
        self.loss = LeastSquaresLoss()
        super(GradientBoostingRegressor, self).fit(X, y)


class GradientBoostingClassifier(GradientBoosting):
    def fit(self, X, y=None):
        # Convert labels from {0, 1} to {-1, 1}
        y = (y * 2) - 1
        self.loss = LogisticLoss()
        super(GradientBoostingClassifier, self).fit(X, y)
