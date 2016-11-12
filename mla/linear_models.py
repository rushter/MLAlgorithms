import logging

import autograd.numpy as np
from autograd import grad

from mla.base import BaseEstimator
from mla.metrics.metrics import mean_squared_error, binary_crossentropy

np.random.seed(1000)


class BasicRegression(BaseEstimator):
    def __init__(self, lr=0.001, penalty='None', C=0.01, tolerance=0.0001, max_iters=1000):
        """

        Parameters
        ----------
        lr : float, default 0.001
            Learning rate.
        penalty : str, {'l1', 'l2', None'}, default None
            Regularization function name.
        C : float, default 0.01
            The regularization coefficient.
        tolerance : float, default 0.0001
            If the gradient descent updates are smaller than `tolerance` then stop optimization process.
        max_iters : int, default 10000
            The maximum number of iterations.
        """
        self.C = C
        self.penalty = penalty
        self.tolerance = tolerance
        self.lr = lr
        self.max_iters = max_iters
        self.errors = []
        self.theta = []
        self.n_samples, self.n_features = None, None
        self.cost_func = None

    def _loss(self, w):
        raise NotImplementedError()

    def init_cost(self):
        raise NotImplementedError()

    def _add_penalty(self, loss, w):
        """Apply regularization to the loss."""
        if self.penalty == "l1":
            loss += self.C * np.abs(w[:-1]).sum()
        elif self.penalty == "l2":
            loss += (0.5 * self.C) * (w[:-1] ** 2).mean()
        return loss

    def _cost(self, X, y, theta):
        prediction = X.dot(theta)
        error = self.cost_func(y, prediction)
        return error

    def fit(self, X, y=None):
        self._setup_input(X, y)
        self.init_cost()
        self.n_samples, self.n_features = X.shape

        # Initialize weights + bias term
        self.theta = np.random.normal(size=(self.n_features + 1), scale=0.5)

        # Add an intercept column
        self.X = self._add_intercept(self.X)

        self._train()

    @staticmethod
    def _add_intercept(X):
        b = np.ones([X.shape[0], 1])
        return np.concatenate([b, X], axis=1)

    def _train(self):
        self.theta, self.errors = self._gradient_descent()
        logging.info(' Theta: %s' % self.theta.flatten())

    def _predict(self, X=None):
        X = self._add_intercept(X)
        return X.dot(self.theta)

    def _gradient_descent(self):
        theta = self.theta
        errors = [self._cost(self.X, self.y, theta)]

        for i in range(1, self.max_iters + 1):
            # Get derivative of the loss function
            cost_d = grad(self._loss)
            # Calculate gradient and update theta
            delta = cost_d(theta)
            theta -= self.lr * delta

            errors.append(self._cost(self.X, self.y, theta))
            logging.info('Iteration %s, error %s' % (i, errors[i]))

            error_diff = np.linalg.norm(errors[i - 1] - errors[i])
            if error_diff < self.tolerance:
                logging.info('Convergence has reached.')
                break
        return theta, errors


class LinearRegression(BasicRegression):
    """Linear regression with gradient descent optimizer."""

    def _loss(self, w):
        loss = self.cost_func(self.y, np.dot(self.X, w))
        return self._add_penalty(loss, w)

    def init_cost(self):
        self.cost_func = mean_squared_error


class LogisticRegression(BasicRegression):
    """Binary logistic regression with gradient descent optimizer."""

    def init_cost(self):
        self.cost_func = binary_crossentropy

    def _loss(self, w):
        loss = self.cost_func(self.y, self.sigmoid(np.dot(self.X, w)))
        return self._add_penalty(loss, w)

    @staticmethod
    def sigmoid(x):
        return 0.5 * (np.tanh(x) + 1)

    def _predict(self, X=None):
        X = self._add_intercept(X)
        return self.sigmoid(X.dot(self.theta))
