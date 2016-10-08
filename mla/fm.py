from mla.base import BaseEstimator
from mla.metrics import mean_squared_error, binary_crossentropy
import autograd.numpy as np
from autograd import elementwise_grad

np.random.seed(9999)

"""
References:
Factorization Machines http://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
"""


class BaseFM(BaseEstimator):
    def __init__(self, n_components=10, max_iter=100, init_stdev=0.1, learning_rate=0.01, reg_v=0.1,
                 reg_w=0.5, reg_w0=0.):
        """Simplified factorization machines implementation using SGD optimizer."""
        self.reg_w0 = reg_w0
        self.reg_w = reg_w
        self.reg_v = reg_v
        self.n_components = n_components
        self.lr = learning_rate
        self.init_stdev = init_stdev
        self.max_iter = max_iter
        self.loss = mean_squared_error
        self.loss_grad = elementwise_grad(mean_squared_error)

    def fit(self, X, y=None):
        self._setup_input(X, y)
        # bias
        self.wo = 0.0
        # Feature weights
        self.w = np.zeros(self.n_features)
        # Factor weights
        self.v = np.random.normal(scale=self.init_stdev, size=(self.n_features, self.n_components))
        self._train()

    def _train(self):
        for epoch in range(self.max_iter):
            y_pred = self._predict(self.X)
            loss = self.loss_grad(self.y, y_pred)
            w_grad = np.dot(loss, self.X) / float(self.n_samples)
            self.wo -= self.lr * (loss.mean() + 2 * self.reg_w0 * self.wo)
            self.w -= self.lr * w_grad + (2 * self.reg_w * self.w)
            self._factor_step(loss)

    def _factor_step(self, loss):
        for ix, x in enumerate(self.X):
            for i in range(self.n_features):
                v_grad = loss[ix] * (x.dot(self.v).dot(x[i])[0] - self.v[i] * x[i] ** 2)
                self.v[i] -= self.lr * v_grad + (2 * self.reg_v * self.v[i])

    def _predict(self, X=None):
        linear_output = np.dot(X, self.w)
        factors_output = np.sum(np.dot(X, self.v) ** 2 - np.dot(X ** 2, self.v ** 2), axis=1) / 2.
        return self.wo + linear_output + factors_output


class FMRegressor(BaseFM):
    pass


class FMClassifier(BaseFM):
    def fit(self, X, y=None):
        super(FMClassifier, self).fit(X, y)
        self.loss = binary_crossentropy
        self.loss_grad = elementwise_grad(binary_crossentropy)

    def predict(self, X=None):
        predictions = self._predict(X)
        return np.sign(predictions)
