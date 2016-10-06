import numpy as np


class BaseEstimator(object):
    X = None
    y = None
    y_required = True

    def _setup_input(self, X, y=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.size == 0:
            raise ValueError('Number of features must be > 0')

        if X.ndim == 1:
            self.n_samples, self.n_features = 1, X.shape
        else:
            self.n_samples, self.n_features = X.shape[0], np.prod(X.shape[1:])

        self.X = X

        if self.y_required:
            if y is None:
                raise ValueError('Missed required argument y')

            if not isinstance(y, np.ndarray):
                y = np.array(y)

            if y.size == 0:
                raise ValueError('Number of features must be > 0')

        self.y = y

    def fit(self, X, y=None):
        self._setup_input(X, y)

    def predict(self, X=None):
        if self.X is not None:
            return self._predict(X)
        else:
            raise ValueError('You must call `fit` before `predict`')

    def _predict(self, X=None):
        raise NotImplementedError()
