from scipy.linalg import svd
import numpy as np
import logging

from mla.base import BaseEstimator

np.random.seed(1000)


class PCA(BaseEstimator):
    y_required = False

    def __init__(self, n_components, solver='svd'):
        """Principal component analysis (PCA) implementation.

        Transforms a dataset of possibly correlated values into n linearly
        uncorrelated components. The components are ordered such that the first
        has the largest possible variance and each following component as the
        largest possible variance given the previous components. This causes
        the early components to contain most of the variability in the dataset.

        Parameters
        ----------
        n_components : int
        solver : str, default 'svd'
            {'svd', 'eigen'}
        """
        self.solver = solver
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X, y=None):
        self.mean = np.mean(X, axis=0)
        self._decompose(X)

    def _decompose(self, X):
        # Mean centering
        X = X.copy()
        X -= self.mean

        if self.solver == 'svd':
            _, s, Vh = svd(X, full_matrices=True)
        elif self.solver == 'eigen':
            s, Vh = np.linalg.eig(np.cov(X.T))
            Vh = Vh.T

        s_squared = s ** 2
        variance_ratio = s_squared / (s_squared).sum()
        logging.info('Explained variance ratio: %s' % (variance_ratio[0:self.n_components]))
        self.components = Vh[0:self.n_components]

    def transform(self, X):
        X = X.copy()
        X -= self.mean
        return np.dot(X, self.components.T)

    def _predict(self, X=None):
        return self.transform(X)
