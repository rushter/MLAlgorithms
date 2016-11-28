import random
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mla.base import BaseEstimator
from mla.kmeans import KMeans


class GaussianMixture(BaseEstimator):
    """Gaussian Mixture Model: clusters with Gaussian prior.

    Finds clusters by repeatedly performing Expectation–Maximization (EM) algorithm
    on the dataset. GMM assumes the datasets is distributed in multivariate Gaussian,
    and tries to find the underlying structure of the Gaussian, i.e. mean and covariance.
    E-step computes the "responsibility" of the data to each cluster, given the mean
    and covariance; M-step computes the mean, covariance and weights (prior of each
    cluster), given the responsibilities. It iterates until the total likelihood
    changes less than the tolerance.


    Parameters
    ----------

    K : int
        The number of clusters into which the dataset is partitioned.
    max_iters: int
        The maximum iterations of assigning points to the perform EM.
        Short-circuited by the assignments converging on their own.
    init: str, default 'random'
        The name of the method used to initialize the first clustering.

        'random' - Randomly select values from the dataset as the K centroids.
        'kmeans' - Initialize the centroids, covariances, weights with KMeams's clusters.
    tolerance: float, default 1e-3
        The tolerance of difference of the two latest likelihood for convergence.
    """

    y_required = False

    def __init__(self, K=4, init='random', max_iters=500, tolerance=1e-3):
        self.K = K
        self.max_iters = max_iters
        self.init = init
        self.assignments = None
        self.likelihood = []
        self.tolerance = tolerance

    def fit(self, X, y=None):
        '''Perform Expectation–Maximization (EM) until converged.'''
        self._setup_input(X, y)
        self._initialize()
        for i in range(self.max_iters):
            self._E_step()
            self._M_step()
            if self._is_converged():
                break

    def _initialize(self):
        """Set the initial weights, means and covs (with full covariance matrix).

        weights: the prior of the clusters (what percentage of data does a cluster have)
        means: the mean points of the clusters
        covs: the covariance matrix of the clusters
        """
        self.weights = np.ones(self.K)
        if self.init == 'random':
            self.means = [self.X[x] for x in random.sample(range(self.n_samples), self.K)]
            self.covs = [np.cov(self.X.T) for _ in range(K)]

        elif self.init == 'kmeans':
            kmeans = KMeans(K=self.K, max_iters=self.max_iters // 3, init='++')
            kmeans.fit(self.X)
            self.assignments = kmeans.predict()
            self.means = kmeans.centroids
            self.covs = []
            for i in np.unique(self.assignments):
                self.weights[int(i)] = (self.assignments == i).sum()
                self.covs.append(np.cov(self.X[self.assignments == i].T))
        else:
            raise ValueError('Unknown type of init parameter')
        self.weights /= self.weights.sum()

    def _E_step(self):
        '''Expectation(E-step) for Gaussian Mixture.'''
        likelihoods = self._get_likelihood(self.X)
        self.likelihood.append(likelihoods.sum())
        weighted_likelihoods = self._get_weighted_likelihood(likelihoods)
        self.assignments = weighted_likelihoods.argmax(axis=1)
        weighted_likelihoods /= weighted_likelihoods.sum(axis=1)[:, np.newaxis]
        self.responsibilities = weighted_likelihoods

    def _M_step(self):
        '''Maximization (M-step) for Gaussian Mixture.'''
        weights = self.responsibilities.sum(axis=0)
        for assignment in range(self.K):
            resp = self.responsibilities[:, assignment][:, np.newaxis]
            self.means[assignment] = (resp * self.X).sum(axis=0) / resp.sum()
            self.covs[assignment] = (self.X - self.means[assignment]).T.dot(
                (self.X - self.means[assignment]) * resp) / weights[assignment]
        self.weights = weights / weights.sum()

    def _is_converged(self):
        """Check if the difference of the latest two likelihood is less than the tolerance."""
        if (len(self.likelihood) > 1) and (self.likelihood[-1] - self.likelihood[-2] <= self.tolerance):
            return True
        return False

    def _predict(self, X):
        '''Get the assignments for X with GMM clusters.'''
        if not X.shape:
            return self.assignments
        likelihoods = self._get_likelihood(X)
        weighted_likelihoods = self._get_weighted_likelihood(likelihoods)
        assignments = weighted_likelihoods.argmax(axis=1)
        return assignments

    def _get_likelihood(self, data):
        n_data = data.shape[0]
        likelihoods = np.zeros([n_data, self.K])
        for c in range(self.K):
            likelihoods[:, c] = multivariate_normal.pdf(data, self.means[c], self.covs[c])
        return likelihoods

    def _get_weighted_likelihood(self, likelihood):
        return self.weights * likelihood

    def plot(self, data=None, ax=None, holdon=False):
        '''Plot contour for 2D data.'''
        if not (len(self.X.shape) == 2 and self.X.shape[1] == 2):
            raise AttributeError("Only support for visualizing 2D data.")

        if ax is None:
            _, ax = plt.subplots()

        if data is None:
            data = self.X
            assignments = self.assignments
        else:
            assignments = self.predict(data)

        COLOR = 'bgrcmyk'
        cmap = lambda assignment: COLOR[int(assignment) % len(COLOR)]

        # generate grid
        delta = .025
        margin = .2
        xmax, ymax = self.X.max(axis=0) + margin
        xmin, ymin = self.X.min(axis=0) - margin
        axis_X, axis_Y = np.meshgrid(np.arange(xmin, xmax, delta),
                                     np.arange(ymin, ymax, delta))

        def grid_gaussian_pdf(mean, cov):
            grid_array = np.array(list(zip(axis_X.flatten(), axis_Y.flatten())))
            return multivariate_normal.pdf(grid_array, mean, cov).reshape(axis_X.shape)

        # plot scatters
        if assignments is None:
            c = None
        else:
            c = [cmap(assignment) for assignment in assignments]
        ax.scatter(data[:, 0], data[:, 1], c=c)

        # plot contours
        for assignment in range(self.K):
            ax.contour(axis_X, axis_Y, grid_gaussian_pdf(self.means[assignment], self.covs[assignment]),
                       colors=cmap(assignment))

        if not holdon:
            plt.show()
