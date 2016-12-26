import logging

from mla.base import BaseEstimator
from scipy.special import expit
import numpy as np

from mla.utils import batch_iterator

np.random.seed(9999)
sigmoid = expit

"""
References:
A Practical Guide to Training Restricted Boltzmann Machines https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
"""


# Warning: It's untested and unfinished implementation.

class RBM(BaseEstimator):
    y_required = False

    def __init__(self, n_hidden=128, learning_rate=0.1, batch_size=10, max_epochs=100):
        """Bernoulli Restricted Boltzmann Machine (RBM)

        Parameters
        ----------

        n_hidden : int, default 128
            The number of hidden units.
        learning_rate : float, default 0.1
        batch_size : int, default 10
        max_epochs : int, default 100
        """
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.n_hidden = n_hidden

    def fit(self, X, y=None):
        self.n_visible = X.shape[1]
        self._init_weights()
        self._setup_input(X, y)
        self._train()

    def _init_weights(self):

        self.W = np.random.randn(self.n_visible, self.n_hidden) * 0.1

        # Bias for visible and hidden units
        self.bias_v = np.zeros(self.n_visible, dtype=np.float32)
        self.bias_h = np.zeros(self.n_hidden, dtype=np.float32)

        self.errors = []

    def _train(self):

        for i in range(self.max_epochs):
            error = 0
            for batch in batch_iterator(self.X, batch_size=self.batch_size):
                positive_hidden = sigmoid(np.dot(batch, self.W) + self.bias_h)
                hidden_states = self._sample(positive_hidden)
                positive_associations = np.dot(batch.T, positive_hidden)

                negative_visible = sigmoid(np.dot(hidden_states, self.W.T) + self.bias_v)
                negative_hidden = sigmoid(np.dot(negative_visible, self.W) + self.bias_h)
                negative_associations = np.dot(negative_visible.T, negative_hidden)

                lr = self.lr / float(batch.shape[0])
                self.W += lr * ((positive_associations - negative_associations) / float(self.batch_size))
                self.bias_h += lr * (negative_hidden.sum(axis=0) - negative_associations.sum(axis=0))
                self.bias_v += lr * (np.asarray(batch.sum(axis=0)).squeeze() - negative_visible.sum(axis=0))

                error += np.sum((batch - negative_visible) ** 2)

            self.errors.append(error)
            logging.info('Iteration %s, error %s' % (i, error))
        logging.debug('Weights: %s' % self.W)
        logging.debug('Hidden bias: %s' % self.bias_h)
        logging.debug('Visible bias: %s' % self.bias_v)

    def _sample(self, X):
        return X > np.random.random_sample(size=X.shape)

    def _predict(self, X=None):
        return sigmoid(np.dot(X, self.W) + self.bias_h)
