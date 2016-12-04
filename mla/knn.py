from collections import Counter

import numpy as np
from scipy.spatial.distance import euclidean

from mla.base import BaseEstimator


class KNNBase(BaseEstimator):

    def __init__(self, k=5, distance_func=euclidean):
        """Base for Nearest neighbors classifier and regressor.

        Parameters
        ----------
        k : int, default 5
            The number of neighbors to take into account. If 0, all the
            training examples are used.
        distance_func : function, default euclidean distance
            A distance function taking two arguments. Any function from
            scipy.spatial.distance will do.
        """

        self.k = None if k == 0 else k  # l[:None] returns the whole list
        self.distance_func = distance_func

    def aggregate(self, neighbors_targets):
        raise NotImplementedError()

    def _predict(self, X=None):

        predictions = [self._predict_x(x) for x in X]

        return np.array(predictions)

    def _predict_x(self, x):
        """Predict the label of a single instance x."""

        # compute distances between x and all examples in the training set.
        distances = (self.distance_func(x, example) for example in self.X)

        # Sort all examples by their distance to x and keep their target value.
        neighbors = sorted(((dist, target)
                           for (dist, target) in zip(distances, self.y)),
                           key=lambda x: x[0])

        # Get targets of the k-nn and aggregate them (most common one or
        # average).
        neighbors_targets = [target for (_, target) in neighbors[:self.k]]

        return self.aggregate(neighbors_targets)


class KNNClassifier(KNNBase):
    """Nearest neighbors classifier.

    Note: if there is a tie for the most common label among the neighbors, then
    the predicted label is arbitrary."""


    def aggregate(self, neighbors_targets):
        """Return the most common target label."""

        most_common_label = Counter(neighbors_targets).most_common(1)[0][0]
        return most_common_label


class KNNRegressor(KNNBase):
    """Nearest neighbors regressor."""

    def aggregate(self, neighbors_targets):
        """Return the mean of all targets."""

        return np.mean(neighbors_targets)
