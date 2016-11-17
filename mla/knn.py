from collections import Counter

import numpy as np
from scipy.spatial.distance import euclidean

from mla.base import BaseEstimator


class KNN(BaseEstimator):

    def __init__(self, k=5, distance_func=euclidean):
        """Nearest neighbors classifier.

        Note: if there is a tie for the most common label among the neighbors,
        then the predicted label is arbitrary.

        Parameters
        ----------
        k : int, default 5
            The number of neighbors to take into account.
        distance_func : function, default euclidean distance
            A distance function taking two arguments. Any function from
            scipy.spatial.distance will do.
        """

        self.k = k
        self.distance_func = distance_func

    def _predict(self, X=None):

        predictions = [self._predict_x(x) for x in X]

        return np.array(predictions)

    def _predict_x(self, x):
        """Predict the label of a single instance x."""

        # compute distances between x and all examples in the training set.
        distances = [self.distance_func(x, example) for example in self.X]

        # Sort all examples by their distance to x and keep their label.
        neighbors = sorted(((dist, label)
                           for (dist, label) in zip(distances, self.y)),
                           key=lambda x: x[0])

        # Get labels of the k-nn and compute the most common one.
        neighbors_labels = [label for (_, label) in neighbors[:self.k]]
        most_common_label = Counter(neighbors_labels).most_common(1)[0][0]

        return most_common_label
