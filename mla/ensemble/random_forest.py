import numpy as np

from mla.base import BaseEstimator
from mla.ensemble.base import information_gain, mse_criterion
from mla.ensemble.tree import Tree
from six.moves import range


class RandomForest(BaseEstimator):
    def __init__(self, n_estimators=10, max_features=None, min_samples_split=10, max_depth=None, criterion=None):
        """

        Parameters
        ----------
        n_estimators : int
            The number of decision tree.
        max_features : int
            The number of features to consider when looking for the best split.
        min_samples_split : int
            The minimum number of samples required to split an internal node.
        max_depth : int
            Maximum depth of the tree.
        criterion : str
            The function to measure the quality of a split.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.trees = []

    def fit(self, X, y):
        self._setup_input(X, y)
        if self.max_features is None:
            self.max_features = int(np.sqrt(X.shape[1]))
        else:
            assert (X.shape[1] > self.max_features)
        self._train()

    def _train(self):
        for tree in self.trees:
            tree.train(self.X, self.y, max_features=self.max_features, min_samples_split=self.min_samples_split,
                       max_depth=self.max_depth)


class RandomForestClassifier(RandomForest):
    def __init__(self, n_estimators=10, max_features=None, min_samples_split=10, max_depth=None, criterion='entropy'):
        super(RandomForestClassifier, self).__init__(n_estimators=n_estimators, max_features=max_features,
                                                     min_samples_split=min_samples_split, max_depth=max_depth,
                                                     criterion=criterion)

        if criterion == 'entropy':
            self.criterion = information_gain
        else:
            raise ValueError()

        # Initialize empty trees
        for _ in range(self.n_estimators):
            self.trees.append(Tree(criterion=self.criterion))

    def _predict(self, X=None):
        y_shape = np.unique(self.y).shape[0]
        predictions = np.zeros((X.shape[0], y_shape))

        for i in range(X.shape[0]):
            row_pred = np.zeros(y_shape)
            for tree in self.trees:
                row_pred += tree.predict_row(X[i, :])

            row_pred /= self.n_estimators
            predictions[i, :] = row_pred
        return predictions


class RandomForestRegressor(RandomForest):
    def __init__(self, n_estimators=10, max_features=None, min_samples_split=10, max_depth=None, criterion='mse'):
        super(RandomForestRegressor, self).__init__(n_estimators=n_estimators, max_features=max_features,
                                                    min_samples_split=min_samples_split, max_depth=max_depth)

        if criterion == 'mse':
            self.criterion = mse_criterion
        else:
            raise ValueError()

        # Initialize empty regression trees
        for _ in range(self.n_estimators):
            self.trees.append(Tree(regression=True, criterion=self.criterion))

    def _predict(self, X=None):
        predictions = np.zeros((X.shape[0], self.n_estimators))
        for i, tree in enumerate(self.trees):
            predictions[:, i] = tree.predict(X)
        return predictions.mean(axis=1)
