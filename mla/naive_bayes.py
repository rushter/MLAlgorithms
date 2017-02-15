import numpy as np
from mla.base import BaseEstimator
from mla.neuralnet.activations import softmax


class NaiveBayesClassifier(BaseEstimator):
    """Gaussian Naive Bayes."""
    # Binary problem.
    n_classes = 2

    def fit(self, X, y=None):
        self._setup_input(X, y)
        # Check target labels
        assert list(np.unique(y)) == [0, 1]

        # Mean and variance for each class and feature combination
        self._mean = np.zeros((self.n_classes, self.n_features), dtype=np.float64)
        self._var = np.zeros((self.n_classes, self.n_features), dtype=np.float64)

        self._priors = np.zeros(self.n_classes, dtype=np.float64)

        for c in range(self.n_classes):
            # Filter features by class
            X_c = X[y == c]

            # Calculate mean, variance, prior for each class
            self._mean[c, :] = X_c.mean(axis=0)
            self._var[c, :] = X_c.var(axis=0)
            self._priors[c] = X_c.shape[0] / float(X.shape[0])

    def _predict(self, X=None):
        # Apply _predict_proba for each row
        predictions = np.apply_along_axis(self._predict_row, 1, X)

        # Normalize probabilities so that each row will sum up to 1.0
        return softmax(predictions)

    def _predict_row(self, x):
        """Predict log likelihood for given row."""
        output = []
        for y in range(self.n_classes):
            prior = np.log(self._priors[y])
            posterior = np.log(self._pdf(y, x)).sum()
            prediction = prior + posterior

            output.append(prediction)
        return output

    def _pdf(self, n_class, x):
        """Calculate Gaussian PDF for each feature."""

        mean = self._mean[n_class]
        var = self._var[n_class]

        numerator = np.exp(-(x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
