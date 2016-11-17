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
        predictions = np.apply_along_axis(self._predict_proba, 1, X)
        # Normalize probabilities
        return softmax(predictions)

    def _predict_proba(self, x):
        """Predict log likelihood for given row."""
        output = []
        for y in range(self.n_classes):
            prior = np.log(self._priors[y])
            posterior = np.sum([self._pdf(y, d, x) for d in range(self.n_features)])
            prediction = prior + posterior

            output.append(prediction)
        return output

    def _pdf(self, n_class, n_feature, x):
        """Calculate probability density function for normal distribution."""
        # Take specific values
        mean = self._mean[n_class, n_feature]
        var = self._var[n_class, n_feature]
        x = x[n_feature]

        # Avoid division by zero
        if var < 1e-15:
            return 0.0

        numerator = np.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
