import logging

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression

from mla.linear_models import LinearRegression, LogisticRegression
from mla.metrics.metrics import *

# Change to DEBUG to see convergence
logging.basicConfig(level=logging.ERROR)


def regression():
    # Generate a random regression problem
    X, y = make_regression(n_samples=10000, n_features=100, n_informative=75, n_targets=1,
                           noise=0.05, random_state=1111, bias=0.5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1111)

    model = LinearRegression(lr=0.01, max_iters=2000, penalty='l2', C=0.03)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print('regression mse', mean_squared_error(y_test, predictions))


def classification():
    # Generate a random binary classification problem.
    X, y = make_classification(n_samples=1000, n_features=100, n_informative=75, random_state=1111, n_classes=2,
                               class_sep=2.5, )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1111)

    model = LogisticRegression(lr=0.01, max_iters=500, penalty='l1', C=0.01)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print('classification accuracy', accuracy(y_test, predictions))


if __name__ == '__main__':
    regression()
    classification()
