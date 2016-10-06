import logging

from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

from mla.datasets import load_boston
from mla.ensemble.random_forest import RandomForestClassifier, RandomForestRegressor
from mla.metrics.metrics import mean_squared_error

logging.basicConfig(level=logging.DEBUG)


def classification():
    X, y = make_classification(n_samples=500, n_features=10, n_informative=10, random_state=1111, n_classes=2,
                               class_sep=2.5, n_redundant=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1111)

    model = RandomForestClassifier(n_estimators=10, max_depth=4)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)[:, 1]
    print('classification, roc auc score: %s' % roc_auc_score(y_test, predictions))


def regression():
    # X, y = make_regression(n_samples=500, n_features=5, n_informative=5, n_targets=1, noise=0.05, random_state=1111,
    #                        bias=0.5)
    X, y = load_boston()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1111)
    model = RandomForestRegressor(n_estimators=50, max_depth=10, max_features=3, )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print('regression, mse: %s' % mean_squared_error(y_test.flatten(), predictions.flatten()))


if __name__ == '__main__':
    # classification()
    regression()
