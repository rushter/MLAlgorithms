import logging

from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_classification

from mla.metrics.metrics import *
from mla.svm.kernerls import *
from mla.svm.svm import SVM

logging.basicConfig(level=logging.DEBUG)


def classification():
    X, y = make_classification(n_samples=1200, n_features=10, n_informative=5, random_state=1111, n_classes=2,
                               class_sep=1.75, )
    # Convert y to {-1, 1}
    y = (y * 2) - 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1111)

    for kernel in [RBF(gamma=0.1), Linear()]:
        model = SVM(max_iter=500, kernel=kernel, C=0.6)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print('Classification accuracy (%s): %s' % (kernel, accuracy(y_test, predictions)))


if __name__ == '__main__':
    classification()
