from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from mla.naive_bayes import NaiveBayesClassifier


def classification():
    # Generate a random binary classification problem.
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=10,
                               random_state=1111, n_classes=2, class_sep=2.5,
                               n_redundant=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                        random_state=1111)

    model = NaiveBayesClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)[:, 1]

    print('classification accuracy', roc_auc_score(y_test, predictions))


if __name__ == '__main__':
    classification()
