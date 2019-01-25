# coding=utf-8
import pytest

from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

from mla.ensemble import RandomForestClassifier
from mla.pca import PCA


@pytest.fixture
def dataset():
    # Generate a random binary classification problem.
    return make_classification(n_samples=1000, n_features=100, n_informative=75,
                               random_state=1111, n_classes=2, class_sep=2.5, )


# TODO: fix
@pytest.mark.skip()
def test_PCA(dataset):
    X, y = dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=1111)
    p = PCA(50, solver='eigen')

    # fit PCA with training set, not the entire dataset
    p.fit(X_train)
    X_train_reduced = p.transform(X_train)
    X_test_reduced = p.transform(X_test)

    model = RandomForestClassifier(n_estimators=25, max_depth=5)
    model.fit(X_train_reduced, y_train)
    predictions = model.predict(X_test_reduced)[:, 1]
    score = roc_auc_score(y_test, predictions)
    assert score >= 0.75
