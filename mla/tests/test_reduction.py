from sklearn.metrics import roc_auc_score

from mla.ensemble import RandomForestClassifier

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_classification


# Generate a random regression problem
X, y = make_classification(n_samples=750, n_features=10,
                           n_informative=8, random_state=1111,
                           n_classes=2, class_sep=2.5, n_redundant=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.12,
                                                    random_state=1111)


def test_random_forest():
    model = RandomForestClassifier(n_estimators=10, max_depth=4)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)[:, 1]
    assert roc_auc_score(y_test, predictions) >= 0.95