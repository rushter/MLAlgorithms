from sklearn.metrics import roc_auc_score

from mla.pca import PCA
from mla.ensemble import RandomForestClassifier

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_classification

# Generate a random binary classification problem.
X, y = make_classification(n_samples=1000, n_features=100, n_informative=75,
                           random_state=1111, n_classes=2, class_sep=2.5, )


def test_PCA():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=1111)
    p = PCA(100, solver='eigen')

    # fit PCA with training data, not the entire dataset
    p.fit(X_train)
    X_train_reduced = p.transform(X_train)
    X_test_reduced = p.transform(X_test)

    model = RandomForestClassifier(n_estimators=10, max_depth=4)
    model.fit(X_train_reduced, y_train)
    predictions = model.predict(X_test_reduced)[:, 1]
    print(roc_auc_score(y_test, predictions))
    assert roc_auc_score(y_test, predictions) >= 0.80
