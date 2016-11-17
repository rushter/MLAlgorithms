
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_classification
from scipy.spatial import distance

from mla import knn
from mla.metrics.metrics import accuracy


X, y = make_classification(n_samples=100, n_features=5, n_informative=5,
                           n_redundant=0, n_repeated=0, n_classes=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state=1111)

clf = knn.KNN(k=5, distance_func=distance.euclidean)

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print('classification accuracy', accuracy(y_test, predictions))
