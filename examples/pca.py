from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from mla.linear_models import LogisticRegression
from mla.metrics import accuracy
from mla.pca import PCA

# logging.basicConfig(level=logging.DEBUG)

X, y = make_classification(n_samples=1000, n_features=100, n_informative=75, random_state=1111, n_classes=2,
                           class_sep=2.5, )

for s in ['svd', 'eigen']:
    p = PCA(15, solver=s)
    p.fit(X)
    X = p.transform(X)
    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1111)
    model = LogisticRegression(lr=0.001, max_iters=2500)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print('Classification accuracy for %s PCA: %s' % (s, accuracy(y_test, predictions)))
