try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_regression

from mla.knn import KNNRegressor
from mla.linear_models import LinearRegression
from mla.metrics.metrics import mean_squared_error
from mla.neuralnet import NeuralNet
from mla.neuralnet.layers import Activation, Dense
from mla.neuralnet.optimizers import Adam
from mla.neuralnet.parameters import Parameters


# Generate a random regression problem
X, y = make_regression(n_samples=1000, n_features=10,
                       n_informative=10, n_targets=1, noise=0.05,
                       random_state=1111, bias=0.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=1111)


def test_linear():
    model = LinearRegression(lr=0.01, max_iters=2000, penalty='l2', C=0.003)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    assert mean_squared_error(y_test, predictions) < 0.25


def test_mlp():
    model = NeuralNet(
        layers=[
            Dense(16, Parameters(init='normal')),
            Activation('linear'),
            Dense(8, Parameters(init='normal')),
            Activation('linear'),
            Dense(1),
        ],
        loss='mse',
        optimizer=Adam(),
        metric='mse',
        batch_size=64,
        max_epochs=150,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    assert mean_squared_error(y_test, predictions.flatten()) < 1.0


def test_knn():
    model = KNNRegressor(k=5)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    assert mean_squared_error(y_test, predictions) < 10000
