import logging

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.metrics import roc_auc_score

from mla.datasets import *
from mla.metrics.metrics import root_mean_squared_log_error, mean_squared_error
from mla.neuralnet import NeuralNet
from mla.neuralnet.constraints import MaxNorm, UnitNorm
from mla.neuralnet.layers import Activation, Dense, Dropout
from mla.neuralnet.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam
from mla.neuralnet.parameters import Parameters
from mla.neuralnet.regularizers import *
from mla.utils import one_hot

logging.basicConfig(level=logging.DEBUG)


def classification():
    X, y = make_classification(n_samples=1000, n_features=100, n_informative=75, random_state=1111, n_classes=2,
                               class_sep=2.5, )
    y = one_hot(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1111)

    model = NeuralNet(
        layers=[
            Dense(256, Parameters(init='uniform', regularizers={'W': L2(0.05)})),
            Activation('relu'),
            Dropout(0.5),
            Dense(128, Parameters(init='normal', constraints={'W': MaxNorm()})),
            Activation('relu'),
            Dense(2),
            Activation('softmax'),
        ],
        loss='categorical_crossentropy',
        optimizer=Adadelta(),
        metric='accuracy',
        batch_size=64,
        max_epochs=25,

    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print('classification accuracy', roc_auc_score(y_test[:, 0], predictions[:, 0]))


def regression():
    X, y = make_regression(n_samples=5000, n_features=25, n_informative=25, n_targets=1, random_state=100, noise=0.05)
    y *= 0.01
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1111)

    model = NeuralNet(
        layers=[
            Dense(64, Parameters(init='normal')),
            Activation('linear'),
            Dense(32, Parameters(init='normal')),
            Activation('linear'),
            Dense(1),
        ],
        loss='mse',
        optimizer=Adam(),
        metric='mse',
        batch_size=256,
        max_epochs=15,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("regression mse", mean_squared_error(y_test, predictions.flatten()))


if __name__ == '__main__':
    classification()
    regression()
