from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

from mla.neuralnet import NeuralNet
from mla.neuralnet.layers import Dense, Activation, Dropout, Parameters
from mla.neuralnet.optimizers import *
from mla.utils import one_hot


def clasifier(optimizer):
    X, y = make_classification(n_samples=1000, n_features=100, n_informative=75, random_state=1111, n_classes=2,
                               class_sep=2.5, )
    y = one_hot(y)

    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1111)

    model = NeuralNet(
        layers=[
            Dense(128, Parameters(init='uniform')),
            Activation('relu'),
            Dropout(0.5),
            Dense(64, Parameters(init='normal')),
            Activation('relu'),
            Dense(2),
            Activation('softmax'),
        ],
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metric='accuracy',
        batch_size=64,
        max_epochs=10,

    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return roc_auc_score(y_test[:, 0], predictions[:, 0])


def test_adadelta():
    assert clasifier(Adadelta()) > 0.95


def test_adam():
    assert clasifier(Adam()) > 0.95


def test_rmsprop():
    assert clasifier(RMSprop()) > 0.95


def test_adagrad():
    assert clasifier(Adagrad()) > 0.95


def test_sgd():
    assert clasifier(SGD(learning_rate=0.0001)) > 0.95
    assert clasifier(SGD(learning_rate=0.0001, nesterov=True, momentum=0.9)) > 0.95
    assert clasifier(SGD(learning_rate=0.0001, nesterov=False, momentum=0.0)) > 0.95
