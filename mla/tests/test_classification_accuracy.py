from sklearn.metrics import roc_auc_score

from mla.ensemble import RandomForestClassifier
from mla.ensemble.gbm import GradientBoostingClassifier
from mla.linear_models import LogisticRegression
from mla.metrics import accuracy
from mla.neuralnet import NeuralNet
from mla.neuralnet.constraints import MaxNorm
from mla.neuralnet.layers import Activation, Dense, Dropout
from mla.neuralnet.optimizers import Adadelta
from mla.neuralnet.parameters import Parameters
from mla.neuralnet.regularizers import L2
from mla.svm.kernerls import RBF, Linear
from mla.svm.svm import SVM
from mla.utils import one_hot

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


# All classifiers except convnet, RNN, LSTM.

def test_linear_model_classification():
    model = LogisticRegression(lr=0.01, max_iters=500, penalty='l1', C=0.01)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    assert roc_auc_score(y_test, predictions) >= 0.95


def test_random_forest_classification():
    model = RandomForestClassifier(n_estimators=10, max_depth=4)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)[:, 1]
    assert roc_auc_score(y_test, predictions) >= 0.95


def test_svm_classification():
    y_signed_train = (y_train * 2) - 1
    y_signed_test = (y_test * 2) - 1

    for kernel in [RBF(gamma=0.1), Linear()]:
        model = SVM(max_iter=250, kernel=kernel)
        model.fit(X_train, y_signed_train)
        predictions = model.predict(X_test)
        assert accuracy(y_signed_test, predictions) >= 0.8


def test_mlp_classification():
    y_train_onehot = one_hot(y_train)
    y_test_onehot = one_hot(y_test)

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
    model.fit(X_train, y_train_onehot)
    predictions = model.predict(X_test)
    assert roc_auc_score(y_test_onehot[:, 0], predictions[:, 0]) >= 0.95


def test_gbm_classification():
    model = GradientBoostingClassifier(n_estimators=25, max_depth=3,
                                       max_features=5, learning_rate=0.1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    assert roc_auc_score(y_test, predictions) >= 0.95


