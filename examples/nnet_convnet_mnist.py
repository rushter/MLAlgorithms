import logging

from mla.datasets import load_mnist
from mla.metrics import accuracy
from mla.neuralnet import NeuralNet
from mla.neuralnet.layers import Activation, Convolution, MaxPooling, Flatten, Dropout, Parameters
from mla.neuralnet.layers import Dense
from mla.neuralnet.optimizers import Adadelta
from mla.utils import one_hot

logging.basicConfig(level=logging.DEBUG)

X_train, X_test, y_train, y_test = load_mnist()

# Normalization
X_train /= 255.
X_test /= 255.

y_train = one_hot(y_train.flatten())
y_test = one_hot(y_test.flatten())
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Approx. 15-20 min. per epoch
model = NeuralNet(
    layers=[
        Convolution(n_filters=32, filter_shape=(3, 3), padding=(1, 1), stride=(1, 1)),
        Activation('relu'),
        Convolution(n_filters=32, filter_shape=(3, 3), padding=(1, 1), stride=(1, 1)),
        Activation('relu'),
        MaxPooling(pool_shape=(2, 2), stride=(2, 2)),
        Dropout(0.5),

        Flatten(),
        Dense(128),
        Activation('relu'),
        Dropout(0.5),
        Dense(10),
        Activation('softmax'),
    ],
    loss='categorical_crossentropy',
    optimizer=Adadelta(),
    metric='accuracy',
    batch_size=128,
    max_epochs=3,
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
print accuracy(y_test, predictions)
