import logging
from itertools import combinations, islice

import numpy as np
from sklearn.model_selection import train_test_split

from mla.metrics import accuracy
from mla.neuralnet import NeuralNet
from mla.neuralnet.layers import Activation, TimeDistributedDense
from mla.neuralnet.layers.recurrent import LSTM
from mla.neuralnet.optimizers import Adam

logging.basicConfig(level=logging.DEBUG)


def addition_dataset(dim=10, n_samples=10000, batch_size=64):
    """Generate binary addition dataset.
    http://devankuleindiren.com/Projects/rnn_arithmetic.php
    """
    binary_format = '{:0' + str(dim) + 'b}'

    # Generate all possible number combinations
    combs = list(islice(combinations(range(2 ** (dim - 1)), 2), n_samples))

    # Initialize empty arrays
    X = np.zeros((len(combs), dim, 2), dtype=np.uint8)
    y = np.zeros((len(combs), dim, 1), dtype=np.uint8)

    for i, (a, b) in enumerate(combs):
        # Convert numbers to binary format
        X[i, :, 0] = list(reversed([int(x) for x in binary_format.format(a)]))
        X[i, :, 1] = list(reversed([int(x) for x in binary_format.format(b)]))

        # Generate target variable (a+b)
        y[i, :, 0] = list(reversed([int(x) for x in binary_format.format(a + b)]))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1111)

    # Round number of examples for batch processing
    train_b = (X_train.shape[0] // batch_size) * batch_size
    test_b = (X_test.shape[0] // batch_size) * batch_size
    X_train = X_train[0:train_b]
    y_train = y_train[0:train_b]

    X_test = X_test[0:test_b]
    y_test = y_test[0:test_b]
    return X_train, X_test, y_train, y_test


def addition_problem(ReccurentLayer):
    X_train, X_test, y_train, y_test = addition_dataset(8, 5000)

    print(X_train.shape, X_test.shape)
    model = NeuralNet(
        layers=[
            ReccurentLayer,
            TimeDistributedDense(1),
            Activation('sigmoid'),
        ],
        loss='mse',
        optimizer=Adam(),
        metric='mse',
        batch_size=64,
        max_epochs=15,
    )
    model.fit(X_train, y_train)
    predictions = np.round(model.predict(X_test))
    predictions = np.packbits(predictions.astype(np.uint8))
    y_test = np.packbits(y_test.astype(np.int))
    print(accuracy(y_test, predictions))


# RNN
# addition_problem(RNN(16, parameters=Parameters(constraints={'W': SmallNorm(), 'U': SmallNorm()})))
# LSTM
addition_problem(LSTM(16))
