import logging

import numpy as np
from autograd import elementwise_grad

from mla.base import BaseEstimator
from mla.metrics.metrics import get_metric
from mla.neuralnet.layers import PhaseMixin
from mla.neuralnet.loss import get_loss
from mla.utils import batch_iterator

np.random.seed(9999)

"""
Architecture inspired from:
https://github.com/fchollet/keras
https://github.com/andersbll/deeppy
"""


class NeuralNet(BaseEstimator):
    def __init__(self, layers, optimizer, loss, max_epochs=10, batch_size=64, random_seed=33, metric='mse',
                 shuffle=True):
        self.shuffle = shuffle
        self.optimizer = optimizer
        self.loss = get_loss(loss)

        # TODO: fix
        if loss == 'categorical_crossentropy':
            self.loss_grad = lambda actual, predicted: -(actual - predicted)
        else:
            self.loss_grad = elementwise_grad(self.loss, 1)
        self.metric = get_metric(metric)
        self.random_seed = random_seed
        self.layers = layers
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self._n_layers = 0
        self.log_metric = True if loss != metric else False
        self.metric_name = metric
        self.bprop_entry = self._find_bprop_entry()
        self.training = False
        self._initialized = False

    def _setup_layers(self, x_shape, ):
        x_shape = list(x_shape)
        x_shape[0] = self.batch_size

        for layer in self.layers:
            layer.setup(x_shape)
            x_shape = layer.shape(x_shape)

        self._n_layers = len(self.layers)
        self._initialized = True
        logging.info('Total parameters: %s' % self.n_params)

    def _find_bprop_entry(self):
        if len(self.layers) > 0 and not hasattr(self.layers[-1], 'parameters'):
            return -1
        return len(self.layers)

    def fit(self, X, y=None):
        if y.ndim == 1:
            # Reshape vector to matrix
            y = y[:, np.newaxis]
        self._setup_input(X, y)
        if not self._initialized:
            self._setup_layers(X.shape)

        self.is_training = True
        # Pass neural network instance to an optimizer
        self.optimizer.optimize(self)
        self.is_training = False

    def update(self, X, y):
        y_pred = self.fprop(X)
        grad = self.loss_grad(y, y_pred)
        for layer in reversed(self.layers[:self.bprop_entry]):
            grad = layer.backward_pass(grad)
        return self.loss(y, y_pred)

    def fprop(self, X):
        """Forward propagation."""
        for layer in self.layers:
            X = layer.forward_pass(X)
        return X

    def _predict(self, X=None):
        y = []
        X_batch = batch_iterator(X, self.batch_size)
        for Xb in X_batch:
            y.append(self.fprop(Xb))
        return np.concatenate(y)

    @property
    def parametric_layers(self):
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                yield layer

    @property
    def parameters(self):
        params = []
        for layer in self.parametric_layers:
            params.append(layer.parameters)
        return params

    def error(self, X=None, y=None):
        training_phase = self.is_training
        if training_phase:
            self.is_training = False
        if X is None and y is None:
            y_pred = self._predict(self.X)
            score = self.metric(self.y, y_pred)
        else:
            y_pred = self._predict(X)
            score = self.metric(y, y_pred)
        if training_phase:
            self.is_training = True
        return score

    @property
    def is_training(self):
        return self.training

    @is_training.setter
    def is_training(self, train):
        self.training = train
        for layer in self.layers:
            if isinstance(layer, PhaseMixin):
                layer.is_training = train

    def shuffle_dataset(self):
        n_samples = self.X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        self.X = self.X.take(indices, axis=0)
        self.y = self.y.take(indices, axis=0)

    @property
    def n_layers(self):
        return self._n_layers

    @property
    def n_params(self):
        return sum([layer.parameters.n_params for layer in self.parametric_layers])

    def reset(self):
        self._initialized = False
