import logging
import time
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from mla.utils import batch_iterator

"""
References:
Gradient descent optimization algorithms  http://sebastianruder.com/optimizing-gradient-descent/index.html
"""


class Optimizer(object):
    def optimize(self, network):
        loss_history = []
        for i in range(network.max_epochs):
            if network.shuffle:
                network.shuffle_dataset()
            start_time = time.time()
            loss = self.train_epoch(network)
            loss_history.append(loss)
            msg = "Epoch:%s, train loss: %s" % (i, loss)
            if network.log_metric:
                msg += ', train %s: %s' % (network.metric_name, network.error())
            msg += ', elapsed: %s sec.' % (time.time() - start_time)
            logging.info(msg)
        return loss_history

    def update(self, network):
        raise NotImplementedError

    def train_epoch(self, network):
        self._setup(network)
        losses = []

        X_batch = batch_iterator(network.X, network.batch_size)
        y_batch = batch_iterator(network.y, network.batch_size)
        for X, y in tqdm(zip(X_batch, y_batch), 'Epoch progress'):
            loss = np.mean(network.update(X, y))
            self.update(network)
            losses.append(loss)
        epoch_loss = np.mean(losses)
        return epoch_loss

    def _setup(self, network):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9, decay=0., nesterov=False):
        self.nesterov = nesterov
        self.decay = decay
        self.momentum = momentum
        self.lr = learning_rate
        self.iteration = 0
        self.velocity = None

    def update(self, network):
        lr = self.lr * (1. / (1. + self.decay * self.iteration))

        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                # Get gradient values
                grad = layer.parameters.grad[n]
                update = self.momentum * self.velocity[i][n] - lr * grad
                self.velocity[i][n] = update
                if self.nesterov:
                    # Adjust using updated velocity
                    update = self.momentum * self.velocity[i][n] - lr * grad
                layer.parameters.step(n, update)
        self.iteration += 1

    def _setup(self, network):
        self.velocity = defaultdict(dict)
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                self.velocity[i][n] = np.zeros_like(layer.parameters[n])


class Adagrad(Optimizer):
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        self.eps = epsilon
        self.lr = learning_rate

    def update(self, network):
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                grad = layer.parameters.grad[n]
                self.accu[i][n] += grad ** 2
                step = self.lr * grad / (np.sqrt(self.accu[i][n]) + self.eps)
                layer.parameters.step(n, -step)

    def _setup(self, network):
        # Accumulators
        self.accu = defaultdict(dict)
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                self.accu[i][n] = np.zeros_like(layer.parameters[n])


class Adadelta(Optimizer):
    def __init__(self, learning_rate=1.0, rho=0.95, epsilon=1e-8):
        self.rho = rho
        self.eps = epsilon
        self.lr = learning_rate

    def update(self, network):
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                grad = layer.parameters.grad[n]
                self.accu[i][n] = self.rho * self.accu[i][n] + (1. - self.rho) * grad ** 2
                step = grad * np.sqrt(self.d_accu[i][n] + self.eps) / np.sqrt(
                    self.accu[i][n] + self.eps)

                layer.parameters.step(n, -step * self.lr)
                # Update delta accumulator
                self.d_accu[i][n] = self.rho * self.d_accu[i][n] + (1. - self.rho) * step ** 2

    def _setup(self, network):
        # Accumulators
        self.accu = defaultdict(dict)
        self.d_accu = defaultdict(dict)
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                self.accu[i][n] = np.zeros_like(layer.parameters[n])
                self.d_accu[i][n] = np.zeros_like(layer.parameters[n])


class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-8):
        self.eps = epsilon
        self.rho = rho
        self.lr = learning_rate

    def update(self, network):
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                grad = layer.parameters.grad[n]
                self.accu[i][n] = (self.rho * self.accu[i][n]) + (1. - self.rho) * (grad ** 2)
                step = self.lr * grad / (np.sqrt(self.accu[i][n]) + self.eps)
                layer.parameters.step(n, -step)

    def _setup(self, network):
        # Accumulators
        self.accu = defaultdict(dict)
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                self.accu[i][n] = np.zeros_like(layer.parameters[n])


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, ):

        self.epsilon = epsilon
        self.beta_2 = beta_2
        self.beta_1 = beta_1
        self.lr = learning_rate
        self.iterations = 0
        self.t = 1

    def update(self, network):
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                grad = layer.parameters.grad[n]
                self.ms[i][n] = (self.beta_1 * self.ms[i][n]) + (1. - self.beta_1) * grad
                self.vs[i][n] = (self.beta_2 * self.vs[i][n]) + (1. - self.beta_2) * grad ** 2
                lr = self.lr * np.sqrt(1. - self.beta_2 ** self.t) / (1. - self.beta_1 ** self.t)

                step = lr * self.ms[i][n] / (np.sqrt(self.vs[i][n]) + self.epsilon)
                layer.parameters.step(n, -step)
        self.t += 1

    def _setup(self, network):
        # Accumulators
        self.ms = defaultdict(dict)
        self.vs = defaultdict(dict)
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                self.ms[i][n] = np.zeros_like(layer.parameters[n])
                self.vs[i][n] = np.zeros_like(layer.parameters[n])
