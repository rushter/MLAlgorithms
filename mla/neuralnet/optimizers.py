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
            if network.verbose:
                msg = "Epoch:%s, train loss: %s" % (i, loss)
                if network.log_metric:
                    msg += ", train %s: %s" % (network.metric_name, network.error())
                msg += ", elapsed: %s sec." % (time.time() - start_time)
                logging.info(msg)
        return loss_history

    def update(self, network):
        """Performs an update of parameters."""
        raise NotImplementedError

    def train_epoch(self, network):
        losses = []

        # Create batch iterator
        X_batch = batch_iterator(network.X, network.batch_size)
        y_batch = batch_iterator(network.y, network.batch_size)

        batch = zip(X_batch, y_batch)
        if network.verbose:
            batch = tqdm(batch, total=int(np.ceil(network.n_samples / network.batch_size)))

        for X, y in batch:
            loss = np.mean(network.update(X, y))
            self.update(network)
            losses.append(loss)

        epoch_loss = np.mean(losses)
        return epoch_loss

    def train_batch(self, network, X, y):
        loss = np.mean(network.update(X, y))
        self.update(network)
        return loss

    def setup(self, network):
        """Creates additional variables.
        Note: Must be called before optimization process."""
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9, decay=0.0, nesterov=False):
        self.nesterov = nesterov
        self.decay = decay
        self.momentum = momentum
        self.lr = learning_rate
        self.iteration = 0
        self.velocity = None

    def update(self, network):
        lr = self.lr * (1.0 / (1.0 + self.decay * self.iteration))

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

    def setup(self, network):
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

    def setup(self, network):
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
                self.accu[i][n] = self.rho * self.accu[i][n] + (1.0 - self.rho) * grad ** 2
                step = grad * np.sqrt(self.d_accu[i][n] + self.eps) / np.sqrt(self.accu[i][n] + self.eps)

                layer.parameters.step(n, -step * self.lr)
                # Update delta accumulator
                self.d_accu[i][n] = self.rho * self.d_accu[i][n] + (1.0 - self.rho) * step ** 2

    def setup(self, network):
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
                self.accu[i][n] = (self.rho * self.accu[i][n]) + (1.0 - self.rho) * (grad ** 2)
                step = self.lr * grad / (np.sqrt(self.accu[i][n]) + self.eps)
                layer.parameters.step(n, -step)

    def setup(self, network):
        # Accumulators
        self.accu = defaultdict(dict)
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                self.accu[i][n] = np.zeros_like(layer.parameters[n])


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):

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
                self.ms[i][n] = (self.beta_1 * self.ms[i][n]) + (1.0 - self.beta_1) * grad
                self.vs[i][n] = (self.beta_2 * self.vs[i][n]) + (1.0 - self.beta_2) * grad ** 2
                lr = self.lr * np.sqrt(1.0 - self.beta_2 ** self.t) / (1.0 - self.beta_1 ** self.t)

                step = lr * self.ms[i][n] / (np.sqrt(self.vs[i][n]) + self.epsilon)
                layer.parameters.step(n, -step)
        self.t += 1

    def setup(self, network):
        # Accumulators
        self.ms = defaultdict(dict)
        self.vs = defaultdict(dict)
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                self.ms[i][n] = np.zeros_like(layer.parameters[n])
                self.vs[i][n] = np.zeros_like(layer.parameters[n])

class Adamax(Optimizer):
    def __init__(self, learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-8):

        self.epsilon = epsilon
        self.beta_2 = beta_2
        self.beta_1 = beta_1
        self.lr = learning_rate
        self.t = 1

    def update(self, network):
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                grad = layer.parameters.grad[n]
                self.ms[i][n] = self.beta_1 * self.ms[i][n] + (1.0 - self.beta_1) * grad
                self.us[i][n] = np.maximum(self.beta_2 * self.us[i][n], np.abs(grad))

                step = self.lr / (1 - self.beta_1 ** self.t) * self.ms[i][n]/(self.us[i][n] + self.epsilon)
                layer.parameters.step(n, -step)
        self.t += 1

    def setup(self, network):
        self.ms = defaultdict(dict)
        self.us = defaultdict(dict)
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                self.ms[i][n] = np.zeros_like(layer.parameters[n])
                self.us[i][n] = np.zeros_like(layer.parameters[n])


class Nadam(Optimizer):

    def __init__(self, learning_rate=0.002, beta_1=0.9, beta_2=0.999, **kwargs):
        self.schedule_decay = kwargs.pop('schedule_decay', 0.004)
        self.epsilon = kwargs.pop('epsilon', K.epsilon())
        learning_rate = kwargs.pop('lr', learning_rate)
        super(Nadam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.m_schedule = K.variable(1., name='m_schedule')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')

    @interfaces.legacy_get_updates_support
    @K.symbolic
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        t = K.cast(self.iterations, K.floatx()) + 1

        # Due to the recommendations in [2], i.e. warming momentum schedule
        momentum_cache_t = self.beta_1 * (1. - 0.5 * (
            K.pow(K.cast_to_floatx(0.96), t * self.schedule_decay)))
        momentum_cache_t_1 = self.beta_1 * (1. - 0.5 * (
            K.pow(K.cast_to_floatx(0.96), (t + 1) * self.schedule_decay)))
        m_schedule_new = self.m_schedule * momentum_cache_t
        m_schedule_next = self.m_schedule * momentum_cache_t * momentum_cache_t_1
        self.updates.append((self.m_schedule, m_schedule_new))

        shapes = [K.int_shape(p) for p in params]
        ms = [K.zeros(shape, name='m_' + str(i))
              for (i, shape) in enumerate(shapes)]
        vs = [K.zeros(shape, name='v_' + str(i))
              for (i, shape) in enumerate(shapes)]

        self.weights = [self.iterations, self.m_schedule] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            # the following equations given in [1]
            g_prime = g / (1. - m_schedule_new)
            m_t = self.beta_1 * m + (1. - self.beta_1) * g
            m_t_prime = m_t / (1. - m_schedule_next)
            v_t = self.beta_2 * v + (1. - self.beta_2) * K.square(g)
            v_t_prime = v_t / (1. - K.pow(self.beta_2, t))
            m_t_bar = (1. - momentum_cache_t) * g_prime + (
                momentum_cache_t_1 * m_t_prime)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))

            p_t = (p - self.learning_rate * m_t_bar / (K.sqrt(v_t_prime) +
                   self.epsilon))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def set_weights(self, weights):
        params = self.weights
        # Override set_weights for backward compatibility of Keras 2.2.4 optimizer
        # since it does not include m_schedule at head of the weight list. Set
        # m_schedule to 1.
        if len(params) == len(weights) + 1:
            weights = [weights[0]] + [np.array(1.)] + weights[1:]
        super(Nadam, self).set_weights(weights)

    def get_config(self):
        config = {'learning_rate': float(K.get_value(self.learning_rate)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'epsilon': self.epsilon,
                  'schedule_decay': self.schedule_decay}
        base_config = super(Nadam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
