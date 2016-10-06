import autograd.numpy as np
from autograd import elementwise_grad
from six.moves import range

from mla.neuralnet.initializations import get_initializer
from mla.neuralnet.layers import Layer, get_activation, ParamMixin
from mla.neuralnet.parameters import Parameters


class RNN(Layer, ParamMixin):
    """Vanilla RNN."""

    def __init__(self, hidden_dim, activation='tanh', inner_init='orthogonal', parameters=None, return_sequences=True):
        self.return_sequences = return_sequences
        self.hidden_dim = hidden_dim
        self.inner_init = get_initializer(inner_init)
        self.activation = get_activation(activation)
        self.activation_d = elementwise_grad(self.activation)
        if parameters is None:
            self._params = Parameters()
        else:
            self._params = parameters
        self.last_input = None
        self.states = None
        self.hprev = None
        self.input_dim = None

    def setup(self, x_shape):
        """
        Parameters
        ----------
        x_shape : np.array(batch size, time steps, input shape)
        """
        self.input_dim = x_shape[2]

        # Input -> Hidden
        self._params['W'] = self._params.init((self.input_dim, self.hidden_dim))
        # Bias
        self._params['b'] = np.full((self.hidden_dim,), self._params.initial_bias)
        # Hidden -> Hidden layer
        self._params['U'] = self.inner_init((self.hidden_dim, self.hidden_dim))

        # Init gradient arrays
        self._params.init_grad()

        self.hprev = np.zeros((x_shape[0], self.hidden_dim))

    def forward_pass(self, X):
        self.last_input = X
        n_samples, n_timesteps, input_shape = X.shape
        states = np.zeros((n_samples, n_timesteps + 1, self.hidden_dim))
        states[:, -1, :] = self.hprev.copy()
        p = self._params

        for i in range(n_timesteps):
            states[:, i, :] = np.tanh(np.dot(X[:, i, :], p['W']) + np.dot(states[:, i - 1, :], p['U']) + p['b'])

        self.states = states
        self.hprev = states[:, n_timesteps - 1, :].copy()
        if self.return_sequences:
            return states[:, 0:-1, :]
        else:
            return states[:, -2, :]

    def backward_pass(self, delta):
        if len(delta.shape) == 2:
            delta = delta[:, np.newaxis, :]
        n_samples, n_timesteps, input_shape = delta.shape
        p = self._params

        # Temporal gradient arrays
        grad = {k: np.zeros_like(p[k]) for k in p.keys()}

        dh_next = np.zeros((n_samples, input_shape))
        output = np.zeros((n_samples, n_timesteps, self.input_dim))

        # Backpropagation through time
        for i in reversed(range(n_timesteps)):
            dhi = self.activation_d(self.states[:, i, :]) * (delta[:, i, :] + dh_next)

            grad['W'] += np.dot(self.last_input[:, i, :].T, dhi)
            grad['b'] += delta[:, i, :].sum(axis=0)
            grad['U'] += np.dot(self.states[:, i - 1, :].T, dhi)

            dh_next = np.dot(dhi, p['U'].T)

            d = np.dot(delta[:, i, :], p['U'].T)
            output[:, i, :] = np.dot(d, p['W'].T)

        # Change actual gradient arrays
        for k in grad.keys():
            self._params.update_grad(k, grad[k])
        return output

    def shape(self, x_shape):
        if self.return_sequences:
            return x_shape[0], x_shape[1], self.hidden_dim
        else:
            return x_shape[0], self.hidden_dim
