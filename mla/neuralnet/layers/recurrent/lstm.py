import autograd.numpy as np
from autograd import elementwise_grad
from six.moves import range

from mla.neuralnet.activations import sigmoid
from mla.neuralnet.initializations import get_initializer
from mla.neuralnet.layers import Layer, get_activation, ParamMixin
from mla.neuralnet.parameters import Parameters

"""
References:
Understanding LSTM Networks http://colah.github.io/posts/2015-08-Understanding-LSTMs/
A Critical Review of Recurrent Neural Networks for Sequence Learning http://arxiv.org/pdf/1506.00019v4.pdf
"""


class LSTM(Layer, ParamMixin):
    def __init__(self, hidden_dim, activation='tanh', inner_init='orthogonal', parameters=None, return_sequences=True):
        self.return_sequences = return_sequences
        self.hidden_dim = hidden_dim
        self.inner_init = get_initializer(inner_init)
        self.activation = get_activation(activation)
        self.activation_d = elementwise_grad(self.activation)
        self.sigmoid_d = elementwise_grad(sigmoid)

        if parameters is None:
            self._params = Parameters()
        else:
            self._params = parameters

        self.last_input = None
        self.states = None
        self.outputs = None
        self.gates = None
        self.hprev = None
        self.input_dim = None
        self.W = None
        self.U = None

    def setup(self, x_shape):
        """
        Naming convention:
        i : input gate
        f : forget gate
        c : cell
        o : output gate

        Parameters
        ----------
        x_shape : np.array(batch size, time steps, input shape)
        """
        self.input_dim = x_shape[2]
        # Input -> Hidden
        W_params = ['W_i', 'W_f', 'W_o', 'W_c']
        # Hidden -> Hidden
        U_params = ['U_i', 'U_f', 'U_o', 'U_c']
        # Bias terms
        b_params = ['b_i', 'b_f', 'b_o', 'b_c']

        # Initialize params
        for param in W_params:
            self._params[param] = self._params.init((self.input_dim, self.hidden_dim))

        for param in U_params:
            self._params[param] = self.inner_init((self.hidden_dim, self.hidden_dim))

        for param in b_params:
            self._params[param] = np.full((self.hidden_dim,), self._params.initial_bias)

        # Combine weights for simplicity
        self.W = [self._params[param] for param in W_params]
        self.U = [self._params[param] for param in U_params]

        # Init gradient arrays for all weights
        self._params.init_grad()

        self.hprev = np.zeros((x_shape[0], self.hidden_dim))
        self.oprev = np.zeros((x_shape[0], self.hidden_dim))

    def forward_pass(self, X):
        n_samples, n_timesteps, input_shape = X.shape
        p = self._params
        self.last_input = X

        self.states = np.zeros((n_samples, n_timesteps + 1, self.hidden_dim))
        self.outputs = np.zeros((n_samples, n_timesteps + 1, self.hidden_dim))
        self.gates = {k: np.zeros((n_samples, n_timesteps, self.hidden_dim)) for k in ['i', 'f', 'o', 'c']}

        self.states[:, -1, :] = self.hprev
        self.outputs[:, -1, :] = self.oprev

        for i in range(n_timesteps):
            t_gates = np.dot(X[:, i, :], self.W) + np.dot(self.outputs[:, i - 1, :], self.U)

            # Input
            self.gates['i'][:, i, :] = sigmoid(t_gates[:, 0, :] + p['b_i'])
            # Forget
            self.gates['f'][:, i, :] = sigmoid(t_gates[:, 1, :] + p['b_f'])
            # Output
            self.gates['o'][:, i, :] = sigmoid(t_gates[:, 2, :] + p['b_o'])
            # Cell
            self.gates['c'][:, i, :] = self.activation(t_gates[:, 3, :] + p['b_c'])

            # (previous state * forget) + input + cell
            self.states[:, i, :] = self.states[:, i - 1, :] * self.gates['f'][:, i, :] + \
                                   self.gates['i'][:, i, :] * self.gates['c'][:, i, :]
            self.outputs[:, i, :] = self.gates['o'][:, i, :] * self.activation(self.states[:, i, :])

        self.hprev = self.states[:, n_timesteps - 1, :].copy()
        self.oprev = self.outputs[:, n_timesteps - 1, :].copy()

        if self.return_sequences:
            return self.outputs[:, 0:-1, :]
        else:
            return self.outputs[:, -2, :]

    def backward_pass(self, delta):
        if len(delta.shape) == 2:
            delta = delta[:, np.newaxis, :]

        n_samples, n_timesteps, input_shape = delta.shape

        # Temporal gradient arrays
        grad = {k: np.zeros_like(self._params[k]) for k in self._params.keys()}

        dh_next = np.zeros((n_samples, input_shape))
        output = np.zeros((n_samples, n_timesteps, self.input_dim))

        # Backpropagation through time
        for i in reversed(range(n_timesteps)):
            dhi = delta[:, i, :] * self.gates['o'][:, i, :] * self.activation_d(self.states[:, i, :]) + dh_next

            og = delta[:, i, :] * self.activation(self.states[:, i, :])
            de_o = og * self.sigmoid_d(self.gates['o'][:, i, :])

            grad['W_o'] += np.dot(self.last_input[:, i, :].T, de_o)
            grad['U_o'] += np.dot(self.outputs[:, i - 1, :].T, de_o)
            grad['b_o'] += de_o.sum(axis=0)

            de_f = (dhi * self.states[:, i - 1, :]) * self.sigmoid_d(self.gates['f'][:, i, :])
            grad['W_f'] += np.dot(self.last_input[:, i, :].T, de_f)
            grad['U_f'] += np.dot(self.outputs[:, i - 1, :].T, de_f)
            grad['b_f'] += de_f.sum(axis=0)

            de_i = (dhi * self.gates['c'][:, i, :]) * self.sigmoid_d(self.gates['i'][:, i, :])
            grad['W_i'] += np.dot(self.last_input[:, i, :].T, de_i)
            grad['U_i'] += np.dot(self.outputs[:, i - 1, :].T, de_i)
            grad['b_i'] += de_i.sum(axis=0)

            de_c = (dhi * self.gates['i'][:, i, :]) * self.activation_d(self.gates['c'][:, i, :])
            grad['W_c'] += np.dot(self.last_input[:, i, :].T, de_c)
            grad['U_c'] += np.dot(self.outputs[:, i - 1, :].T, de_c)
            grad['b_c'] += de_c.sum(axis=0)

            dh_next = dhi * self.gates['f'][:, i, :]

        # TODO: propagate error to the next layer

        # Change actual gradient arrays
        for k in grad.keys():
            self._params.update_grad(k, grad[k])
        return output

    def shape(self, x_shape):
        if self.return_sequences:
            return x_shape[0], x_shape[1], self.hidden_dim
        else:
            return x_shape[0], self.hidden_dim
