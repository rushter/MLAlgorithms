import autograd.numpy as np
from autograd import elementwise_grad

from mla.neuralnet.activations import get_activation
from mla.neuralnet.parameters import Parameters

np.random.seed(9999)


class Layer(object):
    def setup(self, X_shape):
        """Allocates initial weights."""
        pass

    def forward_pass(self, x):
        raise NotImplementedError()

    def backward_pass(self, delta):
        raise NotImplementedError()

    def shape(self, x_shape):
        """Returns shape of the current layer."""
        raise NotImplementedError()


class ParamMixin(object):
    @property
    def parameters(self):
        return self._params


class PhaseMixin(object):
    _train = False

    @property
    def is_training(self):
        return self._train

    @is_training.setter
    def is_training(self, is_train=True):
        self._train = is_train

    @property
    def is_testing(self):
        return not self._train

    @is_testing.setter
    def is_testing(self, is_test=True):
        self._train = not is_test


class Dense(Layer, ParamMixin):
    def __init__(self, output_dim, parameters=None, ):
        """A fully connected layer.

        Parameters
        ----------
        output_dim : int
        """
        self._params = parameters
        self.output_dim = output_dim
        self.last_input = None

        if parameters is None:
            self._params = Parameters()

    def setup(self, x_shape):
        self._params.setup_weights((x_shape[1], self.output_dim))

    def forward_pass(self, X):
        self.last_input = X
        return self.weight(X)

    def weight(self, X):
        W = np.dot(X, self._params['W'])
        return W + self._params['b']

    def backward_pass(self, delta):
        dW = np.dot(self.last_input.T, delta)
        db = np.sum(delta, axis=0)

        # Update gradient values
        self._params.update_grad('W', dW)
        self._params.update_grad('b', db)
        return np.dot(delta, self._params['W'].T)

    def shape(self, x_shape):
        return x_shape[0], self.output_dim


class Activation(Layer):
    def __init__(self, name):
        self.last_input = None
        self.activation = get_activation(name)
        # Derivative of activation function
        self.activation_d = elementwise_grad(self.activation)

    def forward_pass(self, X):
        self.last_input = X
        return self.activation(X)

    def backward_pass(self, delta):
        return self.activation_d(self.last_input) * delta

    def shape(self, x_shape):
        return x_shape


class Dropout(Layer, PhaseMixin):
    """Randomly set a fraction of `p` inputs to 0 at each training update."""

    def __init__(self, p=0.1):
        self.p = p
        self._mask = None

    def forward_pass(self, X):
        assert self.p > 0
        if self.is_training:
            self._mask = np.random.uniform(size=X.shape) > self.p
            y = X * self._mask
        else:
            y = X * (1.0 - self.p)

        return y

    def backward_pass(self, delta):
        return delta * self._mask

    def shape(self, x_shape):
        return x_shape


class TimeStepSlicer(Layer):
    """Take a specific time step from 3D tensor."""

    def __init__(self, step=-1):
        self.step = step

    def forward_pass(self, x):
        return x[:, self.step, :]

    def backward_pass(self, delta):
        return np.repeat(delta[:, np.newaxis, :], 2, 1)

    def shape(self, x_shape):
        return x_shape[0], x_shape[2]


class TimeDistributedDense(Layer):
    """Apply regular Dense layer to every timestep."""

    def __init__(self, output_dim):
        self.output_dim = output_dim
        self.n_timesteps = None
        self.dense = None
        self.input_dim = None

    def setup(self, X_shape):
        self.dense = Dense(self.output_dim)
        self.dense.setup((X_shape[0], X_shape[2]))
        self.input_dim = X_shape[2]

    def forward_pass(self, X):
        n_timesteps = X.shape[1]
        X = X.reshape(-1, X.shape[-1])
        y = self.dense.forward_pass(X)
        y = y.reshape((-1, n_timesteps, self.output_dim))
        return y

    def backward_pass(self, delta):
        n_timesteps = delta.shape[1]
        X = delta.reshape(-1, delta.shape[-1])
        y = self.dense.backward_pass(X)
        y = y.reshape((-1, n_timesteps, self.input_dim))
        return y

    @property
    def parameters(self):
        return self.dense._params

    def shape(self, x_shape):
        return x_shape[0], x_shape[1], self.output_dim
