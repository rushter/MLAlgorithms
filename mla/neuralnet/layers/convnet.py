# coding:utf-8
import autograd.numpy as np

from mla.neuralnet.layers import Layer, ParamMixin
from mla.neuralnet.parameters import Parameters


class Convolution(Layer, ParamMixin):
    def __init__(self, n_filters=8, filter_shape=(3, 3), padding=(0, 0), stride=(1, 1), parameters=None):
        """A 2D convolutional layer.
        Input shape: (n_images, n_channels, height, width)

        Parameters
        ----------
        n_filters : int, default 8
            The number of filters (kernels).
        filter_shape : tuple(int, int), default (3, 3)
            The shape of the filters. (height, width)
        parameters : Parameters instance, default None
        stride : tuple(int, int), default (1, 1)
            The step of the convolution. (height, width).
        padding : tuple(int, int), default (0, 0)
            The number of pixel to add to each side of the input. (height, weight)

        """
        self.padding = padding
        self._params = parameters
        self.stride = stride
        self.filter_shape = filter_shape
        self.n_filters = n_filters
        if self._params is None:
            self._params = Parameters()

    def setup(self, X_shape):
        n_channels, self.height, self.width = X_shape[1:]

        W_shape = (self.n_filters, n_channels) + self.filter_shape
        b_shape = (self.n_filters)
        self._params.setup_weights(W_shape, b_shape)

    def forward_pass(self, X):
        n_images, n_channels, height, width = self.shape(X.shape)
        self.last_input = X
        self.col = image_to_column(X, self.filter_shape, self.stride, self.padding)
        self.col_W = self._params.W.reshape(self.n_filters, -1).T

        out = np.dot(self.col, self.col_W) + self._params.b
        out = out.reshape(n_images, height, width, -1).transpose(0, 3, 1, 2)
        return out

    def backward_pass(self, delta):
        delta = delta.transpose(0, 2, 3, 1).reshape(-1, self.n_filters)

        self._params.d_b = np.sum(delta, axis=0)
        d_W = np.dot(self.col.T, delta)
        self._params.d_W = d_W.transpose(1, 0).reshape(self._params.W.shape)

        d_c = np.dot(delta, self.col_W.T)
        return column_to_image(d_c, self.last_input.shape, self.filter_shape, self.stride, self.padding)

    def shape(self, x_shape):
        height, width = convoltuion_shape(self.height, self.width, self.filter_shape, self.stride, self.padding)
        return x_shape[0], self.n_filters, height, width


class MaxPooling(Layer):
    def __init__(self, pool_shape=(2, 2), stride=(1, 1), padding=(0, 0)):
        """Max pooling layer.
        Input shape: (n_images, n_channels, height, width)

        Parameters
        ----------
        pool_shape : tuple(int, int), default (2, 2)
        stride : tuple(int, int), default (1,1)
        padding : tuple(int, int), default (0,0)
        """
        self.pool_shape = pool_shape
        self.stride = stride
        self.padding = padding

    def forward_pass(self, X):
        self.last_input = X

        out_height, out_width = pooling_shape(self.pool_shape, X.shape, self.stride)
        n_images, n_channels, _, _ = X.shape

        col = image_to_column(X, self.pool_shape, self.stride, self.padding)
        col = col.reshape(-1, self.pool_shape[0] * self.pool_shape[1])

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        self.arg_max = arg_max
        return out.reshape(n_images, out_height, out_width, n_channels).transpose(0, 3, 1, 2)

    def backward_pass(self, delta):
        delta = delta.transpose(0, 2, 3, 1)

        pool_size = self.pool_shape[0] * self.pool_shape[1]
        y_max = np.zeros((delta.size, pool_size))
        y_max[np.arange(self.arg_max.size), self.arg_max.flatten()] = delta.flatten()
        y_max = y_max.reshape(delta.shape + (pool_size,))

        dcol = y_max.reshape(y_max.shape[0] * y_max.shape[1] * y_max.shape[2], -1)
        return column_to_image(dcol, self.last_input.shape, self.pool_shape, self.stride, self.padding)

    def shape(self, x_shape):
        h, w = convoltuion_shape(x_shape[2], x_shape[3], self.pool_shape, self.stride, self.padding)
        return x_shape[0], x_shape[1], h, w


class Flatten(Layer):
    """Flattens multidimensional input into 2D matrix."""

    def forward_pass(self, X):
        self.last_input_shape = X.shape
        return X.reshape((X.shape[0], -1))

    def backward_pass(self, delta):
        return delta.reshape(self.last_input_shape)

    def shape(self, x_shape):
        return x_shape[0], np.prod(x_shape[1:])


def image_to_column(images, filter_shape, stride, padding):
    """Rearrange image blocks into columns.

    Parameters
    ----------

    filter_shape : tuple(height, width)
    images : np.array, shape (n_images, n_channels, height, width)
    padding: tuple(height, width)
    stride : tuple (height, width)

    """
    n_images, n_channels, height, width = images.shape
    f_height, f_width = filter_shape
    out_height, out_width = convoltuion_shape(height, width, (f_height, f_width), stride, padding)
    images = np.pad(images, ((0, 0), (0, 0), padding, padding), mode='constant')

    col = np.zeros((n_images, n_channels, f_height, f_width, out_height, out_width))
    for y in range(f_height):
        y_bound = y + stride[0] * out_height
        for x in range(f_width):
            x_bound = x + stride[1] * out_width
            col[:, :, y, x, :, :] = images[:, :, y:y_bound:stride[0], x:x_bound:stride[1]]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(n_images * out_height * out_width, -1)
    return col


def column_to_image(columns, images_shape, filter_shape, stride, padding):
    """Rearrange columns into image blocks.

    Parameters
    ----------
    columns
    images_shape : tuple(n_images, n_channels, height, width)
    filter_shape : tuple(height, _width)
    stride : tuple(height, width)
    padding : tuple(height, width)
    """
    n_images, n_channels, height, width = images_shape
    f_height, f_width = filter_shape

    out_height, out_width = convoltuion_shape(height, width, (f_height, f_width), stride, padding)
    columns = columns.reshape(n_images, out_height, out_width, n_channels, f_height, f_width).transpose(0, 3, 4, 5, 1,
                                                                                                        2)

    img_h = height + 2 * padding[0] + stride[0] - 1
    img_w = width + 2 * padding[1] + stride[1] - 1
    img = np.zeros((n_images, n_channels, img_h, img_w))
    for y in range(f_height):
        y_bound = y + stride[0] * out_height
        for x in range(f_width):
            x_bound = x + stride[1] * out_width
            img[:, :, y:y_bound:stride[0], x:x_bound:stride[1]] += columns[:, :, y, x, :, :]

    return img[:, :, padding[0]:height + padding[0], padding[1]:width + padding[1]]


def convoltuion_shape(img_height, img_width, filter_shape, stride, padding):
    """Calculate output shape for convolution layer."""
    height = (img_height + 2 * padding[0] - filter_shape[0]) / float(stride[0]) + 1
    width = (img_width + 2 * padding[1] - filter_shape[1]) / float(stride[1]) + 1

    assert height % 1 == 0
    assert width % 1 == 0

    return int(height), int(width)


def pooling_shape(pool_shape, image_shape, stride):
    """Calculate output shape for pooling layer."""
    n_images, n_channels, height, width = image_shape

    height = (height - pool_shape[0]) / float(stride[0]) + 1
    width = (width - pool_shape[1]) / float(stride[1]) + 1

    assert height % 1 == 0
    assert width % 1 == 0

    return int(height), int(width)
