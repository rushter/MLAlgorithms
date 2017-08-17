import sys
import numpy as np

from mla.neuralnet.activations import *

def test_softplus():
    # np.exp(z_max) will overflow
    z_max = np.log(sys.float_info.max) + 1.0e10
    # 1.0 / np.exp(z_min) will overflow
    z_min = np.log(sys.float_info.min) - 1.0e10
    inputs = np.array([0.0, 1.0, -1.0, z_min, z_max])
    # naive implementation of np.log(1 + np.exp(z_max)) will overflow
    # naive implementation of z + np.log(1 + 1 / np.exp(z_min)) will
    # throw ZeroDivisionError
    outputs = np.array([
      np.log(2.0),
      np.log1p(np.exp(1.0)),
      np.log1p(np.exp(-1.0)),
      0.0,
      z_max
    ])

    assert np.allclose(outputs, softplus(inputs))


