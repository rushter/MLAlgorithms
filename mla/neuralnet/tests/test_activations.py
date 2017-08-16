import sys
import numpy as np

from mla.neuralnet.activations import *

def test_softplus():
    # np.exp(z_max) will overflow
    z_max = np.log(sys.float_info.max) + 1.0e10
    # 1.0 / np.exp(z_min) will overflow
    z_min = np.log(sys.float_info.min) - 1.0e10
    inputs = np.array([0.0, 1.0, -1.0, z_min, z_max])
    outputs = np.array([0.69314718,  1.31326169,  0.31326169, 0.0, z_max])

    assert np.array_equal(outputs, softplus(inputs))


