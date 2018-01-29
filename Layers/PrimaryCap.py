import tensorflow as tf
from Squash import *

def PrimaryCap(_inputs,
               _n_channels,
               _dim_vector,
               _kernel_size,
               _strides,
               _padding):
    output = tf.layers.conv2d(inputs      = _inputs,
                              filters     = _n_channels * _dim_vector,
                              kernel_size = _kernel_size,
                              strides     = _strides,
                              padding     = _padding,
                              activation  = tf.nn.relu,
                              name        = 'conv2d')
    _old_shape = output.shape
    _new_shape = [-1, _old_shape[1] * _old_shape[2] *_n_channels, _dim_vector, 1]
    output     = tf.reshape(output, _new_shape)
    output     = squash(_input = output, _axis = -2)
    return output
