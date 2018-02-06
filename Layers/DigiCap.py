import tensorflow as tf
from Squash import *
import numpy

def DigiCap(_inputs,
            _name,
            _num_capsule,
            _dim_vector,
            _num_routing,
            _batch_size):
    _input_shape = _inputs.shape

    # ----- Create variables -----
    with tf.variable_scope(_name):
        W = tf.get_variable(name        = 'W',
                            shape       = (1, _input_shape[1], _num_capsule, _input_shape[2], _dim_vector),
                            dtype       = tf.float32,
                            initializer = tf.random_normal_initializer(0.01))
        # b = tf.get_variable(name        = 'b',
        #                     shape       = (1, _input_shape[1], _num_capsule, 1, 1),
        #                     dtype       = tf.float32,
        #                     initializer = tf.zeros_initializer(),
        #                     trainable   = False)

        biases = tf.get_variable(name  = 'bias',
                                 shape = (1, 1, _num_capsule, _dim_vector, 1))

        b = tf.constant(numpy.zeros([25, _input_shape[1], _num_capsule, 1, 1], dtype = numpy.float32), name = 'b')


    # do tiling for input and W before matmul
    # input => [batch_size, 1152, 10, 8, 1]
    # W => [batch_size, 1152, 10, 8, 16]
    _inputs = tf.reshape(_inputs, [-1, _input_shape[1], 1, _input_shape[2], _input_shape[3]])
    _inputs = tf.tile(_inputs, [1, 1, _num_capsule, 1, 1])
    W       = tf.tile(W, [_batch_size, 1, 1, 1, 1])

    # Calculate u_hat
    _u_hat = tf.matmul(W, _inputs, transpose_a = True)

    _u_hat_stopped = tf.stop_gradient(_u_hat, name = 'stop_gradient')
    for _iter in range(_num_routing):
        with tf.variable_scope('iter_' + str(_iter)):
            _c_ij = tf.nn.softmax(b, dim = 2)

            if _iter == _num_routing - 1:
                _s_j = tf.multiply(_c_ij, _u_hat)
                _s_j = tf.reduce_sum(_s_j, axis = 1, keep_dims = True) + biases
                v_j = squash(_s_j)
            else:
                _s_j = tf.multiply(_c_ij, _u_hat_stopped)
                _s_j = tf.reduce_sum(_s_j, axis = 1, keep_dims = True) + biases
                v_j = squash(_s_j)

                v_j_tiled = tf.tile(v_j, [1, 1152, 1, 1, 1])
                u_produce_v = tf.matmul(_u_hat_stopped, v_j_tiled, transpose_a = True)

                b += u_produce_v
    return v_j