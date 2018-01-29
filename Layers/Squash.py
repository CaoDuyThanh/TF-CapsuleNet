import tensorflow as tf

def squash(_input,
           _axis = -2):
    _squared_norm_input = tf.reduce_sum(tf.square(_input), axis = _axis, keep_dims = True)
    _scale              = (_squared_norm_input / (1 + _squared_norm_input)) / tf.sqrt(_squared_norm_input + 1e-08)
    return _scale * _input