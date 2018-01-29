import tensorflow as tf
from collections import OrderedDict
from Layers.LayerHelper import *

class EncodeModel():
    def __init__(self,
                 _inputs,
                 _state,
                 _batch_size,
                 _name):
        self.inputs     = _inputs
        self.state      = _state
        self.batch_size = _batch_size
        self.name       = _name

        # ===== Create model =====
        # ----- Create net -----
        self.net_name = 'CNN for Feature Extraction'
        self.layers   = OrderedDict()

        # ----- Reshape input -----
        self.layers['input'] = tf.reshape(self.inputs, [-1, 28, 28, 1])

        with tf.variable_scope(self.name):
            # ----- Stack 1 -----
            with tf.variable_scope('stack1'):
                # --- Convolution ---
                self.layers['st1_conv'] = tf.layers.conv2d(inputs      = self.layers['input'],
                                                           filters     = 256,
                                                           kernel_size = [9, 9],
                                                           strides     = [1, 1],
                                                           padding     = 'valid',
                                                           activation  = None,
                                                           name        = 'conv2d')

                # --- Batch normalization ---
                self.layers['st1_batchnorm'] = tf.layers.batch_normalization(inputs   = self.layers['st1_conv'],
                                                                             center   = True,
                                                                             scale    = True,
                                                                             name     = 'batchnorm',
                                                                             training = self.state)

                # --- Relu ---
                self.layers['st1_relu'] = tf.nn.relu(features = self.layers['st1_batchnorm'],
                                                     name     = 'relu')

            # ----- Stack 2 -----
            with tf.variable_scope('stack2'):
                # --- Primary capsule ---
                self.layers['st2_pricap'] = PrimaryCap(_inputs      = self.layers['st1_relu'],
                                                       _n_channels  = 32,
                                                       _dim_vector  = 8,
                                                       _kernel_size = [9, 9],
                                                       _strides     = [2, 2],
                                                       _padding     = 'valid')

                # --- DigiCap ---
                self.layers['st2_digicap'] = DigiCap(_inputs      = self.layers['st2_pricap'],
                                                     _name        = 'digicap',
                                                     _num_capsule = 10,
                                                     _dim_vector  = 16,
                                                     _num_routing = 3,
                                                     _batch_size  = self.batch_size)
                # --- Reshape ---
                _old_shape  = self.layers['st2_digicap'].shape
                self.layers['st2_reshape'] = tf.reshape(self.layers['st2_digicap'], [-1, _old_shape[2], _old_shape[3]])

        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)

    def get_layer(self,
                  _layer_name):
        return self.layers[_layer_name]
