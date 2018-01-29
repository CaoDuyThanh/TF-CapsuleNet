import tensorflow as tf
from collections import OrderedDict

class DecodeModel():
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
        self.layers['input'] = self.inputs

        with tf.variable_scope(self.name):
            # ----- Stack 1 -----
            with tf.variable_scope('stack1'):
                # --- Convolution ---
                self.layers['st1_fc'] = tf.layers.dense(inputs      = self.layers['input'],
                                                        units       = 512,
                                                        activation  = None,
                                                        name        = 'fc')

                # --- Batch normalization ---
                self.layers['st1_batchnorm'] = tf.layers.batch_normalization(inputs   = self.layers['st1_fc'],
                                                                             center   = True,
                                                                             scale    = True,
                                                                             name     = 'batchnorm',
                                                                             training = self.state)

                # --- Relu ---
                self.layers['st1_relu'] = tf.nn.relu(features = self.layers['st1_batchnorm'],
                                                     name     = 'relu')

            # ----- Stack 2 -----
            with tf.variable_scope('stack2'):
                # --- Convolution ---
                self.layers['st2_fc'] = tf.layers.dense(inputs      = self.layers['st1_relu'],
                                                        units       = 1024,
                                                        activation  = None,
                                                        name        = 'fc')

                # --- Batch normalization ---
                self.layers['st2_batchnorm'] = tf.layers.batch_normalization(inputs   = self.layers['st2_fc'],
                                                                             center   = True,
                                                                             scale    = True,
                                                                             name     = 'batchnorm',
                                                                             training = self.state)

                # --- Relu ---
                self.layers['st2_relu'] = tf.nn.relu(features = self.layers['st2_batchnorm'],
                                                     name     = 'relu')

            # ----- Stack 3 -----
            with tf.variable_scope('stack3'):
                # --- Convolution ---
                self.layers['st3_fc'] = tf.layers.dense(inputs      = self.layers['st2_relu'],
                                                        units       = 784,
                                                        activation  = None,
                                                        name        = 'fc')

                # --- Sigmoid ---
                self.layers['st3_sig'] = tf.nn.sigmoid(x    = self.layers['st3_fc'],
                                                       name = 'sigmoid')

                # --- Reshape ---
                self.layers['st3_reshape'] = tf.reshape(tensor = self.layers['st3_sig'],
                                                        shape  = [-1, 28, 28, 1],
                                                        name   = 'reshape')

        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)

    def get_layer(self,
                  _layer_name):
        return self.layers[_layer_name]
