import tensorflow as tf
from Models.EncodeModel import *
from Models.DecodeModel import *

class CapsuleModel():
    def __init__(self):
        # ===== Create tensor variables to store input / output data =====
        self.input         = tf.placeholder(tf.float32, shape = [None, 784], name = 'input')
        self.output        = tf.placeholder(tf.int32, shape = [None], name = 'output')
        self.batch_size    = tf.placeholder(tf.int32, shape = (), name = 'batch_size')
        self.state         = tf.placeholder(tf.bool, shape = (), name = 'state')
        self.learning_rate = tf.placeholder(tf.float32, shape = (), name = 'learning_rate')

        # ===== Create models =====
        self.encode_net = EncodeModel(_inputs     = self.input,
                                      _state      = self.state,
                                      _batch_size = self.batch_size,
                                      _name       = 'encode')
        _encode_params  = self.encode_net.params
        _encode_latent  = self.encode_net.get_layer('st2_reshape')
        _v_length       = tf.sqrt(tf.reduce_sum(tf.square(_encode_latent), axis = 2) + 1e-08)
        _prob           = tf.nn.softmax(_v_length, dim = 1)
        _pred           = tf.argmax(_prob, axis = 1, output_type = tf.int32)
        _encode_label   = tf.gather_nd(_encode_latent,
                                       tf.transpose([tf.range(self.batch_size), _pred,]))

        self.decode_net = DecodeModel(_inputs     = _encode_label,
                                      _state      = self.state,
                                      _batch_size = self.batch_size,
                                      _name       = 'decode')
        _decode_params  = self.decode_net.params
        _decode_output  = self.decode_net.get_layer('st3_sig')
        _decode_recon   = self.decode_net.get_layer('st3_reshape')

        # ----- Loss function -----
        # --- Train ---
        # --- The margin loss ---
        _max_l = tf.square(tf.maximum(0., 0.9 - _v_length))
        _max_r = tf.square(tf.maximum(0., _v_length - 0.1))
        _t_c   = tf.one_hot(self.output, depth = 10)
        _l_c   = _t_c * _max_l + 0.5 * (1 - _t_c) * _max_r
        _pre_loss   = tf.reduce_mean(tf.reduce_mean(_l_c, axis = 1))
        _recon_loss = tf.reduce_mean(tf.square(self.input - _decode_output))
        self.loss  = _pre_loss + _recon_loss * 0.392
        _adam_opti = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        _params    = _encode_params + _decode_params
        _grads     = tf.gradients(self.loss, _params)

        _update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(_update_ops): # Update batchnorm moving average before update parameters
            self.optimizer = Optimizer(_optimizer_opt = _adam_opti,
                                       _grads         = _grads,
                                       _params        = _params)
        def _train_func(_session, _state, _learning_rate, _batch_size,
                        _batch_x, _batch_y):
            return _session.run([self.loss, self.optimizer.ratio, _v_length, _pred, _prob, _max_l, _max_r, self.optimizer.train_opt],
                                feed_dict = {
                                    'state:0':         _state,
                                    'learning_rate:0': _learning_rate,
                                    'batch_size:0':    _batch_size,
                                    'input:0':         _batch_x,
                                    'output:0':        _batch_y,
                                })
        self.train_func = _train_func

        # --- Valid ---
        self.prec = tf.reduce_mean(tf.cast(tf.equal(self.output, _pred), tf.float32))
        def _valid_func(_session, _state, _batch_size,
                        _batch_x, _batch_y):
            return _session.run([self.prec],
                                feed_dict = {
                                    'state:0':      _state,
                                    'batch_size:0': _batch_size,
                                    'input:0':      _batch_x,
                                    'output:0':     _batch_y,
                                })
        self.valid_func = _valid_func

        # --- Reconstruct ---
        def _recon_func(_session, _state, _batch_size,
                        _batch_x, _batch_y):
            return _session.run([_decode_recon],
                                feed_dict = {
                                    'state:0':      _state,
                                    'batch_size:0': _batch_size,
                                    'input:0':      _batch_x,
                                    'output:0':     _batch_y
                                })
        self.recon_func = _recon_func