import tensorflow as tf

class Optimizer():
    def __init__(self,
                 _optimizer_opt,
                 _grads,
                 _params):
        # ===== Update params =====
        _pairs = list(zip(_grads, _params))
        self.train_opt = _optimizer_opt.apply_gradients(grads_and_vars = _pairs)

        # ===== Compute ratio =====
        _train_ops            = tf.get_collection(tf.GraphKeys.TRAIN_OP)
        _num_train            = len(tf.get_collection(tf.GraphKeys.TRAIN_OP))
        _params_before_update = [tf.norm(_param, name = 'params_before_update') + 1e-06
                                 for _param in _params]
        with tf.control_dependencies(_train_ops[:_num_train]):
            _params_after_update = [tf.norm(_param, name='params_after_update')
                                    for _param in _params]
            _absdiff             = tf.abs(tf.subtract(_params_after_update, _params_before_update), name = 'absdiff')
            self.ratio = tf.reduce_mean(tf.divide(_absdiff, _params_before_update, name = 'ratio'))
