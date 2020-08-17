from keras.callbacks import Callback
import tensorflow as tf
import numpy as np

class SubTensorBoard(TensorBoard):
    """https://github.com/keras-team/keras/issues/2823"""
    def __init__(self, *args, **kwargs):
        super(SubTensorBoard, self).__init__(*args, **kwargs)

    def lr_getter(self):
        # Get vals
        decay = self.model.optimizer.decay
        lr = self.model.optimizer.lr
        iters = self.model.optimizer.iterations # only this should not be const
        beta_1 = self.model.optimizer.beta_1
        beta_2 = self.model.optimizer.beta_2
        # calculate
        lr = lr * (1. / (1. + decay * K.cast(iters, K.dtype(decay))))
        t = K.cast(iters, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(beta_2, t)) / (1. - K.pow(beta_1, t)))
        return np.float32(K.eval(lr_t))

    def on_epoch_end(self, episode, logs = {}):
        logs.update({"lr": self.lr_getter()})
        super(SubTensorBoard, self).on_epoch_end(episode, logs)
        
class SGDLearningRateTracker(Callback):
    """https://github.com/keras-team/keras/issues/2823"""
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        _lr = tf.to_float(optimizer.lr, name='ToFloat')
        _decay = tf.to_float(optimizer.decay, name='ToFloat')
        _iter = tf.to_float(optimizer.iterations, name='ToFloat')

        lr = K.eval(_lr * (1. / (1. + _decay * _iter)))
        print('\nLR: {:.6f}\n'.format(lr))