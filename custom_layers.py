import keras.backend as K
from keras.layers import Layer

class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, beta=.5, *args, **kwargs):
        self.is_placeholder = True
        self.beta = beta
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - self.beta * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs
		
    def get_config(self):
        config = dict(beta=self.beta)
        base_config = super(KLDivergenceLayer, self).get_config()
        base_config.update(config)
        return base_config