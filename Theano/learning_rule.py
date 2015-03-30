"""
Additional learning rules
"""
import numpy as np

from theano import config
from theano import tensor as T

from theano.compat.python2x import OrderedDict

class RMSPropMomentum(object):
    """
    RMSProp with momentum.

    """

    def __init__(self,
                 init_momentum,
                 averaging_coeff=0.95,
                 stabilizer=1e-2):

        init_momentum = float(init_momentum)
        assert init_momentum >= 0.
        assert init_momentum <= 1.
        averaging_coeff = float(averaging_coeff)
        assert averaging_coeff >= 0.
        assert averaging_coeff <= 1.
        stabilizer = float(stabilizer)
        assert stabilizer >= 0.
        self.momentum = init_momentum
        self.averaging_coeff = averaging_coeff
        self.stabilizer = stabilizer

    def get_updates(self, learning_rate, params, grads, lr_scalers=None):

        updates = OrderedDict()
        for param_i, grad_i in zip(params, grads):
            temp = param_i.get_value()
            avg_grad_sqr = np.zeros_like(temp)
            momentum = np.zeros_like(temp)

            new_avg_grad_sqr = self.averaging_coeff * avg_grad_sqr \
                + (1 - self.averaging_coeff) \
                * T.sqr(grad_i)

            rms_grad_t = T.sqrt(new_avg_grad_sqr)
            rms_grad_t = T.maximum(rms_grad_t, self.stabilizer)
            normalized_grad = grad_i / (rms_grad_t)
            new_momentum = self.momentum * momentum \
                - learning_rate * normalized_grad

            #updates[avg_grad_sqr] = new_avg_grad_sqr
            #updates[momentum] = new_momentum
            #momentum = new_momentum
            updates[param_i] = param_i + new_momentum

        return updates 
