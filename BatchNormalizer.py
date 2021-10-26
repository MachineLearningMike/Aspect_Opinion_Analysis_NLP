import tensorflow as tf 

from Config import *

class BatchNormalizer():

    def __init__(self, dim_batch, dim_input):
        self.gamma = tf.Variable( tf.ones_initializer()(shape = (1, dim_input), dtype = configWeightDType ), trainable = True)
        self.beta = tf.Variable( tf.zeros_initializer()(shape = (1, dim_input), dtype = configWeightDType ), trainable = True)

        self.dim_batch = dim_batch
        self.dim_input = dim_input

    def Initialize(self):
        pass

    def Finalize(self):
        pass

    def __GetWeightsTensorList(self):
        list = []
        list.append(self.gamma)
        list.append(self.beta)

        return list

    def __SetWeightsTensorList(self, list):
        self.gamma = list[0]
        self.beta = list[1]

    weights = property(__GetWeightsTensorList, __SetWeightsTensorList)

    def NWeights(self):
        sum = 0
        for tensor in self.weights:
            n = 1; k = 0
            for _ in range(len(tensor.shape)):
                n = n * tensor.shape[k]
            sum += n

        return sum
    
    def FeedForward(self, x):
        assert x.shape == [self.dim_batch, self.dim_input]

        mu = tf.reduce_mean(x, axis = 0, keepdims = True)
        assert mu.shape == [1, self.dim_input]
        std = tf.sqrt( tf.reduce_mean(tf.pow( x - mu, 2), axis = 0, keepdims = True) )
        assert std.shape == [1, self.dim_input]
        x_hat = tf.add( x, - mu) / ( std + 1e-8 )
        assert x_hat.shape == [self.dim_batch, self.dim_input]
        out = tf.multiply(self.gamma, x_hat) + self.beta
        assert out.shape == [self.dim_batch, self.dim_input]

        return out
    
