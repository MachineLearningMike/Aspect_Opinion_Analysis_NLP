import tensorflow as tf
import numpy as np

from keras import backend as K
from keras.layers import Layer

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.kernel = self.add_weight(name = 'kerner',
            shape = (input_shape[1], self.output_dim),
            initializer='uniform',
            trainable=True)
        super(MyLayer, self).build(input_shape)
    
    def call(self, x):
        #return K.dot(x, self.kernel)
        x = tf.py_function(self.py_mul, inp=[x], Tout=[tf.float32])
        print( type(x) )
        return x

    def py_mul(self, x):
        print( type(x) )
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0])

tf.compat.v1.enable_eager_execution()
layer = MyLayer(1)

t2 = tf.constant([[3]], dtype=tf.float32)
a = layer(t2)

t = tf.compat.v1.placeholder(tf.float32, shape=(1,1))
a = layer(t)

t2 = tf.constant([[3]], dtype=tf.float32)
a = layer(t2)

t3 = tf.constant([[3]], dtype=tf.float32)
a = layer(t3)

t4 = tf.compat.v1.placeholder(tf.float32, shape=(1,1))
a = layer(t4)

t5 = tf.compat.v1.placeholder(tf.float32, shape=(2,1))
a = layer(t5)

print(a)


