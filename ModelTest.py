import tensorflow as tf
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model, Sequential

from Database import Database
from Parser import Parser
from W2VEC import W2VEC
from Config import *


tf.compat.v1.disable_eager_execution()

database = Database('./Database/')

x = tf.keras.Input(shape=(), dtype=tf.string) # batch_size=1, name='', dtype=tf.string, sparse=False, tensor=None, ragged=True)
y = Parser(database)(x)
n = Model( inputs = x, outputs = y )
z = W2VEC( output_dim = 300, w2vecPath = './Embedders/', database = database )(y)
m = Model( inputs = x, outputs = z)

