from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np

from tensorflow.keras import Input, Model, optimizers, losses
from tensorflow.keras.layers import Layer, Dense, Activation, Embedding, LSTM
from tensorflow.keras.models import Model, Sequential

tokens = [ ['I', 'am', 'happy.'], ['How', 'much', 'is', 'it', '?'], ['For', 'me', 'it', 'is', 'enough', '.'] ]
#tensor = tf.Variable(tokens, dtype = tf.string)

padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(tokens, dtype = '<U100', padding='post', value = '')

print(padded_inputs)


raw_inputs = [
  [83, 91, 1, 645, 1253, 927],
  [73, 8, 3215, 55, 927],
  [711, 632, 71]
]

padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(raw_inputs, padding='post')
print(padded_inputs)

masking_layer = tf.keras.layers.Masking()
# Simulate the embedding lookup by expanding the 2D input to 3D,
# with embedding dimension of 10.
unmasked_embedding = tf.cast(
    tf.tile(tf.expand_dims(padded_inputs, axis=-1), [1, 1, 10]),
    tf.float32)

masked_embedding = masking_layer(unmasked_embedding)
print(masked_embedding._keras_mask)


embedding = Embedding(input_dim=5000, output_dim=16, mask_zero=True)
masked_output = embedding(padded_inputs)
print(masked_output._keras_mask)

inputs = tf.keras.Input(shape=(None,), dtype='int32')
x = Embedding(input_dim=5000, output_dim=16, mask_zero=True)(inputs)
outputs = LSTM(32)(x)

model = tf.keras.Model(inputs, outputs)
y = model(padded_inputs)
print(y)





class TemporalSplit(tf.keras.layers.Layer):
  """Split the input tensor into 2 tensors along the time dimension."""
  def __init__(self, mask_zero = False):
      super(tf.keras.layers.Layer, TemporalSplit).__init__( mask_zero = mask_zero)

  def call(self, inputs):
    # Expect the input to be 3D and mask to be 2D, split the input tensor into 2
    # subtensors along the time axis (axis 1).
    return inputs
    
  def compute_mask(self, inputs, mask=None):
    # Also split the mask into 2 if it presents.
    mask = tf.Variable( [[ True, True, True, True, True, True], [ True, True, True, True, True, False], [ True,True, True, False, False, False]], dtype = tf.bool)

    return mask

#print(unmasked_embedding._keras_mask)
ts = TemporalSplit(mask_zero = True)
masked_embedding = ts(masked_embedding)
print(masked_embedding._keras_mask)
print(masked_embedding)