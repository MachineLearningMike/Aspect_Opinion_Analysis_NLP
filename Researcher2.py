from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np

from tensorflow.keras import Input, Model, optimizers, losses
from tensorflow.keras.layers import Layer, Dense, Activation
from tensorflow.keras.models import Model, Sequential

from stanfordnlp.server import CoreNLPClient

from Database import Database
from Parser import Parser
from BERT import BERT
from W2VEC import W2VEC
from RNSCN import RNSCN
from Config import *


database = Database('./Database/')

MAXTOKENS = 100
NLabelClasses = database.NLabelClass()

parser = Parser(database, maxTokens = MAXTOKENS)
database.tokenizer = parser

dataset = database.CreateSentenceAndOneHotLabelsNumpy( save = True )
onehostsForOther = database.GetOnehotsForOther()

paddedDataset = dataset.padded_batch(2, padded_shapes = ([], [None, None]) ) #, padding_values = ('', onehostsForOther) )

tokens = [ 'I am happy.', 'How much is it?', "For me, it's enough." ]
tensor = tf.Variable(tokens, dtype = tf.string)

@tf.function
def test(tensor):
    inputs = Input(shape = (), name = 'sentences')
    #inputs = tf.compat.v1.placeholder(shape = (None,), name = 'sentences', dtype = tf.string)
    x = parser(inputs)
    y = embedder(x[0])
    z = rnscn( (y, x[1], x[2] ))
    model = Model( inputs = inputs, outputs = z )

    #model(tensor)

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__(dynamic = True)
        self.parser = Parser(database, maxTokens = MAXTOKENS)
        self.embedder = BERT("./Embedders/", database, maxTokens = MAXTOKENS, embLayer = 7, firstNHiddens = 768, testMode = False)
        self.rnscn = RNSCN( database, maxTokens = MAXTOKENS, flexibleMaxTokens = False, parser = parser, embedder = embedder, dim_hidden = 7, topDown = False, seqver=0 )

    #@tf.function
    def call(self, inputs):
        retParser = self.parser(inputs)
        parTokenList = retParser[0]
        tokenEmbList = self.embedder(parTokenList)

        parDepMapList = retParser[1]
        parTokenMapList = retParser[2]
        retRnscn = self.rnscn( ( tokenEmbList, parDepMapList, parTokenMapList ) )

        return retRnscn
        #return tokenEmbList # parTokenList # retParser # retRnscn

parser = Parser(database, maxTokens = MAXTOKENS)
embedder = BERT("./Embedders/", database, embLayer = 7, firstNHiddens = 768, testMode = False)
#embedder = W2VEC("./Embedders/", database, maxTokens = MAXTOKENS, output_dim = 300, testMode = False)
rnscn = RNSCN( database, maxTokens = MAXTOKENS, flexibleMaxTokens = False, parser = parser, embedder = embedder, dim_hidden = 768, topDown = False, seqver=0 )
dense1 = Dense( 768, activation='tanh', kernel_regularizer = tf.keras.regularizers.l2(l=0.01) , bias_regularizer = tf.keras.regularizers.l2(l=0.01) )
dense2 = Dense( NLabelClasses, activation='tanh', kernel_regularizer = tf.keras.regularizers.l2(l=0.01) , bias_regularizer = tf.keras.regularizers.l2(l=0.01) )
softmax = Activation( 'softmax' )

id_sentences = Input( shape = (), name = 'sentences', dtype = tf.string )
parTokens, parDepMapList, parTokenMapList = parser(id_sentences)
tokenEmbList = embedder(parTokens)
rnscnHidden = rnscn((tokenEmbList, parDepMapList, parTokenMapList))
retDense1 = dense1(rnscnHidden)
retDens2 = dense2(retDense1)
logits = softmax(retDens2)

model = Model ( inputs = id_sentences, outputs = logits )

model.compile( optimizer = optimizers.RMSprop(), \
    loss = losses.CategoricalCrossentropy(from_logits = True), \
    metrics = ['categorical_accuracy'])

ret = model(tensor)

import os
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


model.fit(paddedDataset, epochs = 5, callbacks = [cp_callback])

#ret = model(tensor) # calling this is only possible when model.call has no @tf.function decoration. Check for why. The sublayers are dynamic = True.
#model.summarize()


