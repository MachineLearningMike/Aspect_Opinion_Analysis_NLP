from BERT import BERT
from W2VEC import W2VEC

from Config import *
import os

class Embedder():

    def __init__(self, embType, path, database, embLayer = 1, firstNHiddens = 768, testMode = False):
        self.database = database
        self.useEmb = embType
        self.testMode = testMode

        self.body = None
        if embType == emb.bert :
            self.body = BERT(path, database, embLayer=embLayer, firstNHiddens=firstNHiddens, testMode=testMode)
        elif embType == emb.w2vec :
            self.body = W2VEC(path, database, firstNHiddens = 300, testMode = False)

        if self.body == None :
            self.dim_wordVector = 1 * firstNHiddens
            self.dummyMode = False # Client modules behave differently if dummyMode.
        else: 
            self.dim_wordVector = self.body.dim_wordVector
            self.dummyMode = self.body.dummyMode

    def GetTokenEmbList(self, parTokenList):
        if self.body == None :
            ones = tf.Variable( tf.ones_initializer()( shape = [self.dim_wordVector,]), dtype = configWeightDType )
            tokenEmbList = [ones] * len(parTokenList)
            embTokenList = parTokenList
            mapping = [ [n] for n in range(len(parTokenList)) ]

            return tokenEmbList, embTokenList, mapping
        else:
            return self.body.GetTokenEmbList(parTokenList)
