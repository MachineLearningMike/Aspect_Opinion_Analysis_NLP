from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np

from tensorflow.keras import Input, Model, optimizers, losses
from tensorflow.keras.layers import Layer, Dense, Activation
from tensorflow.keras.models import Model, Sequential

import gensim
from gensim.utils import simple_preprocess

from Config import *

import os

class W2VEC(Layer):

    useLargeBert = False
   
    def __init__(self, w2vecPath, database, maxTokens, output_dim, testMode = False, **kwargs):
        self.output_dim = output_dim
        self.maxTokens = maxTokens
        self.zeroVec = np.zeros( shape = (self.output_dim,), dtype=np.float)

        super(W2VEC, self).__init__(trainable = True, dynamic = True, **kwargs)

        self.database = database

        assert os.path.isdir(w2vecPath)

        abs_w2vec_path = os.path.abspath(w2vecPath)
        self.model = gensim.models.KeyedVectors.load_word2vec_format(\
            os.path.join(abs_w2vec_path, 'GoogleNews-vectors-negative300.bin'),\
            binary=True )

        self.dim_wordVector = 300

        self.testMode = testMode


    def build(self, input_shape):

        self.w_comma = self.add_weight(name = 'comma', shape = (1, self.output_dim), initializer = 'uniform', trainable = True, dtype = configWeightDType)
        self.w_period = self.add_weight(name = 'period', shape = (1, self.output_dim), initializer = 'uniform', trainable = True, dtype = configWeightDType)
        self.w_qmark = self.add_weight(name = 'qmart', shape = (1, self.output_dim), initializer = 'uniform', trainable = True, dtype = configWeightDType)
        self.w_emark = self.add_weight(name = 'emark', shape = (1, self.output_dim), initializer = 'uniform', trainable = True, dtype = configWeightDType)
        self.w_unknown = self.add_weight(name = 'unknown', shape = (1, self.output_dim), initializer = 'uniform', trainable = True, dtype = configWeightDType)

        super(W2VEC, self).build(input_shape)

    def call(self, inputs, training = False):
        ret = tf.map_fn(self.WrapGetTokenEmbList, inputs, dtype = (tf.string, configWeightDType) ) 

        return ret

    def compute_output_shape(self, input_shape):
        output_shape = ( input_shape[0], input_shape[1] + self.output_dim )

        return output_shape

    def WrapGetTokenEmbList(self, id_tokens):
        if tf.executing_eagerly():
            tokenEmbs = self.GetTokenEmbList(id_tokens[0], id_tokens[1])
        else:
            tokenEmbs = tf.py_function(self.GetTokenEmbList, inp = [id_tokens[0], id_tokens[1]], Tout = configWeightDType) 

        tokenEmbs2 = (id_tokens[0], tf.convert_to_tensor(tokenEmbs, dtype = configWeightDType) )

        return tokenEmbs2
       
    def GetTokenEmbList(self, id, parTokens):
        print( 'WORD2VEC embedding ======================================: ', id.numpy().decode()) 

        assert tf.executing_eagerly()

        if isinstance(parTokens, tf.Tensor):
            parTokens = parTokens.numpy()
        if isinstance(parTokens, np.ndarray):
            parTokens = parTokens[parTokens != b'']
            parTokens = [elem.decode() for elem in parTokens]
        assert isinstance(parTokens, list)

        sentence = ''
        for token in parTokens:
            sentence += (token + ' ')
        sentence = sentence[:-1]

        embTokenList = self.__GetTokenList__(sentence)
        compatible, mapping, embTokenList = self.__GetTokenMapping__(parTokens, embTokenList)

        tokenEmbList = []
        if compatible :
            for embTokens in mapping :
                tokenEmbList.append( self.__AverageWordVector__( embTokenList, embTokens ) )
        else:
            raise Exception('Incompatible sentence.')       

        NNullTokens = self.maxTokens - len(tokenEmbList)
        if NNullTokens >= 0 :
            tokenEmbList = [self.zeroVec.copy() for _ in range(NNullTokens)] + tokenEmbList
        else:
            tokenEmbList = tokenEmbList[:NNullTokens] # NNullTokens is negative

        return tokenEmbList

    def __GetTokenList__(self, text):
        nText = text
        if nText[-1] == '\n':
            nText = nText[:-1]
        tokenList = simple_preprocess(text, deacc = True)

        return tokenList

    def __GetWordVector__(self, embTokenList, tokenId):

        if tokenId >= 0 :
            word =  embTokenList[ tokenId ]
            if self.model.__contains__(word):   # Word found in embToken may not found in self.model.
                vector = self.model[word] # What's the hit rate?
            else:
                # Most frequent tokens in this category, i think.
                if word == ',' : vector = self.w_comma[0]
                elif word == '.' : vector = self.w_period[0]
                elif word == '?' : vector = self.w_qmark[0]
                elif word == '!' : vector = self.w_emark[0]
                else: vector = self.w_unknown[0]
                vector = vector.numpy()
        else: # We are now dealing with a token that is not found in embTokenList.
            vector = self.w_unknown[0] # This is a parameter to be trained.
            vector = vector.numpy()

        return vector

        # Do not attempt the following block. self.vocab is not what we expect it to be.
        """
        # embTokenList may now contain tokens that doesn't belong to W2VEC vacabulary. so, blow...
        try:
            if word in self.vocab :
                if word == ',' : vector = self.w_comma
                elif word == '.' : vector = self.w_period
                elif word == '?' : vector = self.w_qmark
                elif word == '!' : vector = self.w_emark
                else: vector = self.w_unknown
            else:
                vector = self.model[word]

            return vector

        except:
            print('REPORT02', word)
        
        """

    def __AverageWordVector__(self, embTokenList, tokenIdList):
        assert len(tokenIdList) > 0
        sumVector = self.zeroVec.copy() # np.zeros( shape = (self.output_dim), dtype = np.float)
        for tokenId in tokenIdList:
            vector = self.__GetWordVector__(embTokenList, tokenId)
            #assert not np.isnan( vector ).any()
            sumVector += vector
            #assert sumVector.shape == (self.output_dim,)
        
        return sumVector / len(tokenIdList)

    def __GetTokenMapping__(self, parTokens, embTokens):
        compatible = False; mapping = []

        hop = 0 # Start from 0 rather than 1. It will give a chance to add parToken ',' or the likes to W2Vec embTokens.
        compatible, mapping, embTokens = self.__GetTokenMapingWithHop__(parTokens, embTokens, hop)
        while compatible == False and  hop < len(embTokens):
            compatible, mapping, embTokens = self.__GetTokenMapingWithHop__(parTokens, embTokens, hop)
            hop += 1
        
        if not compatible :
            compatible = compatible # for breakpoint

        return compatible, mapping, embTokens


    def __OneToManyMappingFromFirst__(self, token, candSubstrings):

        mapping = []; 
        
        token = token.lower()
        token = ''.join ( c for c in token if c.isalpha() ) # not isalnum()

        concat = ''
        for candId in range(len(candSubstrings)):
        
            substring = candSubstrings[candId]
            substring = ''.join ( c for c in substring if c.isalpha() ) # not isalnum()

            concat += substring

            if token == concat :
                for id in range(candId + 1) :
                    mapping.append(id)
                break

        return mapping


    def __GetTokenMapingWithHop__(self, parTokens, embTokens, hop = 1):
        compatible = True 
        mapping = [None] * len(parTokens) 
        searchPoint = 0    

        for parId in range(len(parTokens)):

            nestMap = self.__OneToManyMappingFromFirst__(parTokens[parId], embTokens[searchPoint:])
            if len(nestMap) > 0 :
                for id in nestMap: nestMap[id] += searchPoint
            elif parId + 1 < len(parTokens) and searchPoint + hop < len(embTokens):
                for hopPoint in range(searchPoint + hop, len(embTokens)):
                    hoppedMap = self.__OneToManyMappingFromFirst__(parTokens[parId + 1], embTokens[hopPoint:])
                    if len(hoppedMap) > 0:
                        for id in range(len(hoppedMap)): hoppedMap[id] += hopPoint
                        # Throw away hoppedMap. Tokens before hoppedMap are mapped to.
                        if searchPoint < hoppedMap[0] :
                            for id in range(searchPoint, hoppedMap[0]):
                                nestMap.append(id)
                        else:
                            # Insert parToken ',' and the likes, which are skipped by W2Vec embTokens.
                            embTokens.insert( hoppedMap[0], parTokens[parId] )
                            nestMap.append( hoppedMap[0] )
                        break
            
            if len(nestMap) > 0 :
                mapping[parId] = nestMap
                searchPoint = nestMap[-1] + 1
            else:
                # compatible = False
                # break
                # These two lines couple with __GetWordVector__

                embTokens.insert( searchPoint, parTokens[parId].lower() ) # embTokens are all lower-cased.
                mapping[parId] = [searchPoint]
                searchPoint = searchPoint + 1
        
        if searchPoint < len(embTokens):
             compatible = False
        
        return compatible, mapping, embTokens


"""
# Unit Test

from Database import Database
database = Database('./Database/')
O = W2VEC(300, "./Embedders/", database, 10, testMode = False)


tokens = [ [',', ',', ',', ',', ','], ['hello', ',', 'I', 'am', 'happy'], ['', '', 'stay', 'up', '.'] ]
tensor = tf.Variable(tokens, dtype = tf.string)
print(tensor.shape.dims)
ret = O(tensor)
print(ret)

#@tf.function
def test():
    x = tf.keras.layers.Input(shape=(), dtype=tf.string) # batch_size=1, name='', dtype=tf.string, sparse=False, tensor=None, ragged=True)
    #x = tf.compat.v1.placeholder(shape=(4,), dtype=tf.string)
    y = O(x)
    m = Model( inputs = x, outputs = y)
    ret = m(tensor)

#test()

class MyLayer(Layer):
    def __init__(self):
        super(MyLayer, self).__init__()
        self.embedderLayer = W2VEC(300, "./Embedders/", database, 10, testMode = False)

    #@tf.function --- Failure
    def call(self, inputs):
        parTokenEmbs = self.embedderLayer( inputs )

        return parTokenEmbs

m = MyLayer()
ret = m(tensor)
print(ret)

"""