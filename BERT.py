from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np

from tensorflow.keras import Input, Model, optimizers, losses
from tensorflow.keras.layers import Layer, Dense, Activation
from tensorflow.keras.models import Model, Sequential

import torch
from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch, BertTokenizer, BertModel
#from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM

from Config import *

import os

class BERT(Layer):

    useLargeBert = False
   
    def __init__(self, bertPath, database, embLayer = 1, firstNHiddens = 768, testMode = False, **kwargs):
        super(BERT, self).__init__(trainable = False, dynamic = True, **kwargs)


        self.database = database

        assert embLayer in range(12)
        assert 0 <= firstNHiddens and firstNHiddens <= 768

        self.embLayer = embLayer
        self.output_dim = firstNHiddens
        self.firstNHiddens = firstNHiddens

        self.dim_wordVector = 1 * firstNHiddens
        self.zeroVec = np.zeros( shape = (self.dim_wordVector,), dtype=np.float)

        if bertPath == None:
            self.dummyMode = True; return
        else:
            self.dummyMode = False

        assert os.path.isdir(bertPath)

        abs_bert_path = os.path.abspath(bertPath)
        abs_ckptPath = os.path.join(abs_bert_path, 'bert_model.ckpt')
        abs_configPath = os.path.join(abs_bert_path, 'bert_config.json')
        abs_modelPath = os.path.join(abs_bert_path, 'pytorch_model.bin')

        self.Tokenizer = BertTokenizer.from_pretrained(abs_bert_path, cache_dir=None)

        #convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
        #    abs_ckptPath, abs_configPath, abs_modelPath )

        if self.useLargeBert == False:
            self.Model = BertModel.from_pretrained(abs_bert_path)
        else:
            self.Model = BertModel.from_pretrained('bert-large-uncased')

        self.testMode = testMode

        self.Model.eval() 

    def build(self, input_shape):

        super(BERT, self).build(input_shape)

    def call(self, inputs, training = False):
        ret = tf.map_fn(self.WrapGetTokenEmbList, inputs, dtype = configWeightDType )

        return ret

    def compute_output_shape(self, input_shape):
        #output_shape = input_shape + self.output_dim
        output_shape = tf.TensorShape( [input_shape[0], None, self.output_dim] )

        return output_shape

    def WrapGetTokenEmbList(self, tokens):
        if tf.executing_eagerly():
            if isinstance(tokens, tf.Tensor):
                tokens = tokens.numpy()

            nOrgTokens = len(tokens)

            if isinstance(tokens, np.ndarray):
                tokens = tokens[tokens != b'']
                tokens = [token.decode() for token in tokens]
            assert isinstance(tokens, list)

            tokenEmbs = self.GetTokenEmbList(tokens)

            NNullTokens = nOrgTokens - len(tokenEmbs)
            if NNullTokens >= 0 :
                tokenEmbs = [self.zeroVec.copy() for _ in range(NNullTokens)] + tokenEmbs
            else:
                tokenEmbs = tokenEmbs[:NNullTokens] # NNullTokens is negative

        else:
            tokenEmbs = tf.py_function(self.GetTokenEmbList, inp = [tokens], Tout = configWeightDType) 

        tokenEmbs2 = tf.convert_to_tensor(tokenEmbs, dtype = configWeightDType)

        return tokenEmbs2
        
    def GetTokenEmbList(self, tokens):
        """
        inputs:
            tokens: List of tokens of numpy.str data type.
        outputs:
            tokenEmbs: List of token embeddings
        """

        print( 'BERT embedding ======================================:' ) 

        assert tf.executing_eagerly()

        sentence = ''
        for token in tokens:
            sentence += (token + ' ')
        sentence = sentence[:-1]

        embTokens = self.__GetTokenList__(sentence)
        compatible, mapping = self.__GetTokenMapping__(tokens, embTokens)

        layersExtended = self.__GetEmbLayersExtended__( sentence )

        tokenEmbs = []
        if compatible :
            for embTokens in mapping :
                tokenEmbs.append( self.__AverageWordVector__( layersExtended, embTokens ) )
        else:
            raise Exception('Incompatible sentence.')

        return tokenEmbs


    def __GetTokenList__(self, text):
        """
        Return: excludes "[CLS]" / "[SEP]"
        """
        nText = text
        if nText[-1] == '\n':
            nText = nText[:-1]
        tokenList = self.Tokenizer.tokenize(text)

        return tokenList


    def __Tokenize__(self, text):
        # tokenize as expected by the model
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = self.Tokenizer.tokenize(marked_text)
        indexed_tokens = self.Tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.LongTensor([indexed_tokens])
        segments_tensors = torch.LongTensor([segments_ids])
        
        return tokens_tensor, segments_tensors

    def __LookupForTokenId__(self, token, tokensListSen):
        tokenId = -1
        for id in range(len(tokensListSen)):
            if tokensListSen[id] == token:
                tokenId = id; break
        if tokenId < 0:
            raise Exception('Token not found.')
        else:
            return tokenId

    def __GetEmbLayersExtended__(self, text):
        """Return: ndarrys of shape (layers, tokens, hiddens)"""

        tokens_tensor, segments_tensors = self.__Tokenize__(text)

        # Predict hidden states features for each layer
        with torch.no_grad():
            layers, _ = self.Model(tokens_tensor, segments_tensors)
        
        layers = np.array([torch.Tensor.numpy(layer) for layer in layers]) # layers[layer, sentence, token, hidden]
        layers = layers[:, 0, :, :] # Squeeze automatically the sentence dimension. We have a single sentence.
        layers = layers[self.embLayer, :, :] # resulting in [token, hidden]

        return layers

    def __GetWordVector__(self, embLayersExtended, tokenId):
        assert 0 <= tokenId and tokenId < embLayersExtended.shape[1] - 1 # [token, hidden].
        assert embLayersExtended.shape[1] >= 3 # Read the line below.
        embLayers = embLayersExtended[:, :self.firstNHiddens]

        tid = tokenId + 1
        embLayers = embLayers[tid, :] # token dimension is squeezed out automatically, resulting in [hidden]

        dim = embLayers.shape[0]
        assert dim == self.dim_wordVector
        vector = embLayers.reshape(dim,)

        return vector

    def __AverageWordVector__(self, embLayersExtended, tokenIdList):
        assert len(tokenIdList) > 0
        sumVector = self.zeroVec.copy()  #np.zeros( shape = (self.dim_wordVector), dtype = np.float) # np.ndarray(shape = (self.dim_wordVector), dtype=np.float) # defaults to 0.0
        for tokenId in tokenIdList:
            vector = self.__GetWordVector__(embLayersExtended, tokenId)
            assert not np.isnan( vector ).any()
            sumVector += vector
            assert sumVector.shape == (self.dim_wordVector,)
        
        return sumVector / len(tokenIdList)

    def __GetTokenMapping__(self, parTokens, embTokens):
        compatible = False; mapping = []

        hop = 1
        while compatible == False and  hop < len(embTokens):
            compatible, mapping = self.__GetTokenMapingWithHop__(parTokens, embTokens, hop)
            hop += 1
        
        if not compatible :
            compatible = compatible # for breakpoing

        return compatible, mapping


    def __OneToManyMappingFromFirstIncludingPoundedTokens__(self, token, candSubstrings):
        """
        If token = 'attentive' and candSubstrings = [t1, ..., tn, 'at', '##ten', '##tive', ...]
        then mapping = [0, 1, ..., n, n+1, n+2]
        n > 0 is abnormal. BERT doesn't tokenize that way.
        """

        mapping = []; token = token.lower(); firstPoundToken = 0

        concat = ''
        for candId in range(len(candSubstrings)):
        
            substring = candSubstrings[candId]

            if len(substring) >= 2 and substring[0] == '#' and substring[1] == '#':
                substring = substring[2:]
                if firstPoundToken == 0 : firstPoundToken == 1

            concat += substring

            if token == concat :
                for id in range(candId + 1) :
                    mapping.append(id)
                break
        
        if firstPoundToken > 1 :
            print('REPORT01:', token, candSubstrings)

        return mapping


    def __GetTokenMapingWithHop__(self, parTokens, embTokens, hop = 1):
        compatible = True 
        mapping = [None] * len(parTokens) 
        searchPoint = 0    

        for parId in range(len(parTokens)):

            nestMap = self.__OneToManyMappingFromFirstIncludingPoundedTokens__(parTokens[parId], embTokens[searchPoint:])
            if len(nestMap) > 0 :
                for id in nestMap: nestMap[id] += searchPoint
            elif parId + 1 < len(parTokens) and searchPoint + hop < len(embTokens):
                for hopPoint in range(searchPoint + hop, len(embTokens)):
                    hoppedMap = self.__OneToManyMappingFromFirstIncludingPoundedTokens__(parTokens[parId + 1], embTokens[hopPoint:])
                    if len(hoppedMap) > 0:
                        for id in range(len(hoppedMap)): hoppedMap[id] += hopPoint
                        # Throw away hoppedMap. Tokens before hoppedMap are mapped to.
                        for id in range(searchPoint, hoppedMap[0]):
                            nestMap.append(id)
                        break
            if len(nestMap) > 0 :
                mapping[parId] = nestMap
                searchPoint = nestMap[-1] + 1
            else:
                compatible = False
                break
        
        if searchPoint < len(embTokens):
            compatible = False
        
        return compatible, mapping

    def __Norm__(self, array):
        return np.sum(array * array) ** .5
    def Related(self, array_a, array_b):
        return np.sum(array_a * array_b) / max(self.__Norm__(array_a), self.__Norm__(array_b)) ** 2

    def MeanOverTokens(self, layers):
        """  encoded_layers: ndarray of shape (#layers, token, hiddensize) """
        return np.mean(layers, axis = 1)
        
    def Compare(self, seq_a, seq_a1, seq_b, seq_b1 = -1):
        vector_a = self.__GetEmbLayersExtended__(seq_a); vector_a = self.MeanOverTokens(vector_a)
        vector_a1 = self.__GetEmbLayersExtended__(seq_a1); vector_a1 = self.MeanOverTokens(vector_a1)
        
        related_aa1 = self.Related(vector_a, vector_a1)

        print('type of rep:', type(vector_a), vector_a.shape)
        
        vector_b = self.__GetEmbLayersExtended__(seq_b); vector_b = self.MeanOverTokens(vector_b)
        if seq_b1 == -1: 
            seq_b1 = seq_a1; vector_b1 = vector_a1
        else: 
            vector_b1 = self.__GetEmbLayersExtended__(seq_b1); vector_b1 = self.MeanOverTokens(vector_b1)
        
        related_bb1 = self.Related(vector_b, vector_b1)
        
        if related_aa1 > related_bb1: # don't use absolute.
            text = "'" + seq_a + "' : '" + seq_a1 + "' > \n'" + seq_b + "' : '" + seq_b1 + "'."
            rate = related_aa1 / related_bb1
            high = related_aa1; low = related_bb1
        else:
            text = "'" + seq_b + "' : '" + seq_b1 + "' > \n'" + seq_a + "' : '" + seq_a1 + "'." 
            high = related_bb1; low = related_aa1
            rate = high / low
            
        print(text)
        print( 'Rate = %0.3f / %0.3f = %0.2f' % (high, low, abs(rate)) )
        
        return text, high, low, rate

"""

# Unit test.
from Database import Database
database = Database('./Database/')
O = BERT("./Embedders/", database, 10, embLayer = 7, firstNHiddens = 768, testMode = False)

tokens = [ [',', ',', ',', ',', ','], ['hello', ',', 'I', 'am', 'happy'], ['', '', 'stay', 'up', '.'] ]
tensor = tf.Variable(tokens, dtype = tf.string)

print(tensor.shape.dims)
ret = O(tensor)
print(ret)

#@tf.function
def test():
    x = tf.keras.layers.Input(shape=(), dtype=tf.string) # batch_size=1, name='', dtype=tf.string, sparse=False, tensor=None, ragged=True)
    y = O(x)
    m = Model( inputs = x, outputs = y)
    ret = m(tensor)

#test() --- Failure

class MyLayer(Layer):
    def __init__(self):
        super(MyLayer, self).__init__()
        self.embedderLayer = BERT("./Embedders/", database, 10, embLayer = 7, firstNHiddens = 768, testMode = False)

    #@tf.function --- Failure
    def call(self, inputs):
        parTokenEmbs = self.embedderLayer( inputs )

        return parTokenEmbs

m = MyLayer()
ret = m(tensor)
print(ret)

"""