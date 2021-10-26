from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np

from tensorflow.keras import Input, Model, optimizers, losses
from tensorflow.keras.layers import Layer, Dense, Activation
from tensorflow.keras.models import Model, Sequential

import json

from stanfordnlp.server import CoreNLPClient

from Database import Database
from Config import *

class Parser(Layer):

    dependencyOfChoice = 'basicDependencies' #'enhancedPlusPlusDependencies'

    def __init__(self, database, maxTokens, dummyMode = False, testMode = False, **kwargs):
        super(Parser, self).__init__(trainable = False, dynamic = True, **kwargs)

        assert isinstance(database, Database)
        self.database = database
        self.maxTokens = maxTokens

        if dummyMode:
            self.dummyMode = True
            return
        else:
            self.dummyMode = False

        self.client = CoreNLPClient(annotators=['tokenize','ssplit','pos','lemma','ner','depparse','coref'],\
             timeout=30000, memory='3G', output_format='json') #----------------------------------------------- output_format='json'
        
        self.parameters = self.__LoadParameterDictionary__( self.database.pParameters )

        self.parTokenList = None
        self.parDepMapList = None
        self.parTokenMapList = None

        self.testMode = testMode

    def call(self, inputs, training = False):
        ret = tf.map_fn(self.WrapGetTokenList, inputs, dtype = (tf.string, tf.string, tf.string)) # types for id, parTokenList, parDepMapListJson, parTokenMapListJson
        parTokenList = ret[0].numpy()
        nMax = max( [ len ( [token for token in sen if not token.decode() == "" ] ) for sen in parTokenList ] )
        parTokenList = tf.convert_to_tensor([ sen[-nMax:] for sen in parTokenList ])

        return (parTokenList, ret[1], ret[2])

    def compute_output_shape(self, input_shape):
        output_shape = ( tf.TensorShape([input_shape[0], None]), tf.TensorShape([input_shape[0]]), tf.TensorShape([input_shape[0]]) )

        return output_shape

    def WrapGetTokenList(self, sentence):
        if tf.executing_eagerly():
            sentence = tf.squeeze(sentence)
            parTokenList, parDepMapList, parTokenMapList = self.GetTokenList(sentence)

            NNullTokens = self.maxTokens - len(parTokenList)
            if NNullTokens >= 0 :
                parTokenList = [''] * NNullTokens + parTokenList
            else:
                parTokenList = parTokenList[:NNullTokens] # NNullTokens is negative

        else:
            #tokens = tf.py_function(self.GetTokenListAndKeepState, inp = [sentence], Tout = [tf.string]*self.maxTokens) #tokens is a list of token tensors.
            parTokenList, parDepMapList, parTokenMapList = tf.py_function(self.GetTokenList, inp = [sentence], Tout = tf.string) #*self.maxTokens) #tokens is a list of token tensors.

        return ( tf.convert_to_tensor(parTokenList, dtype = tf.string), \
            tf.convert_to_tensor(json.dumps(parDepMapList), dtype = tf.string), \
            tf.convert_to_tensor(json.dumps(parTokenMapList), dtype = tf.string) )
        
    def GetTokenList(self, sentence): 
        """
        input: 
            sentence: Tensor of tf.string, numpy bytes, or list of numpy.str data type
        
        output:
            parTokenList: list of tokens of numpy.str data type
            parDepMapList: list of dependency maps
            parTokenMapList: list of token maps

        """
        assert tf.executing_eagerly()
       
        if isinstance(sentence, tf.Tensor):
            sentence = sentence.numpy().decode()
        elif isinstance(sentence, bytes):
            sentence = sentence.decode()
        assert isinstance(sentence, str)

        print( 'Parsing ========: ', sentence ) 

        normalSen = self.__NormalizeSentence__(sentence)
        parTokenList, parDepMapList, parTokenMapList = self.__GetTokenList__(normalSen)       

        return parTokenList, parDepMapList, parTokenMapList

    def __GenerateAnnotation__(self, sentence):
        annotation = self.client.annotate(sentence)
    
        return annotation

    def __NormalizeSentence__(self, line):
        """
        Goal: Create a LEAST modified sentence that allows a one-to-many (0..n) mapping from parser tokens to embedder tokens, while it's usually many-to-many.
        It's required for the following reasons:
        embedding(parser token) = average( embedding of the embedder tokens that the parser token maps to.)
        eg: embedding('attentive') = average( emebdding('at'), embedding('##ten'), emebdding('tive') )
        eg: embedding(n't) = average( embedding('n'), emebdding("'"), embedding('t') )
        eg: For one-to-many mapping, we must modify "can't" to "can not". Y? there is no one-to-many mapping from the parser tokern "ca" to the embedder tokens ["can", "'", "t"]
        eg: We don't have to modify "don't" to "do not". Y? the parser tokens ["do", "n't"] ARE one-to-many mapped to the embedder tokens ["do", "n", "'", "t"].
        This mechanism is required to fill the gap between the tokenizers of the parser (stanford) and the embedder (bert).
        Together with parToEmbTokenMapping function, this will ensure the dataset sentences can be used between the parser and the embedder.
        """
        annotationSen = self.__GenerateAnnotation__(line)

        words = []
        tokenMapList = self.__GetTokenMapList__(annotationSen)
        for tmap in tokenMapList:
            w_org = tmap['originalText']
            w_word = tmap['word']
            w_lemma = tmap['lemma']

            """ # We can exclude these check thanks to the parToEmbTokenMapping. We have to minimize modification.
            if  w_org.lower() == "'m" and w_lemma == "be":
                token = "am"
            elif w_org.lower() == "'re" and w_lemma == "be":
                token = "are"
            elif w_org.lower() == "'s" and w_lemma == "be":
                token = "is" # or "was": no way.
            elif w_org.lower() == "'ve" and w_lemma == "have":
                token = "have"
            elif w_org.lower() == "'s" and w_lemma == "have":
                token = "has"
            """

            if w_org.lower() == "n't" and w_lemma == "not":               
                """ I thought BERT created, from "don't", ["do", "n", "'", "t"]. 
                # It actually creates ["don", "'", "t"]. So, please comment out this block.
                # It means we have to convert "I don't go" to "I do not go".
                if len(words) > 0:
                    if words[-1].lower() == 'do': # Keep contraction.
                        if words[-1] == 'D':
                            words[-1] = "Don't"
                        else:
                            words[-1] = "don't"
                        continue 
                    elif words[-1].lower() == 'does': # Keep contraction.
                        if words[-1] == 'D':
                            words[-1] = "Doesn't"
                        else:
                            words[-1] = "doesn't"
                        continue  
                    elif words[-1].lower() == 'have': # Keep contraction.
                        if words[-1] == 'H':
                            words[-1] = "Haven't"
                        else:
                            words[-1] = "haven't"
                        continue  
                    elif words[-1].lower() == 'has': # Keep contraction.
                        if words[-1] == 'H':
                            words[-1] = "Hasn't"
                        else:
                            words[-1] = "hasn't"
                        continue  
                    elif words[-1].lower() == 'is': # Keep contraction.
                        if words[-1] == 'I':
                            words[-1] = "Isn't"
                        else:
                            words[-1] = "isn't"
                        continue  
                    elif words[-1].lower() == 'are': # Keep contraction.
                        if words[-1] == 'A':
                            words[-1] = "Aren't"
                        else:
                            words[-1] = "aren't"
                        continue  
                    elif words[-1].lower() == 'should': # Keep contraction.
                        if words[-1] == 'S':
                            words[-1] = "Should't"
                        else:
                            words[-1] = "should't"
                        continue  
                    elif words[-1].lower() == 'could': # Keep contraction.
                        if words[-1] == 'C':
                            words[-1] = "Could't"
                        else:
                            words[-1] = "could't"
                        continue  
                    """

                token = "not"

                if len(words) > 0:
                    if words[-1].lower() == 'ca':   # for "can't"
                        if words[-1][0] == 'C':
                            words[-1] = 'Can'
                        else:
                            words[-1] = 'can'
                    elif words[-1].lower() == 'wo': # for "won't"
                        if words[-1][0] == 'W':
                            words[-1] = 'Will'
                        else:
                            words[-1] = 'will'
                    elif words[-1].lower() == 'sha': # for "shan't"
                        if words[-1][0] == 'S':
                            words[-1] = 'Shall'
                        else:
                            words[-1] = 'shall'
            else:
                if (tmap['originalText']).isalnum(): #.isalpha():  # Care should be taken here. It detect "n't", "'re"
                    token = tmap['originalText']
                else:
                    token = tmap['lemma']

            words.append(token)

        sen = ''
        for word in words:
            if len(sen) > 0 and not word[0:1].isalnum(): # detects special character.
                sen = sen[:-1] # remove the last space.
            sen += (word + ' ')

        sen = sen[:-1] # remove back the last space.
        sen += '\n' # make it a line. The parser and embedder doesn't care of it.


        if sen.find("n't") >= 0:
            sen = sen # for breakpoint

        return sen

    def __LoadParameterDictionary__(self, path): # dict
        parameters = {}

        if self.database.Exists(path):
            parameters = self.database.LoadJsonData(path)
        else:

            dsFull, dsEmpty = self.database.CreateFullDatasetsFromCombinedFile(trainPortion = 1.0, testPortion = 0.0, addLastPeriod = False, returnSizeOnly = False, save = False)

            annotations = []
            for dataset_record in dsFull:
                senIdOneBased, sentence, aspect, opinion, check = self.database.DecodeComibinedDatasetRecord(dataset_record)
                annotations.append(self.__GenerateAnnotation__(sentence))

            dependencies = set()
            for ann in annotations:
                for b in ann['sentences'][0][self.dependencyOfChoice]:
                    dependencies.add(b['dep'])

            parameters['dependenciesList'] = sorted(list(dependencies))

            self.database.SaveJsonData(parameters, path)
            del dsFull, dsEmpty, dependencies

        return parameters

    def __GenerateAnnotations__(self, sourcePath, destPath):
        print('Producing annotations of file: ' + sourcePath)

        lines = self.database.GetListOfLines(sourcePath, addLastPeriod = True)

        annotations = []
        for line in lines:
            ann = self.__GenerateAnnotation__(line)
            annotations.append(ann)
        
        print('Saving annotations to file: ' + destPath)
        self.database.SaveJsonData(annotations, destPath)

    def __GetDepMapList__(self, annotation):
        return annotation['sentences'][0][self.dependencyOfChoice]

    def __GetTokenMapList__(self, annotation):
        return annotation['sentences'][0]['tokens']

    def GetRootNode(self, depMapList):
        rootNode = -1
        for depMap in depMapList:
            if depMap['dep'] == 'ROOT':
                rootNode = depMap['governor'] # token id, incl. 0
                break
        if rootNode < 0:
            raise Exception('Rood node not found.')
        else:
            return rootNode

    def GetDependentNodeList(self, governor, depMapList ):
        depList = []
        for depMap in depMapList:
            if depMap['governor'] == governor:
                depList.append(depMap['dependent'])
        return depList

    def __GetTokenList__(self, sentence):
        """
        Better call this method with normalized sentences only.
        """
        tList = []
        annotation = self.__GenerateAnnotation__(sentence)
        depMapList = self.__GetDepMapList__(annotation)
        tokenMapList = self.__GetTokenMapList__(annotation)
        for tmap in tokenMapList:
            token = tmap['originalText'] # + "/" + tmap['word'] + "/" + tmap['lemma']
            tList.append(token)

        return tList, depMapList, tokenMapList

    def LookupForDependencyLabel(self, governorNode, dependentNode, depMapListSen, parTokenListSen):
        depLabel = ''
        for depMap in depMapListSen:
           if depMap['governor'] == governorNode and depMap['dependent'] == dependentNode:
                depLabel = depMap['dep']
                break
        if depLabel == '':
            raise Exception("Couldn't find dependency lable for governor {} and dependent {}".format(governorNode, dependentNode))
        elif self.testMode:
            govIndex = governorNode - 1; depIndex = dependentNode - 1
            print("({} {}) ----> ({} {}) : {}".format(govIndex, parTokenListSen[govIndex], depIndex, parTokenListSen[depIndex], depLabel))

        return depLabel

    def LookupForDependencyId(self, dependencyLabel):
        dependenciesList = self.parameters['dependenciesList']
        depId = -1
        for num in range(len(dependenciesList)):
            if dependenciesList[num] == dependencyLabel:
                depId = num; break
        if depId < 0:
            raise Exception("Couldn't find dependency label in dependency dictionary.")
        else:
            return depId
   
"""
# Unit test
tf.compat.v1.enable_eager_execution() # This is by default.

database = Database('./Database/')
O = Parser(database, maxTokens = 10)

text = ['hello Mike', 'hello, world', 'feel free', 'how long']
tensor = tf.Variable(text)

print(tensor.shape.dims)
ret = O(tensor)
print(ret)

@tf.function
def test():
    x = tf.keras.layers.Input(shape=(), dtype=tf.string) # batch_size=1, name='', dtype=tf.string, sparse=False, tensor=None, ragged=True)
    y = O(x)
    m = Model( inputs = x, outputs = y)
    ret = m(tensor)

#test() --- Failutre

class MyLayer(Layer):
    def __init__(self):
        super(MyLayer, self).__init__()
        self.parserLayer = Parser(database, maxTokens = 10)

    #@tf.function --- Failure
    def call(self, inputs):
        parTokens = self.parserLayer( inputs )

        return parTokens

m = MyLayer()
ret = m(tensor)
print(ret)
"""




#O.GenerateTranslationPairFile()
#examples = O.ReadTranslationPairFile(trainPortion = 0.7, testPortion = 0.3)
#cnt = 0

"""
O.LoadGenerateDepMapList(trainNotTest = False)

param = O.parameters
trainDeps = param['trainDependenciesList']
print(trainDeps)
testDeps = param['testDependenciesList']
print(testDeps) 
deps = param['dependenciesList']
print(deps)
print(len(trainDeps), len(testDeps), len(deps))

sen = "Can't you go."
tList, tString, _, _ = O.GetTokenList(sen)
ann = O.GenerateAnnotation(sen)
nSen = O.NormalizeSentence(ann)

sen = "I can't go."
tList, tString, _, _ = O.GetTokenList(sen)
ann = O.GenerateAnnotation(sen)
nSen = O.NormalizeSentence(ann)

sen = "I don't go."
tList, tString, _, _ = O.GetTokenList(sen)
ann = O.GenerateAnnotation(sen)
nSen = O.NormalizeSentence(ann)

sen = "I'm a student."
tList, tString, _, _ = O.GetTokenList(sen)
print(tList)
sen = "It doesn't work!"
tList, tString, _, _ = O.GetTokenList(sen)
print(tList)
sen = "It's my fault."
tList, tString, _, _ = O.GetTokenList(sen)
print(tList)
sen = "He's got a ball."
tList, tString, _, _ = O.GetTokenList(sen)
print(tList)
sen = "I couldn't move."
tList, tString, _, _ = O.GetTokenList(sen)
print(tList)
sen = "You aren't a student."
tList, tString, _, _ = O.GetTokenList(sen)
print(tList)
sen = "You aint a student."
tList, tString, _, _ = O.GetTokenList(sen)
print(tList)
sen = "It isn't mine."
tList, tString, _, _ = O.GetTokenList(sen)
print(tList)
sen = "You're a student."
tList, tString, _, _ = O.GetTokenList(sen)
print(tList)
sen = "You've a ball."
tList, tString, _, _ = O.GetTokenList(sen)
print(tList)

sen = "You can't go."
tList, tString, _, _ = O.GetTokenList(sen)
print(tList)

sen = "It's Ahmed's birthday today."
tList, tString, _, _ = O.GetTokenList(sen)
print(tList)

ann = O.GenerateAnnotation(sen)
nsen = O.NormalizeSentence(ann)
print(nsen)

"""

