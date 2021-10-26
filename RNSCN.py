from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np

from tensorflow.keras import Input, Model, optimizers, losses
from tensorflow.keras.layers import Layer, Dense, Activation
from tensorflow.keras.models import Model, Sequential

import json

from BERT import BERT
from Parser import Parser
from Config import *


class Context():
    def __init__(self, tokenEmbList, parDepMapList, parTokenMapList, hiddensList, relationsList):
        self.tokenEmb = tokenEmbList
        self.parDepMap = parDepMapList
        self.parTokenMap = parDepMapList
        self.hiddens = hiddensList
        self.relations = relationsList

class RNSCN(Layer):
    def __init__(self, database, maxTokens, flexibleMaxTokens, parser, embedder, dim_hidden, topDown = False, seqver = 0, **kwargs):
        super(RNSCN, self).__init__(trainable = True, dynamic = True, **kwargs)

        self.database = database
        self.maxTokens = maxTokens
        self.flexibleMaxTokens = flexibleMaxTokens
        self.embedder = embedder
        assert isinstance(parser, Parser)
        self.parser = parser

        n_dependency_types = len(self.parser.parameters['dependenciesList'])

        self.n_dependency_types = n_dependency_types
        self.dim_wordVec = self.embedder.dim_wordVector
        self.dim_hidden = dim_hidden
        self.topDown = topDown
        self.seqver = seqver

        self.zeroVecConst = np.zeros( shape = (self.dim_wordVec,), dtype=np.float)
        self.zeroVecTensorConst = tf.constant( tf.zeros( shape = [self.dim_wordVec, 1], dtype = configWeightDType ), name = 'zeroVecConst' )
        self.zeroHiddenConst = tf.constant( tf.zeros_initializer()(shape = [self.dim_hidden, 1], dtype = configWeightDType ), name = 'zeroHiddenConst' )

    def build(self, input_shape):

        assert input_shape[0][-1] == self.dim_wordVec
        assert input_shape[0][0] == input_shape[1][0]
        assert input_shape[1][0] == input_shape[2][0]

        self.w_wordvec_hidden = self.add_weight(name = 'w_wordvec_hidden', shape = (self.dim_hidden, self.dim_wordVec), initializer = 'uniform', trainable = True, dtype = configWeightDType)
        self.b_wordvec_hidden = self.add_weight(name = 'b_wordvec_hidden', shape = (self.dim_hidden, 1), initializer = 'zeros', trainable = True, dtype = configWeightDType)
        self.w_hidden_relation = self.add_weight(name = 'w_hidden_relation', shape = (self.dim_hidden, self.dim_hidden), initializer = 'uniform', trainable = True, dtype = configWeightDType)

        if self.seqver == 3 : dim_relInput = self.dim_hidden * 2; dim_relOutput = self.dim_hidden
        else: dim_relInput = self.dim_hidden; dim_relOutput = self.dim_hidden

        self.ws_relation_hidden = []
        for i in range(self.n_dependency_types):
            w = self.add_weight(name = 'ws_relation_hidden_' + str(i), shape = (dim_relOutput, dim_relInput), initializer = 'uniform', trainable = True, dtype = configWeightDType)
            self.ws_relation_hidden.append(w)

        if self.seqver == 3 :
            self.bs_relation_hidden = []
            for i in range(self.n_dependency_types):
                w = self.add_weight(name = 'bs_relation_hidden_' + str(i), shape = (dim_relOutput, 1), initializer = 'zero', trainable = True, dtype = configWeightDType)
                self.bs_relation_hidden.append(w)                

        #self.w_relation_probability = self.add_weight(name = 'w_relation_probability', shape = (self.n_dependency_types, self.dim_hidden), initializer = 'uniform', trainable = True, dtype = configWeightDType)
        #self.b_relation_probability = self.add_weight(name = 'b_relation_probability', shape = (self.n_dependency_types, 1), initializer = 'zero', trainable = True, dtype = configWeightDType)

        super(RNSCN, self).build(input_shape)

    def call(self, inputs, training = True): # (parTokenList, parDepMapList, parTokenMapList)
        ret = tf.map_fn(self.WrapGenerateHiddensSen, inputs, dtype = configWeightDType )

        return ret

    def compute_output_shape(self, input_shape):
        shape2 = list(input_shape[0])
        shape2[-2] = None
        shape2[-1] = self.dim_hidden
        shape2 = tf.TensorShape(shape2)

        return shape2

    def WrapGenerateHiddensSen(self, input):
        tokenEmbList = input[0]; parDepMapList = input[1]; parTokenMapList = input[2]

        print( 'RNSCN =============================================: ', self.topDown )

        if tf.executing_eagerly():
            if isinstance(tokenEmbList, tf.Tensor):
                tokenEmbList = tokenEmbList.numpy()
            assert tokenEmbList.shape[-1] == self.dim_wordVec
            nOrgTokens = len(tokenEmbList)
            tokenEmbList = list( tokenEmbList[ tokenEmbList.any(axis=1) ] ) # remove zero vectors.
            tokenEmbList = list(tokenEmbList)
            assert isinstance(tokenEmbList, list)

            parDepMapList = json.loads(parDepMapList.numpy())
            parTokenMapList = json.loads(parTokenMapList.numpy())

            hiddenList = self.GenerateGenerateHiddensSen(tokenEmbList, parDepMapList, parTokenMapList)

            if not self.flexibleMaxTokens :
                NNullTokens = nOrgTokens - len(hiddenList)
                if NNullTokens >= 0 :
                    zeroVectorsList = [ tf.identity(self.zeroHiddenConst) for _ in range(NNullTokens)] 
                    hiddenList = zeroVectorsList + hiddenList
                else:
                    hiddenList = hiddenList[:NNullTokens] # NNullTokens is negative

            print("Len(hiddenList) = ", len(hiddenList), hiddenList[0].shape)

            hiddenList = tf.squeeze( hiddenList, [-1] ) # We had to use one more dimension to make them a matrix. Remove it now.

        else:
            #tokens = tf.py_function(self.GetTokenListAndKeepState, inp = [sentence], Tout = [tf.string]*self.maxTokens) #tokens is a list of token tensors.
            hiddenList = tf.py_function(self.GenerateGenerateHiddensSen, inp = [tokenEmbList, parDepMapList, parTokenMapList], Tout = configWeightDType) #*self.maxTokens) #tokens is a list of token tensors.

        hiddenList = tf.convert_to_tensor(hiddenList, dtype = configWeightDType) # This must not create a new variable in graph execution.

        return hiddenList

    def GenerateGenerateHiddensSen(self, tokenEmbList, parDepMapList, parTokenMapList):

        assert tf.executing_eagerly()

        hiddenList = [None] * len(parTokenMapList)
        relationList = [None] * len(parTokenMapList)
        
        ctx = Context( tokenEmbList, parDepMapList, parTokenMapList, hiddenList, relationList )

        rootNode = self.parser.GetRootNode( ctx.parDepMap )
        topNodesList = self.parser.GetDependentNodeList(rootNode, ctx.parDepMap )
        if len(topNodesList) != 1:
            raise Exception("More than 1 top nodes found.")

        if self.topDown and self.seqver == 0 :
            ctx.relations[ 0 ] = []
            self.ProcessNodeTopDown(topNodesList[0], 0, self.zeroHiddenConst, ctx ) # The goal is to populate self.hiddenList and self.relationList
            # ., governorDepNode, governorHidden
        elif self.seqver == 0 :
            self.ProcessNodeBottomUp(topNodesList[0], ctx) # The goal is to populate self.hiddenList and self.relationList
        elif self.seqver == 3 :
            self.ProcessNodeBottomUpSequential3(topNodesList[0], ctx)

        hiddenList = ctx.hiddens

        return hiddenList

        #return success, self.hiddenList, self.relationList
    

    def ProcessNodeBottomUp(self, focusDepNode, ctx):
        """
        focusDepNode: token # in the parser's annotion. 0 for ROOT.
        A parser token list doesn't include ROOT, requiring index [focusDepNode - 1].
        This complexity comes from the parser introducing ROOT token with token id of 0 in its dependency tree.
        Returns the hidden state of focusDepNode, and fills in self.hiddenListSen and self.relationListSen at its and its subordidate nodes's slots, respectively.
        """
        dependentNodesList = self.parser.GetDependentNodeList(focusDepNode, ctx.parDepMap )

        if len(dependentNodesList) > 0:
            text = 'dependency: ' + str(focusDepNode - 1) + ' ---> '
            for dep in dependentNodesList: text += (str( dep-1 ) + ', ')
            print(text)

        if focusDepNode <= 0 :
            focusWordVec = self.zeroVecTensorConst
        else:
            if focusDepNode - 1 < self.maxTokens :
                focusWordVec = ctx.tokenEmb[ focusDepNode - 1 ]
            else :
                focusWordVec = self.zeroVecConst
            #assert not np.isnan(focusWordVec).any()
            focusWordVec = tf.convert_to_tensor([ focusWordVec ], dtype = configWeightDType ) # [] in requires here, decendents, to make it a matrix.
            tf.debugging.assert_all_finite(focusWordVec, message = 'wordVec is a nan at 2.')
            focusWordVec = tf.transpose( focusWordVec )

        focusLinearPart = tf.matmul(self.w_wordvec_hidden, focusWordVec, name = 'matmul-01')
        focusHiddenSimple = tf.tanh ( tf.add( focusLinearPart, self.b_wordvec_hidden ) )

        if len(dependentNodesList) == 0:    # leaf node
            if focusDepNode > 0: # non-ROOT node
                focusHidden = focusHiddenSimple
            else:
                raise Exception('Root node with node dependents.')
        else:
            ctx.relations[ focusDepNode - 1 ] = [] #

            # sumDependentInfluence is initiall set to zero, as it will accumulate on an addition basis.
            sumDependentInfluence = tf.Variable( tf.zeros( shape = [self.dim_hidden, 1], dtype = configWeightDType ) )

            for dependentDepNode in dependentNodesList:                
                dependendHidden = self.ProcessNodeBottomUp(dependentDepNode, ctx) # Recursive call. =================================== 
                
                dependentRole = tf.matmul( self.w_hidden_relation, dependendHidden, name = 'matmul-02' )
                focusRole = tf.matmul(self.w_wordvec_hidden, focusWordVec, name = 'matmul-03') # keep it here for clarity, though it's redundant.
                dependentRelation = tf.tanh( tf.add( dependentRole, focusRole ) ) # Why add?
                tf.debugging.assert_all_finite(dependentRelation, message = 'dependentRelation is a nan.')
                ctx.relations[ focusDepNode - 1 ].append( (dependentDepNode - 1, dependentRelation) ) # Generate relations.
                
                depLabel = self.parser.LookupForDependencyLabel(focusDepNode, dependentDepNode, ctx.parDepMap, ctx.parTokenMap) # dependency-ralated operation.
                depId = self.parser.LookupForDependencyId(depLabel)

                dependentInfluence = tf.matmul( self.ws_relation_hidden[depId], dependentRelation, name = 'matmul-04')
                sumDependentInfluence = tf.add( sumDependentInfluence, dependentInfluence )

            focusHidden = tf.tanh ( tf.add( sumDependentInfluence, focusHiddenSimple ) )
            #assert goveHidden.shape == [self.dim_hidden, 1]

        ctx.hiddens[ focusDepNode - 1 ] = focusHidden # Generate hiddens.
        print("Node finished: ", focusDepNode - 1)
        tf.debugging.assert_all_finite(focusHidden, message = 'ficusHidden is a nan.')

        return focusHidden

    def ProcessNodeBottomUpSequential3(self, focusDepNode, ctx):
        """
        focusDepNode: token # in the parser's annotion. 0 for ROOT.
        A parser token list doesn't include ROOT, requiring index [focusDepNode - 1].
        This complexity comes from the parser introducing ROOT token with token id of 0 in its dependency tree.
        Returns the hidden state of focusDepNode, and fills in self.hiddenListSen and self.relationListSen at its and its subordidate nodes's slots, respectively.
        """
        dependentNodesList = self.parser.GetDependentNodeList(focusDepNode, ctx.parDepMap )

        if len(dependentNodesList) > 0:
            text = 'dependency: ' + str(focusDepNode - 1) + ' ---> '
            for dep in dependentNodesList: text += (str( dep-1 ) + ', ')
            print(text)

        if focusDepNode <= 0 :
            focusWordVec = focusWordVec = self.zeroVecTensorConst
        else:
            if focusDepNode - 1 < self.maxTokens :
                focusWordVec = ctx.tokenEmb[ focusDepNode - 1 ]
            else :
                focusWordVec = self.zeroVecConst
            #assert not np.isnan(focusWordVec).any()
            focusWordVec = tf.convert_to_tensor([ focusWordVec ], dtype = configWeightDType )
            tf.debugging.assert_all_finite(focusWordVec, message = 'wordVec is a nan at 2.')
            focusWordVec = tf.transpose( focusWordVec )

        focusLinearPart = tf.matmul(self.w_wordvec_hidden, focusWordVec, name = 'matmul-01')
        focusHiddenSimple = tf.tanh ( tf.add( focusLinearPart, self.b_wordvec_hidden ) )

        if len(dependentNodesList) == 0:    # leaf node
            if focusDepNode > 0: # non-ROOT node
                focusHidden = focusHiddenSimple
            else:
                raise Exception('Root node with node dependents.')
        else:
            ctx.relations[ focusDepNode - 1 ] = [] #

            focusHidden = focusHiddenSimple

            for dependentDepNode in dependentNodesList:          
                dependentHidden = self.ProcessNodeBottomUpSequential3(dependentDepNode, ctx) # Recursive call. =================================== 

                depLabel = self.parser.LookupForDependencyLabel(focusDepNode, dependentDepNode, ctx.parDepMap, ctx.parTokenMap) # dependency-ralated operation.
                depId = self.parser.LookupForDependencyId(depLabel)

                focusHidden = tf.matmul( ctx.relations[depId], tf.concat([focusHidden, dependentHidden], axis = 0), name = 'matmul-04')
                focusHidden = tf.add( focusHidden, self.bs_relation_hidden[depId] )
                focusHidden = tf.tanh( focusHidden )

            #assert focusHidden.shape == [self.dim_hidden, 1]

        ctx.hiddens[ focusDepNode - 1 ] = focusHidden # Generate hiddens.
        print("Node finished: ", focusDepNode - 1)
        tf.debugging.assert_all_finite(focusHidden, message = 'ficusHidden is a nan.')

        return focusHidden

    def ProcessNodeTopDown(self, focusDepNode, governorDepNode, governorHidden, ctx):
        """
        focusDepNode, governorDepNode: token # in the parser's annotion. 0 for ROOT.
        A parser token list doesn't include ROOT, requiring index [governorDepNode - 1].
        This complexity comes from the parser introducing ROOT token with token id of 0 in its dependency tree.
        Returns the hidden state of focusDepNode, and fills in self.hiddenListSen and self.relationListSen at its and its subordidate nodes's slots, respectively.
        """
        dependentNodesList = self.parser.GetDependentNodeList(focusDepNode, ctx.parDepMap )

        if len(dependentNodesList) > 0:
            text = 'dependency: ' + str(focusDepNode - 1) + ' ---> '
            for dep in dependentNodesList: text += (str( dep-1 ) + ', ')
            print(text)

        if focusDepNode <= 0 :
            focusWordVec = focusWordVec = self.zeroVecTensorConst
        else:
            if focusDepNode - 1 < self.maxTokens :
                focusWordVec = ctx.tokenEmb[ focusDepNode - 1 ]
            else :
                focusWordVec = self.zeroVecConst
            #assert not np.isnan(focusWordVec).any()
            focusWordVec = tf.convert_to_tensor([ focusWordVec ], dtype = configWeightDType )
            assert focusWordVec.shape == [1, self.dim_wordVec]
            tf.debugging.assert_all_finite(focusWordVec, message = 'wordVec is a nan at 2.')
            focusWordVec = tf.transpose( focusWordVec )
            assert focusWordVec.shape == [self.dim_wordVec, 1]
        
        ctx.relations[ focusDepNode - 1 ] = []

        focusLinearPart = tf.matmul(self.w_wordvec_hidden, focusWordVec, name = 'matmul-01')
        focusHiddenSimple = tf.tanh( tf.add( focusLinearPart, self.b_wordvec_hidden ) )
        assert focusHiddenSimple.shape == [self.dim_hidden, 1]

        governorRole = tf.matmul( self.w_hidden_relation, governorHidden, name = 'matmul-02' )
        focusRole = tf.matmul(self.w_wordvec_hidden, focusWordVec, name = 'matmul-03') # keep it here for clarity, though it's redundant.
        governorRelation = tf.tanh( tf.add( governorRole, focusRole ) ) # Why add?
        tf.debugging.assert_all_finite(governorRelation, message = 'governorRelation is a nan.')
        assert governorRelation.shape == [self.dim_hidden, 1]
        ctx.relations[ focusDepNode - 1 ].append( (governorDepNode - 1, governorRelation) ) # Generate relations.

        depLabel = self.parser.LookupForDependencyLabel(governorDepNode, focusDepNode, ctx.parTokenMap, ctx.parTokenMap) # dependency-ralated operation.
        depId = self.parser.LookupForDependencyId(depLabel)

        governorInfluence = tf.matmul(  self.ws_relation_hidden[depId], governorRelation , name = 'matmul-04')
        focusHidden = tf.tanh ( tf.add( governorInfluence, focusHiddenSimple ) )
        assert focusHidden.shape == [self.dim_hidden, 1]

        tf.debugging.assert_all_finite(focusHidden, message = 'hidden is a nan.')
        ctx.hiddens[ focusDepNode - 1 ] = focusHidden # Generate hiddens.
        print("Node finished: ", focusDepNode - 1)

        for dependentDepNode in dependentNodesList:
            self.ProcessNodeTopDown(dependentDepNode, focusDepNode, focusHidden, ctx) # Recursive call. =================================== 

    def GetParTokenList(self, sentence):
        normalizedLine = self.parser.NormalizeSentence(sentence)
        parTokenList, _, _ = self.parser.GetTokenList(normalizedLine)

        return parTokenList

    def PrintParToEmbMapping(self, parTokenList, embTokenList, parToEmbMapping, topNodesList):
        text = "parTokens: "            
        for id in range(len(parTokenList)):
            text += ( '(' + str(id) + ' ' + parTokenList[id].lower() + ') ' )
        print(text)

        text = "embTokens: "
        for id in range(len(embTokenList)):
            text += ( '(' + str(id) + ' ' + embTokenList[id] + ') ' )
        print(text)

        text = 'parToEmb mapping: '
        for parId in range(len(parToEmbMapping)):
            text += (str(parId) + '-')
            if len(parToEmbMapping[parId]) <= 1: text +=  str(parToEmbMapping[parId][0]) # the length is one, not zero.
            else:
                text += '('
                for embId in range(len(parToEmbMapping[parId])):
                    text += (str(parToEmbMapping[parId][embId]) + ' ')
                text = text[:-1]; text += ')'
            text += ' '
        print(text)

        print('top node = ', topNodesList[0])


    def GenerateLabelFile(self, trainNotTest = True):
        labelList = []

        dataset = self.database.CreateTextLinesDataset(trainNotTest, addLastPeriod = True)

        for dataset_record in dataset:
            sentence, aspect, opinion, lineId = self.database.DecodeTextLineDatasetRecord(dataset_record)

            consistent, aspectList, opinionList, wrongAspList, wrongOpnList = self.database.GetRefeinedLabels(sentence, aspect, opinion)
            if not consistent:
                print( "Inconsistent")
                print( sentence )
                print( "Wrong aspects: ", wrongAspList)
                print( "Wrong opinions: ", wrongOpnList)
                
            normalizedLine = self.parser.NormalizeSentence(sentence)
            parTokenList, _, _ = self.parser.GetTokenList(normalizedLine, True)
            tokenLabelStringList, tokenLabelNumeralList = self.database.GetTokenLabelList(parTokenList, aspectList, opinionList)

            labelList.append(tokenLabelNumeralList)

        if trainNotTest: pFile = self.database.pTrainTokenLabelClass
        else: pFile = self.database.pTestTokenLabelClass 
        self.database.SaveJsonData(labelList, pFile)

class RNSCNBlock():
    def __init__(self, database, embedder, parser, dim_hidden, useRNSCN = rnscn.up, testMode = False):

        self.upward = None; self.downward = None

        self.represent = None

        cnt = 0; dim_out = 0
        if useRNSCN == rnscn.no :
            cnt += 0; dim_out += embedder.dim_wordVector
            self.represent = self.upward
        if useRNSCN == rnscn.up :
            self.upward = RNSCN( database, embedder, parser, dim_hidden = dim_hidden, topdown = False )
            cnt += self.upward.nWeightTensors; dim_out += self.upward.dim_hidden
            self.represent = self.upward
        elif useRNSCN == rnscn.down :
            self.downward =  RNSCN( database, embedder, parser, dim_hidden = dim_hidden, topdown = True )
            cnt += self.downward.nWeightTensors; dim_out += self.downward.dim_hidden
            self.represent = self.downward
        elif useRNSCN == rnscn.bidir :
            self.upward = RNSCN( database, embedder, parser, dim_hidden = dim_hidden, topdown = False )
            cnt += self.upward.nWeightTensors; dim_out += self.upward.dim_hidden
            self.downward = RNSCN( database, embedder, parser, dim_hidden = dim_hidden, topdown = True )
            cnt += self.downward.nWeightTensors; dim_out += self.downward.dim_hidden
            self.represent = self.upward
        elif useRNSCN == rnscn.sup3 :
            self.upward = RNSCN( database, embedder, parser, dim_hidden = dim_hidden, topdown = False, seqver = 3 )
            cnt += self.upward.nWeightTensors; dim_out += self.upward.dim_hidden
            self.represent = self.upward

        if self.represent == None : 
            self.represent = RNSCN( database, embedder, parser, dim_hidden = dim_hidden, topdown = False, dummyMode = True )

        self.nWeightTensors = cnt
        self.dim_out = dim_out

    def Initialize(self):
        pass

    def Finalize(self):
        pass

    def __GetWeightsTensorList(self):
        list = []

        if self.upward != None: list = list + self.upward.weights
        if self.downward != None: list = list + self.downward.weights

        return list

    def __SetWeightsTensorList(self, list):
        cnt = 0

        if self.upward != None: self.upward.weights = list[cnt:]; cnt += self.upward.nWeightTensors
        if self.downward != None: self.downward.weights = list[cnt:]; cnt += self.downward.nWeightTensors
        
        self.nWeightTensors = cnt

    weights = property(__GetWeightsTensorList, __SetWeightsTensorList)

    def NWeights(self):
        sum = 0
        for tensor in self.weights:
            n = 1; k = 0
            for _ in range(len(tensor.shape)):
                n = n * tensor.shape[k]
            sum += n
            
        return sum

    def OnStartEpoch(self, accumEpoch):
        self.represent.OnStartEpoch(accumEpoch)

    def OnEndEpoch(self, accumEpoch):
        self.represent.OnEndEpoch(accumEpoch)

    def RemoveCache(self):
        self.represent.RemoveCache()

    def GetParTokenList(self, sentence):
        return self.represent.GetParTokenList(sentence)

    def GenerateStatesSingleSetence(self, tokenEmbList):
        hiddenListUpward = []; hiddenListDownward = []; hiddenListRnscn = []

        if self.upward != None :
            print( 'RNSCN bottom-up =====================================')
            successUpward, hiddenListUpward, _ = self.upward.GenerateStatesSingleSetence(tokenEmbList)
            if not successUpward: raise Exception('RNSCN failed: ', tokenEmbList)
            if self.downward == None :
                return successUpward, hiddenListUpward

        if self.downward != None :
            print( 'RNSCN top-down =======================================')
            successDownward, hiddenListDownward, _  = self.downward.GenerateStatesSingleSetence(tokenEmbList)
            if self.upward == None :
                return successDownward, hiddenListDownward
      
        if self.upward == None and self.downward == None :
            print( 'RNSCN skipping =======================================')
            assert tokenEmbList[0].shape == (self.dim_out,)
            for tokenEmb in tokenEmbList: 
                hidden = tf.expand_dims( tf.Variable(tokenEmb, dtype = configWeightDType), axis = 1 )
                hiddenListRnscn.append( hidden )
            return True, hiddenListRnscn

        assert len(hiddenListUpward) == len(hiddenListDownward)
            
        for n in range(len(hiddenListDownward)):
            hiddenListRnscn.append( tf.concat( [ hiddenListUpward[n], hiddenListDownward[n] ], axis = 0 ) )

        return successUpward and successDownward, hiddenListRnscn

"""
# Unit Test
embedder = Embedder('./BERT_base_uncased/', lastNLayers = 2, firstNHiddens = 3)
from Database import Database
database = Database('./Database')
parser = Parser(database)
O = RNSCN(database, embedder, parser, 8)
w = O.weights
print(type(w), O.NofWeights())

P = RNSCN(database, embedder, parser, 8)
P.weights = w
w2 = P.weights

"""
