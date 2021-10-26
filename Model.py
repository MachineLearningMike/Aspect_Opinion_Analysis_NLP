import os
import sys
import datetime
import tensorflow as tf
import numpy as np

from Database import Database
from Parser import Parser
from Embedder import Embedder
from RNSCN import RNSCNBlock
from GRU import GRUBlock
from MultiFCN import MultiFCN
from VisualClass import VisualClass

from Autoencoder import Autoencoder
from Supervisor import Supervisor

from Config import *

from VisualClass import *

class Model():
    
    def __init__(self, metaParams, filebase = '', testMode = False ):
        if metaParams == None or metaParams.get('dataPath', None) == None:
            print('Invalid Model Metaparameters (dataPath).')
            return

        filebase, folder = self.BuildFileNames( metaParams )

        os.makedirs(folder, exist_ok = True)
        self.database = Database( metaParams['dataPath'], filebase )

        dummyParser = Parser(self.database, dummyMode = True, testMode = testMode)
        dummyEmbedder = Embedder( emb.bert, path = None, database = self.database ) # bertPath = None signals dummyMode
        dummyRnscn = RNSCNBlock(self.database, dummyEmbedder, dummyParser, -1)

        if self.database.Exists(self.database.pTrainHistory):
            history = self.database.LoadBinaryData(self.database.pTrainHistory)
            fileMetaParams, _, _, _, _ = history

            createFromNewParams = False

            if  metaParams.get('dataPath', None) != None and metaParams['dataPath'] != fileMetaParams['dataPath'] or \
                metaParams.get('embPath', None) != None and metaParams['embPath'] != fileMetaParams['embPath'] :

                dummyRnscn.RemoveCache(); createFromNewParams = True

            if  metaParams.get('bertLayer', None) != None and metaParams['bertLayer'] != fileMetaParams['bertLayer'] or \
                metaParams.get('embDim', None) != None and metaParams['embDim'] != fileMetaParams['embDim'] or \
                metaParams.get('rnscnDim', None) != None and metaParams['rnscnDim'] != fileMetaParams['rnscnDim'] or \
                metaParams.get('gruDim', None) != None and metaParams['gruDim'] != fileMetaParams['gruDim'] or \
                metaParams.get('RNSCN', None) != None and metaParams['RNSCN'] != fileMetaParams['RNSCN'] or \
                metaParams.get('GRU', None) != None and metaParams['GRU'] != fileMetaParams['GRU'] :

                createFromNewParams = True

            if createFromNewParams:
                self.__Create__( metaParams = metaParams, filebase = filebase, testMode = testMode )
            else:
                # Do not remove. We inherit checkpoint files and caches.
                self.__Create__( metaParams = fileMetaParams, filebase = filebase, testMode = testMode )
        
        else:
            createFromNewParams = False        

            if  metaParams.get('dataPath', None) != None and \
                metaParams.get('embPath', None) != None and \
                metaParams.get('bertLayer', None) != None and \
                metaParams.get('embDim', None) != None and \
                metaParams.get('rnscnDim', None) != None and \
                metaParams.get('gruDim', None) != None and \
                metaParams.get('RNSCN', None) != None and \
                metaParams.get('GRU', None) != None :

                #self.RemoveCheckpointFiles(); dummyRnscn.RemoveCache()
                self.__Create__( metaParams = metaParams, filebase = filebase, testMode = testMode )

            else:
                print("Metaparameters missing.")
                return


    def __Create__(self, metaParams, filebase = '', testMode = False ):
        #print("Model Parameters: {}".format( metaParams))
 
        dataPath = metaParams['dataPath']

        useEmbedder = metaParams['Embedder']
        embPath = metaParams['embPath']
        embLayer = metaParams['bertLayer']
        embFirstNHiddens = metaParams['embDim']

        useRNSCN = metaParams['RNSCN']
        rnscnDimHidden = metaParams['rnscnDim']

        useGRU = metaParams['GRU']
        gruDimHidden = metaParams['gruDim']

        self.metaParams = metaParams
        self.metaParams['parser'] = 'CoreNLP'

        self.database = Database(dataPath, filebase = filebase)
        self.parser = Parser(self.database, testMode = testMode)
        self.embedder = Embedder(useEmbedder, embPath, self.database, embLayer = embLayer, firstNHiddens = embFirstNHiddens, testMode = testMode)
        self.rnscnBlock = RNSCNBlock(self.database, self.embedder, self.parser, dim_hidden = rnscnDimHidden, useRNSCN = useRNSCN, testMode = testMode)
        self.gruBlock = GRUBlock(self.database, self.rnscnBlock.dim_out, dim_hidden = gruDimHidden, normalizeLayer = True, useGRU = useGRU)

        dim_outputs = [self.gruBlock.dim_out * 2, self.gruBlock.dim_out * 2, self.gruBlock.dim_out, self.database.NLabelClass()]
        activations = ['tanh', 'tanh', 'tanh', 'tanh']
        self.multifcn = MultiFCN(dim_batch = 1, dim_input = self.gruBlock.dim_out, dim_outputs = dim_outputs, activations = activations, useLayerNormalizer = True)

        #self.autoencoder = Autoencoder()
        #self.supervisor = Supervisor()

        self.nWeightTensors = self.rnscnBlock.nWeightTensors + self.rnscnBlock.nWeightTensors + self.multifcn.nWeightTensors

        self.testMode = testMode

        self.optimizer = tf.keras.optimizers.Adadelta(learning_rate = 0.5)

        self.lossHistoryTrain = []; self.scoreHistoryTrain = []
        self.lossHistoryTest = []; self.scoreHistoryTest = []

    @classmethod
    def BuildFileNames( cls, metaParams ):
        return Database.BuildFileNames( metaParams )


    def Train(self, shuffleBuffer = 1000, miniBatchSize = 20, epochs = 5, logToFile = False):

        if logToFile :
            sys.stdout = open( self.database.pLog, 'a+')
            print("\nTraining Session ============================================================================================")

        start_time = datetime.datetime.now()
        print ('Start time: ', start_time)

        if  self.metaParams.get('shuffle', None) != None and self.metaParams['shuffle'] != shuffleBuffer or \
            self.metaParams.get('batch', None) != None and self.metaParams['batch'] != miniBatchSize :

            pass

        self.metaParams['shuffle'] = shuffleBuffer
        self.metaParams['batch'] = miniBatchSize

        self.metaParams['tensors'] = self.nWeightTensors

        dsTrain, dsTest = self.database.CreateFullDatasetsFromCombinedFile(trainPortion = 0.7, testPortion = 0.3, addLastPeriod = False, returnSizeOnly = False, save = False)

        dsTrain = dsTrain.shuffle(buffer_size = shuffleBuffer, reshuffle_each_iteration = True).batch(batch_size = miniBatchSize, drop_remainder = True)

        accumEpoch = 0; step = 0
        epochRange = range(epochs)

        self.optimizer = tf.keras.optimizers.Adadelta(learning_rate = 0.5) # This is unusual.

        for epoch in epochRange:   
            tf.compat.v1.global_variables_initializer()

            self.rnscnBlock.OnStartEpoch(accumEpoch)

            print("\n=============================================== epoch : ", accumEpoch, "===========================\n")
            time = datetime.datetime.now()
            print ('Start time: ', time)

            lossHistoryTrain, scoreHistoryTrain, step = self.LearnFromEpoch(dsTrain, step)
            self.lossHistoryTrain = self.lossHistoryTrain + lossHistoryTrain
            self.scoreHistoryTrain = self.scoreHistoryTrain + scoreHistoryTrain
            
            lossHistoryTest, scoreHistoryTest = self.EvaluateOnDataset( dsTest, shuffleBuffer, miniBatchSize )
            self.lossHistoryTest = self.lossHistoryTest + lossHistoryTest
            self.scoreHistoryTest = self.scoreHistoryTest + scoreHistoryTest

            #_, avgHitRateTrain, _ = Model.GetAverageHistory(lossHistoryTrain, scoreHistoryTrain, [1, 3, 5])
            #self.LogAccuracy( scoreHistoryTrain, 'Train', [0,1,2,3,4,5,6] )
            #_, avgHitRateTest, _ = self.GetAverageHistory(lossHistoryTest, scoreHistoryTest, [1, 3, 5])
            #self.LogAccuracy( scoreHistoryTrain, 'Test', [0,1,2,3,4,5,6] )

            self.metaParams['epochs'] = accumEpoch
            print( "Epoch: {}, Summary: {}", accumEpoch, self.metaParams)
          
            time = datetime.datetime.now()
            print ('End time: ', time)

        if logToFile: sys.stdout.close()


    def LogAccuracy(self, scoreHistory, title, classList):
        hitRateList = self.database.GetHitRate(scoreHistory, classList)
        precisionArray, recallArray = self.database.GetF1ArrayClassTimesBatch(scoreHistory)
        nClasses = len(self.database.tokenLabelStringList)

        print( '\nAccuracy logging: ', title, '===================================================')

        print( 'Average hit rate: ', round( np.average(hitRateList), 2) )

        for c in range(nClasses):
            print( 'Average f1 precision for ', self.database.GetTokenLabelString(c), ' = ', round( np.nanmean(precisionArray[c, :]), 2) )

        for c in range(nClasses):
            print( 'Average f1 recall for ', self.database.GetTokenLabelString(c), ' = ', round( np.nanmean(recallArray[c, :]), 2) )

        print( 'Accuracy logging ========================================================\n')


    def LearnFromEpoch(self, dataset, step):

        lossHistory = []; scoreHistory = []

        for miniBatch in dataset :
            mbGradient, mbLoss, mbScore = self.LearnFromMiniBatch(batch = miniBatch, step = step)

            lossHistory.append(mbLoss)
            scoreHistory.append(mbScore)

            step += 1

        return lossHistory, scoreHistory, step


    def LearnFromMiniBatch(self, batch, step):
        batchLoss = 0.0; batchScores = []

        sumGradient = []
        for weight in self.weightsSnapshot:
            sumGradient.append( tf.zeros_like( weight ) )

        batchSize = 0
        for dataset_record in batch:
            batchSize += 1

            with tf.GradientTape() as tape:
                tape.watch(self.weightsSnapshot)

                loss, score = self.GetLossForSingleExample(dataset_record)
                batchLoss += loss.numpy()
                batchScores.append(score)

                tf.debugging.assert_all_finite(loss, message = 'Loss is a nan.')
                print( '\nloss =', loss.numpy() )

            grad = tape.gradient(loss, self.weightsSnapshot)

            for n in range(len(sumGradient)):
                if grad[n] is not None:
                    tf.debugging.assert_all_finite(grad[n], message = 'Gradient is a nan.')
                    sumGradient[n] = tf.add( sumGradient[n], grad[n] )
                else: pass # grad[n] is None. No change to sumGradient[n]

        self.DoGreatForGradient()

        self.optimizer.apply_gradients( zip(sumGradient, self.weightsSnapshot) )

        return sumGradient, batchLoss / batchSize, batchScores


    def DoGreatForGradient(self):
        pass

    def EvaluateOnDataset(self, dataset, shuffleBuffer, miniBatchSize):
        lossHistory = []; scoreHistory = []

        dataset = dataset.shuffle(buffer_size = shuffleBuffer, reshuffle_each_iteration = True).batch(batch_size = miniBatchSize, drop_remainder = True)

        for miniBatch in dataset :

            batchLoss = 0.0; batchSize = 0; batchScores = []

            for dataset_record in miniBatch :
                batchSize += 1

                loss, score = self.GetLossForSingleExample(dataset_record)
                batchLoss += loss.numpy()
                batchScores.append(score)
                    
                tf.debugging.assert_all_finite(loss, message = 'Loss is a nan.')
                print( '\nloss =', loss.numpy() )
            
            lossHistory.append( batchLoss / batchSize )
            scoreHistory.append( batchScores )


        return lossHistory, scoreHistory

    def GetLossForSingleExample(self, dataset_record):
        senIdOneBased, sentence, aspect, opinion, check = self.database.DecodeComibinedDatasetRecord(dataset_record)
        consistent, aspectList, opinionList, wrongAspList, wrongOpnList = self.database.GetRefeinedLabels(sentence, aspect, opinion)

        if self.testMode:
            print('============================= Sentence: ', str(senIdOneBased), '===================================')
            print("\n{} \n{} \n{}".format(sentence, aspect, opinion))

            if not consistent:
                print( "Sentence inconsistent with labels. Skipping.=========")
                print( "Wrong aspects: ", wrongAspList)
                print( "Wrong opinions: ", wrongOpnList)

        probDistList = self.GetProbabilityDistributionList(sentence)

        parTokenList = self.rnscnBlock.GetParTokenList( sentence)
        tokenLabelClassList, tokenLabelStringList = self.database.GetTokenLabelList(parTokenList, aspectList, opinionList)

        if self.testMode:
            print( 'True label class : ', tokenLabelClassList)
            print( 'True label string: ', tokenLabelStringList)

        assert len(probDistList) == len(tokenLabelClassList)
        trueDistList = [None] * len(probDistList)
        nClass = self.database.NLabelClass()

        scoreList = []
        lossTotal = tf.constant(value = 0.0, dtype = configWeightDType )

        for tokenId in range(len(trueDistList)):

            trueClassDist = tf.Variable(tf.zeros( shape = [nClass], dtype = configWeightDType ) )
            sensitivity = 1.0
            trueClass = tokenLabelClassList[tokenId]
            trueClassDist[ trueClass  ].assign( sensitivity )

            probClassDist = probDistList[tokenId]
            probClass = tf.argmax(probClassDist).numpy()

            scoreList.append( (trueClass, probClass ) )

            text = 'PredDist: '
            for n in range(nClass):
                if n == trueClass:
                    text = text + " [{0:2.0f}]".format(probClassDist[n].numpy() * 100)
                else:
                    text = text + "  {0:2.0f} ".format(probClassDist[n].numpy() * 100)
            if trueClass != probClass: text = text + '   No'
            print(text)

            a = - tf.multiply( trueClassDist, tf.math.log(probClassDist) )
            assert a.shape == [self.database.NLabelClass()]
            tf.debugging.assert_all_finite(a, message = 'a is a nan.')
            assert len(probDistList) > 0
            # a = tf.square(a) # to accelarate learning, trading off stability.
            a = tf.reduce_sum(a)
            loss = a / ( 2.0 * len(probDistList) )
            tf.debugging.assert_all_finite(loss, message = 'Loss is a nan.')
            lossTotal = tf.add( lossTotal, loss )

        return lossTotal, scoreList

    def Predict(self, sentence):
        probDistList = self.GetProbabilityDistributionList(sentence)
        predLabelStringList = [None] * len(probDistList)

        for tokenId in range(len(probDistList)):
            probDist = probDistList[tokenId]
            predClass = tf.argmax(probDist)
            assert predClass.shape == []
            predLabelString = self.database.GetTokenLabelString(predClass)
            predLabelStringList[tokenId] = predLabelString

        return predLabelStringList

    def GetProbabilityDistributionList(self, sentence):

        parTokenList = self.parser.ParseSentenceAndKeepState(sentence)
        parTokenEmbListSen, embTokenList, parToEmbMapping = self.embedder.GetTokenEmbList( parTokenList )

        if self.testMode:
            self.PrintParToEmbMapping(parTokenList, embTokenList, parToEmbMapping)

        _, hiddenListRnscn = self.rnscnBlock.GenerateStatesSingleSetence(parTokenEmbListSen)

        hiddenListGru = self.gruBlock.GenerateStates( hiddenListRnscn )

        hiddenMFCN = self.multifcn.FeedForwardList( hiddenListGru )

        probDistList = self.GetSoftMax(hiddenMFCN)

        return probDistList

    def GetSoftMax(self, hiddenList):
        print( 'SoftMax ==============================================')
        probDistList = [None] * len(hiddenList)
        nClass = self.database.NLabelClass()
        for tokenId in range(len(hiddenList)):
            logits = hiddenList[tokenId]
            assert logits.shape == [ 1, nClass ]
            logits = tf.reshape(logits, [-1])
            probDist = tf.nn.softmax(logits = logits, axis = 0)
            assert probDist.shape == [ nClass ]
            probDistList[tokenId] = probDist
        
        return probDistList

    
    def PrintParToEmbMapping(self, parTokenList, embTokenList, parToEmbMapping):
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




