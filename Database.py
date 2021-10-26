import os
import json
import pickle
import random
import tensorflow as tf
import numpy as np

from Config import *

from VisualClass import *

class Database():

    sentencesFname = 'sentences'
    aspectLabelsFname = 'aspect_labels'
    opinionLabelsFname = 'opinion_labels'
    annotationSuffix = '_ann'
    cleanSuffix = '_clean'
    removedSuffix = '_removed'
    traintestPrefix = 'all_data_'
    trainPrefix = 'train_'
    testPrefix = 'test_'
    parameters = 'parameters'
    dependency = 'dependency'
    weights = 'weights'
    combined = 'combined'
    tokenLableClass = 'tokenLabelClass'
    history = 'history'
    historyImage = 'history'
    historySheet = 'history'
    cacheName = 'cache'
    databaseSuffix = '_database'
    embedderSuffix = '_embedder'
    parserSuffix = '_parser'
    log = 'log'
    dataFolder = ''
    sentagpair = 'sentagpair'

    def __init__(self, homePath, filebase = '', fileNameSuffix = '', tokenizer = None):
        self.homePath = homePath
        self.tokenizer = tokenizer

        absHomePath = os.path.abspath(self.homePath)
        assert os.path.exists(absHomePath)

        path = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.sentencesFname) + '.txt'
        assert os.path.exists(path)
        self.pTrainSentences = path
        
        path = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.sentencesFname) + '.txt'
        assert os.path.exists(path)
        self.pTestSentences = path

        self.pTrainAnnotations = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.sentencesFname + self.annotationSuffix) + '.txt'
        #assert os.path.exists(self.pathToTrainAnnotations)
        self.pTestAnnotations = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.sentencesFname + self.annotationSuffix) + '.txt'
        #assert os.path.exists(self.pathToTestAnnotations)
        
        path = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.aspectLabelsFname) + '.txt'
        assert os.path.exists(path)
        self.pTrainAspectLables = path
        path = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.aspectLabelsFname) + '.txt'
        assert os.path.exists(path)
        self.pTestAspectLables = path

        path = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.opinionLabelsFname) + '.txt'
        assert os.path.exists(path)
        self.pTrainOpinionLables = path
        path = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.opinionLabelsFname) + '.txt'
        assert os.path.exists(path)
        self.pTestOpinionLables = path

        path = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.sentencesFname + self.cleanSuffix) + '.txt'
        self.pTrainSentencesClean = path
        path = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.sentencesFname + self.cleanSuffix) + '.txt'
        self.pTestSentencesClean = path

        path = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.sentencesFname + self.removedSuffix) + '.txt'
        self.pTrainSentencesRemoved = path
        path = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.sentencesFname + self.removedSuffix) + '.txt'
        self.pTestSentencesRemoved = path

        path = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.dependency) + '.txt'
        self.pTrainDependency = path
        path = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.dependency) + '.txt'
        self.pTestDependency = path

        path = os.path.join( absHomePath, self.dataFolder, self.traintestPrefix + self.combined) + '.txt'
        self.pTraintestCombined = path
        path = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.combined) + '.txt'
        self.pTrainCombined = path
        path = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.combined) + '.txt'
        self.pTestCombined = path
        path = os.path.join( absHomePath, self.dataFolder, self.traintestPrefix + self.combined) + '.bin'
        self.pTrainTestCombinedBin = path

        path = os.path.join( absHomePath, self.dataFolder, self.sentagpair) + '.txt'
        self.pSentagpair = path

        path = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.tokenLableClass) + '.txt'
        self.pTrainTokenLabelClass = path
        path = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.tokenLableClass) + '.txt'
        self.pTestTokenLabelClass = path

        path = os.path.join( absHomePath, self.dataFolder, filebase, self.trainPrefix + self.history) + '.bin'
        self.pTrainHistory = path
        path = os.path.join( absHomePath, self.dataFolder, filebase, self.testPrefix + self.history) + '.bin'
        self.pTestHistory = path

        path = os.path.join( absHomePath, self.dataFolder, filebase, self.historySheet + fileNameSuffix) + '.csv'
        self.pHistorySheet = path
        path = os.path.join( absHomePath, self.dataFolder, self.historySheet + fileNameSuffix) + '.csv'
        self.pHistorySheetTotal = path

        path = os.path.join( absHomePath, self.dataFolder, filebase, self.historyImage + fileNameSuffix) + '.png'
        self.pHistoryImage = path

        path = os.path.join( absHomePath, self.dataFolder, filebase, self.cacheName + self.databaseSuffix) + '.bin'
        self.pCacheDatabase = path

        path = os.path.join( absHomePath, self.dataFolder, filebase, self.cacheName + self.embedderSuffix) + '.bin'
        self.pCacheEmbedder = path

        path = os.path.join( absHomePath, self.dataFolder, filebase, self.cacheName + self.parserSuffix) + '.bin'
        self.pCacheParser = path
        
        path = os.path.join( absHomePath, self.dataFolder, filebase, self.parameters) + '.txt'
        self.pParameters = path

        path = os.path.join( absHomePath, self.dataFolder, filebase, self.weights) + '.bin'
        self.pWeights = path

        path = os.path.join( absHomePath, self.dataFolder, filebase, self.log) + '.txt'
        self.pLog = path

        self.lbOther = 'Oth'; self.lbBeginAsp = 'bAsp'; self.lbInsideAsp = 'iAsp'
        self.lbBeginPosOpn = 'bOpn+'; self.lbInsidePosOpn = 'iOpn+'; self.lbBeginNegOpn = 'bOpn-'; self.lbInsideNegOpn = 'iOpn-'
        
        self.tokenLabelStringList = \
            [ self.lbOther, self.lbBeginAsp, self.lbInsideAsp, self.lbBeginPosOpn, self.lbInsidePosOpn, self.lbBeginNegOpn, self.lbInsideNegOpn ]


    @classmethod
    def BuildFileNames( self, metaParams ):
        verString = 'v' + str(configVersion)

        filebase = verString + '-' \
            + str( metaParams['Embedder'] ) + '-' \
            + str( metaParams['RNSCN'] ) + '.' \
            + str( metaParams['rnscnDim'] ) + '-' \
            + str( metaParams['GRU'] ) + '.' \
            + str( metaParams['gruDim'] ) + '-bat.' \
            + str( metaParams['batch'] ) 

        if configVersion == 0.83 :
            filebase = verString + '-' \
                + str( metaParams['Embedder'] ) + '.' + str(metaParams['bertLayer']) + '-'\
                + str( metaParams['RNSCN'] ) + '.' \
                + str( metaParams['rnscnDim'] ) + '-' \
                + str( metaParams['GRU'] ) + '.' \
                + str( metaParams['gruDim'] ) + '-bat.' \
                + str( metaParams['batch'] ) 
                        
        
        folder = os.path.abspath( metaParams['dataPath'] + filebase)

        return filebase, folder

    def GetListOfLines(self, path, addLastPeriod = False):
        try:
            file = open(path, 'rt')
            text = file.read()
            file.close()
        except:
            raise Exception("Couldn't open/read/close file: " + path)
        
        lines = text.split('\n')

        nlines = []
        if addLastPeriod == True:
            for line in lines:
                if line[-1] != '?' and line[-1] != '.' :
                    line = line + '.'
                nlines.append(line)

        if addLastPeriod == False:
            return lines
        else:
            return nlines

    def SaveJsonData(self, data, path):
        try:
            file = open(path, 'wt+')
            json.dump(data, file)
            file.close()
        except:
            raise Exception("Couldn't open/save/close file:" + path)

    def LoadJsonData(self, path):
        try:
            file = open(path, 'rt')
            data = json.load(file)
            file.close()
        except:
            raise Exception("Couldn't open/read/close file:" + path)
        
        return data

    def SaveBinaryData(self, data, path):
        try:
            file = open(path, 'wb+')
            pickle.dump(data, file)
            file.close()
        except:
            raise Exception("Couldn't open/read/close file:" + path)

    def LoadBinaryData(self, path):
        try:
            file = open(path, 'rb')
            data = pickle.load(file)
            file.close()
        except:
            raise Exception("Couldn't open/read/close file:" + path)

        return data

    def Exists(self, path):
        return os.path.exists(path)

    def CreateSentenceAndOneHotLabelsNumpy(self, save = False):

        if self.Exists(self.pTrainTestCombinedBin) :
            sentences, onehots = self.LoadBinaryData(self.pTrainTestCombinedBin)
        else:
            pCombined = self.pTraintestCombined

            comLines = self.GetListOfLines(pCombined, addLastPeriod = False)
            lineCount = len(comLines)

            senPreIds = [n for n in range( 0, int(lineCount/5) )]
            random.shuffle(senPreIds)

            sentences = []; aspects = []; opinions = []; onehots = []

            senNo = 0; senCount = len(senPreIds)
            while senNo < senCount:
                senId = senPreIds[senNo] + 1
                lineId = (senId - 1) * 5
                
                senOrgId = comLines[lineId]; lineId += 1; senOrgId = senOrgId.strip()
                sentence = comLines[lineId]; lineId += 1; sentence = " ".join(sentence.split())
                aspect = comLines[lineId]; lineId += 1; aspect = " ".join(aspect.split())
                opinion = comLines[lineId]; lineId += 1; opinion = " ".join(opinion.split())

                #check = comLines[lineId]; lineId += 1
                consistent, aspectList, opinionList, wrongAspList, wrongOpnList = self.GetRefeinedLabels(sentence, aspect, opinion)
                if consistent: check = ""
                else: check = "ERROR: asp: {}, opn: {}".format(wrongAspList, wrongOpnList); print(check)

                parTokenList, _, _ = self.tokenizer.GetTokenList(sentence)
                parTokenList = (" ".join(parTokenList)).split() # Remove ""s, which are generated by tokenizer for maxTokens.

                tokenLabelClassList, _ = self.GetTokenLabelList(parTokenList, aspectList, opinionList)
                assert len(tokenLabelClassList) == len(parTokenList)
                
                nTokens = len(parTokenList)
                nPlaces = nTokens # self.tokenizer.maxTokens
                nClasses = self.NLabelClass()

                onehotsSen = [None] * nPlaces

                for placeId in range(nPlaces) :
                    onehotsToken = [0] * nClasses
                    tokenId = placeId - (nPlaces - nTokens)
                    if tokenId >= 0:
                        trueClass = tokenLabelClassList[tokenId]
                        onehotsToken[ trueClass ] = 1
                    onehotsSen[placeId] = onehotsToken

                sentences.append( sentence )
                #aspects.append(aspect)
                #opinions.append(opinion)
                onehots.append( onehotsSen )

                senNo += 1

            if save :
                self.SaveBinaryData( ( sentences, onehots ), self.pTrainTestCombinedBin )

        ret = tf.data.Dataset.from_generator( lambda: iter( zip( sentences, onehots ) ), output_types = ( tf.string, tf.int32 ) )
        
        return ret

    def CreateFullDatasetsFromCombinedFile(self, trainPortion = 0.7, testPortion = 0.3, addLastPeriod = False, returnSizeOnly = False, save = False):
        assert trainPortion >= 0 and testPortion >= 0 and trainPortion + testPortion <= 1.0

        pCombined = self.pTraintestCombined

        comLines = self.GetListOfLines(pCombined, addLastPeriod = False)
        lineCount = len(comLines)

        trainSenCount = int( lineCount * trainPortion / 5 )
        testSenCount = int(  lineCount * (trainPortion + testPortion) / 5 )

        if returnSizeOnly: 
            return trainSenCount, testSenCount - trainSenCount

        senPreIds = [n for n in range( 0, int(lineCount/5) )]
        random.shuffle(senPreIds)

        sentenceIdsTrain = []; sentencesTrain = []; aspectsTrain = []; opinionsTrain = []; checksTrain = []
        sentenceIdsTest = []; sentencesTest = []; aspectsTest = []; opinionsTest = []; checksTest = []

        senNo = 0; senCount = len(senPreIds)
        while senNo < senCount:
            senId = senPreIds[senNo] + 1
            lineId = (senId - 1) * 5
            
            senOrgId = comLines[lineId]; lineId += 1; senOrgId = senOrgId.strip()
            sentence = comLines[lineId]; lineId += 1; sentence = " ".join(sentence.split())


            aspect = comLines[lineId]; lineId += 1; aspect = " ".join(aspect.split())

            opinion = comLines[lineId]; lineId += 1; opinion = " ".join(opinion.split())

            #check = comLines[lineId]; lineId += 1
            consistent, aspectList, opinionList, wrongAspList, wrongOpnList = self.GetRefeinedLabels(sentence, aspect, opinion)
            if consistent: check = ""
            else: check = "ERROR: asp: {}, opn: {}".format(wrongAspList, wrongOpnList); print(check)

            if senNo < trainSenCount : 
                sentenceIdsTrain.append(senOrgId)
                sentencesTrain.append(sentence)
                aspectsTrain.append(aspect)
                opinionsTrain.append(opinion)
                checksTrain.append(check)
            elif senNo < testSenCount : 
                sentenceIdsTest.append(senOrgId)
                sentencesTest.append(sentence)
                aspectsTest.append(aspect)
                opinionsTest.append(opinion)
                checksTest.append(check)

            senNo += 1

        if save :
            self.__SaveCombinedFile__(sentenceIdsTrain, sentencesTrain, aspectsTrain, opinionsTrain, checksTrain, self.pTrainCombined)
            self.__SaveCombinedFile__(sentenceIdsTest, sentencesTest, aspectsTest, opinionsTest, checksTest, self.pTestCombined)
            self.__SaveCombinedFile__(sentenceIdsTrain+sentenceIdsTest, sentencesTrain+sentencesTest, aspectsTrain+aspectsTest,\
                opinionsTrain+opinionsTest, checksTrain+checksTest, self.pTraintestCombined)

        concatList = []
        for id, sen, asp, opn, check in zip(sentenceIdsTrain, sentencesTrain, aspectsTrain, opinionsTrain, checksTrain):
            concatList.append( ( id, sen, asp, opn, check ) )
        trainDataset = tf.data.Dataset.from_tensor_slices(concatList)

        concatList = []; lineId = 0
        for id, sen, asp, opn, check in zip(sentenceIdsTest, sentencesTest, aspectsTest, opinionsTest, checksTest):
            concatList.append( ( id, sen, asp, opn, check ) )
        testDataset = tf.data.Dataset.from_tensor_slices(concatList)

        return trainDataset, testDataset


    def DecodeComibinedDatasetRecord(self, dataset_record):
        lineId = int(dataset_record[0].numpy())
        sentence = dataset_record[1].numpy().decode()
        aspect = dataset_record[2].numpy().decode()
        opinion = dataset_record[3].numpy().decode()
        check = dataset_record[4].numpy().decode()

        return lineId, sentence, aspect, opinion, check


    def __SaveCombinedFile__(self, idsList, sentencesList, aspectsList, opinionsList, checksList, path):

        combinedList = []
        linefeed = '\n' 
        for id, sen, asp, opn, check in zip(idsList, sentencesList, aspectsList, opinionsList, checksList):
            combinedList.append(id + linefeed)
            combinedList.append(sen + linefeed)
            combinedList.append(asp + linefeed)
            combinedList.append(opn + linefeed)
            combinedList.append(check + linefeed)

        if len(combinedList) > 0 and len(combinedList[-1]) > 0 and combinedList[-1][-1] == linefeed :
            combinedList[-1] = combinedList[-1][:-1] # remove last linefeed.

        combinedFile = open(path, 'wt+')
        combinedFile.writelines(combinedList)
        combinedFile.close()


    def GetRefeinedLabels(self, sentence, aspect, opinion):
        wrongAspList = []; wrongOpnList = []

        aspectList = self.__GetRefinedAspectList__(aspect)
        opinionList = self.__GetRefinedOpinionList__(opinion)

        consistent = True
        sentence = sentence.lower()

        for asp in aspectList:
            if 0 > sentence.find(asp.lower()):
                consistent = False
                wrongAspList.append(asp)
        
        for opnText, opnScore in opinionList:
            if 0 > sentence.find(opnText.lower()):
                consistent = False
                wrongOpnList.append(opnText)
       
        return consistent, aspectList, opinionList, wrongAspList, wrongOpnList

    def __GetRefinedAspectList__(self, aspect):
        aList = aspect.split(','); nList = []
        for aspect in aList:
            asp = aspect.lower().strip()

            if asp != 'nil':
                nList.append(asp)

        return nList

    def __GetRefinedOpinionList__(self, opinion):
        oList = opinion.split(','); nList = []
        for opinion in oList:
            opnStr = opinion.lower().strip()
            if opnStr != 'nil':
                if opnStr.find('+1') >= 0:
                    opnScore = 1
                elif opnStr.find('-1') >= 0:
                    opnScore = -1
                else: opnScore = -1 # For integrity.
                opnStr = opnStr.replace('+1', ''); opnStr = opnStr.replace('-1', ''); opnStr = opnStr.strip()            
                nList.append( (opnStr, opnScore) )
        
        return nList

    def GetTokenLabelList(self, tokenList, aspList, opnList):
        nO = self.__GetTokenLabelClass__(self.lbOther) # for other
        tokenLabelStringList = [self.lbOther] * len(tokenList)
        tokenLabelClassList = [nO] * len(tokenList)

        for asp in aspList:
            aspTokenList = asp.split()
            if len(aspTokenList) <= 0: continue
            location = self.__FindSeries__( tokenList, aspTokenList )
            if location >= 0:
                BA = self.lbBeginAsp; IA = self.lbInsideAsp
                #if tokenLabelStringList[location] != self.lbOther and tokenLabelStringList[location] != BA: raise Exception('Label inconsistency!')
                tokenLabelStringList[location] = BA
                #if tokenLabelClassList[location] != nO and tokenLabelClassList[location] != self.GetTokenLabelClass(BA) : raise Exception('Label inconsistency!')
                tokenLabelClassList[location] = self.__GetTokenLabelClass__(BA)
                for inc in range( 1, len(aspTokenList) ):
                    #if tokenLabelStringList[location + inc] != self.lbOther and tokenLabelStringList[location + inc] != IA: raise Exception('Label inconsistency!')
                    tokenLabelStringList[location + inc] = IA
                    #if tokenLabelClassList[location + inc] != nO and tokenLabelClassList[location + inc] != self.GetTokenLabelClass(IA) : raise Exception('Label inconsistency!')
                    tokenLabelClassList[location + inc] = self.__GetTokenLabelClass__(IA)
        
        for opn in opnList:
            opnString, opnScore = opn
            opnTokenList = opnString.split()
            if len(opnTokenList) <= 0: continue
            location = self.__FindSeries__( tokenList, opnTokenList )
            if location >= 0:
                if opnScore > 0: BO = self.lbBeginPosOpn; IO = self.lbInsidePosOpn
                else: BO = self.lbBeginNegOpn; IO = self.lbInsideNegOpn

                #if tokenLabelStringList[location] != self.lbOther and tokenLabelStringList[location] != BO : raise Exception('Label inconsistency!')
                tokenLabelStringList[location] = BO
                
                #if tokenLabelClassList[location] != nO and tokenLabelClassList[location] != self.GetTokenLabelClass(BO) : raise Exception('Label inconsistency!')
                tokenLabelClassList[location] = self.__GetTokenLabelClass__(BO)
                
                for inc in range( 1, len(opnTokenList) ):
                    #if tokenLabelStringList[location + inc] != self.lbOther and tokenLabelStringList[location + inc] != IO : raise Exception('Label inconsistency!')
                    tokenLabelStringList[location + inc] = IO
                
                    #if tokenLabelClassList[location + inc] != nO and tokenLabelClassList[location + inc] != self.GetTokenLabelClass(IO) : raise Exception('Label inconsistency!')
                    tokenLabelClassList[location + inc] = self.__GetTokenLabelClass__(IO)

        return tokenLabelClassList, tokenLabelStringList

    def __FindSeries__(self, superSeries, subSeries):       
        location = -1
        if len(superSeries) > 0 and len(subSeries) > 0 :
            if len(subSeries) > len(superSeries):
                location = -1
            else:
                start = 0
                while len(superSeries) - start >= len(subSeries):
                    if self.__FindHeadSeries__(superSeries[start:], subSeries):
                        location = start; break
                    start += 1
        return location

    def __FindHeadSeries__(self, superSeries, subSeries):
        assert len(superSeries) > 0
        assert len(subSeries) > 0
        
        found = True
        if len(subSeries) > len(superSeries):
            found = False
        else:               
            for a, b in zip(superSeries, subSeries):
                if a.lower() != b.lower():
                    found = False
                    break
            return found

    def NLabelClass(self):
        return len(self.tokenLabelStringList)

    def GetOnehotsForOther(self):
        ret = [0] * self.NLabelClass()
        ret[ self.__GetTokenLabelClass__( self.lbOther) ] = 1

        return ret       

    def GetTokenLabelString(self, numeral):
        return self.tokenLabelStringList[numeral]

    def __GetTokenLabelClass__(self, string):
        numeral = 0; count = len(self.tokenLabelStringList)
        while numeral < count and self.tokenLabelStringList[numeral] != string:
            numeral += 1
        if numeral >= count:
            raise Exception('Token label string not found: ', string)
        else:
            return numeral

    def GetHitRate(self, scoreHistory, classList):
        hitRate = []
        for batchScore in scoreHistory:
            cntHit = 0; cntMiss = 0
            for sentenceScore in batchScore:
                for trueClass, predClass in sentenceScore:
                    if trueClass in classList :
                        if trueClass == predClass: cntHit += 1
                        else: cntMiss += 1
            hitRate.append( np.float32(1.0 * cntHit / (cntHit + cntMiss)) )

        return hitRate

    def GetF1ArrayClassTimesBatch(self, scoreHistory) :
        classes = len(self.tokenLabelStringList)
        batches = len(scoreHistory)
        precision = np.zeros( (classes, batches), dtype = np.float32 ); precision[:] = np.nan
        recall = np.zeros( (classes, batches), dtype = np.float32 ); recall[:] = np.nan

        for batchId in range(batches) :
            batchScore = scoreHistory[batchId]

            relavant = [0] * classes; truePositive = [0] * classes; falsePositive = [0] * classes

            for sentenceScore in batchScore :
                for trueClass, predClass in sentenceScore: # one for each token in the sentence.
                    # A trueClass is observed. [Positive (in a class)] = [Predicted as true (in the class)]
                    # This token is truely belongs to trueClass, increasing # of relavant tokens of trueClass. 
                    relavant[trueClass] += 1

                    if trueClass == predClass : # [True positive in predClass]
                    # true positive in predClass: this token is predicted as predClass, and it's true.
                        truePositive[predClass] += 1
                    else: # [False positive in predClass]
                    # false positive in predClass: this token is predicted as predClass, and it's false.
                        falsePositive[predClass] += 1

            for clsId in range(classes) :
                precision[clsId, batchId] = 1.0 * truePositive[clsId] / (truePositive[clsId] + falsePositive[clsId] + 1e-30)
                recall[clsId, batchId] = 1.0 * truePositive[clsId] / (relavant[clsId] + 1e-30)

        return precision, recall

    @classmethod
    def VisualizeTrainHistory(cls, metaParams, avgLastNBatches = 30, imageSize = (4, 3), classList = [1], showNotSave = True, fileNameSuffix = '') :
        filebase, folder = cls.BuildFileNames( metaParams )
        database = Database(homePath = metaParams['dataPath'],  filebase = filebase, fileNameSuffix = fileNameSuffix )

        history = database.LoadBinaryData(database.pTrainHistory)
        metaParams, lossHistoryTrain, scoreHistoryTrain, lossHistoryTest, scoreHistoryTest = history

        def remove(key):
            if metaParams.get(key, None) != None : metaParams.pop(key)
        remove('lossTrain'); remove('lossTest'); remove('hitRateTrain'); remove('hitRateTest')
        for n in range(len(lossHistoryTrain)) : lossHistoryTrain[n] = np.float32(lossHistoryTrain[n])
        for n in range(len(lossHistoryTest)) : lossHistoryTest[n] = np.float32(lossHistoryTest[n])
        
        #history = (metaParams, lossHistoryTrain, scoreHistoryTrain, lossHistoryTest, scoreHistoryTest)
        #database.SaveBinaryData(history, database.pTrainHistory)

        title = "Evaluation on Test Data: Accuracy by steps: " + str(classList)
        fromIndex = int(  (metaParams['epochs'] - 1) *  len(scoreHistoryTest) / metaParams['epochs']  ) # Aims at the last epoch.
        seriesDict = cls.GetSeriesDict(database, 'test', scoreHistoryTest[fromIndex:], lossHistoryTest[fromIndex:], classList, avgLastNBatches)

        path = database.pHistorySheetTotal
        if not database.Exists(path) :
            cls.WriteCSVTotalSummary(title, seriesDict, metaParams, imageSize = imageSize, showNotSave = showNotSave, path = path, HeaderNotBody=True)
        cls.WriteCSVTotalSummary(title, seriesDict, metaParams, imageSize = imageSize, showNotSave = showNotSave, path = path, HeaderNotBody=False)

        if showNotSave == False : path = database.pHistoryImage
        else: path = ''
        cls.Plot(title, seriesDict, metaParams, imageSize = imageSize, showNotSave = showNotSave, path = path)


    @classmethod
    def WriteCSVTotalSummary(cls, title, seriesDict, metaParams, imageSize, showNotSave, path = '', HeaderNotBody = False):
        testCaseFactors = ['version', 'Embedder', 'bertLayer', 'embDim', 'RNSCN', 'rnscnDim', 'GRU', 'gruDim', 'batch']

        if HeaderNotBody : file = open(path, 'w+')
        else : file = open(path, 'at+')

        if HeaderNotBody: pass
        else: file.write('\n')

        # Write line header here.

        if HeaderNotBody :
            for key in testCaseFactors: file.write(key + ',')
        else :
            def writeParam(key) :
                file.write(str(metaParams[key]) + ',')
            for key in testCaseFactors: writeParam(key)

        ParamVecDict = {
            'emb.no': (1, 0, 0, 0), 'emb.bert': (0, 1, 0, 0), 'emb.w2vec': (0, 0, 1, 0), 'emb.bword': (0, 0, 0, 1),
            'rnscn.no': (1, 0, 0, 0), 'rnscn.up': (0, 1, 0, 0), 'rnscn.down': (0, 0, 1, 0), 'rnscn.bidir': (0, 0, 0, 1),
            'gru.no': (1, 0, 0, 0), 'gru.right': (0, 1, 0, 0), 'gru.left': (0, 0, 1, 0), 'gru.bidir': (0, 0, 0, 1),
        }

        var = 1
        for key in testCaseFactors:
            key2 = str(metaParams[key])
            if ParamVecDict.get(key2, None) != None:
                if HeaderNotBody :
                    for _ in range(4): file.write(',' + str(var)); var += 1
                    pass
                else:
                    def writeParamVec(vec) :
                        for v in vec : file.write(',' + str(v))
                    writeParamVec(ParamVecDict[key2])
        file.write(','); 
        
        if HeaderNotBody : 
            file.write('Sum(p,r)')
        else:
            sum = 0
            for key in seriesDict :
                if key.find('.pre.') >= 0 or key.find('.rec.') >= 0 : #or key.find('.f1.'):
                    sum += np.nansum(seriesDict[key])
            file.write(str(sum))
        file.write(',')


        for key in seriesDict :
            if HeaderNotBody : file.write(key); file.write(','); file.write('mean'); file.write(','); file.write('std')
            else:
                file.write(''); file.write(','); file.write( str(np.nanmean(seriesDict[key])) ); file.write(','); file.write( str(np.nanstd(seriesDict[key])) )
            col = 1
            for data in seriesDict[key] :
                file.write( ', ' )
                if HeaderNotBody :  file.write( str(col) ); col += 1
                else: file.write( str(data) )
            file.write( ', ' )
        file.close()

    @classmethod
    def Plot(cls, title, seriesDict, metaParams, imageSize, showNotSave, path = ''):
        vc = VisualClass()
        vc.PlotStepHistory(title, seriesDict, metaParams, imageSize = imageSize, path = path)

    @classmethod
    def GetSeriesDict(cls, database, prefix, scoreHistory, lossHistory, classList, avgLastNBatches):
        seriesDict = {}

        precisionArray, recallArray = database.GetF1ArrayClassTimesBatch(scoreHistory)
        fArray = 2 * (precisionArray * recallArray) / (precisionArray + recallArray + 1e-30)
        avgLastNBatches = min( avgLastNBatches, precisionArray.shape[1] )

        for clsId in classList : # range(nClasses):
            seriesDict[prefix + '.pre.' + database.tokenLabelStringList[clsId] + ': ' +  str(round(np.nanmean(precisionArray[clsId, -avgLastNBatches:]), 2)) + '/' + str(round(np.nanstd(precisionArray[clsId, -avgLastNBatches:]), 3)) ] = precisionArray[clsId, :]
            seriesDict[prefix + '.rec.' + database.tokenLabelStringList[clsId] + ': ' +  str(round(np.nanmean(recallArray[clsId, -avgLastNBatches:]), 2)) + '/' + str(round(np.nanstd(recallArray[clsId, -avgLastNBatches:]), 3)) ] = recallArray[clsId, :]
            seriesDict[prefix + '.f1.' + database.tokenLabelStringList[clsId] + ': ' +  str(round(np.nanmean(fArray[clsId, -avgLastNBatches:]), 2)) + '/' + str(round(np.nanstd(fArray[clsId, -avgLastNBatches:]), 3)) ] = fArray[clsId, :]

        seriesDict[prefix + 'AvgPre: ' +  str(round(np.nanmean(precisionArray[classList, -avgLastNBatches:]),3)) + '/' + str(round(np.nanstd(precisionArray[classList, -avgLastNBatches:]), 4))] \
            = np.nanmean(precisionArray[classList, :], axis = 0)
        seriesDict[prefix + 'AvgRec: ' +  str(round(np.nanmean(recallArray[classList, -avgLastNBatches:]),3)) + '/' + str(round(np.nanstd(recallArray[classList, -avgLastNBatches:]), 4))] \
            = np.nanmean(recallArray[classList, :], axis = 0)
        seriesDict[prefix + 'AvgF1: ' +  str(round(np.nanmean(fArray[classList, -avgLastNBatches:]),3)) + '/' + str(round(np.nanstd(fArray[classList, -avgLastNBatches:]), 4))] \
            = np.nanmean(fArray[classList, :], axis = 0)

        avgLoss, avgHitRate, hitRateSeries = cls.GetAverageHistory(database, lossHistory, scoreHistory, classList)
        seriesDict[prefix + 'AvgLoss: ' + str(round(avgLoss, 3))] = lossHistory
        seriesDict[prefix + 'AvgHits: ' + str(round(avgHitRate, 3))] = hitRateSeries

        return seriesDict

    @classmethod
    def GetAverageHistory(self, database, lossHistory, scoreHistory, classList):

        lossSereis = lossHistory
        
        avgLoss = 0.0; cnt = 0
        for loss in lossSereis:
            if loss is not None:
                avgLoss += loss
                cnt += 1

        assert cnt > 0
        avgLoss = avgLoss / cnt

        hitRateSeries = database.GetHitRate(scoreHistory, classList)

        avgHitRate = 0.0; cnt = 0
        for hitRate in hitRateSeries:
            if hitRate is not None:
                avgHitRate += hitRate
                cnt += 1

        assert cnt > 0
        avgHitRate = avgHitRate / cnt

        return avgLoss, avgHitRate, hitRateSeries


# Unit Test


# Unit Test
#O = Database('./Database/', 'filebase')
#O.CreateFullDatasetsFromCombinedFile(trainPortion = .7, testPortion = .3, addLastPeriod = False, returnSizeOnly = False)
