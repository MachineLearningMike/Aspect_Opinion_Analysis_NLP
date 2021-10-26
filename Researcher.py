import os
import sys
import datetime

from Model import Model
from Config import *

sourcefiles = [
    'Database.py',
    'BERT.py',
    'GRU.py',
    'LayerNormalizer.py',
    'BatchNormalizer.py',
    'Model.py',
    'Parser.py',
    'RNSCN.py',
    'Researcher.py',
    'MultiFCN.py'
]


import os
from Database import Database

def GetTotalEffectiveLines(textFileList):
    database = Database('./Database/')
    sum = 0
    for filename in textFileList:
        path = os.path.join('./', filename)
        lines = database.GetListOfLines(path, addLastPeriod=False)
        cnt = 0
        for line in lines:
            if len(line.split()) > 0: cnt += 1
        sum += cnt

    return sum

import tensorflow as tf


def Run( dataPath, useEmb, embPath, bertLayer, embDim, useRNSCN, rnscnDim, useGRU, gruDim, miniBatch, epochs, logToFile ):
    
    metaParams = {}
    metaParams['version'] = configVersion
    metaParams['dataPath'] = dataPath 
    metaParams['Embedder'] = useEmb; metaParams['embPath'] = embPath; metaParams['bertLayer'] = bertLayer; metaParams['embDim'] = embDim
    metaParams['RNSCN'] = useRNSCN; metaParams['rnscnDim'] = rnscnDim
    metaParams['GRU'] = useGRU; metaParams['gruDim'] = gruDim
    metaParams['batch'] = miniBatch


    O = Model( metaParams = metaParams, testMode = True)
    #_ = O.GetProbabilityDistributionList("The other times I've gone its romantic date heaven, you can walk in get a booth by the windows, be treated like a VIP in a not-crowded place, with great food and service.")
    O.Train(shuffleBuffer = 1000, miniBatchSize = miniBatch, epochs = epochs, logToFile = logToFile)

    fileNameSuffix = '_06'
    Database.VisualizeTrainHistory(metaParams, avgLastNBatches = 10, imageSize = (15, 5), classList = [1, 3, 5, 2, 4, 6], showNotSave = False, fileNameSuffix = fileNameSuffix)


dataPath = './Database/'
embPath = './Embedders/';  bertLayer = 1; bertDim = 768; w2vecDim = 300
rnscnDim = 768; gruDim = 1000
miniBatch = 100; epochs = 2


for bertLayer in range(0, 12) :
    Run( dataPath, emb.bert, embPath, bertLayer, bertDim, rnscn.no, rnscnDim, gru.no, gruDim, miniBatch, epochs, logToFile = False )
    Run( dataPath, emb.bert, embPath, bertLayer, bertDim, rnscn.up, rnscnDim, gru.no, gruDim, miniBatch, epochs, logToFile = True )


#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.no, rnscnDim, gru.no, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.no, rnscnDim, gru.left, gruDim, miniBatch, epochs, logToFile = True )
##Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.no, rnscnDim, gru.right, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.no, rnscnDim, gru.bidir, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.up, rnscnDim, gru.no, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.up, rnscnDim, gru.left, gruDim, miniBatch, epochs, logToFile = True )
##Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.up, rnscnDim, gru.right, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.up, rnscnDim, gru.bidir, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.down, rnscnDim, gru.no, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.down, rnscnDim, gru.left, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.down, rnscnDim, gru.right, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.down, rnscnDim, gru.bidir, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.bidir, rnscnDim, gru.no, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.bidir, rnscnDim, gru.left, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.bidir, rnscnDim, gru.right, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.bidir, rnscnDim, gru.bidir, gruDim, miniBatch, epochs, logToFile = True )


#Run( dataPath, emb.no, embPath,  embLayers, bertDim, rnscn.no, rnscnDim, gru.left, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.no, embPath,  embLayers, bertDim, rnscn.no, rnscnDim, gru.right, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.no, embPath,  embLayers, bertDim, rnscn.no, rnscnDim, gru.bidir, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.no, embPath,  embLayers, bertDim, rnscn.up, rnscnDim, gru.no, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.no, embPath,  embLayers, bertDim, rnscn.up, rnscnDim, gru.left, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.no, embPath,  embLayers, bertDim, rnscn.up, rnscnDim, gru.right, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.no, embPath,  embLayers, bertDim, rnscn.up, rnscnDim, gru.bidir, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.no, embPath,  embLayers, bertDim, rnscn.down, rnscnDim, gru.no, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.no, embPath,  embLayers, bertDim, rnscn.down, rnscnDim, gru.left, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.no, embPath,  embLayers, bertDim, rnscn.down, rnscnDim, gru.right, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.no, embPath,  embLayers, bertDim, rnscn.down, rnscnDim, gru.bidir, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.no, embPath,  embLayers, bertDim, rnscn.bidir, rnscnDim, gru.no, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.no, embPath,  embLayers, bertDim, rnscn.bidir, rnscnDim, gru.left, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.no, embPath,  embLayers, bertDim, rnscn.bidir, rnscnDim, gru.right, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.no, embPath,  embLayers, bertDim, rnscn.bidir, rnscnDim, gru.bidir, gruDim, miniBatch, epochs, logToFile = True )

#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.no, rnscnDim, gru.no, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.no, rnscnDim, gru.left, gruDim, miniBatch, epochs, logToFile = True )
##Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.no, rnscnDim, gru.right, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.no, rnscnDim, gru.bidir, gruDim, miniBatch, epochs, logToFile = True )
#dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.up, rnscnDim, gru.no, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.up, rnscnDim, gru.left, gruDim, miniBatch, epochs, logToFile = True )
##Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.up, rnscnDim, gru.right, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.up, rnscnDim, gru.bidir, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.down, rnscnDim, gru.no, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.down, rnscnDim, gru.left, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.down, rnscnDim, gru.right, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.down, rnscnDim, gru.bidir, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.bidir, rnscnDim, gru.no, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.bidir, rnscnDim, gru.left, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.bidir, rnscnDim, gru.right, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.bidir, rnscnDim, gru.bidir, gruDim, miniBatch, epochs, logToFile = True )

#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.no, rnscnDim, gru.no, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.no, rnscnDim, gru.left, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.no, rnscnDim, gru.right, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.no, rnscnDim, gru.bidir, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.up, rnscnDim, gru.no, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.up, rnscnDim, gru.left, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.up, rnscnDim, gru.right, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.up, rnscnDim, gru.bidir, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.down, rnscnDim, gru.no, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.down, rnscnDim, gru.left, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.down, rnscnDim, gru.right, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.down, rnscnDim, gru.bidir, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.bidir, rnscnDim, gru.no, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.bidir, rnscnDim, gru.left, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.bidir, rnscnDim, gru.right, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.bidir, rnscnDim, gru.bidir, gruDim, miniBatch, epochs, logToFile = True )


dataPath = './Database/'
embPath = './Embedders/';  embLayers = 1; bertDim = 768; w2vecDim = 300
rnscnDim = 1000; gruDim = 1000
miniBatch = 100; epochs = 2

#Run( dataPath, emb.no, embPath,  embLayers, bertDim, rnscn.no, rnscnDim, gru.left, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.no, embPath,  embLayers, bertDim, rnscn.no, rnscnDim, gru.right, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.no, embPath,  embLayers, bertDim, rnscn.no, rnscnDim, gru.bidir, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.no, embPath,  embLayers, bertDim, rnscn.up, rnscnDim, gru.no, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.no, embPath,  embLayers, bertDim, rnscn.up, rnscnDim, gru.left, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.no, embPath,  embLayers, bertDim, rnscn.up, rnscnDim, gru.right, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.no, embPath,  embLayers, bertDim, rnscn.up, rnscnDim, gru.bidir, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.no, embPath,  embLayers, bertDim, rnscn.down, rnscnDim, gru.no, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.no, embPath,  embLayers, bertDim, rnscn.down, rnscnDim, gru.left, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.no, embPath,  embLayers, bertDim, rnscn.down, rnscnDim, gru.right, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.no, embPath,  embLayers, bertDim, rnscn.down, rnscnDim, gru.bidir, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.no, embPath,  embLayers, bertDim, rnscn.bidir, rnscnDim, gru.no, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.no, embPath,  embLayers, bertDim, rnscn.bidir, rnscnDim, gru.left, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.no, embPath,  embLayers, bertDim, rnscn.bidir, rnscnDim, gru.right, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.no, embPath,  embLayers, bertDim, rnscn.bidir, rnscnDim, gru.bidir, gruDim, miniBatch, epochs, logToFile = True )

#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.no, rnscnDim, gru.no, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.no, rnscnDim, gru.left, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.no, rnscnDim, gru.right, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.no, rnscnDim, gru.bidir, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.up, rnscnDim, gru.no, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.up, rnscnDim, gru.left, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.up, rnscnDim, gru.right, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.up, rnscnDim, gru.bidir, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.down, rnscnDim, gru.no, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.down, rnscnDim, gru.left, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.down, rnscnDim, gru.right, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.down, rnscnDim, gru.bidir, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.bidir, rnscnDim, gru.no, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.bidir, rnscnDim, gru.left, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.bidir, rnscnDim, gru.right, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.bidir, rnscnDim, gru.bidir, gruDim, miniBatch, epochs, logToFile = True )

#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.no, rnscnDim, gru.no, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.no, rnscnDim, gru.left, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.no, rnscnDim, gru.right, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.no, rnscnDim, gru.bidir, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.up, rnscnDim, gru.no, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.up, rnscnDim, gru.left, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.up, rnscnDim, gru.right, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.up, rnscnDim, gru.bidir, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.down, rnscnDim, gru.no, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.down, rnscnDim, gru.left, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.down, rnscnDim, gru.right, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.down, rnscnDim, gru.bidir, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.bidir, rnscnDim, gru.no, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.bidir, rnscnDim, gru.left, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.bidir, rnscnDim, gru.right, gruDim, miniBatch, epochs, logToFile = True )
#Run( dataPath, emb.w2vec, embPath,  embLayers, bertDim, rnscn.bidir, rnscnDim, gru.bidir, gruDim, miniBatch, epochs, logToFile = True )


dataPath = './Database/'
embPath = './Embedders/';  embLayers = 1; bertDim = 768; w2vecDim = 300
rnscnDim = 500; gruDim = 1000  # -------------------------------------------- rnscnDim 768 -> 500
miniBatch = 100; epochs = 2

#Run( dataPath, emb.bert, embPath,  embLayers, bertDim, rnscn.up, rnscnDim, gru.no, gruDim, miniBatch, epochs, logToFile = True )

