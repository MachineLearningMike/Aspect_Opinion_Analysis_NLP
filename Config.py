import tensorflow as tf
from enum import Enum

configVersion = 0.83
configWeightDType = tf.float32

class CacheTag(Enum):
    Cache = 1
    Real = 2

class rnscn(Enum):
    no = 1
    up = 2
    down = 3
    bidir = 4
    sup3 = 5

class gru(Enum):
    no = 1
    right = 2
    left = 3
    bidir = 4

class emb(Enum):
    no = 1
    bert = 2
    w2vec = 3

