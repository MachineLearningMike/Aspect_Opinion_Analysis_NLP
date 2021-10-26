import tensorflow as tf

class Trainee() :
    def __init__(self, inputShape, outputShape, paramDict = None) :
        self.inputShape = inputShape
        self.outputShape = outputShape
        self.nWeightTensors = 0

    def __GetWeightsTensorList(self):
        list = []

        return list

    def __SetWeightsTensorList(self, list):
        pass

    weights = property(__GetWeightsTensorList, __SetWeightsTensorList)

    def NWeights(self):
        sum = 0
        for tensor in self.weights:
            n = 1; k = 0
            for _ in range(len(tensor.shape)):
                n = n * tensor.shape[k]
            sum += n

        return sum

    def ProduceOutput(self, input) :
        assert input.shape == self.inputShape
        output = tf.constant()
        assert output.shape == self.outputShape
