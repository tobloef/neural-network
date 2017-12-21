import numpy as np
from mathUtils import *

class Network(object):
    """
    Model for a feedforward Neural Network that use backpropagation with stochastic gradient decent.
    """

    def __init__(self, layerSizes, biasVectors, weightMatrices):
        """
        Initialise the network with a list of layer sizes and lists for biases and weights for the neurons in the network. The first layer is the input layer and the last layer is the output layer.
        """
        
        self.layerSizes = layerSizes
        self.biasVectors = biasVectors
        self.weightMatrices = weightMatrices

    @staticmethod
    def generateRandomNetwork(layerSizes):
        """
        Initialise a new network with random weights and biases. Input and output layers are included in the layerSizes list. The random weights and biases are generated using a Gaussian distribution, so the results are more probable to be around 0.
        """
        
        biasVectors = []
        """Generate biases for each neuron in each layer, except the input layer."""
        for size in layerSizes[1:]:
            """
            np.random.randn generates arrays of arrays of random numbers, based on the paramters.
            np.random.randn(3,2) will generate an array of 3 arrays with 2 random numbers.
            """
            biasVectors.append(np.random.randn(size, 1))
        """Generate weights for connections between layers."""
        weightMatrices = []
        for size, prevSize in zip(layerSizes[:-1], layerSizes[1:]):
            weightMatrices.append(np.random.randn(prevSize, size))
        return Network(layerSizes, biasVectors, weightMatrices)

    def getOutputs(self, inputs):
        """Return a vector of the network's outputs based on the given inputs, using feedforward."""
        
        activations = inputs
        for biasVector, weightMatrix in zip(self.biasVectors, self.weightMatrices):
            """
            For every layer, get the bias vector and the weight matrix. Then get dot product between the weight matrix and the output vector and add the bias vector. This is the activation vector for the current layer.
            """
            zVector = np.dot(weightMatrix, activations) + biasVector
            activations = sigmoid(zVector)
        return activations

    def train(self, data, epochs, batchSize, rate, testData=None):
        """
        Train the neural network using stochastic gradient descent. Smaller batches of random samples from the training are used to reduce the training time. The training date is a list of tuples (inputs, expected outputs). The learning rate is how much to change the values each batch.
        """

        print("Training network with shape {}, batch size {} and learning rate {} for {} epochs...".format(self.layerSizes, batchSize, rate, epochs))
        for e in range(epochs):
            np.random.shuffle(data)
            batches = []
            for i in range(0, len(data), batchSize):
                batches.append(data[i:i+batchSize])
            for batch in batches:
                self._tuneNetwork(batch, rate)
            if (testData):
                result = self._evaluate(testData)
                print("Epoch #{} completed with {:.2f}% correctness.".format(e+1, 100/len(testData)*result))
            else:
                print("Epoch #{} completed.".format(e))

    def _tuneNetwork(self, batch, rate):
        """
        Tune the weights and biases of the network by using backpropagation with gradient descend.
        """
        
        """
        Setup matrix and vector based on the weight matrix and bias vector filled with zeroes. This is used for storing each change to make for each vector, for each set of training date.
        """
        sumBiasVectors = []
        for biasVector in self.biasVectors:
            sumBiasVectors.append(np.zeros(biasVector.shape))
        sumWeightMatrices = []
        for weightMatrix in self.weightMatrices:
            sumWeightMatrices.append(np.zeros(weightMatrix.shape))
        for inputs, expected in batch:
            """
            Get a matrix/vector with the required changes to the network, based on that set of training data, and add it to a set of matrix/vector totalling the changes needed from all the training data.
            """ 
            deltaBiasVectors, deltaWeightMatrices = self._backpropagate(inputs, expected)
            newSumBiasVectors = []
            for totalBiasVector, deltaBiasVector in zip(sumBiasVectors, deltaBiasVectors):
                newSumBiasVectors.append(totalBiasVector + deltaBiasVector)
            sumBiasVectors = newSumBiasVectors
            newSumWeightMatrices = []
            for totalWeightMatrix, deltaWeightMatrix in zip(sumWeightMatrices, deltaWeightMatrices):
                newSumWeightMatrices.append(totalWeightMatrix + deltaWeightMatrix)
            sumWeightMatrices = newSumWeightMatrices
        """
        Take each change for each set of training data, get the average of these and subtract them from the current weights and biases. Then use these as the new weights and biases.
        """
        newBiasVectors = []
        for biasVector, totalBiasVector in zip(self.biasVectors, sumBiasVectors):
            newBiasVectors.append(biasVector - (rate/len(batch)) * totalBiasVector)
        newWeightMatrices = []
        for weightMatrix, totalWeightMatrix in zip(self.weightMatrices, sumWeightMatrices):
            newWeightMatrices.append(weightMatrix - (rate/len(batch)) * totalWeightMatrix)
        self.biasVectors = newBiasVectors
        self.weightMatrices = newWeightMatrices

    def _backpropagate(self, inputs, expected):
        """
        Return a tuple with gradient of the cost function for each bias and weight, in the format (vector of bias changes, matrix of weight changes), for the specified set of training data.
        """

        deltaBiasVectors = []
        for biasVector in self.biasVectors:
            deltaBiasVectors.append(np.zeros(biasVector.shape))
        deltaWeightMatrices = []
        for weightMatrix in self.weightMatrices:
            deltaWeightMatrices.append(np.zeros(weightMatrix.shape))
        """Store all activations for the entire network, starting with the input layer."""
        activationVector = inputs
        activationVectors = [inputs]
        """Find the z-vector for layer in the network"""
        zVectors = []
        for biasVector, weightMatrix in zip(self.biasVectors, self.weightMatrices):
            zVector = np.dot(weightMatrix, activationVector) + biasVector
            zVectors.append(zVector)
            activationVector = sigmoid(zVector)
            activationVectors.append(activationVector)
        """
        * Start with output compared to expected, tune weights and biases based on the derivative of the cost function with respect to the weight/bias.
        * Then move onto each hidden layer and the input layer.
        """
        deltaBiasVector = (activationVectors[-1] - expected) * 2 * sigmoidDerivative(zVectors[-1])
        deltaBiasVectors[-1] = deltaBiasVector
        deltaWeightMatrices[-1] = np.dot(deltaBiasVector, activationVectors[-2].transpose())

        for l in range(-2, -len(self.layerSizes), -1):
            # Equivalent to https://i.imgur.com/8PQQ28r.png, because deltaBiasVector is * 1 instead
            weightMatrix = self.weightMatrices[l+1].transpose()
            sigmoidDeriv = sigmoidDerivative(zVectors[l])
            deltaBiasVector = np.dot(weightMatrix, deltaBiasVector) * sigmoidDeriv
            deltaBiasVectors[l] = deltaBiasVector
            deltaWeightMatrices[l] = np.dot(deltaBiasVector, activationVectors[l-1].transpose())
        return (deltaBiasVectors, deltaWeightMatrices)

    def _evaluate(self, testData):
        """Test the network with the specified test data and return the number of correct guesses."""
        correctGuesses = 0
        for inputs, expected in testData:
            """Increment correct guesses if the most active output is the expected one."""
            outputs = self.getOutputs(inputs)
            guess = np.argmax(outputs)
            if (guess == expected):
                correctGuesses += 1
        return correctGuesses