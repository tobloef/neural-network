import mnist_loader
from network import Network

trainingData, validationData, testData = map(list, mnist_loader.load_data_wrapper())

net = Network.generateRandomNetwork([784, 30, 10])
net.train(trainingData, 20, 10, 3.0, testData=testData)