
from network import *

torch.set_printoptions(linewidth=120)

print("-- CNN --")

train_set = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor()
    ])
)
"""train_set = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor()
    ])
)"""

testNetwork(trainSet=train_set, batchSize=100,
            numOfEpochs=25, learningRate=0.002, dinamicLearningRate=True)

""" RECORD 10 epochs: 91.47 %
testNetwork(trainSet=train_set, batchSize=100,
            numOfEpochs=10, learningRate=0.002, dinamicLearningRate=True) """
