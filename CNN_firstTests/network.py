import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from datetime import datetime


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        # (4) hidden linear layer
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)
        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)
        # (6) output layer
        t = self.out(t)
        #t = F.softmax(t, dim=1)

        return t


def testNetwork(trainSet, batchSize=1000, numOfEpochs=10, learningRate=0.01, printEpochData=True,
                dinamicLearningRate=False, dinamicBatchSize=False):
    print("-- TESTING CNN --")
    net = Network()
    print("[PARAMETERS]:")
    if(dinamicBatchSize):
        print("   Batch size:              Dinamic")
    else:
        print("   Batch size:             ", batchSize)
    print("   Number of epochs:       ", numOfEpochs,
          "\n   Learning rate:          ", learningRate,
          "\n   Dinamic learning rate:  ", dinamicLearningRate,
          "\n", net, "\n")

    starting_time = datetime.now()
    total_correct = 0
    for i in range(1, numOfEpochs+1):
        if dinamicLearningRate:
            _learningRate = learningRate*1.5 - \
                learningRate * ((i-1)/numOfEpochs)/2
        else:
            _learningRate = learningRate
        optimizer = optim.Adam(net.parameters(), lr=_learningRate)
        total_correct = 0
        _batchSize = 0
        if(dinamicBatchSize):
            progress = (i)/numOfEpochs
            if(i == 1):
                _batchSize = 100
            elif(progress <= 0.5):
                _batchSize = 1000
            elif(progress <= 0.8):
                _batchSize = 500
            else:
                _batchSize = 100
        else:
            _batchSize = batchSize
        train_loader = DataLoader(trainSet, batch_size=_batchSize)
        starting_epoch_time = datetime.now()
        for images, labels in train_loader:
            preds = net(images)  # Pass Batch
            loss = F.cross_entropy(preds, labels)  # Calculate Loss
            optimizer.zero_grad()
            loss.backward()  # Calculate Gradients
            optimizer.step()  # Update Weights
            total_correct += preds.argmax(dim=1).eq(labels).sum().item()
        if(printEpochData):
            print("[epoch", i, "] ",
                  "total corrects=", total_correct, "/", len(trainSet),
                  "accuracy=", round(total_correct*100/len(trainSet), 2), "%",
                  "time=", datetime.now()-starting_epoch_time,
                  "learning rate=", round(_learningRate, 4),
                  "batch size=", _batchSize)
    print("total execution time=", datetime.now()-starting_time,
          "final accuracy=", round(total_correct*100/len(trainSet), 2), "\n")


if __name__ == "__main__":
    print("-- network tests --")
    train_set = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )
    net = Network()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    train_loader = DataLoader(train_set, batch_size=50)
    total_correct = 0
    for images, labels in train_loader:
        preds = net(images)  # Pass Batch
        loss = F.cross_entropy(preds, labels)  # Calculate Loss
        optimizer.zero_grad()
        loss.backward()  # Calculate Gradients
        optimizer.step()  # Update Weights
        total_correct += preds.argmax(dim=1).eq(labels).sum().item()
    print("[epoch0]",
          "total corrects=", total_correct,
          "accuracy=", round(total_correct/len(train_set), 4))
