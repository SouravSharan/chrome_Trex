from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


#main_dir = "/home/harsha/Desktop/data"

#data = ImageFolder(root=main_dir,transform=ToTensor())


data_transform = transforms.Compose([
        #transforms.Scale((64,64)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             #std=[0.229, 0.224, 0.225])
    ])
datafolder = datasets.ImageFolder(root='/home/harsha/Desktop/data',
                                           transform=data_transform)
dataset_loader = DataLoader(datafolder,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)

use_gpu = torch.cuda.is_available()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size = 5, padding=2),nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=5, padding=2), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        #self.layer3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, padding=2), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))

        self.fc = nn.Linear(85*85*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


cnn = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters())

#Train the Model
for epoch in range(10):
    for i, (images, labels) in enumerate(dataset_loader):
        #images = images.resize(64,6)
        images = Variable(images)
        #print(images)
        labels = Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()

        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, 10, i + 1, len(dataset_loader) // 4, loss.data[0]))

# Test the Model

cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in dataset_loader:
    images = Variable(images)
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Trained Model
torch.save(cnn.state_dict(), './cnn.pkl')