# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 14:53:39 2020

@author: bhatt
"""

import sys
sys.path.append("./vision")
from torchvision import datasets
from torchvision import transforms
sys.path.remove("./vision")

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import datetime


# this is (almost) the same as the model commented below it
# this example demonstrates that we can create our own custom networks
# 
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * 8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out

"""
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size = 3, padding = 1),
    nn.Tanh(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 8, kernel_size = 3, padding = 1),
    nn.Tanh(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(8 * 8 * 8, 32),
    nn.Tanh(),
    nn.Linear(32, 2))
"""

data_path = "./data-unversioned/p1ch7/"

# these values are obtained from loading_cifar_data.py
mean = torch.Tensor([0.4915, 0.4823, 0.4468])
std = torch.Tensor([0.2470, 0.2435, 0.2616])

cifar10 = datasets.CIFAR10(
    data_path, train=True, download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]))

cifar10_val = datasets.CIFAR10(
    data_path, train=False, download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]))

label_map = {0: 0, 2: 1}
class_names = ["airplane", "bird"]
cifar2 = [(img, label_map[label]) for img, label in cifar10 if label in [0, 2]]
cifar2_val = [(img, label_map[label]) for img, label in cifar10_val if label in [0, 2]]



conv = nn.Conv2d(3, 1, kernel_size = 3, padding = 1)
#print(conv, conv.weight.shape, conv.bias.shape)

# convolution layers can't act on boundaries on their own
# in consequence, the output of the layer is a smaller image
# padding = 1 allows us to do the boundaries
# it adds imaginary entries with default values 0

img, _ = cifar2[0]
output = conv(img.unsqueeze(0))
#print(img.unsqueeze(0).shape, output.shape)


# testing convolution on an image
with torch.no_grad():
    conv.bias.zero_()
    conv.weight.fill_(1.0 / 9.0)
    
output = conv(img.unsqueeze(0))
#plt.imshow(output[0, 0].detach(), cmap='gray')
#plt.show()
    

# testing another convolution, this is an edge detection kernel vertically
conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
with torch.no_grad():
    conv.weight[:] = torch.tensor([[-1.0, 0.0, 1.0],
                                   [-1.0, 0.0, 1.0],
                                   [-1.0, 0.0, 1.0]])
    conv.bias.zero_()
    
    
pool = nn.MaxPool2d(2)
output = pool(img.unsqueeze(0))
#print(img.unsqueeze(0).shape, output.shape)


model = Net()
numel_list = [p.numel() for p in model.parameters()]
#print(sum(numel_list), numel_list)


def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            
            if(epoch == 1 or epoch % 10 == 0):
               print(f"Epoch {epoch}, Loss {loss / len(train_loader)}, Time {datetime.datetime.now()}")



train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)
optimizer = optim.SGD(model.parameters(), lr = 1e-2)
loss_fn = nn.CrossEntropyLoss()

training_loop(
    n_epochs = 100,
    optimizer = optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader = train_loader)
