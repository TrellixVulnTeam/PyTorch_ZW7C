# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 18:01:33 2020

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

n_out = 2
model = nn.Sequential(
    nn.Linear(3072, 1024),
    nn.Tanh(),
    nn.Linear(1024, 512),
    nn.Tanh(),
    nn.Linear(512, 128),
    nn.Tanh(),
    nn.Linear(128, 2))

loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-2
optimizer = optim.SGD(model.parameters(), lr = learning_rate)
n_epochs = 100

train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64,
                                           shuffle=True)


for epoch in range(n_epochs):
    for imgs, labels in train_loader:
        batch_size = imgs.shape[0]
        outputs = model(imgs.view(batch_size, -1))
        loss = loss_fn(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch}, Loss {loss}")
    
    
val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64,
shuffle=False)
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in val_loader:
        batch_size = imgs.shape[0]
        outputs = model(imgs.view(batch_size, -1))
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
    print(f"Accuracy: {correct/total*100}%")
    
    
# our neural network has 70-80% accuracy
# we lose a lot of spatial information by flattening the tensor
# this causes our resulting tensor to fail to be translation-invariant
# in consequence, the model has to manually learn more than it needs to
# we can use convolution layers to learn this spatial information
# thus saving training time and allowing our network to generalize