# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 00:27:09 2020

@author: bhatt
"""

import sys
sys.path.append("./vision")
from torchvision import datasets
from torchvision import transforms
sys.path.remove("./vision")

import torch
import matplotlib.pyplot as plt

data_path = "./data-unversioned/p1ch7/"
cifar10 = datasets.CIFAR10(data_path, train=True, download=False)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=False)


img, label = cifar10[99]
#plt.imshow(img)
#plt.show()

#print(dir(transforms))

to_tensor = transforms.ToTensor()
img_t = to_tensor(img)
#print(img_t.shape, img_t.dtype)
#print(img_t.min(), img_t.max())
# notice that the datatype is float, not an integer
# it takes on values from 0 to 1
#plt.imshow(img_t.permute(1, 2, 0))
#plt.show()

tensor_cifar10 = datasets.CIFAR10(data_path, train=True, download=False,
                                  transform = transforms.ToTensor())

imgs = torch.stack([img_t for img_t, _ in tensor_cifar10], dim = 3)
mean = imgs.view(3, -1).mean(dim=1)
std = imgs.view(3, -1).std(dim=1)
print(mean, std)

transformed_cifar10 = datasets.CIFAR10(
    data_path, train=True, download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]))

img_t, _ = transformed_cifar10[99]
#plt.imshow(img_t.permute(1, 2, 0))
#plt.show()
# note that, since we have transformed the image, it does not look the same anymore