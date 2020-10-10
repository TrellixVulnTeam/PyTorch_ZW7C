# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 18:04:06 2020

@author: bhatt
"""

import torch
import numpy as np
import csv
import torch.optim as optim
import torch.nn as nn
from collections import OrderedDict
import matplotlib.pyplot as plt
import sys

wine_path = "./data/p1ch4/tabular-wine/winequality-white.csv"
wineq_numpy = np.loadtxt(wine_path, dtype = np.float32, delimiter=";", skiprows = 1)

col_list = next(csv.reader(open(wine_path), delimiter=";"))

wine_samples = torch.from_numpy(wineq_numpy)
data = wine_samples[:, :-1] # this is our input sample data
target = wine_samples[:, -1].float().unsqueeze(1)
#target_onehot = torch.zeros(target.shape[0], 10)
#target_onehot.scatter_(1, target, 1.0) # this is our one-shot output data

num_samples = wine_samples.shape[0]
num_validation_samples = int(0.2 * num_samples)

shuffled_indices = torch.randperm(num_samples)
train_indices = shuffled_indices[:-num_validation_samples]
validation_indices = shuffled_indices[-num_validation_samples:]

train_samples = data[train_indices]
train_results = target[train_indices]
val_samples = data[validation_indices]
val_results = target[validation_indices]

model = nn.Sequential(OrderedDict([
    ('l1', nn.Linear(11, 6)),
    ('a1', nn.Tanh()),
    ('l2', nn.Linear(6, 3)),
    ('a2', nn.Tanh()),
    ('output', nn.Linear(3, 1))
    ]))

learning_rate = 1e-2
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


def training_loop(n_epochs, optimizer, model, loss_fn, t_u_train, t_u_val,
    t_c_train, t_c_val):
    for epoch in range(1, n_epochs + 1):
        t_p_train = model(t_u_train)
        loss_train = loss_fn(t_p_train, t_c_train)
        t_p_val = model(t_u_val)
        loss_val = loss_fn(t_p_val, t_c_val)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        if epoch == 1 or epoch % 200 == 0:
            print(f"Epoch {epoch}, Training loss {loss_train.item():.4f},"
                  + f" Validation loss {loss_val.item():.4f}")
            
training_loop(
    n_epochs = 5000,
    optimizer = optimizer,
    model = model,
    loss_fn = nn.MSELoss(),
    t_u_train = train_samples,
    t_u_val = val_samples,
    t_c_train = train_results,
    t_c_val = val_results)


print(col_list)
plt.plot(data[:,0].numpy(), target[:, 0].numpy(), 'b.')
plt.plot(data[:,0].numpy(), model(data).detach().numpy(), 'r.')

