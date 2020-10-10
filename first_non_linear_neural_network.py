# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 20:31:44 2020

@author: bhatt
"""

import torch
import torch.optim as optim
import torch.nn as nn
from collections import OrderedDict
import matplotlib.pyplot as plt


t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c).unsqueeze(1)
t_u = torch.tensor(t_u).unsqueeze(1)
t_un = 0.1 * t_u

n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)

shuffled_indices = torch.randperm(n_samples)
train_indices = shuffled_indices[: -n_val]
val_indices = shuffled_indices[-n_val:]

train_tu = t_u[train_indices]
train_tc = t_c[train_indices]
val_tu = t_u[val_indices]
val_tc = t_c[val_indices]
train_tun = 0.1 * train_tu
val_tun = 0.1 * val_tu



seq_model = nn.Sequential(OrderedDict([
    ('hidden_linear', nn.Linear(1, 8)),
    ('hidden_activation', nn.Tanh()),
    ('output_linear', nn.Linear(8, 1))
    ]))
#print(seq_model)
#for name, param in seq_model.named_parameters():
#    print(name, param.shape)
#print([param.shape for param in seq_model.parameters()])


learning_rate = 1e-2
optimizer = optim.SGD(seq_model.parameters(), lr=learning_rate)

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
        if epoch == 1 or epoch % 1000 == 0:
            print(f"Epoch {epoch}, Training loss {loss_train.item():.4f},"
                  + f" Validation loss {loss_val.item():.4f}")
            
training_loop(
    n_epochs = 3000,
    optimizer = optimizer,
    model = seq_model,
    loss_fn = nn.MSELoss(),
    t_u_train = train_tun,
    t_u_val = val_tun,
    t_c_train = train_tc,
    t_c_val = val_tc)
print()

print('output', seq_model(val_tun))
print('answer', val_tc)
print('hidden', seq_model.hidden_linear.weight.grad)


t_range = torch.arange(20., 90.).unsqueeze(1)
fig = plt.figure()

plt.xlabel("Fahrenheit")
plt.ylabel("Celsius")
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.plot(t_range.numpy(), seq_model(0.1 * t_range).detach().numpy(), 'c-')
plt.plot(t_u.numpy(), seq_model(0.1 * t_u).detach().numpy(), 'kx')