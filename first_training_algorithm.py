# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 14:39:54 2020

@author: bhatt
"""

import torch
import matplotlib.pyplot as plt

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)

#plt.plot(t_u, t_c, color="blue", marker='.', linestyle = "None")
#plt.show()

def model(t_u: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return w * t_u + b

def dmodel_dw(t_u: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return t_u

def dmodel_db(t_u: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> float:
    return 1.0

def loss_fn(t_p: torch.Tensor, t_c: torch.Tensor) -> torch.Tensor:
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()

def dloss_fn(t_p: torch.Tensor, t_c: torch.Tensor) -> torch.Tensor:
    dsq_diffs = 2 * (t_p - t_c) / t_p.size(0)
    return dsq_diffs

def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw = dloss_dtp * dmodel_dw(t_u, w, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w, b)
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])

w = torch.ones(())
b = torch.zeros(())

t_p = model(t_u, w, b)
loss = loss_fn(t_p, t_c)

#print(t_p, loss)

delta = 0.1
loss_rate_of_change_w = (loss_fn(model(t_u, w + delta, b), t_c)
                         - loss_fn(model(t_u, w - delta, b), t_c)) / (2.0 * delta)
loss_rate_of_change_b = (loss_fn(model(t_u, w, b + delta), t_c)
                         - loss_fn(model(t_u, w, b - delta), t_c)) / (2.0 * delta)

learning_rate = 10**-2

w = w - learning_rate * loss_rate_of_change_w
b = b - learning_rate * loss_rate_of_change_b

def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        w, b = params
        t_p = model(t_u, w, b)
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w, b)
        
        params = params - learning_rate * grad
        
        print(f"Epoch {epoch}:\nLoss = {float(loss)}\nGrad: {grad}")
        
    return params
"""   
results = training_loop(
    n_epochs = 100,
    learning_rate = 1e-4, # note that if this is not small enough, loss diverges
    params = torch.tensor([1.0, 0.0]),
    t_u = t_u,
    t_c = t_c)

print(results)
# there's an issue here - the gradient for w and b may have different learning rates
# one solution is to define two learning rates for them
# however, this becomes too cumbersome as the parameters increases
# a more convenient solution is to normalize w and b wrt each other
"""

# we normalize t_u by dividing by 10 for now
t_un = 0.1 * t_u

params = training_loop(
    n_epochs = 5000,
    learning_rate = 1e-2, # note that if this is not small enough, loss diverges
    params = torch.tensor([1.0, 0.0]),
    t_u = t_un, # this is the only change relative to the commented out section
    t_c = t_c)

print(params)

t_p = model(t_un, *params)
plt.xlabel("Temperature (F)")
plt.ylabel("Temperature (C)")
plt.plot(t_u.numpy(), t_p.detach().numpy())
plt.plot(t_u.numpy(), t_c.numpy(), 'o')