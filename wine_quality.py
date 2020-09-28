# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 13:27:32 2020

@author: bhatt
"""

import torch
import numpy as np
import csv

wine_path = "./data/p1ch4/tabular-wine/winequality-white.csv"
wineq_numpy = np.loadtxt(wine_path, dtype = np.float32, delimiter=";", skiprows = 1)

col_list = next(csv.reader(open(wine_path), delimiter=";"))

wineq = torch.from_numpy(wineq_numpy)

data = wineq[:, :-1]
target = wineq[:, -1].long()

target_onehot = torch.zeros(target.shape[0], 10)
target_onehot.scatter_(1, target.unsqueeze(1), 1.0)

data_mean = torch.mean(data, dim = 0)
data_var = torch.var(data, dim = 0)

data_normalized = (data - data_mean) / torch.sqrt(data_var)

bad_data = data[target <= 3]
mid_data = data[(target > 3) & (target < 7)]
good_data = data[target >= 7]

bad_mean = torch.mean(bad_data, dim = 0)
mid_mean = torch.mean(mid_data, dim = 0)
good_mean = torch.mean(good_data, dim = 0)

for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
    print(f"{i}: {args}")
    
total_sulfur_threshold = 141.83
total_sulfur_data = data[:,6]
predicted_indexes = torch.lt(total_sulfur_data, total_sulfur_threshold)

actual_indexes = target > 5

print(predicted_indexes.sum(), actual_indexes.sum())

n_matches = torch.sum(actual_indexes & predicted_indexes).item()
n_predicted = torch.sum(predicted_indexes).item()
n_actual = torch.sum(actual_indexes).item()

print(n_matches, n_matches / n_predicted, n_matches / n_actual)