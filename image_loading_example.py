# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 21:42:56 2020

@author: bhatt
"""

import imageio
import torch
import os


batch_size = 3
batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)

data_dir = "./data/p1ch4/image-cats/"
filenames = [name for name in os.listdir(data_dir) if os.path.splitext(name)[-1] == ".png"]

for i, filename in enumerate(filenames):
    filepath = os.path.join(data_dir, filename)
    img_arr = imageio.imread(filepath) # img_arr is in the shape [256, 256, 3], or width, height, channel
    img_t = torch.from_numpy(img_arr)
    img_t = img_t.permute(2, 0, 1) # pytorch uses the convention channel, width, height for image data
    img_t = img_t[:3] # some images have more than 3 channels, we take the first 3 only
    batch[i] = img_t


batch = batch.float() # most tensor operations are performed on floats
batch /= 255 # normalizes the floats, RGB values max out at 255

print(batch[0])