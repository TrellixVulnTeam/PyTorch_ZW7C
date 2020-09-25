# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 22:04:26 2020

@author: bhatt
"""

import imageio
import torch

dir_path = "./data/p1ch4/volumetric-dicom/2-LUNG 3.0  B70f-04083"
vol_arr = imageio.volread(dir_path, "DICOM")
print(vol_arr.shape)
# notice that the image has no channel information
# we'll have to create one ourselves

vol = torch.from_numpy(vol_arr).float()
vol = torch.unsqueeze(vol, 0)
print(vol.shape)