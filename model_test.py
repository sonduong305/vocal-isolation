from model import TestModel, VocalModel
import numpy as np 
import torch
model = VocalModel()

a = torch.rand((1, 513, 128))
print(model.forward(a).shape)