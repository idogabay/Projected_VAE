import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision
import os
from vae_utils import *
from pokemon_dataset import PokemonDataset




epochs = 300
x_shape = (3,60,60)
lr = 1e-4
batch_size = 128
path = os.path.join("..","data","pokemon")
poke_data = PokemonDataset(root=path, rgb=True)#'/content/drive/MyDrive/pokemon/pokemon', rgb=True)
#poke_data = PokemonDataset(root='/content/drive/MyDrive/pokemon/pokemon', rgb=True)
dataloader = DataLoader(poke_data, batch_size=128, shuffle=True, drop_last=True)
num_of_classes = 18
loss_type = "bce"
optimizer_type = "Adam"
z_dim = 1000
beta = 3

all_total_losses = 0
total_losses = training_loop(epochs,x_shape,z_dim,lr,beta,dataloader,num_of_classes,loss_type,optimizer_type)
title = "loss for beta= "+str(beta)+"z_dim= "+str(z_dim)
plot_loss(total_losses,title)
