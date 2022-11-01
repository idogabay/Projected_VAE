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
import our_datasets



epochs = 5
x_shape = (3,256,256)
lr = 1e-4
batch_size = 2
pics_path ="/home/user_118/datasets/Flowers/resized_images"
weights_save_path = "/home/user_118/datasets/Flowers/weights"
dataset_name = "flowers"
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
dataset = our_datasets.Flowers_dataset(pics_path,transform)   #PokemonDataset(root=path, rgb=True)#'/content/drive/MyDrive/pokemon/pokemon', rgb=True)
#poke_data = PokemonDataset(root='/content/drive/MyDrive/pokemon/pokemon', rgb=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
#num_of_classes = 18
loss_type = "bce"
optimizer_type = "Adam"
z_dim = 1000
beta = 3
device = set_device()
model = Vae_cnn_1(z_dim=z_dim,x_shape=x_shape,device=device).to(device)

all_total_losses = 0
total_losses, weights_path = training_loop(model,device,epochs,x_shape,z_dim,lr,beta,dataloader,loss_type,optimizer_type,weights_save_path,dataset_name)
title = "loss for beta= "+str(beta)+"z_dim= "+str(z_dim)
plot_loss(total_losses,title)
generate_samples(num_of_samples=5,model=model,weights_path=weights_path)
