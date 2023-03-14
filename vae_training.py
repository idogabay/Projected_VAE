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


'''
G_kwargs = {'class_name': 'pg_modules.networks_....Generator', 'cond': False, 'synthesis_kwargs': {'lite': False}}
G_opt_kwargs = {'class_name': 'torch.optim.Adam', 'betas': [0, 0.99], 'eps': 1e-08, 'lr': 0.0002}
'''
epochs =100
x_shape = (3,256,256)
lr = 1e-3
batch_size = 20
pics_path ="/home/user_118/datasets/Flowers/resized_images"
weights_save_path = "/home/user_118/datasets/Flowers/weights"
dataset_name = "flowers"

transform = torchvision.transforms.Compose([
    torchvision.transforms.ColorJitter(brightness = 0.2,contrast = 0.2,saturation = 0.2,hue = 0.1),
    torchvision.transforms.RandomHorizontalFlip(p=0.3),
    torchvision.transforms.RandomVerticalFlip(p=0.3),
    #torchvision.transforms.RandomRotation(degrees = 15,interpolation = torchvision.transforms.InterpolationMode.BILINEAR),
    torchvision.transforms.ToTensor()
    ])

dataset = our_datasets.Flowers_dataset(pics_path,transform)   #PokemonDataset(root=path, rgb=True)#'/content/drive/MyDrive/pokemon/pokemon', rgb=True)
#poke_data = PokemonDataset(root='/content/drive/MyDrive/pokemon/pokemon', rgb=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
#num_of_classes = 18
loss_type = "mse"
optimizer_type = "Adam"
z_dim =200
beta = 0.01
device = set_device()
model = Vae_cnn_1(z_dim=z_dim,x_shape=x_shape,device=device).to(device)

all_total_losses = 0
kl_losses,recon_losses,total_losses, weights_path = training_loop(model,device,epochs,x_shape,z_dim,lr,beta,dataloader,loss_type,optimizer_type,weights_save_path,dataset_name)
#title = "loss for beta= "+str(beta)+"z_dim= "+str(z_dim)
title = "kl"
plot_loss(kl_losses,title)
title = "recon"
plot_loss(recon_losses,title)
title = "total"
plot_loss(total_losses,title)
#weights_path = '/home/user_118/datasets/Flowers/weights/flowersimage_size256_beta_0.01_epochs_500_z_dim_200_loss_type_mse_optimizer_type_Adam.pth'
generate_samples(num_of_samples=17,model=model,weights_path=weights_path)
