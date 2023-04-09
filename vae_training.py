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
from projector import F_RandomProj
from datetime import datetime
import json

def main():
    torch.set_float32_matmul_precision('high')
    epochs = 10
    x_shape = (3,256,256)
    lr = 0.0008 #for adam lr of 0.0005 is the optimal
    batch_size = 35
    architecture = "pvae"
    augmentations = ["horizontalflip"]
    pics_path ="/home/ido/datasets/projected_vae/pokemon/resized_images"
    weights_save_path = "/home/ido/datasets/projected_vae/pokemon/weights"
    dataset_name = "pokemon"
    now = datetime.now()
    transform = torchvision.transforms.Compose([
        #torchvision.transforms.Resize((256,256)),
        #torchvision.transforms.ColorJitter(brightness = 0.2,contrast = 0.2,saturation = 0.2,hue = 0.1),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        #torchvision.transforms.RandomVerticalFlip(p=0.3),
        #torchvision.transforms.RandomRotation(degrees = 15,interpolation = torchvision.transforms.InterpolationMode.BILINEAR),
        torchvision.transforms.ToTensor()
        ])
    dataset = our_datasets.Pokemon_dataset(pics_path,transform)   #PokemonDataset(root=path, rgb=True)#'/content/drive/MyDrive/pokemon/pokemon', rgb=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    loss_type = "mse"
    optimizer_type = "Adam"
    z_dim =256
    beta = 0.05 #for adam 0.01.
    proj_type = 2
    projected = True
    device = set_device()
    #model = Vae_cnn_1(z_dim=z_dim,x_shape=x_shape,device=device).to(device)

    ## projected ##
    outs_shape = {"0":[40, 64, 128, 128],
                  "1":[40, 64, 64, 64],
                  "2":[40, 128, 32, 32],
                  "3":[40, 256, 16, 16]
                  }
    ###############
    
    ######################################
    #hyper parameters for loop
    betas = [0.02,0.05,0.1,0.2,0.3,0.4]
    optimizers = ["Adam","SGD"]
    losses = ["mse","bce","l1"]
    datasets = ["pokemon","flowers,landscapes"]
    output_base_path = "./output_images"
    lrs = [ 0.005,0.001,0.0005,0.0001]
    proj_types = [0,1,2]
    architecture = "pvae"
    augmentations = ["horizontalflip"]
    #####################################
    model = ProjectedVAE(z_dim=z_dim,outs_shape=outs_shape,device=device,projected=projected,proj_type=proj_type)#.to(device)
    model = torch.compile(model.to(device)).to(device)
    weights_path = ""
    sub_folder = now.strftime("%d/%m/%Y %H:%M:%S") #weights_path.split("/")[-1][:-4]
    
    for beta in betas:
        for optimizer_type in optimizers:
            for loss_type in losses:
                for lr in lrs:
                    for dataset_name in datasets:
                        for proj_type in proj_types:
                                        #json build
                                        json_data = {}
                                        json_data['projected'] = projected
                                        json_data['lr'] = lr
                                        json_data['beta'] = beta
                                        json_data['optimazer'] = optimizer_type
                                        json_data['loss_type'] = loss_type
                                        json_data['proj_types'] = proj_types
                                        json_data['augmentations'] = augmentations
                                        json_data['dataset_name'] = dataset_name
                                        json_data['architecture'] = architecture
                                        if dataset_name == "pokemon":
                                            transform = torchvision.transforms.Compose([
                                                    torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                                    #torchvision.transforms.RandomRotation(degrees = 10,interpolation = torchvision.transforms.InterpolationMode.BILINEAR),
                                                    torchvision.transforms.ToTensor()
                                            ])
                                            pics_path ="/home/ido/datasets/projected_vae/pokemon/resized_images"

                                        elif dataset_name == "flowers":
                                            transform = torchvision.transforms.Compose([
                                                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                                torchvision.transforms.RandomVerticalFlip(p=0.5),
                                                #torchvision.transforms.RandomRotation(degrees = 90,interpolation = torchvision.transforms.InterpolationMode.BILINEAR),
                                                torchvision.transforms.ToTensor()
                                            ])
                                            pics_path ="/home/ido/datasets/projected_vae/flowers/resized_images"
                                            
                                        elif dataset_name == "landscapes":
                                            transform = torchvision.transforms.Compose([
                                                    torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                                    #torchvision.transforms.RandomRotation(degrees = 10,interpolation = torchvision.transforms.InterpolationMode.BILINEAR),
                                                    torchvision.transforms.ToTensor()
                                            ])
                                            pics_path ="/home/ido/datasets/projected_vae/landscapes/resized_images"
                                        dataset = our_datasets.Pokemon_dataset(pics_path,transform)
                                        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
                                        # restart model
                                        model = model.to("cpu")
                                        del model
                                        model = ProjectedVAE(z_dim=z_dim,outs_shape=outs_shape,device=device,projected=projected,proj_type=proj_type)#.to(device)
                                        model = torch.compile(model.to(device)).to(device)
                                        
                                        # training loop
                                        kl_losses,recon_losses,total_losses, weights_path,end_epoch,lr_history = training_loop(
                                            model,device,epochs,x_shape,z_dim,lr,beta,dataloader,
                                            loss_type,optimizer_type,weights_save_path,dataset_name,str(sub_folder),json_data
                                        )

                                        # create and save samples
                                        output_path = os.path.join(output_base_path,str(sub_folder))
                                        if not os.path.exists(output_path):
                                            os.makedirs(output_path)
                                        generate_samples(num_of_samples=50,model=model,weights_path=weights_path,output_path=output_path)

                                        #json build
                                        json_data['kl_losses'] = kl_losses
                                        json_data['recon_losses'] = recon_losses
                                        json_data['total_losses'] = total_losses
                                        json_data['weights_path'] = weights_path
                                        json_data['end_epoch'] = end_epoch
                                        json_data['lr_history'] = lr_history
                                        json_data['output_path'] =  output_path
                                        json_string = json.dumps(json_data,indent=4)
                                        json_name = str(sub_folder)+".json"
                                        json_path = os.path.join(output_path,json_name)
                                        
                                        # json save
                                        with open(json_path, "w") as outfile:
                                            outfile.write(json_string)
                                        # json_data[''] = 
                                        # json_data[''] = 
                                        # json_data[''] = 
                                        # json_data[''] = 
                                        # json_data[''] = 




    #weights_path = '/home/ido/datasets/projected_vae/pokemon/weights/pokemon_image_size_256_beta_0.01_epochs_500_z_dim_256_loss_type_mse_optimizer_type_Adam.pth'

    #title = "loss for beta= "+str(beta)+"z_dim= "+str(z_dim)
    # title = "kl"
    # plot_loss(kl_losses,title)
    # title = "recon"
    # plot_loss(recon_losses,title)
    # title = "total"
    # plot_loss(total_losses,title)





if __name__ == "__main__":
    main()