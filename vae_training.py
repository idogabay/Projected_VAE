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
    epochs = 500
    x_shape = (3,256,256)
    lr =2e-4 #for adam lr of 0.0005 is the optimal
    batch_size = 30
    #architecture = "pvae"
    augmentations = ["horizontalflip"]
    pics_path ="/home/ido/datasets/projected_vae/obama/resized_images"
    weights_save_path = "/home/ido/datasets/projected_vae/obama/weights"
    dataset_name = "flowers"
    
    
    #hyper parameters
    projected = True
    transform = torchvision.transforms.Compose([
        #torchvision.transforms.Resize((256,256)),
        #torchvision.transforms.ColorJitter(brightness = 0.1,contrast = 0,saturation = 0.1,hue = 0.05),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        #torchvision.transforms.RandomVerticalFlip(p=0.3),
        #torchvision.transforms.RandomRotation(degrees = 5,interpolation = torchvision.transforms.InterpolationMode.BILINEAR),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    dataset = our_datasets.Pokemon_dataset(pics_path,transform)   #PokemonDataset(root=path, rgb=True)#'/content/drive/MyDrive/pokemon/pokemon', rgb=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    loss_type = "mse"
    optimizer_type = "Adam"
    z_dim =256
    big_z = True
    beta = 0.05 #for adam 0.01.
    proj_type = 2
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
    betas = [0.1]#[2,5]#[0.1,0.5,1]
    datasets = ["flowers"]
    #big_zs = [True,False]
    dataset_sizes = [9000]
    output_base_path = "./output_images"
    #lrs = [2e-4,5e-5,1e-5] #2e-4
    #proj_types = [2,1,0]
    #architecture = "pvae"
    augmentations = ["horizontalflip"]
    #####################################
    weights_path = "/home/ido/datasets/projected_vae/obama/weights/obama_date_24-04-2023__time_01-27-58.pth"
    if projected:
        model = ProjectedVAE(z_dim=z_dim,outs_shape=outs_shape,device=device,big_z=big_z)#.to(device)
    else:
        model = Vae_cnn_1(z_dim=4*z_dim,x_shape=x_shape,device=device)#.to(device)
    model = torch.compile(model.to(device))
    
    
    #imgs_output = "/home/ido/git_repos/Projected_VAE/batch_images"
    #generate_samples(100,model,weights_path,imgs_output)


    # ## calculate FID
    dataset_images_path = "/home/ido/datasets/projected_vae/pokemon/resized_images"
    generated_imgs_path = "/home/ido/git_repos/Projected_VAE/output_images/date_17-04-2023__time_18-32-56"
    print(calc_fid(dataset_images_path,generated_imgs_path))





## grid search


    # sub_folder = now.strftime("date_%d-%m-%Y__time_%H-%M-%S") #weights_path.split("/")[-1][:-4]
    iter_count = 0
    skip = 0
    generated_pics_dir = "./batch_generated"
    pics_path_base ="/home/ido/datasets/projected_vae/"
    weights_save_path_base = "/home/ido/datasets/projected_vae"
    for datset_size in dataset_sizes:
        for beta in betas:
                #for dataset_name in datasets:
                #for lr in lrs:
                    iter_count+=1
                    if iter_count >skip:
                                torch.cuda.empty_cache()
                                del dataset
                                del dataloader

                                json_data = {}
                                json_data['projected'] = projected
                                json_data['lr'] = lr
                                json_data['beta'] = beta
                                json_data['optimazer'] = optimizer_type
                                json_data['loss_type'] = loss_type
                                json_data['proj_types'] = proj_type
                                json_data['dataset_name'] = dataset_name
                                json_data['datset_size'] = datset_size
                                #json_data['architecture'] = architecture

                                pics_path = pics_path_base+"/"+dataset_name+"/resized_images"
                                weights_save_path = weights_save_path_base+"/"+dataset_name+"/weights"
                                dataset = our_datasets.Pokemon_dataset(pics_path,transform,dataset_size=datset_size)
                                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
                                
                                # restart model
                                model = model.to("cpu")
                                del model
                                if projected:
                                    model = ProjectedVAE(z_dim=z_dim,outs_shape=outs_shape,device=device,big_z=big_z)#.to(device)
                                else:
                                    model = Vae_cnn_1(z_dim=4*z_dim,x_shape=x_shape,device=device)
                                model = torch.compile(model.to(device)).to(device)
                                
                                # training loop
                                kl_losses,recon_losses,total_losses, weights_path,end_epoch,lr_history,timestamp,fid_history = training_loop(
                                    model,device,epochs,lr,beta,dataloader,
                                    loss_type,optimizer_type,weights_save_path,dataset_name,
                                    json_data,pics_path, generated_pics_dir
                                    )

                                # create and save samples
                                output_path = os.path.join(output_base_path,timestamp)
                                if not os.path.exists(output_path):
                                    os.makedirs(output_path)
                                generate_samples(num_of_samples=50,model=model,weights_path=weights_path,output_path=output_path,to_print=True)

                                #json build
                                json_data['augmentations'] = augmentations
                                json_data['kl_losses'] = kl_losses
                                json_data['recon_losses'] = recon_losses
                                json_data['total_losses'] = total_losses
                                json_data['weights_path'] = weights_path
                                json_data['end_epoch'] = end_epoch
                                json_data['lr_history'] = lr_history
                                json_data['output_path'] =  output_path
                                json_data['fid_history'] = fid_history
                                json_string = json.dumps(json_data,indent=4)
                                json_name = str(timestamp)+".json"
                                json_path = os.path.join(output_path,json_name)
                                
                                # json save
                                with open(json_path, "w") as outfile:
                                    outfile.write(json_string)


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
