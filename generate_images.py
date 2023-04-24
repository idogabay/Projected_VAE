import torch
from torch.utils.data import DataLoader
import torchvision
import os
from vae_utils import *
import our_datasets


def main():

    torch.set_float32_matmul_precision('high')
    x_shape = (3,256,256)
    z_dim =256
    device = set_device()

    ## projected ##
    outs_shape = {"0":[40, 64, 128, 128],
                  "1":[40, 64, 64, 64],
                  "2":[40, 128, 32, 32],
                  "3":[40, 256, 16, 16]
                  }
    batch_size = 30
    projected = True
    #architecture = "pvae"
    
    ### FILL ALL ###
    pics_path =""
    weights_path = ""
    output_path =""
    
    
    
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    dataset = our_datasets.Pokemon_dataset(pics_path,transform) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    if projected:
        model = ProjectedVAE(z_dim=z_dim,outs_shape=outs_shape,device=device)
    else:
        model = Vae_cnn_1(z_dim=4*z_dim,x_shape=x_shape,device=device)
    model = torch.compile(model.to(device))


    torch.cuda.empty_cache()

    

    # create and save samples
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    generate_samples(num_of_samples=50,model=model,weights_path=weights_path,output_path=output_path,to_print=True)


if __name__ == "__main__":
    main()
