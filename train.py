import torch
from torch.utils.data import DataLoader
import torchvision
import os
from vae_utils import *
import our_datasets
import json


def main():
    torch.set_float32_matmul_precision('high')
    epochs = 500
    x_shape = (3,256,256)
    z_dim =256
    beta = 0.01
    device = set_device()

    ## projected ##
    outs_shape = {"0":[40, 64, 128, 128],
                  "1":[40, 64, 64, 64],
                  "2":[40, 128, 32, 32],
                  "3":[40, 256, 16, 16]
                  }
    projected = True

    ### FILL ALL ###
    pics_path =""
    weights_save_path = ""
    dataset_name = ""
    weights_path = ""
    output_base_path = ""
    generated_pics_dir = "./batch_generated"
    pics_path_base =""
    weights_save_path_base = ""
    
    if projected:
        model = ProjectedVAE(z_dim=z_dim,outs_shape=outs_shape,device=device)
    else:
        model = Vae_cnn_1(z_dim=4*z_dim,x_shape=x_shape,device=device)
    model = torch.compile(model.to(device))


    torch.cuda.empty_cache()

    pics_path = pics_path_base+"/"+dataset_name+"/resized_images"
    weights_save_path = weights_save_path_base+"/"+dataset_name+"/weights"
    dataset = our_datasets.Pokemon_dataset(pics_path,transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # create and save samples
    output_path = os.path.join(output_base_path,timestamp)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    generate_samples(num_of_samples=50,model=model,weights_path=weights_path,output_path=output_path,to_print=True)

    #json build
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


if __name__ == "__main__":
    main()
