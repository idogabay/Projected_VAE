#aaaaaaa
#bbbbb
import numpy as np
import torch
#import cv2
import torchvision
import projector
from PIL import Image



def main():
    path0 = "/home/ido/datasets/projected_vae/100-shot-obama/resized_images/0.jpg"
    path1 = "/home/ido/datasets/projected_vae/100-shot-obama/resized_images/1.jpg"
    img0 = Image.open(path0)
    img1 = Image.open(path1)
    transform = torchvision.transforms.ToTensor()
    t0 = torch.unsqueeze(transform(img0),dim=0)
    t1 = torch.unsqueeze(transform(img1),dim=0)
    t = torch.cat((t0,t1),dim=0)
    print(t.shape)
    # min_pic = torch.min(torch.min(t[:,0],dim=2)[0],dim=1)[0]
    min_pic = torch.min(t[:,0],dim=2)[0]
    print(min_pic.shape)
    min_pic = torch.min(min_pic,dim=1)[0]
    min_pic = torch.min(min_pic,dim=0)[0]
    print(min_pic)




if __name__ == "__main__":
    main()
