from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import csv
import numpy as np
import pandas as pd
import torch
import os
import json
import torchvision


class Pokemon_dataset(Dataset):
    def __init__(self, images_root,transform):
        self.images_root = images_root
        #self.csv = np.array(pd.read_csv(csv_file))
        self.all_images = os.listdir(self.images_root)
        self.all_images.sort()
        #print(self.all_images)
        self.transform = transform

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        #print("holaaaaaaaaaaaaaaaaaaaaaaaa")
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.all_images[idx]
        image_path = os.path.join(str(self.images_root),str(image_name))
        #print(image_path)
        image = Image.open(image_path)
        return self.transform(image)



class Flowers_dataset(Dataset):
    def __init__(self, images_root,transform):
        self.images_root = images_root
        #self.csv = np.array(pd.read_csv(csv_file))
        self.all_images = os.listdir(self.images_root)
        self.all_images.sort()
        self.transform = transform

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.all_images[idx]
        image_path = os.path.join(str(self.images_root),str(image_name))
        image = Image.open(image_path)
        return self.transform(image)



class landscape_dataset(Dataset):
    def __init__(self, images_root,csv_file,transform):
        self.images_root = images_root
        #self.csv = np.array(pd.read_csv(csv_file))
        self.all_images = os.listdir(self.images_root)
        self.all_images.sort()
        self.transform = transform

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.all_images[idx]
        image_path = os.path.join(str(self.images_root),str(image_name))
        image = Image.open(image_path)
        return self.transform(image)


#### to be edit
class FFHQ_dataset(Dataset):
    def __init__(self, images_root,csv_file,transform):
        self.images_root = images_root
        self.csv = np.array(pd.read_csv(csv_file))
        self.transform = transform

    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.csv[idx][0]
        image_path = os.path.join(str(self.images_root),str(self.csv[idx][0]))
        image = Image.open(image_path)
        return self.transform(image)

class FID_dataset(Dataset):
    def __init__(self, images_root, num_images,
                 transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])):
        self.images_root = images_root
        self.transform = transform
        self.num_images = num_images
    def __len__(self):
        return len(self.num_images)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = str(idx)+".jpg"
        image_path = os.path.join(str(self.images_root),image_name)
        image = Image.open(image_path)
        return self.transform(image)


