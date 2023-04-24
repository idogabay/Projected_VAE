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
from torchvision.models.feature_extraction import create_feature_extractor
from torch.nn.utils import spectral_norm
import networks_fastgan
import dnnlib
from networks_fastgan import Generator
from PIL import Image
from projector import F_RandomProj
from datetime import datetime
import math
import piqa
#from pg_modules.blocks import DownBlock, DownBlockPatch, conv2d
from functools import partial
import our_datasets
from random import sample
import torch.nn.functional as F

def calc_fid(data_path,generated_path):
    num_pics = 30
    pics_names = os.listdir(data_path)
    samples = sample(pics_names,num_pics)

    transform = torchvision.transforms.ToTensor()
    pics = torch.zeros((1,3,256,256))
    generates = torch.zeros((1,3,256,256))
    for i in range(num_pics):
        pic_name = str(i)+".jpg"
        pic_path = os.path.join(data_path,samples[i])
        generated_pic_path = os.path.join(generated_path,pic_name)
        pic = Image.open(pic_path)
        generate = Image.open(generated_pic_path)
        if i == 0:
            pics = torch.unsqueeze(transform(pic),0)
            generates = torch.unsqueeze(transform(generate),0)
        else:
            pics = torch.cat((pics,torch.unsqueeze(transform(pic),dim=0)))
            generates = torch.cat((generates,torch.unsqueeze(transform(pic),dim=0)),dim = 0)

    min_pic = [0,0,0]
    max_pic = [0,0,0]
    max_range_pic = [0,0,0]
    min_generated = [0,0,0]
    max_generated = [0,0,0]
    max_range_generated = [0,0,0]
    for i in range(3):
        min_pic[i] = torch.min(torch.min(torch.min(pics[:,i],dim=2)[0],dim=1)[0],dim=0)[0]
        min_generated[i] = torch.min(torch.min(torch.min(generates[:,i],dim=2)[0],dim=0)[0],dim=0)[0]
        max_pic[i] = torch.min(torch.max(torch.max(pics[:,i],dim=2)[0],dim=0)[0],dim=0)[0]
        max_generated[i] = torch.min(torch.max(torch.max(generates[:,i],dim=2)[0],dim=0)[0],dim=0)[0]
    for i in range(3):
        max_range_pic[i] = max_pic[i]-min_pic[i]
        max_range_generated[i] = max_generated[i]-min_generated[i]  
    for i in range(3):
        pics[:,i] = (pics[:,i]-min_pic[i])/max_range_pic[i]
        generates[:,i] = (generates[:,i]-min_generated[i])/max_range_generated[i]

    pics[pics>1] = 1
    pics[pics<0] = 0
    generates[generates>1] = 1
    generates[generates<0] = 0
    fid_metric = piqa.FID()
    pics_features = fid_metric.features(x=pics)
    generates_features = fid_metric.features(x=generates)
    fid = fid_metric(pics_features,generates_features)
    del fid_metric
    del pics
    del generates
    del pics_features
    del generates_features
    return fid


def NormLayer(c, mode='group'):
    if mode == 'group':
        return nn.GroupNorm(c//2, c)
    elif mode == 'batch':
        return nn.BatchNorm2d(c)


def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))

class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes, separable=False):
        super().__init__()
        if not separable:
            self.main = nn.Sequential(
                conv2d(in_planes, out_planes, 4, 2, 1),
                NormLayer(out_planes),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.main = nn.Sequential(
                SeparableConv2d(in_planes, out_planes, 3),
                NormLayer(out_planes),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AvgPool2d(2, 2),
            )

    def forward(self, feat):
        return self.main(feat)



class DownBlockPatch(nn.Module):
    def __init__(self, in_planes, out_planes, separable=False):
        super().__init__()
        self.main = nn.Sequential(
            DownBlock(in_planes, out_planes, separable),
            conv2d(out_planes, out_planes, 1, 1, 0, bias=False),
            NormLayer(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, feat):
        return self.main(feat)


### CSM

# the original implementation from the tutorial - leave untouched (for your own sake), copy-paste what you need to another cell

# reparametrization trick
# the original implementation from the tutorial - leave untouched (for your own sake), copy-paste what you need to another cell

def reparameterize(mu, logvar, device=torch.device("cpu")):
        """
        This function applies the reparameterization trick:
        z = mu(X) + sigma(X)^0.5 * epsilon, where epsilon ~ N(0,I)
        :param mu: mean of x
        :param logvar: log variance of x
        :param device: device to perform calculations on
        :return z: the sampled latent variable
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(device)
        return mu + eps * std


# encoder - Q(z|X)
class VaeCnnEncoder(torch.nn.Module):
    """
       This class builds the encoder for the VAE
       :param x_dim: input dimensions
       :param hidden_size: hidden layer size
       :param z_dim: latent dimensions
       :param device: cpu or gpu
       """

    def __init__(self, z_dim,x_shape, device):
        super(VaeCnnEncoder, self).__init__()
        self.z_dim = z_dim
        self.device = device
        
        ###block1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=128)
        self.relu1 = nn.LeakyReLU()
        
        ###block2
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64*4, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=64*4)
        self.relu2 = nn.LeakyReLU()

        ###block3
        self.conv3 = nn.Conv2d(in_channels=64*4, out_channels=64*8, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=64*8)
        self.relu3 = nn.LeakyReLU()

        #self.features = nn.Sequential(nn.Linear(x_dim, self.hidden_size),nn.ReLU())
        ###here need to concate
        self.fc1 = nn.Linear(self._get_conv_out(x_shape), self.z_dim)#nn.Linear(self.hidden_size, self.z_dim, bias=True)  # fully-connected to output mu
        self.fc2 = nn.Linear(self._get_conv_out(x_shape), self.z_dim)#nn.Linear(self.hidden_size, self.z_dim, bias=True)  # fully-connected to output logvar

    def _get_conv_out(self, shape):
        """
        Helper function to automatically calculate the conv layers output.
        """
        o = self.conv3(self.conv2(self.conv1(torch.zeros(1, *shape))))
        return int(np.prod(o.size()))
    
    # reparametrization trick

    def bottleneck(self, h):
        """
        This function takes features from the encoder and outputs mu, log-var and a latent space vector z
        :param h: features from the encoder
        :return: z, mu, log-variance
        """
        mu, logvar = self.fc1(h), self.fc2(h)
        # use the reparametrization trick as torch.normal(mu, logvar.exp()) is not differentiable
        z = reparameterize(mu, logvar, device=self.device)
        return z, mu, logvar

    def forward(self, x):
        """
        This is the function called when doing the forward pass:
        z, mu, logvar = VaeEncoder(X)
        """
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = torch.flatten(x,1)
        #x = torch.cat([x, labels], dim=1)
        z, mu, logvar = self.bottleneck(x)
        return z, mu, logvar

    
class VaeCnnDecoder(torch.nn.Module):
    """
       This class builds the decoder for the VAE
       :param x_dim: input dimensions
       :param hidden_size: hidden layer size
       :param z_dim: latent dimensions
       """

    def __init__(self, z_dim):
        super(VaeCnnDecoder, self).__init__()
        self.z_dim = z_dim
        ###block0
        self.fc0 = nn.Linear(self.z_dim, 16*16*64*4)
        self.bn0 = nn.BatchNorm1d(16*16*64*4)
        
        ###block1
        self.deconv1 = nn.ConvTranspose2d(in_channels=64 * 4,out_channels= 64 * 2, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(64 * 2)
        self.relu1 = nn.LeakyReLU()
        
        ###block2
        self.deconv2 = nn.ConvTranspose2d(in_channels=64 * 2,out_channels= 64, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU()
        
        ###block3
        self.deconv3 = nn.ConvTranspose2d(in_channels=64,out_channels= 64, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.LeakyReLU()

        ###block4
        self.deconv4 = nn.ConvTranspose2d(in_channels=64,out_channels= 3, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.sigmoid4 = nn.Sigmoid()


    def forward(self, x):
        """
        This is the function called when doing the forward pass:
        x_reconstruction = VaeDecoder(z)
        """
        x = self.bn0(self.fc0(x))
        x = x.reshape(x.shape[0],64*4,16,16)
        x = self.relu1(self.bn1(self.deconv1(x)))
        x = self.relu2(self.bn2(self.deconv2(x)))
        x = self.relu3(self.bn3(self.deconv3(x)))
        x = self.sigmoid4(self.deconv4(x))
        return x
    

class Vae_cnn_1(torch.nn.Module):
    def __init__(self, z_dim,x_shape, device):
        super(Vae_cnn_1, self).__init__()
        self.device = device
        self.z_dim = z_dim
        self.encoder = encoder_pg(start_sz = x_shape[1], end_sz=8, separable=False, patch=False ,z_dim=z_dim,device=self.device,projected=False)
        self.decoder = Generator(synthesis_kwargs={'lite': False},z_dim=self.z_dim)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        x = self.decoder(z)
        return x

    def sample(self,num_samples):
        """
        This functions generates new data by sampling random variables and decoding them.
        Vae.sample() actually generatess new data!
        Sample z ~ N(0,1)
        """
        z = torch.randn(num_samples, self.z_dim).to(self.device)
        return self.decode(z)

    def forward(self, x):
        """
        This is the function called when doing the forward pass:
        return x_recon, mu, logvar, z = Vae(X)
        """
        z,mu, logvar = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, mu, logvar



def beta_loss_function(recon_x, x, mu, logvar, loss_type='bce',beta = 1,projected = False):
    """
    This function calculates the loss of the VAE.
    loss = reconstruction_loss - 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param recon_x: the reconstruction from the decoder
    :param x: the original input
    :param mu: the mean given X, from the encoder
    :param logvar: the log-variance given X, from the encoder
    :param loss_type: type of loss function - 'mse', 'l1', 'bce'
    :return: VAE loss
    """
    if loss_type == 'mse':
        recon_error = (x-recon_x)**2
        recon_error = recon_error.sum() / recon_error.shape[0]
    elif loss_type == 'l1':
        recon_error = F.l1_loss(recon_x, x, reduction='sum')
    elif loss_type == 'bce':
        recon_error = F.binary_cross_entropy(recon_x, x, reduction='sum')
    else:
        raise NotImplementedError

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = (recon_error + beta*kl) / x.size(0)
    return kl,recon_error,total_loss

def set_device():
    if torch.cuda.is_available():
        print("device set to cuda:0")
        torch.cuda.current_device()
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def training_loop(model,device,epochs,lr,beta,dataloader,
                  loss_type,optimizer_type, weights_save_path,dataset_name,
                  json_data, pics_root_dir, generated_pics_root_dir):
    # training
    # optimizer
    if optimizer_type == "Adam":
        vae_optim = torch.optim.Adam(params=model.parameters(), lr=lr)
    else:
        vae_optim = torch.optim.SGD(params=model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=vae_optim,step_size=10,gamma=0.95,last_epoch=-1,verbose=False)
    now = datetime.now()
    current_time = now.strftime("date_%d-%m-%Y__time_%H-%M-%S")
    weights_name = dataset_name+"_"+str(current_time)+".pth"
    
    weights_full_path = os.path.join(weights_save_path,weights_name)
    print("Starting :","\n",json_data)

    # save the losses from each epoch, we might want to plot it later
    recon_losses = []
    kl_losses = []
    total_losses = []
    lr_history = []
    fid_history = []
    lpips_gistory = []
    print("start training")
    # here we go
    min_fid = torch.ones((1))
    min_fid[0] = 999999999
    min_fid = min_fid[0]
    fid = min_fid
    last_epoch_min = 0
    end_epoch = 0
    not_a_number = False

    for epoch in range(epochs):
        epoch_start_time = time.time()
        batch_total_losses = []
        batch_kl_losses = []
        batch_recon_losses = []
        torch.save(model.state_dict(), weights_full_path)
        current_lr = 0
        batch = torch.zeros((1))
        mu = 0
        logvar = 1
        for param_group in vae_optim.param_groups:
            current_lr = param_group['lr']
            lr_history.append(current_lr)
        for batch_i, batch in enumerate(dataloader):

            # forward pass
            batch = batch.to(device)
            x_recon, mu, logvar = model(batch)#, c)
            kl,recon,total_loss = beta_loss_function(x_recon, batch, mu, logvar, loss_type=loss_type, beta=beta)#.permute(0, 2, 3, 1)
            
            # optimization (same 3 steps everytime)
            vae_optim.zero_grad()
            total_loss.backward()
            vae_optim.step()
            
            # save loss
            batch_total_losses.append(total_loss.cpu().item())
            batch_kl_losses.append(kl.cpu().item())
            batch_recon_losses.append(recon.cpu().item())
            if math.isnan(kl) or math.isnan(recon) or math.isnan(total_loss):
                not_a_number = True
                break 
         
        x_recon = x_recon.detach().cpu()
        x_recon = x_recon.to("cpu")
        batch = batch.detach().cpu()
        batch = batch.to("cpu")
        to_pil = torchvision.transforms.ToPILImage()
        for i in range(x_recon.shape[0]):
            img = denormalized(x_recon[i])
            pil = to_pil(img)
            img_name = "./batch_recon/"+str(i)+".jpg"
            im = pil.save(img_name)
            
            img = denormalized(batch[i])
            pil = to_pil(img)
            img_name = "./batch_images/"+str(i)+".jpg"
            im = pil.save(img_name)

        #delete
        del batch
        if not_a_number:
            print("loss is not a number - break")
            break
        if epoch>10 and epoch % 5 == 0:
            generate_samples(30,model,None,"./batch_generated")
            fid = calc_fid(pics_root_dir, generated_pics_root_dir)

        loss = np.mean(batch_total_losses)
        if fid<min_fid:
            best_mu = mu
            best_logvar = logvar
            min_fid = fid
            last_epoch_min = epoch
            state = model.state_dict()
            torch.save(state, weights_full_path)
            print("new min FID:",min_fid,"epoch:",epoch)
        scheduler.step()
        total_losses.append(loss)
        kl_losses.append(np.mean(batch_kl_losses))
        recon_losses.append(np.mean(batch_recon_losses))
        fid_history.append(fid.item())
        if (epoch+1)%10 ==0:
            print("epoch: {}| kl {:.3f}| recon {:.3f} |total_loss {:.3f}| epoch time: {:.3f} sec"\
                .format((epoch+1),kl_losses[-1],recon_losses[-1],total_losses[-1], time.time() - epoch_start_time))
        if epoch-last_epoch_min > 100:
            print("stop improving at epoch:",last_epoch_min,".\nbreaking loop")
            end_epoch = last_epoch_min
            break
        end_epoch = epoch
    return kl_losses,recon_losses,total_losses,weights_full_path,end_epoch,lr_history,current_time,fid_history


    
def plot_loss(losses,title):
    plt.plot(losses)
    plt.xlabel('epochs')
    plt.ylabel('loss value')
    plt.title(title)
    plt.show()

def denormalized(t1,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]):
    channels = t1.shape[0]
    t2 = torch.zeros((t1.shape))
    for i in range(channels):
        t2[i] = t1[i]*std[i] + mean[i]
    return t2

def generate_samples(num_of_samples,model,weights_path = None,output_path = "./batch_generated",to_print = False):
    if weights_path != None:
        state = torch.load(weights_path,map_location=set_device())
        model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        samples = model.sample(num_of_samples)
        transform = torchvision.transforms.ToPILImage()
        for i,sample in enumerate(samples):
            img = transform(denormalized(sample))
            img = img.save(output_path+"/"+str(i)+".jpg")
    model.train()    
    if to_print:
        print("done")

#encoder new 06/11/22- Q(z|X)
class VaeCnnEncoder_06_11(torch.nn.Module):
    """
       This class builds the encoder for the VAE
       :param x_dim: input dimensions
       :param hidden_size: hidden layer size
       :param z_dim: latent dimensions
       :param device: cpu or gpu
       """

    def __init__(self, z_dim,x_shape, device):
        super(VaeCnnEncoder_06_11, self).__init__()
        self.z_dim = z_dim
        self.device = device
        drop_p = 0.1
        ###block1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), stride=(2, 2), padding=0)
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout2d(p=drop_p)

        ###block2
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=0)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout2d(p=drop_p)

        ###block3
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=0)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout2d(p=drop_p)
        
        ###block4
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=0)
        self.bn4 = nn.BatchNorm2d(num_features=64)
        self.relu4 = nn.LeakyReLU()
        self.dropout4 = nn.Dropout2d(p=drop_p)
        
        ###block5
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=0)
        self.bn5 = nn.BatchNorm2d(num_features=64)
        self.relu5 = nn.LeakyReLU()
        self.dropout5 = nn.Dropout2d(p=drop_p)

        ###block6
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=0)
        self.bn6 = nn.BatchNorm2d(num_features=64)
        self.relu6 = nn.LeakyReLU()
        self.dropout6 = nn.Dropout2d(p=drop_p)

        ###block7
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=0)
        self.bn7 = nn.BatchNorm2d(num_features=64)
        self.relu7 = nn.LeakyReLU()
        self.dropout7 = nn.Dropout2d(p=drop_p)
        
        ###block8
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=0)
        self.bn8 = nn.BatchNorm2d(num_features=128)
        self.relu8 = nn.LeakyReLU()
        self.dropout8 = nn.Dropout2d(p=drop_p)
        
        ###block9
        self.conv9 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=0)
        self.bn9 = nn.BatchNorm2d(num_features=128)
        self.relu9 = nn.LeakyReLU()
        self.dropout9 = nn.Dropout2d(p=drop_p)
        
        ### block 10
        self.conv10 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=0)
        self.bn10 = nn.BatchNorm2d(num_features=32)
        self.relu10 = nn.LeakyReLU()
        
        self.fc1 = nn.Linear(288, self.z_dim)
        self.fc2 = nn.Linear(288, self.z_dim)

    def _get_conv_out(self, shape):
        """
        Helper function to automatically calculate the conv layers output.
        """
        o = self.conv3(self.conv2(self.conv1(torch.zeros(1, *shape))))
        return int(np.prod(o.size()))
    

    def bottleneck(self, h):
        """
        This function takes features from the encoder and outputs mu, log-var and a latent space vector z
        :param h: features from the encoder
        :return: z, mu, log-variance
        """
        mu, logvar = self.fc1(h), self.fc2(h)
        # use the reparametrization trick as torch.normal(mu, logvar.exp()) is not differentiable
        z = reparameterize(mu, logvar, device=self.device)
        return z, mu, logvar

    def forward(self, x):
        """
        This is the function called when doing the forward pass:
        z, mu, logvar = VaeEncoder(X)
        """
        x = self.dropout1(self.relu1(self.bn1(self.conv1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.conv2(x))))
        x = self.dropout3(self.relu3(self.bn3(self.conv3(x))))
        x = self.dropout4(self.relu4(self.bn4(self.conv4(x))))
        x = self.dropout5(self.relu5(self.bn5(self.conv5(x))))
        x = self.dropout6(self.relu6(self.bn6(self.conv6(x))))
        x = self.dropout7(self.relu7(self.bn7(self.conv7(x))))
        x = self.dropout8(self.relu8(self.bn8(self.conv8(x))))
        x = self.dropout9(self.relu9(self.bn9(self.conv9(x))))
        x = self.relu10(self.bn10(self.conv10(x)))
        x = torch.flatten(x,1)
        z, mu, logvar = self.bottleneck(x)
        return z, mu, logvar

  
class VaeCnnDecoder_06_11(torch.nn.Module):
    """
       This class builds the decoder for the VAE
       :param x_dim: input dimensions
       :param hidden_size: hidden layer size
       :param z_dim: latent dimensions
       """

    def __init__(self, z_dim):
        super(VaeCnnDecoder_06_11, self).__init__()
        self.z_dim = z_dim
        ### here need to concate label
        ###block0
        drop_p = 0.1
        self.fc0 = nn.Linear(self.z_dim, 512)
        self.bn0 = nn.BatchNorm1d(512) 
        self.relu0 = nn.LeakyReLU()
        self.dropout0 = nn.Dropout1d(p=drop_p)
        ###here need to reshape signal
        ###block1
        self.deconv1 = nn.ConvTranspose2d(in_channels=32,out_channels= 128, kernel_size=(3, 3), stride=(2,2), padding=0,output_padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout2d(p=drop_p)
        
        ###block2
        self.deconv2 = nn.ConvTranspose2d(in_channels=128,out_channels= 128, kernel_size=(3, 3), stride=(2,2), padding=2,output_padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout2d(p=drop_p)

        ###block2.5
        self.deconv2_5 = nn.ConvTranspose2d(in_channels=128,out_channels= 128, kernel_size=(3, 3), stride=(2, 2), padding=2,output_padding=1)
        self.bn2_5 = nn.BatchNorm2d(128)
        self.relu2_5 = nn.LeakyReLU()
        self.dropout2_5 = nn.Dropout2d(p=drop_p)
        
        ###block3
        self.deconv3 = nn.ConvTranspose2d(in_channels=128,out_channels= 64, kernel_size=(3, 3), stride=(2, 2), padding=2,output_padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout2d(p=drop_p)
        
        ###block4
        self.deconv4 = nn.ConvTranspose2d(in_channels=64,out_channels= 64, kernel_size=(3, 3),stride=(2, 2), padding=2,output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.LeakyReLU()
        self.dropout4 = nn.Dropout2d(p=drop_p)
        
                
        ###block5
        self.deconv5 = nn.ConvTranspose2d(in_channels=64,out_channels= 64, kernel_size=(3, 3), stride=(2, 2), padding=2,output_padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.LeakyReLU()
        self.dropout5 = nn.Dropout2d(p=drop_p)
        
        ###block6
        self.deconv6 = nn.ConvTranspose2d(in_channels=64,out_channels= 32, kernel_size=(3, 3), stride=(1, 1), padding=2)
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.LeakyReLU()
        self.dropout6 = nn.Dropout2d(p=drop_p)

                
        ###block7
        self.deconv7 = nn.ConvTranspose2d(in_channels=32,out_channels= 16, kernel_size=(3, 3), stride=(1, 1), padding=2)
        self.bn7 = nn.BatchNorm2d(16)
        self.relu7 = nn.LeakyReLU()
        self.dropout7 = nn.Dropout2d(p=drop_p)

        ###block8
        self.deconv8 = nn.ConvTranspose2d(in_channels=16,out_channels= 8, kernel_size=(3, 3), stride=(1,1), padding=1)
        self.bn8 = nn.BatchNorm2d(8)
        self.relu8 = nn.LeakyReLU()
        self.dropout8 = nn.Dropout2d(p=drop_p)


        ###block9
        self.deconv9 = nn.ConvTranspose2d(in_channels=8,out_channels= 3, kernel_size=(3, 3), stride=(1, 1), padding=0)
        self.sigmoid9 = nn.Sigmoid()
        

    def forward(self, x):
        """
        This is the function called when doing the forward pass:
        x_reconstruction = VaeDecoder(z)
        """
        x = self.dropout0(self.relu0(self.bn0(self.fc0(x))))
        x = x.reshape(x.shape[0],32,4,4)
        x = self.dropout1(self.relu1(self.bn1(self.deconv1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.deconv2(x))))
        x = self.dropout2_5(self.relu2_5(self.bn2_5(self.deconv2_5(x))))
        x = self.dropout3(self.relu3(self.bn3(self.deconv3(x))))
        x = self.dropout4(self.relu4(self.bn4(self.deconv4(x))))
        x = self.dropout5(self.relu5(self.bn5(self.deconv5(x))))
        x = self.dropout6(self.relu6(self.bn6(self.deconv6(x))))
        x = self.dropout7(self.relu7(self.bn7(self.deconv7(x))))
        x = self.dropout8(self.relu8(self.bn8(self.deconv8(x))))
        x = self.sigmoid9(self.deconv9(x))
        return x



class encoder_pg(nn.Module):
    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8,
                head=None, separable=False, patch=False,z_dim=100,
                device='cpu',projected = False,big_z = True):
        super().__init__()
        self.device = device
        self.projected = projected
        channel_dict = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64,
                        256: 3, 512: 16, 1024: 8}
        self.z_dim = z_dim
        # interpolate for start sz that are not powers of two
        if start_sz not in channel_dict.keys():
            sizes = np.array(list(channel_dict.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz
        self.big_z = big_z

        # if given ndf, allocate all layers with the same ndf
        if ndf is None:
            nfc = channel_dict
        else:
            nfc = {k: ndf for k, v in channel_dict.items()}

        # for feature map discriminators with nfc not in channel_dict
        # this is the case for the pretrained backbone (midas.pretrained)
        if nc is not None and head is None:
            nfc[start_sz] = nc

        layers = []

        # Head if the initial input is the full modality
        if head:
            layers += [conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                       nn.LeakyReLU(0.2, inplace=True)]

        # Down Blocks
        DB = partial(DownBlockPatch, separable=separable) if patch else partial(DownBlock, separable=separable)
        while start_sz > end_sz:
            layers.append(DB(nfc[start_sz],  nfc[start_sz//2]))
            start_sz = start_sz // 2

        layers.append(conv2d(in_channels = nfc[end_sz], out_channels = 8, kernel_size =4, padding=0, bias=False))

        ### i added 
        #net= 
        #x = torch.flatten(net(x),1)
        self.main = nn.Sequential(*layers)

        self.fc1 = nn.Linear(200, self.z_dim)#nn.Linear(self.hidden_size, self.z_dim, bias=True)  # fully-connected to output mu
        self.fc2 = nn.Linear(200, self.z_dim)#nn.Linear(self.hidden_size, self.z_dim, bias=True)  # fully-connected to output logvar
   
    def bottleneck(self, h):
        """
        This function takes features from the encoder and outputs mu, log-var and a latent space vector z
        :param h: features from the encoder
        :return: z, mu, log-variance
        """ 
        mu, logvar = self.fc1(h), self.fc2(h)
        # use the reparametrization trick as torch.normal(mu, logvar.exp()) is not differentiable
        if self.projected == False or self.big_z == False:
            z = reparameterize(mu, logvar, device=self.device)
        else:
            z = None
        return z, mu, logvar


    def forward(self, x):
    #c):
        x = torch.flatten(self.main(x),1)
        return self.bottleneck(x)









# TODO: insert projected part int the vae for the purpose of saving the weights as one model.



class ProjectedVAE(nn.Module):
    def __init__(self, z_dim,outs_shape,device, big_z):
        super(ProjectedVAE, self).__init__()
        self.device = device
        self.z_dim = z_dim
        self.outs_shape = outs_shape
        self.big_z = big_z
        self.projector = F_RandomProj(proj_type = 2).eval()
        self.projector.requires_grad_(False)
        self.encoder0 = encoder_pg(start_sz = outs_shape["0"][2],
                            end_sz=8, separable=False,
                            patch=False ,z_dim=z_dim,
                            device=self.device,projected=True,big_z=self.big_z)
        self.encoder1 = encoder_pg(start_sz = outs_shape["1"][2],
                            end_sz=8, separable=False,
                            patch=False ,z_dim=z_dim,
                            device=self.device,projected=True,big_z=self.big_z)
        self.encoder2 = encoder_pg(start_sz = outs_shape["2"][2],
                                   end_sz=8, separable=False,
                                   patch=False ,z_dim=z_dim,
                                   device=self.device,projected=True,big_z=self.big_z)
        self.encoder3 = encoder_pg(start_sz = outs_shape["3"][2],
                                   end_sz=8, separable=False,
                                   patch=False ,z_dim=z_dim,
                                   device=self.device,projected=True,big_z=self.big_z)
        self.decoder = Generator(z_dim=4*self.z_dim,synthesis_kwargs={'lite': False})#VaeCnnDecoder_06_11(self.z_dim)#,self.cond_dim)

        # params:
        # 
        
    def project(self,x):
        outs = self.projector(x)
        return outs

    def encode(self, outs):
        z_0,mu_0, logvar_0 = self.encoder0(outs['0'])
        z_1,mu_1, logvar_1 = self.encoder1(outs['1'])
        z_2,mu_2, logvar_2 = self.encoder2(outs['2'])
        z_3,mu_3, logvar_3 = self.encoder3(outs['3'])
        mu = torch.cat([mu_0,mu_1,mu_2,mu_3],dim=1)
        logvar = torch.cat([logvar_0,logvar_1,logvar_2,logvar_3],dim=1)
        if self.big_z == False:
            z = torch.cat([z_0,z_1,z_2,z_3],dim=1)
        else:
            z = reparameterize(mu,logvar,self.device)
        return z, mu, logvar

    def decode(self, z):
        x = self.decoder(z)
        return x

    def sample(self,num_samples):
        """
        This functions generates new data by sampling random variables and decoding them.
        Vae.sample() actually generatess new data!
        Sample z ~ N(0,1)
        """
        multiplyer = 4
        z = torch.randn(num_samples, multiplyer*self.z_dim).to(self.device)
        return self.decode(z)

    def forward(self, x, memory_save = False):#, c):
        """
        This is the function called when doing the forward pass:
        return x_recon, mu, logvar, z = Vae(X)
        """
        with torch.no_grad():
            x = self.project(x)
        for i in range(4):
            x[str(i)].requires_grad = True
        z, mu, logvar = self.encode(x)
        x_recon = self.decode(z)        
        return x_recon, mu, logvar



