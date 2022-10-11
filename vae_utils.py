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

    def __init__(self, z_dim=10,x_shape = (3,60,60), device=torch.device("cpu")):
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

    def __init__(self, z_dim=10):#, cond_dim = 10):
        super(VaeCnnDecoder, self).__init__()
        self.z_dim = z_dim
        ### here need to concate label
        ###block0
        self.fc0 = nn.Linear(self.z_dim, 64 * 4 * 4 * 4)
        self.bn0 = nn.BatchNorm1d(64 * 4 * 4 * 4)
        
        ###here need to reshape signal
        ###block1
        self.deconv1 = nn.ConvTranspose2d(in_channels=64 * 4,out_channels= 64 * 2, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(64 * 2)
        self.relu1 = nn.LeakyReLU()
        
        ###block2
        self.deconv2 = nn.ConvTranspose2d(in_channels=64 * 2,out_channels= 64, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU()
        
        ###block3
        self.deconv3 = nn.ConvTranspose2d(in_channels=64,out_channels= 64, kernel_size=(3, 3), stride=(2, 2), padding=2, output_padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.LeakyReLU()

        ###block4
        self.deconv4 = nn.ConvTranspose2d(in_channels=64,out_channels= 3, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.sigmoid4 = nn.Sigmoid()

        # self.decoder = nn.Sequential(nn.Linear(self.z_dim, self.hidden_size),
        #                              nn.ReLU(),
        #                              nn.Linear(self.hidden_size, self.x_dim),
        #                              nn.Sigmoid())
        # why we use sigmoid? becaue the pixel values of images are in [0,1] and sigmoid(x) does just that!
        # if you don't work with images, you don't have to use that.


    def forward(self, x):
        """
        This is the function called when doing the forward pass:
        x_reconstruction = VaeDecoder(z)
        """
        #x = torch.cat([x,labels],dim=1)
        x = self.bn0(self.fc0(x))
        x = x.reshape(x.shape[0],64*4,4,4)
        x = self.relu1(self.bn1(self.deconv1(x)))
        x = self.relu2(self.bn2(self.deconv2(x)))
        x = self.relu3(self.bn3(self.deconv3(x)))
        x = self.sigmoid4(self.deconv4(x))


        
        #x = self.decoder(x)
        return x
    

class Vae_cnn_1(torch.nn.Module):
    def __init__(self, z_dim=10,x_shape=(3,60,60), device=torch.device("cpu")):
        super(Vae_cnn_1, self).__init__()
        self.device = device
        self.z_dim = z_dim
        #self.cond_dim = cond_dim

        #if self.cond_dim is not None:
        #x_dim+= cond_dim
        self.encoder = VaeCnnEncoder(z_dim,x_shape, self.device)#,self.cond_dim)
        
        #if self.cond_dim is not None:
        #self.z_dim+=cond_dim
        self.decoder = VaeCnnDecoder(self.z_dim)#,self.cond_dim)
        

    def encode(self, x):
        z, mu, logvar = self.encoder(x)
        return z, mu, logvar

    def decode(self, z):#,labels):
        x = self.decoder(z)#,labels)
        return x

    def sample(self,num_samples=1):
        """
        This functions generates new data by sampling random variables and decoding them.
        Vae.sample() actually generatess new data!
        Sample z ~ N(0,1)
        """
        #if self.cond_dim is not None:
        z = torch.randn(num_samples, self.z_dim).to(self.device)
        #if x_cond is not None:
        #print(self.z_dim)
        #labels = labels_to_one_hots(labels, self.cond_dim).to(device) ###check if neseccery 
        #else:
        #     label = torch.randint(0,9, num_samples)
        #     label = labels_to_one_hots(label, self.cond_dim).to(device)
        #z = torch.cat([z, labels], dim=1)
        # #else:
        # z = torch.randn(num_samples, self.z_dim).to(self.device)
        return self.decode(z)#,labels)

    def forward(self, x):
        """
        This is the function called when doing the forward pass:
        return x_recon, mu, logvar, z = Vae(X)
        """
        z, mu, logvar = self.encode(x)
        #if x_cond is not None:
        #z = torch.cat([z, x_cond], dim=1)
        x_recon = self.decode(z)#,labels)
        #x_recon = self.decode(z)
        return x_recon, mu, logvar, z



def beta_loss_function_pokemon(recon_x, x, mu, logvar, loss_type='bce',beta = 1):
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
        recon_error = F.mse_loss(recon_x, x, reduction='sum')
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
    return total_loss



def training_loop(epochs,x_shape,z_dim,lr,beta,dataloader,num_of_classes,loss_type,optimizer_type) :
    # training
    # check if there is gpu avilable, if there is, use it
    if torch.cuda.is_available():
        torch.cuda.current_device()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    #print("running calculations on: ", device)
    # load the data
    #dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    # create our model and send it to the device (cpu/gpu)
    vae = Vae_cnn_1(z_dim=z_dim,x_shape=x_shape,device=device).to(device)
    # optimizer
    if optimizer_type == "Adam":
        vae_optim = torch.optim.Adam(params=vae.parameters(), lr=lr)
    else:
        vae_optim = torch.optim.SGD(params=vae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=vae_optim,mode='min',factor=0.8,patience=10,verbose = True)
    fname = "pokemon_cnn_beta_"+str(beta)+"_vae_"+str(epochs)+"_epochs_dim_"+str(z_dim)+"_loss_type_"+str(loss_type)+"_optimizer_type_"+str(optimizer_type)+".pth"
    path = os.path.join("..","data")
    fname = os.path.join(path,fname)
    #print(fname)
    # save the losses from each epoch, we might want to plot it later
    # recon_losses = []
    # kl_losses = []
    total_losses = []

    # here we go
    for epoch in range(epochs):
        epoch_start_time = time.time()
        batch_total_losses = []
        # batch_kl_losses = []
        # batch_recon_losses = []
        for batch_i, (batch,labels) in enumerate(dataloader):
            # forward pass
            # x = batch[0].to(device).view(-1, X_DIM) # just the images
            # x = batch.to(device).view(-1, X_DIM) # just the images
            # x_cond = labels_to_one_hots(labels, num_of_classes).to(device)
            # x = torch.cat([x,x_cond ], dim=1)
            batch = batch.to(device)
            labels = labels.to(device)
            x_recon, mu, logvar, z = vae(batch)
            # calculate the loss
            # recon_loss,kl,
            total_loss = beta_loss_function_pokemon(x_recon, batch, mu, logvar, loss_type=loss_type, beta=beta)#.permute(0, 2, 3, 1)
            # optimization (same 3 steps everytime)
            vae_optim.zero_grad()
            total_loss.backward()
            vae_optim.step()
            # save loss
            batch_total_losses.append(total_loss.data.cpu().item())
            # batch_kl_losses.append(kl)
            # batch_recon_losses.append(recon_loss)
        loss = np.mean(batch_total_losses)
        scheduler.step(loss)
        total_losses.append(loss)
        # kl_losses.append(np.mean(batch_kl_losses))
        # recon_losses.append(np.mean(batch_recon_losses))
        if epoch%50 ==0:
            print("epoch: {}| total_loss {:.5f}| epoch time: {:.3f} sec".format(epoch,total_losses[-1], time.time() - epoch_start_time))
    torch.save(vae.state_dict(), fname)
    # return recon_losses,kl_losses,total_losses
    return total_losses

    
def plot_loss(losses,title):
    plt.plot(losses)
    plt.xlabel('epochs')
    plt.ylabel('loss value')
    plt.title(title)
    plt.show()



def generate_samples(num_of_samples, beta,model,weights_path,epochs = 300, z_dim = 1000):
    #weights_path = "pokemon_cnn_beta_3_vae_300_epochs_dim_7500_loss_type_bce_optimizer_type_Adam.pth"#"/content/drive/MyDrive/pokemon/weights/pokemon_cnn_beta_"+str(beta)+"_vae_"+str(epochs)+"_epochs.pth"
    #weights_path = "pokemon_cnn_beta_"+str(beta)+"_vae_"+str(epochs)+"_epochs_dim_"+str(z_dim)+"_loss_type_"+str(loss_type)+"_optimizer_type_"+str(optimizer_type)+".pth"
    #path = os.path.join("..","data")
    #weights_path = os.path.join(path,weights_path)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    with torch.no_grad():
        samples = model.sample(num_of_samples)
        fig = plt.figure(figsize=(20 ,12))
        for i,sample in enumerate(samples):
            ax = fig.add_subplot(3, 6, i + 1)
            sample = sample.permute(1, 2, 0).data.cpu().numpy()
            ax.imshow(sample)
            ax.set_axis_off()
