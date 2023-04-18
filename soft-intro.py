import torch
# imports for the tutorial
import os
import time
import numpy as np
import random
#import matplotlib.pyplot as plt

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import vae_utils
import our_datasets


def reparameterize(mu, logvar):
    """
    This function applies the reparameterization trick:
    z = mu(X) + sigma(X)^0.5 * epsilon, where epsilon ~ N(0,I)
    :param mu: mean of x
    :param logvar: log variaance of x
    :return z: the sampled latent variable
    """
    device = mu.device
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std).to(device)
    return mu + eps * std

def calc_reconstruction_loss(x, recon_x, loss_type='mse', reduction='sum'):
    """

    :param x: original inputs
    :param recon_x:  reconstruction of the VAE's input
    :param loss_type: "mse", "l1", "bce", "gaussian"
    :param reduction: "sum", "mean", "none"
    :return: recon_loss
    """
    recon_x = recon_x.view(x.size(0), -1)
    x = x.view(x.size(0), -1)
    if reduction not in ['sum', 'mean', 'none']:
        raise NotImplementedError
    if loss_type == 'mse':
        recon_error = F.mse_loss(recon_x, x, reduction='none')
        recon_error = recon_error.sum(1)
        if reduction == 'sum':
            recon_error = recon_error.sum()
        elif reduction == 'mean':
            recon_error = recon_error.mean()
    elif loss_type == 'l1':
        recon_error = F.l1_loss(recon_x, x, reduction=reduction)
    elif loss_type == 'bce':
        recon_error = F.binary_cross_entropy(recon_x, x, reduction=reduction)
    else:
        raise NotImplementedError
    return recon_error


def calc_kl(logvar, mu, mu_o=10, is_outlier=False, reduce='sum'):
    """
    Calculate kl-divergence
    :param logvar: log-variance from the encoder
    :param mu: mean from the encoder
    :param mu_o: negative mean for outliers (hyper-parameter)
    :param is_outlier: if True, calculates with mu_neg
    :param reduce: type of reduce: 'sum', 'none'
    :return: kld
    """
    if is_outlier:
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp() + 2 * mu * mu_o - mu_o.pow(2)).sum(1)
    else:
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
    if reduce == 'sum':
        kl = torch.sum(kl)
    elif reduce == 'mean':
        kl = torch.mean(kl)
    return kl



def train_soft_intro_vae_toy(images_root,transform,z_dim=2, lr_e=2e-4, lr_d=2e-4, batch_size=32, n_iter=30000, num_vae=0, 
                             save_interval=1, recon_loss_type="mse", beta_kl=1.0, beta_rec=1.0,
                             beta_neg=1.0, test_iter=5000, seed=-1, pretrained=None, scale=1,
                             device=torch.device("cpu"), dataset="8Gaussians", gamma_r=1e-8):
    # if seed != -1:
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     torch.backends.cudnn.deterministic = True
    #     print("random seed: ", seed)

    # --------------build models -------------------------
    train_set = our_datasets.Pokemon_dataset(images_root,transform)# ToyDataset(distr=dataset) # TODO complete
    scale *= train_set.range  # the scale of the 2d grid ([-1, 1] for Gaussians, [-2, 2] for the rest)

    model = vae_utils.ProjectedVAE()# TODO complete #SoftIntroVAESimple(x_dim=2, zdim=z_dim, n_layers=3, num_hidden=256).to(device)
    if pretrained is not None:
        load_model(model, pretrained)
    print(model)

    optimizer_e = optim.Adam(model.encoder.parameters(), lr=lr_e)
    optimizer_d = optim.Adam(model.decoder.parameters(), lr=lr_d)

    milestones = (10000, 15000)
    e_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_e, milestones=milestones, gamma=0.1)
    d_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=milestones, gamma=0.1)

    start_time = time.time()
    dim_scale = 0.5  # normalizing factor, 's' in the paper

    for it in range(n_iter):
        batch = train_set.next_batch(batch_size=batch_size, device=device)
        # save models
        if it % save_interval == 0 and it > 0:
            save_epoch = (it // save_interval) * save_interval
            save_checkpoint(model, save_epoch, it, '')

        model.train()
        # --------------train----------------
        if it < num_vae:
            # vanilla VAE training, optimizeing the ELBO for both encoder and decoder
            batch_size = batch.size(0)

            real_batch = batch.to(device)

            # =========== Update E, D ================
            real_mu, real_logvar, z, rec = model(real_batch)

            loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction="mean")
            loss_kl = calc_kl(real_logvar, real_mu, reduce="mean")
            loss = beta_rec * loss_rec + beta_kl * loss_kl

            optimizer_e.zero_grad()
            optimizer_d.zero_grad()
            loss.backward()
            optimizer_e.step()
            optimizer_d.step()

            if it % test_iter == 0:
                info = "\nIter: {}/{} : time: {:4.4f}: ".format(it, n_iter, time.time() - start_time)
                info += 'Rec: {:.4f}, KL: {:.4f} '.format(loss_rec.data.cpu(), loss_kl.data.cpu())
                print(info)
        else:
            # soft-intro-vae training
            if len(batch.size()) == 3:
                batch = batch.unsqueeze(0)

            b_size = batch.size(0)

            # generate random noise to produce 'fake' later
            noise_batch = torch.randn(size=(b_size, z_dim)).to(device)
            real_batch = batch.to(device)

            # =========== Update E ================
            for param in model.encoder.parameters():
                param.requires_grad = True
            for param in model.decoder.parameters():
                param.requires_grad = False

            # generate 'fake' data
            fake = model.sample(noise_batch)
            # optimize for real data
            real_mu, real_logvar = model.encode(real_batch)
            z = reparameterize(real_mu, real_logvar)
            rec = model.decoder(z)  # reconstruction
            # we also want to see what is the reconstruction error from mu
            _, _, _, rec_det = model(real_batch, deterministic=True)

            loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction="mean")
            # reconstruction error from mu (not optimized, only to observe)
            loss_rec_det = calc_reconstruction_loss(real_batch, rec_det.detach(), loss_type=recon_loss_type,
                                                    reduction="mean")

            # KLD loss for the real data
            lossE_real_kl = calc_kl(real_logvar, real_mu, reduce="mean")

            # prepare the fake data for the expELBO
            fake_mu, fake_logvar, z_fake, rec_fake = model(fake.detach())
            # we also consider the reconstructions as 'fake' data, as they are output of the decoder
            rec_mu, rec_logvar, z_rec, rec_rec = model(rec.detach())
            
            # KLD loss for the fake data
            fake_kl_e = calc_kl(fake_logvar, fake_mu, reduce="none")
            rec_kl_e = calc_kl(rec_logvar, rec_mu, reduce="none")
            
            # reconstruction loss for the fake data
            loss_fake_rec = calc_reconstruction_loss(fake, rec_fake, loss_type=recon_loss_type, reduction="none")
            loss_rec_rec = calc_reconstruction_loss(rec, rec_rec, loss_type=recon_loss_type, reduction="none")
            
            # expELBO
            exp_elbo_fake = (-2 * dim_scale * (beta_rec * loss_fake_rec + beta_neg * fake_kl_e)).exp().mean()
            exp_elbo_rec = (-2 * dim_scale * (beta_rec * loss_rec_rec + beta_neg * rec_kl_e)).exp().mean()
            
            # total loss
            lossE = dim_scale * (beta_kl * lossE_real_kl + beta_rec * loss_rec) + 0.25 * (exp_elbo_fake + exp_elbo_rec)
            
            optimizer_e.zero_grad()
            lossE.backward()
            optimizer_e.step()

            # ========= Update D ==================
            for param in model.encoder.parameters():
                param.requires_grad = False
            for param in model.decoder.parameters():
                param.requires_grad = True

            # generate fake
            fake = model.sample(noise_batch)
            rec = model.decoder(z.detach())
            # ELBO loss for real -- just the reconstruction, KLD for real doesn't affect the decoder
            loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction="mean")
            
            # prepare fake data for ELBO
            rec_mu, rec_logvar = model.encode(rec)
            z_rec = reparameterize(rec_mu, rec_logvar)
            
            fake_mu, fake_logvar = model.encode(fake)
            z_fake = reparameterize(fake_mu, fake_logvar)

            rec_rec = model.decode(z_rec.detach())
            rec_fake = model.decode(z_fake.detach())

            loss_rec_rec = calc_reconstruction_loss(rec.detach(), rec_rec, loss_type=recon_loss_type, reduction="mean")
            loss_rec_fake = calc_reconstruction_loss(fake.detach(), rec_fake, loss_type=recon_loss_type, reduction="mean")

            fake_kl = calc_kl(fake_logvar, fake_mu, reduce="mean")
            rec_kl = calc_kl(rec_logvar, rec_mu, reduce="mean")

            lossD = beta_rec * loss_rec + 0.5 * beta_kl * (fake_kl + rec_kl) + \
                            gamma_r * 0.5 * beta_rec * (loss_rec_rec + loss_rec_fake)
            lossD = dim_scale * lossD

            optimizer_d.zero_grad()
            lossD.backward()
            optimizer_d.step()
            
            if it % test_iter == 0:
                info = "\nIter: {}/{} : time: {:4.4f}: ".format(it, n_iter, time.time() - start_time)

                info += 'Rec: {:.4f} ({:.4f}), '.format(loss_rec.data.cpu(), loss_rec_det.data.cpu())
                info += 'Kl_E: {:.4f}, expELBO_R: {:.4f}, expELBO_F: {:.4f}, '.format(lossE_real_kl.data.cpu(),
                                                                                exp_elbo_rec.data.cpu(),
                                                                                exp_elbo_fake.cpu())
                info += 'Kl_F: {:.4f}, KL_R: {:.4f},'.format(fake_kl.data.cpu(), rec_kl.data.cpu())
                info += ' DIFF_Kl_F: {:.4f}'.format(-lossE_real_kl.data.cpu() + fake_kl.data.cpu())

                print(info)

            if torch.isnan(lossE) or torch.isnan(lossD):
                #plt.close('all')
                raise SystemError("loss is NaN.")
        e_scheduler.step()
        d_scheduler.step()

        if it % test_iter == 0 and it > 0 or it == n_iter - 1:
            print("plotting...")
            model.eval()
            #fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            noise_batch = torch.randn(size=(1024, z_dim)).to(device)
            plot_fake_batch = model.sample(noise_batch)
            plot_fake_batch = plot_fake_batch.data.cpu().numpy()
            #ax.scatter(plot_fake_batch[:, 0], plot_fake_batch[:, 1], s=8, c='g', label="fake")
            #ax.set_xlim((-scale * 2, scale * 2))
            #ax.set_ylim((-scale * 2, scale * 2))
            #ax.set_axis_off()
            # f_name = dataset + "_bkl_" + str(beta_kl) + "_bneg_" + str(beta_neg) + "_brec_" + str(
            #         beta_rec) + "_seed_" + str(seed) + "_iter_" + str(it) + ".png"
            #plt.savefig(f_name, bbox_inches='tight')
            #plt.close()
            if it == n_iter - 1:
                # f_name = dataset + "_bkl_" + str(beta_kl) + "_bneg_" + str(beta_neg) + "_brec_" + str(
                #         beta_rec) + "_seed_" + str(seed) + "_iter_" + str(it) + "_real.png"
            #    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                plot_batch = train_set.next_batch(batch_size=1024, device=device)
                plot_batch = plot_batch.data.cpu().numpy()
                # ax.scatter(plot_batch[:, 0], plot_batch[:, 1], s=8, label="true dist")
                # ax.set_xlim((-scale * 2, scale * 2))
                # ax.set_ylim((-scale * 2, scale * 2))
                # ax.set_axis_off()
                # plt.savefig(f_name, bbox_inches='tight')
                # plt.close()
                # f_name = dataset + "_bkl_" + str(beta_kl) + "_bneg_" + str(beta_neg) + "_brec_" + str(
                #         beta_rec) + "_seed_" + str(seed) + "_iter_" + str(it) + ".png"
                # print("plotting density...")
            #    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                # test_grid = setup_grid(range_lim=scale * 2, n_pts=1024, device=torch.device('cpu'))
                # plot_vae_density(model, ax, test_grid, n_pts=1024, batch_size=256, colorbar=False,
                #                  beta_kl=1.0, beta_recon=1.0, set_title=False, device=device)
                # ax.set_axis_off()
                # f_name = "density_" + f_name
                # plt.savefig(f_name, bbox_inches='tight')
                # plt.close()
            model.train()
    # plot_samples_density(train_set, model, scale, device)
    # plt.show()
    return model