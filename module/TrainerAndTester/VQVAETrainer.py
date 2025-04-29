import random
from typing import Dict

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from module.Dataset import *
from module.Model import *
from module.TrainerAndTester.util.lpips import LPIPS

class VQVAETrainer:
    def __init__(self, init_dataset_cfg: Dict[str, str], init_model_cfg: Dict[str, str]):
        # --- Setup dataset ---
        dataset_name = init_dataset_cfg['dataset_name']
        dataset_root = init_dataset_cfg['dataset_root']
        self.img_channel_num = init_dataset_cfg['img_channel_num']
        self.dataset = globals()[dataset_name]()
        self.dataset.FindData(dataset_root)
        
        # --- Setup model ---
        self.model_info = {}
        self.model_info['model_type'] = globals()[init_model_cfg['model_name']]
        self.model_info['discriminator_type'] = globals()[init_model_cfg['discriminator_name']]
        self.model_info['autoencoder_params'] = init_model_cfg['autoencoder_params']
        self.save_pt_dir = init_model_cfg['save_pt_dir']
        self.save_onnx_dir = init_model_cfg['save_onnx_dir']

    def Train(self, in_train_cfg: Dict[str, str]):
        # --- Setup seed ---
        seed = in_train_cfg['train']['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # --- Hyperparameter and setting ---
        lr = in_train_cfg['train']['lr']
        epoch_num = in_train_cfg['train']['epoch_num']
        device = in_train_cfg['train']['device']
        if device.startswith('cuda'):
            torch.cuda.manual_seed_all(seed)
        save_period = in_train_cfg['save_period']
        Path(self.save_pt_dir).mkdir(parents=True, exist_ok=True)
        Path(self.save_onnx_dir).mkdir(parents=True, exist_ok=True)
        save_sample_dir = in_train_cfg['train']['save_sample_dir']
        Path(save_sample_dir).mkdir(parents=True, exist_ok=True)

        # --- Data ---
        data_loader_cfg = in_train_cfg['data_loader']
        data_loader = DataLoader(
            self.dataset,
            data_loader_cfg['batch_size'],
            data_loader_cfg['shuffle_mode'],
            num_workers=data_loader_cfg['worker_num'],
            pin_memory=True)

        # --- Models ---
        model = self.model_info['model_type'](
            self.img_channel_num,
            self.model_info['autoencoder_params']).to(device)
        discriminator = self.model_info['discriminator_type'](self.img_channel_num).to(device)

        # --- Hyperparameter, optimizer and loss function ---
        disc_step_start = in_train_cfg['train']['disc_start']
        acc_step_num = in_train_cfg['train']['accumulating_step_num']
        img_save_step_num = in_train_cfg['train']['img_save_step_num']
        optimizer_g = torch.optim.AdamW(model.parameters(), lr, betas=(0.5, 0.999))
        optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr, betas=(0.5, 0.999))
        recon_criterion = torch.nn.MSELoss() # L1/L2 loss for Reconstruction
        disc_criterion = torch.nn.MSELoss() # Disc Loss can even be BCEWithLogits
        lpips_model = LPIPS().eval().to(device) # No need to freeze lpips as lpips.py takes care of that

        # --- Training Loop ---
        step_count = 0
        img_save_count = 0
        for epoch in range(epoch_num):
            recon_losses = []
            codebook_losses = []
            perceptual_losses = []
            disc_losses = []
            gen_losses = []
            losses = []

            optimizer_g.zero_grad()
            optimizer_d.zero_grad()
            pbar = tqdm(data_loader, desc=f'Epoch {epoch+1}/{epoch_num}')
            for batch in pbar:
                step_count += 1
                imgs = batch.to(device)
                current_batch_size = imgs.shape[0]

                # --- Fetch autoencoders output(reconstructions) ---
                model_output = model(imgs)
                output, z, quantize_losses = model_output

                # --- Save sample image
                if step_count % img_save_step_num == 0 or step_count == 1:
                    sample_size = min(8, current_batch_size)
                    save_output = torch.clamp(output[:sample_size], -1., 1.).detach().cpu()
                    save_output = ((save_output + 1) / 2)
                    save_input = ((imgs[:sample_size] + 1) / 2).detach().cpu()
                    
                    grid = torchvision.utils.make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
                    sample_img = torchvision.transforms.ToPILImage()(grid)
                    sample_img.save(Path(save_sample_dir).joinpath(f'autoencoder_sample_{img_save_count}.png'))
                    img_save_count += 1
                    sample_img.close()

                # --- Update loss and backpropagation ---
                
                ######### Optimize Generator ##########
                # L2 Loss
                recon_loss = recon_criterion(output, imgs) 
                recon_losses.append(recon_loss.item())
                recon_loss = recon_loss / acc_step_num
                g_loss = (recon_loss +
                        (in_train_cfg['train']['codebook_weight'] * quantize_losses['codebook_loss'] / acc_step_num) +
                        (in_train_cfg['train']['commitment_beta'] * quantize_losses['commitment_loss'] / acc_step_num))
                codebook_losses.append(in_train_cfg['train']['codebook_weight'] * quantize_losses['codebook_loss'].item())
                # Adversarial loss only if disc_step_start steps passed
                if step_count > disc_step_start:
                    disc_fake_pred = discriminator(model_output[0])
                    disc_fake_loss = disc_criterion(
                        disc_fake_pred,
                        torch.ones(disc_fake_pred.shape, device=disc_fake_pred.device))
                    gen_losses.append(in_train_cfg['train']['disc_weight'] * disc_fake_loss.item())
                    g_loss += in_train_cfg['train']['disc_weight'] * disc_fake_loss / acc_step_num
                lpips_loss = torch.mean(lpips_model(output, imgs))
                perceptual_losses.append(in_train_cfg['train']['perceptual_weight'] * lpips_loss.item())
                g_loss += in_train_cfg['train']['perceptual_weight']*lpips_loss / acc_step_num
                losses.append(g_loss.item())
                g_loss.backward()
                #####################################
                
                ######### Optimize Discriminator #######
                if step_count > disc_step_start:
                    fake = output
                    disc_fake_pred = discriminator(fake.detach())
                    disc_real_pred = discriminator(imgs)
                    disc_fake_loss = disc_criterion(
                        disc_fake_pred,
                        torch.zeros(disc_fake_pred.shape, device=disc_fake_pred.device))
                    disc_real_loss = disc_criterion(
                        disc_real_pred,
                        torch.ones(disc_real_pred.shape, device=disc_real_pred.device))
                    disc_loss = in_train_cfg['train']['disc_weight'] * (disc_fake_loss + disc_real_loss) / 2
                    disc_losses.append(disc_loss.item())
                    disc_loss = disc_loss / acc_step_num
                    disc_loss.backward()
                    if step_count % acc_step_num == 0:
                        optimizer_d.step()
                        optimizer_d.zero_grad()
                #####################################
                
                if step_count % acc_step_num == 0:
                    optimizer_g.step()
                    optimizer_g.zero_grad()

                # --- Logging ---
                if len(disc_losses) > 0:
                    pbar.set_postfix({
                        'Recon Loss': f'{np.mean(recon_losses):.4f}',
                        'Perceptual Loss' : f'{np.mean(perceptual_losses):.4f}',
                        'Codebook Loss': f'{np.mean(codebook_losses):.4f}',
                        'G Loss': f'{np.mean(gen_losses):.4f}',
                        'D Loss': f'{np.mean(disc_losses):.4f}'
                    })
                else:
                    pbar.set_postfix({
                        'Recon Loss': f'{np.mean(recon_losses):.4f}',
                        'Perceptual Loss' : f'{np.mean(perceptual_losses):.4f}',
                        'Codebook Loss': f'{np.mean(codebook_losses):.4f}',
                    })
            
            # --- Step the optimizer ---
            optimizer_d.step()
            optimizer_d.zero_grad()
            optimizer_g.step()
            optimizer_g.zero_grad()

        # --- Save checkpoints ---
        torch.save(model.state_dict(), Path(self.save_pt_dir).joinpath('vqvae_autoencoder_final.pt'))
        torch.save(discriminator.state_dict(), Path(self.save_pt_dir).joinpath('vqvae_discriminator_final.pt'))
        print(f'[INFO] Saved model checkpoints at epoch {epoch}')
