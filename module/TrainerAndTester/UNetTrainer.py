import random
from typing import Dict

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from module.Dataset import *
from module.Model import *

class UNetTrainer:
    def __init__(self, init_dataset_cfg: Dict[str, str], init_model_cfg: Dict[str, str]):
        # --- Setup dataset ---
        dataset_name = init_dataset_cfg['dataset_name']
        dataset_root = init_dataset_cfg['dataset_root']
        self.dataset = globals()[dataset_name]()
        self.dataset.FindData(dataset_root)
        
        # --- Setup model ---
        self.model_info = {}
        self.model_info['model_type'] = globals()[init_model_cfg['model_name']]
        self.model_info['scheduler_type'] = globals()[init_model_cfg['scheduler_name']]
        self.model_info['diffusion_params'] = init_model_cfg['diffusion_params']
        self.model_info['autoencoder_params'] = init_model_cfg['autoencoder_params']
        self.model_info['ldm_params'] = init_model_cfg['ldm_params']
        self.save_pt_dir = init_model_cfg['save_pt_dir']
        self.save_onnx_dir = init_model_cfg['save_onnx_dir']

    def Train(self, in_train_cfg: Dict[str, str]):
        # --- Hyperparameter and setting ---
        lr = in_train_cfg['train']['lr']
        epoch_num = in_train_cfg['train']['epoch_num']
        device = in_train_cfg['train']['device']
        digit_cond_cfg = self.model_info['ldm_params']['condition_config']['digit_condition_config']
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
            self.model_info['autoencoder_params']['z_channels'],
            self.model_info['ldm_params']).to(device)
        scheduler = self.model_info['scheduler_type'](
            num_timesteps = self.model_info['diffusion_params']['num_timesteps'],
            beta_start = self.model_info['diffusion_params']['beta_start'],
            beta_end = self.model_info['diffusion_params']['beta_end'])

        # --- Hyperparameter, optimizer and loss function ---
        optimizer = torch.optim.AdamW(model.parameters(), lr)
        criterion = torch.nn.MSELoss()

        # --- Training Loop ---
        for epoch in range(epoch_num):
            losses = []

            #pbar = tqdm(data_loader, desc=f'Epoch {epoch+1}/{epoch_num}')
            #for batch in pbar:
            for batch in data_loader:
                optimizer.zero_grad()
                latents, conds = batch
                latents = latents.to(device)
                current_batch_size = latents.shape[0]

                # --- Handling conditional input ---
                digit_condition = torch.nn.functional.one_hot(
                    conds['digit_label'],
                    digit_cond_cfg['num_classes']).to(device)
                digit_drop_prob = digit_cond_cfg['cond_drop_prob']
                digit_drop_mask = torch.zeros(
                    (current_batch_size, 1),
                    device=device).float().uniform_(0, 1) > digit_drop_prob
                conds['digit_label'] = digit_condition * digit_drop_mask # Drop condition

                # --- Diffusion ---
                noise = torch.randn_like(latents).to(device)
                t = torch.randint(0, self.model_info['diffusion_params']['num_timesteps'], (current_batch_size,)).to(device)
                noisy_latents = scheduler.add_noise(latents, noise, t)
                pred_noise = model(noisy_latents, t, cond_input=conds)

                # --- Update loss and backpropagation ---
                loss = criterion(pred_noise, noise)
                losses.append(loss.item())
                loss.backward()

                # --- Logging ---
                #pbar.set_postfix({
                #    'Loss': f'{np.mean(losses):.4f}',
                #})
            
                # --- Step the optimizer ---
                optimizer.step()
            print(f'[INFO] Epoch {epoch+1}/{epoch_num} | Loss: {np.mean(losses):.4f}')

        # --- Save checkpoints ---
        torch.save(model.state_dict(), Path(self.save_pt_dir).joinpath('unet_final.pt'))
        print(f'[INFO] Saved model checkpoints at epoch {epoch}')
