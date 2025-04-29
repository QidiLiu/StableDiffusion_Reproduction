import random
from typing import Dict

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from module.Dataset import *
from module.Model import *

class UNetTester:
    def __init__(self, init_dataset_cfg: Dict[str, str], init_model_cfg: Dict[str, str]):
        # --- Setup model ---
        self.model_info = {}
        self.model_info['model_type'] = globals()[init_model_cfg['model_name']]
        self.model_info['scheduler_type'] = globals()[init_model_cfg['scheduler_name']]
        self.model_info['diffusion_params'] = init_model_cfg['diffusion_params']
        self.model_info['autoencoder_params'] = init_model_cfg['autoencoder_params']
        self.model_info['ldm_params'] = init_model_cfg['ldm_params']
        self.save_pt_dir = init_model_cfg['save_pt_dir']
        self.save_onnx_dir = init_model_cfg['save_onnx_dir']

    def Test(self, in_test_cfg: Dict[str, str]):
        # --- Setup seed ---
        seed = in_test_cfg['test']['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # --- Hyperparameter and setting ---
        device = in_test_cfg['test']['device']
        if device.startswith('cuda'):
            torch.cuda.manual_seed_all(seed)
        latent_size = in_test_cfg['img_size'] // (2 ** sum(self.model_info['autoencoder_params']['down_sample']))
        sample_num = in_test_cfg['test']['sample_num']
        sample_grid_row_num = in_test_cfg['test']['sample_grid_row_num']
        save_sample_dir = in_test_cfg['test']['save_sample_dir']
        Path(save_sample_dir).mkdir(parents=True, exist_ok=True)

        # --- Models ---
        model = self.model_info['model_type'](
            self.model_info['autoencoder_params']['z_channels'],
            self.model_info['ldm_params']).to(device)
        model.load_state_dict(torch.load(
            Path(self.save_pt_dir).joinpath('unet_final.pt'),
            device,
            weights_only=True))
        model.eval()
        vae = globals()[in_test_cfg['vae']['type']](
            in_test_cfg['img_channel_num'],
            self.model_info['autoencoder_params']).to(device)
        vae.load_state_dict(torch.load(
            Path(in_test_cfg['vae']['weight_dir']).joinpath('vqvae_autoencoder_final.pt'),
            device,
            weights_only=True))
        vae.eval()
        scheduler = self.model_info['scheduler_type'](
            num_timesteps = self.model_info['diffusion_params']['num_timesteps'],
            beta_start = self.model_info['diffusion_params']['beta_start'],
            beta_end = self.model_info['diffusion_params']['beta_end'])

        # --- Testing loop ---
        with torch.no_grad():
            xt = torch.randn((sample_num,
                            self.model_info['autoencoder_params']['z_channels'],
                            latent_size,
                            latent_size)).to(device)
            for i in tqdm(reversed(range(self.model_info['diffusion_params']['num_timesteps']))):
                # --- Reversed diffusion ---
                pred_noise = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
                xt, _ = scheduler.sample_prev_timestep(xt, pred_noise, torch.as_tensor(i).to(device))

                # --- Decode ---
                imgs = vae.decode(xt)
                imgs = torch.clamp(imgs, -1.0, 1.0).detach().cpu()
                imgs = (imgs + 1) / 2
                sample_grid = torchvision.utils.make_grid(imgs, nrow=sample_grid_row_num)
                sample_grid_img = torchvision.transforms.ToPILImage()(sample_grid)
                sample_grid_img.save(Path(save_sample_dir).joinpath(f'sample_{i}.png'))
                sample_grid_img.close()
