import pickle
from typing import Dict

import torch
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from module.Dataset import *
from module.Model import *

class VQVAETester:
    def __init__(self, init_dataset_cfg: Dict[str, str], init_model_cfg: Dict[str, str]):
        # --- Setup dataset ---
        self.dataset_root = init_dataset_cfg['dataset_root']
        self.img_channel_num = init_dataset_cfg['img_channel_num']
        
        # --- Setup model ---
        self.model_info = {}
        self.model_info['model_type'] = globals()[init_model_cfg['model_name']]
        self.model_info['autoencoder_params'] = init_model_cfg['autoencoder_params']
        self.save_pt_dir = init_model_cfg['save_pt_dir']

    def Test(self, in_test_cfg: Dict[str, str]):
        # --- Hyperparameter and setting ---
        device = in_test_cfg['test']['device']
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2 * x - 1), # Transform tensor to range [-1, 1]
            transforms.Lambda(lambda x: x.unsqueeze(0)),
        ])

        # --- Models ---
        model = self.model_info['model_type'](
            self.img_channel_num,
            self.model_info['autoencoder_params']).to(device)
        model.load_state_dict(torch.load(
            Path(self.save_pt_dir).joinpath('vqvae_autoencoder_final.pt'),
            device,
            weights_only=True))
        model.eval()

        # --- Testing Loop ---
        with torch.no_grad():
            for path in tqdm(Path(self.dataset_root).rglob('*')):
                if (path.suffix.lower() == '.png'):
                    latent_pathobj = Path(str(path).replace('mnist', 'mnist_latent'))
                    latent_pathobj = latent_pathobj.with_suffix('.pkl')
                    Path(latent_pathobj.parent).mkdir(parents=True, exist_ok=True)
                    img = Image.open(path)
                    tensor = preprocess(img)
                    encoded_output, _ = model.encode(tensor.to(device))
                    pickle.dump(encoded_output.squeeze(0).cpu(), open(latent_pathobj, 'wb'))


