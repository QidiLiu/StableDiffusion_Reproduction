import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset

class MnistLatentDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.paths = []

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, in_idx: int):
        out_tensor = pickle.load(open(self.paths[in_idx], 'rb'))

        return out_tensor
    
    def FindData(self, in_dataset_root: str):
        adding_paths = []
        for path in Path(in_dataset_root).rglob('*'):
            if (path.suffix.lower() == '.pkl'):
                adding_paths.append(str(path))
        assert len(adding_paths) > 0, '[ERROR] No image was found in init_dataset_dir'
        self.paths += adding_paths
