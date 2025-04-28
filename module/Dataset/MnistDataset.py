from pathlib import Path

from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class MnistDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.paths = []
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, in_idx: int):
        img = Image.open(self.paths[in_idx])
        out_tensor = self.preprocess(img)
        out_tensor = (2 * out_tensor) - 1 # Transform tensor to range [-1, 1]

        return out_tensor
    
    def FindData(self, in_dataset_root: str):
        adding_paths = []
        for path in Path(in_dataset_root).rglob('*'):
            if (path.suffix.lower() == '.png'):
                adding_paths.append(str(path))
        assert len(adding_paths) > 0, '[ERROR] No image was found in init_dataset_dir'
        self.paths += adding_paths
