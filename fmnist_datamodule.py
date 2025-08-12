from lightning.pytorch import LightningDataModule
from pathlib import Path
from torch.utils.data import DataLoader
from fmnist_dataset import load_dataset, download_dataset
from torchvision.transforms.v2 import Compose, Transform
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class FashionMNISTDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: Path,
                 train_transform: Compose | Transform | None = None,
                 val_test_transform: Compose | Transform | None = None,
                 train_batch_size: int = 32,
                 train_num_workers: int = 4,
                 val_batch_size: int = 32,
                 val_num_workers: int = 4,
                 test_batch_size: int = 64,
                 test_num_workers: int = 4,
                 split_seed: int = 33):
        super().__init__()
        self.data_dir = data_dir
        self.train_transform = train_transform
        self.val_test_transform = val_test_transform
        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.val_batch_size = val_batch_size
        self.val_num_workers = val_num_workers
        self.test_batch_size = test_batch_size
        self.test_num_workers = test_num_workers
        self.split_seed = split_seed

    def setup(self, stage=None):
        if not self.train_transform or not self.val_test_transform:
            config = resolve_data_config({}, model=self.trainer.model.model)
            default_transform = create_transform(**config)
            self.train_transform = default_transform
            self.val_test_transform = default_transform
        self.datasets = load_dataset(self.data_dir,
                                     train_transforms=self.train_transform,
                                     val_test_transforms=self.val_test_transform,
                                     split_seed=self.split_seed)
        
    def prepare_data(self):
        download_dataset(self.data_dir)

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            self.train_batch_size,
            shuffle=True,
            num_workers=self.train_num_workers
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.datasets["val"],
            self.val_batch_size,
            shuffle=False,
            num_workers=self.val_num_workers
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            self.test_batch_size,
            shuffle=False,
            num_workers=self.test_num_workers
        )
