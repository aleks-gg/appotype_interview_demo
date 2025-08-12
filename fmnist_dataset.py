from torchvision.datasets import FashionMNIST
from pathlib import Path
from torch.utils.data import Subset
from torch.utils.data import Dataset
import pandas as pd
from torchvision.transforms.v2 import Compose, Transform


from sklearn.model_selection import train_test_split

def load_dataset(data_dir: Path,
                 train_transforms: Compose | Transform | None = None,
                 val_test_transforms: Compose | Transform | None = None,
                 split_seed: int = 33) -> dict[str, Dataset]:
    """
    Loads previously downloaded FashionMNIST dataset.
    Splits the train dataset into train and val, saves the split indices and loads them on subsequent runs.

    Args:
        data_dir (Path): Path to the dataset directory.
        train_transforms (Compose | Transform): Transform to be applied to the training data.
        val_test_transforms (Compose | Transform): Transform to be applied to the validation and test data.
        split_seed (int): Random seed for train/val split.

    Returns:
        dict[str, Dataset]: Return a dictionary containing the train, validation, and test datasets.
    """
    train_ds_presplit = FashionMNIST(data_dir, train=True, download=False, transform=train_transforms)
    val_ds_presplit = FashionMNIST(data_dir, train=True, download=False, transform=val_test_transforms)
    
    split_cache_path = data_dir / "splits.csv"

    if split_cache_path.exists():
        df = pd.read_csv(split_cache_path)
        train_indices = df[df['split'] == 'train']['index'].tolist()
        val_indices = df[df['split'] == 'val']['index'].tolist()
        
        train_ds = Subset(train_ds_presplit, train_indices)
        val_ds = Subset(val_ds_presplit, val_indices)
    else:
        indices = list(range(len(train_ds_presplit)))
        train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=split_seed)
        
        train_ds = Subset(train_ds_presplit, train_indices)
        val_ds = Subset(val_ds_presplit, val_indices)
        
        df_train = pd.DataFrame({'index': train_indices, 'split': 'train'})
        df_val = pd.DataFrame({'index': val_indices, 'split': 'val'})
        df_splits = pd.concat([df_train, df_val])
        df_splits.to_csv(split_cache_path, index=False)

    test_ds = FashionMNIST(data_dir, train=False, download=False, transform=val_test_transforms)
    return {"train": train_ds, "test": test_ds, "val": val_ds}


def download_dataset(data_dir: Path) -> None:
    """Downloads the FashionMNIST dataset for further use."""
    FashionMNIST(data_dir, train=True, download=True)
    FashionMNIST(data_dir, train=False, download=True)


if __name__ == "__main__":
    ds_dir = Path(__file__).parent / "dataset"
    ds_dir.mkdir(parents=True, exist_ok=True)
    download_dataset(ds_dir)
    datasets = load_dataset(ds_dir)
