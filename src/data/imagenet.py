# src/data/imagenet.py
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from src.base.base_dataset import JsaDataset

import os
from dotenv import load_dotenv
load_dotenv()

DATASET_KEY = {
    "image_key": "image",
    "index_key": "index",
    "label_key": "label",
}
TRAINING_DATASET_PATH = os.getenv("IMAGENET_TRAINING_DATASET_PATH", "./data/imagenet/train")
VALIDATION_DATASET_PATH = os.getenv("IMAGENET_VALIDATION_DATASET_PATH", "./data/imagenet/val") 

def check_path(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"The dataset path for {name} does not exist: {path}. "
            f"Please set the environment variable IMAGENET_{name.upper()}_DATASET_PATH to the correct path." 
        )

class ImageNetDataset(JsaDataset):
    def __init__(self, root, transform=None):
        super().__init__()
        check_path(root, "training" if root == TRAINING_DATASET_PATH else "validation")
        self.ds = ImageFolder(
            root=root,
            transform=transform,
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        # x: [3, 224, 224], label: int
        # index: int
        x, label = self.ds[index]
        return {
            "image": x,
            "label": label,
            "index": index,
        }

class ImageNetDataModule(LightningDataModule):
    def __init__(self, batch_size=64, num_workers=4,image_size=256,random_crop=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.random_crop = random_crop

    def setup(self, stage=None):
        transform_train = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.RandomCrop(self.image_size) if self.random_crop else transforms.CenterCrop(self.image_size),
                # transforms.RandomHorizontalFlip(), # Maybe add this later
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        transform_val = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        if stage == "fit" or stage is None:
            self.train_set = ImageNetDataset(TRAINING_DATASET_PATH, transform=transform_train)
            self.val_set = ImageNetDataset(VALIDATION_DATASET_PATH, transform=transform_val)
        if stage == "test" or stage is None:
            self.test_set = ImageNetDataset(VALIDATION_DATASET_PATH, transform=transform_val)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        
if __name__ == "__main__":
    dm = ImageNetDataModule(batch_size=64, num_workers=4)
    dm.setup()
    train_dataset = dm.train_set
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of classes: {len(train_dataset.ds.classes)}")
    train_loader = dm.train_dataloader()
    for batch in train_loader:
        print(batch["image"].shape)  # [64, 3, 256, 256]
        print(batch["label"].shape)  # [64]
        print(batch["index"].shape)  # [64]
        break