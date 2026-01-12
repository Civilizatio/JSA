# src/data/mnist.py
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from src.base.base_dataset import JsaDataset


class CIFAR10Dataset(JsaDataset):
    def __init__(self, root: str, train=True):
        super().__init__()
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.ds = datasets.CIFAR10(
            root=root,
            train=train,
            download=True,
            transform=transform,
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        # x: [3, 32, 32], label: int
        # index: int
        x, label = self.ds[index]
        return {
            "image": x,
            "label": label,
            "index": index,
        }


class CIFAR10DataModule(LightningDataModule):
    def __init__(self, root="./data/cifar10", batch_size=64, num_workers=4):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full = CIFAR10Dataset(self.root, train=True)
            self.train_set, self.val_set = random_split(full, [45000, 5000])
        if stage == "test" or stage is None:
            self.test_set = CIFAR10Dataset(self.root, train=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True
        )
        


if __name__ == "__main__":
    data_module = CIFAR10DataModule()
    data_module.setup()
    train_loader = data_module.train_dataloader()
    for batch in train_loader:
        x = batch["image"]
        print(x.shape)
        y = batch["label"]
        print(y)
        idx = batch["index"]
        print(idx)
        break
        
    dataset = CIFAR10Dataset(root="./data/cifar10", train=True)
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample shape: {sample['image'].shape}, Label: {sample['label']}, Index: {sample['index']}")