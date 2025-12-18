# src/data/binary_mnist.py
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from src.base.base_dataset import JsaDataset

class BinaryMNISTDataset(JsaDataset):
    def __init__(self, root: str, train=True):
        self.ds = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), lambda x: (x > 0.5).float()]),
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        x, label = self.ds[index]
        return x, label, index
    
class BinaryMNISTDataModule(LightningDataModule):
    def __init__(self, root="./data", batch_size=64, num_workers=4):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full = BinaryMNISTDataset(self.root, train=True)
            self.train_set, self.val_set = random_split(full, [50000, 10000])
        if stage == "test" or stage is None:
            self.test_set = BinaryMNISTDataset(self.root, train=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size, num_workers=self.num_workers
        )


if __name__ == "__main__":
    data_module = BinaryMNISTDataModule()
    data_module.setup()
    train_loader = data_module.train_dataloader()
    for batch in train_loader:
        x, y, idx = batch
        print(x.shape, y.shape, idx.shape)
        break