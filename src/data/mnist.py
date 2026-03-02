# src/data/mnist.py
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from src.base.base_dataset import JsaDataset


class MNISTDataset(JsaDataset):
    def __init__(self, root: str, train=True):
        self.ds = datasets.MNIST(
            root=root, train=train, download=True, transform=transforms.ToTensor()
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        # x: [1, 28, 28], label: int
        # index: int
        x, label = self.ds[index]
        return {
            self.IMAGE_KEY: x,
            self.LABEL_KEY: label,
            self.INDEX_KEY: index,
        }



class MNISTDataModule(LightningDataModule):
    def __init__(self, root="./data/mnist", batch_size=64, num_workers=4):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full = MNISTDataset(self.root, train=True)
            self.train_set, self.val_set = random_split(full, [50000, 10000])
        if stage == "test" or stage is None:
            self.test_set = MNISTDataset(self.root, train=False)

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
    data_module = MNISTDataModule()
    data_module.setup()
    train_loader = data_module.train_dataloader()
    for batch in train_loader:
        x = batch[JsaDataset.IMAGE_KEY]
        y = batch[JsaDataset.LABEL_KEY]
        idx = batch[JsaDataset.INDEX_KEY]
        print(x.shape, y.shape, idx.shape)
        break

    dataset = MNISTDataset(root="./data/mnist", train=True)
    print(f"Dataset length: {len(dataset)}")
    x, label, index = dataset[0]
    print(f"Sample shape: {x.shape}, Label: {label}, Index: {index}")
