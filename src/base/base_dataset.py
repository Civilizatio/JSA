# src/base/base_dataset.py
from torch.utils.data import Dataset
from abc import ABC, abstractmethod


class JsaDataset(Dataset, ABC):
    """Base dataset for JSA framework

    Must return (data, index) or (data, label, index) in __getitem__ method,
    for using cache mechanism in JSA training.
    """
    IMAGE_KEY = "image"
    LABEL_KEY = "label"
    INDEX_KEY = "index"

    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """ Return sample AND index (AND label if supervised) """
        pass
