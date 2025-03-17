import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import lightning as L


class MNISTDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        num_workers,
        transform=transforms.ToTensor(),
        val_split=0.2,
        seed=42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.val_split = val_split
        self.seed = torch.Generator().manual_seed(seed)

    def prepare_data(self):
        # download data
        datasets.MNIST(root=self.data_dir, train=True, download=True)
        datasets.MNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage):
        entire_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            transform=self.transform,
            download=False,
        )

        train_set_size = int(len(entire_dataset) * (1 - self.val_split))
        valid_set_size = len(entire_dataset) - train_set_size

        self.train_set, self.valid_set = data.random_split(
            entire_dataset, [train_set_size, valid_set_size], generator=self.seed
        )

        self.test_set = datasets.MNIST(
            root=self.data_dir,
            train=False,
            transform=self.transform,
            download=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
