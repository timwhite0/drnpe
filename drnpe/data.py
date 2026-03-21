from abc import abstractmethod

import lightning
from torch.utils.data import DataLoader, random_split


class BaseDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        num_batches: int,
        batch_size: int,
        train_split: float,
        val_split: float,
    ):
        super().__init__()
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split

    @abstractmethod
    def generate_dataset(self, num_samples):
        """Generate a TensorDataset of (z, x) pairs.

        Args:
            num_samples: Number of samples to generate

        Returns:
            TensorDataset(z, x)
        """

    def setup(self, stage: str):
        size_total = self.num_batches * self.batch_size
        size_train = int(self.train_split * size_total)
        size_val = int(self.val_split * size_total)
        size_test = size_total - size_train - size_val

        self.train_dataset = self.generate_dataset(size_train)

        valtest = self.generate_dataset(size_val + size_test)
        self.val_dataset, self.test_dataset = random_split(
            valtest, [size_val, size_test]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
