import lightning
import torch
from torch import Tensor
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset, random_split


class GaussianDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        prior_stdev: float,
        z_noise_stdev: float,
        likelihood_stdev: float,
        x_noise_stdev: float,
        num_observations: int,
        num_batches: int,
        batch_size: int,
        train_split: float,
        val_split: float,
    ):
        super().__init__()
        self.prior_stdev = prior_stdev
        self.z_noise_stdev = z_noise_stdev
        self.likelihood_stdev = likelihood_stdev
        self.x_noise_stdev = x_noise_stdev
        self.num_observations = num_observations
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split

    def generate_dataset(self, num_samples, z_noise_stdev, x_noise_stdev):
        z = Normal(0.0, self.prior_stdev).sample([num_samples])
        x = (
            Normal(z, self.likelihood_stdev)
            .sample([self.num_observations])
            .permute(1, 0)
        )
        z_noised = z + z_noise_stdev * torch.randn_like(z)
        x_noised = x + x_noise_stdev * torch.randn_like(x)
        return TensorDataset(Tensor(z_noised), Tensor(x_noised))

    def setup(self, stage: str):
        size_total = self.num_batches * self.batch_size
        size_train = int(self.train_split * size_total)
        size_val = int(self.val_split * size_total)
        size_test = size_total - size_train - size_val

        # Generate training dataset with noise
        self.train_dataset = self.generate_dataset(
            size_train, self.z_noise_stdev, self.x_noise_stdev
        )

        # Generate validation and test datasets without noise
        valtest = self.generate_dataset(
            size_val + size_test, z_noise_stdev=0.0, x_noise_stdev=0.0
        )
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
