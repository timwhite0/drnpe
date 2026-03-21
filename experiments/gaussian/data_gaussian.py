import torch
from data import BaseDataModule
from torch import Tensor
from torch.distributions import Normal
from torch.utils.data import TensorDataset


class GaussianDataModule(BaseDataModule):
    def __init__(
        self,
        prior_stdev: float,
        likelihood_stdev: float,
        num_observations: int,
        num_batches: int,
        batch_size: int,
        train_split: float,
        val_split: float,
    ):
        super().__init__(
            num_batches=num_batches,
            batch_size=batch_size,
            train_split=train_split,
            val_split=val_split,
        )
        self.prior_stdev = prior_stdev
        self.likelihood_stdev = likelihood_stdev
        self.num_observations = num_observations

    def generate_dataset(self, num_samples):
        z = Normal(0.0, self.prior_stdev).sample([num_samples])
        x_raw = (
            Normal(z, self.likelihood_stdev)
            .sample([self.num_observations])
            .permute(1, 0)
        )
        # Compute sufficient statistics: sample mean and biased sample variance
        sample_mean = x_raw.mean(dim=1, keepdim=True)
        sample_var = x_raw.var(dim=1, unbiased=False, keepdim=True)
        x = torch.cat([sample_mean, sample_var], dim=1)
        return TensorDataset(Tensor(z), Tensor(x))
