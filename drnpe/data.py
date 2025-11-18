import lightning
from torch import Tensor
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset, random_split


class GaussianDataModule(lightning.LightningDataModule):
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
        super().__init__()
        self.prior_stdev = prior_stdev
        self.likelihood_stdev = likelihood_stdev
        self.num_observations = num_observations
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split

    def generate_dataset(self):
        z = Normal(0.0, self.prior_stdev).sample([self.num_batches * self.batch_size])
        x = (
            Normal(z, self.likelihood_stdev)
            .sample([self.num_observations])
            .permute(1, 0)
        )
        return TensorDataset(Tensor(z), Tensor(x))

    def setup(self, stage: str):
        dataset = self.generate_dataset()

        size_total = len(dataset)
        size_train = int(self.train_split * size_total)
        size_val = int(self.val_split * size_total)
        size_test = size_total - size_train - size_val

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [size_train, size_val, size_test]
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
