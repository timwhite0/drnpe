import lightning
import torch
import torch.nn.functional as F
from networks import LocationScaleNet
from torch import optim
from torch.distributions import Normal


class Encoder(lightning.LightningModule):
    def __init__(
        self,
        objective: str,
        kl_ball_threshold: float,
        initial_lambda: float,
        x_dim: int,
        num_hidden_channels: int,
        lr: float,
    ):
        super().__init__()

        self.objective = objective
        self.kl_ball_threshold = kl_ball_threshold
        self.initial_lambda = initial_lambda
        self.x_dim = x_dim
        self.num_hidden_channels = num_hidden_channels
        self.lr = lr

        lambda_softplus_inverse = torch.log(
            torch.exp(torch.tensor(self.initial_lambda)) - 1.0
        )
        if objective == "primal":
            self.lambda_softplus_inverse = lambda_softplus_inverse
        elif objective == "dual":
            self.lambda_softplus_inverse = torch.nn.Parameter(lambda_softplus_inverse)

        self.net = LocationScaleNet(
            x_dim=self.x_dim, num_hidden_channels=self.num_hidden_channels
        )

    def compute_loss(self, batch, batch_idx, mode):
        z, x = batch

        mu, log_sigma = self.net(x)
        sigma = torch.exp(log_sigma)

        logq = Normal(mu, sigma).log_prob(z)
        lam = F.softplus(self.lambda_softplus_inverse) + 1e-8

        logw = -(1.0 / lam) * logq

        m = torch.max(logw)
        log_Ew = m + torch.log(torch.mean(torch.exp(logw - m)))

        if self.objective == "primal":
            loss = ((logw - log_Ew).exp() * -logq).mean()
        elif self.objective == "dual":
            loss = lam * (log_Ew + self.kl_ball_threshold)

        self.log(
            f"{mode}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self.compute_loss(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.compute_loss(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        return self.compute_loss(batch, batch_idx, mode="test")

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
