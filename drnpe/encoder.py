import lightning
import torch
import torch.nn.functional as F
from networks import ConditionalSplineFlow, LocationScaleNet
from torch import optim
from torch.distributions import Normal


class EncoderNPE(lightning.LightningModule):
    def __init__(self, x_dim: int, num_hidden_channels: int, lr: float):
        super().__init__()
        self.x_dim = x_dim
        self.num_hidden_channels = num_hidden_channels
        self.lr = lr
        self.net = LocationScaleNet(
            x_dim=self.x_dim, num_hidden_channels=self.num_hidden_channels
        )

    def compute_loss(self, batch, batch_idx, mode):
        z, x = batch

        mu, log_sigma = self.net(x)
        sigma = torch.exp(log_sigma)

        logq = Normal(mu, sigma).log_prob(z)

        loss = -logq.mean()

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


class EncoderDRNPE(EncoderNPE):
    def __init__(
        self,
        kl_ball_threshold: float,
        initial_lambda: float,
        x_dim: int,
        z_noise_stdev: float,
        x_noise_stdev: float,
        num_hidden_channels: int,
        lr: float,
        objective: str = "drnpe primal",
    ):
        super().__init__(x_dim=x_dim, num_hidden_channels=num_hidden_channels, lr=lr)
        self.objective = objective
        self.kl_ball_threshold = kl_ball_threshold
        self.initial_lambda = initial_lambda
        self.z_noise_stdev = z_noise_stdev
        self.x_noise_stdev = x_noise_stdev
        lambda_softplus_inverse = torch.log(
            torch.exp(torch.tensor(self.initial_lambda)) - 1.0
        )
        if objective == "drnpe primal":
            self.register_buffer("lambda_softplus_inverse", lambda_softplus_inverse)
        elif objective == "drnpe dual":
            self.lambda_softplus_inverse = torch.nn.Parameter(lambda_softplus_inverse)

    def compute_loss(self, batch, batch_idx, mode):
        z, x = batch
        if mode == "train":
            z = z + self.z_noise_stdev * torch.randn_like(z)
            x = x + self.x_noise_stdev * torch.randn_like(x)

        mu, log_sigma = self.net(x)
        sigma = torch.exp(log_sigma)

        logq = Normal(mu, sigma).log_prob(z)

        lam = F.softplus(self.lambda_softplus_inverse) + 1e-8
        logw = -(1.0 / lam) * logq
        m = torch.max(logw)
        log_Ew = m + torch.log(torch.mean(torch.exp(logw - m)))

        if self.objective == "drnpe primal":
            loss = ((logw - log_Ew).exp() * -logq).mean()
        elif self.objective == "drnpe dual":
            loss = lam * (log_Ew + self.kl_ball_threshold)
            self.log(
                f"{mode}_lambda",
                lam,
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )

        self.log(
            f"{mode}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            f"{mode}_logEw",
            log_Ew,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        return loss


class EncoderNPEFlow(lightning.LightningModule):
    """NPE encoder using Neural Spline Flow for the variational distribution."""

    def __init__(
        self,
        z_dim: int,
        x_dim: int,
        num_hidden_channels: int,
        lr: float,
        num_transforms: int = 8,
        num_bins: int = 8,
        num_blocks: int = 1,
        tail_bound: float = 3.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.z_dim = z_dim
        self.x_dim = x_dim
        self.lr = lr

        self.flow = ConditionalSplineFlow(
            z_dim=z_dim,
            x_dim=x_dim,
            num_hidden_channels=num_hidden_channels,
            num_transforms=num_transforms,
            num_bins=num_bins,
            num_blocks=num_blocks,
            tail_bound=tail_bound,
        )

    def compute_loss(self, batch, batch_idx, mode):
        z, x = batch

        logq = self.flow.log_prob(z, x)

        loss = -logq.mean()

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


class EncoderDRNPEFlow(EncoderNPEFlow):
    """DRNPE encoder using Neural Spline Flow for the variational distribution."""

    def __init__(
        self,
        kl_ball_threshold: float,
        initial_lambda: float,
        z_dim: int,
        x_dim: int,
        z_noise_stdev: float,
        x_noise_stdev: float,
        num_hidden_channels: int,
        lr: float,
        objective: str = "drnpe primal",
        num_transforms: int = 8,
        num_bins: int = 8,
        num_blocks: int = 1,
        tail_bound: float = 25.0,
        init_checkpoint: str = None,
    ):
        super().__init__(
            z_dim=z_dim,
            x_dim=x_dim,
            num_hidden_channels=num_hidden_channels,
            lr=lr,
            num_transforms=num_transforms,
            num_bins=num_bins,
            num_blocks=num_blocks,
            tail_bound=tail_bound,
        )

        self.objective = objective
        self.kl_ball_threshold = kl_ball_threshold
        self.initial_lambda = initial_lambda
        self.z_noise_stdev = z_noise_stdev
        self.x_noise_stdev = x_noise_stdev

        lambda_softplus_inverse = torch.log(
            torch.exp(torch.tensor(self.initial_lambda)) - 1.0
        )
        if objective == "drnpe primal":
            self.register_buffer("lambda_softplus_inverse", lambda_softplus_inverse)
        elif objective == "drnpe dual":
            self.lambda_softplus_inverse = torch.nn.Parameter(lambda_softplus_inverse)

        # Load flow weights from pre-trained NPE checkpoint if provided
        if init_checkpoint is not None:
            self._load_flow_weights(init_checkpoint)

    def _load_flow_weights(self, checkpoint_path: str):
        """Load flow weights from a pre-trained NPE checkpoint."""
        state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
        # Filter to only flow weights (keep the 'flow.' prefix)
        flow_state_dict = {k: v for k, v in state_dict.items() if k.startswith("flow.")}
        print(
            f"Loading {len(flow_state_dict)} flow weights from checkpoint: {checkpoint_path}"
        )
        # Load into self, which has self.flow as the ConditionalSplineFlow
        self.load_state_dict(flow_state_dict, strict=False)

    def compute_loss(self, batch, batch_idx, mode):
        z, x = batch

        if mode == "train":
            z = z + self.z_noise_stdev * torch.randn_like(z)
            x = x + self.x_noise_stdev * torch.randn_like(x)

        logq = self.flow.log_prob(z, x)

        lam = F.softplus(self.lambda_softplus_inverse) + 1e-8
        logw = -(1.0 / lam) * logq
        m = torch.max(logw)
        log_Ew = m + torch.log(torch.mean(torch.exp(logw - m)))

        if self.objective == "drnpe primal":
            loss = ((logw - log_Ew).exp() * -logq).mean()
        elif self.objective == "drnpe dual":
            loss = lam * (log_Ew + self.kl_ball_threshold)
            self.log(
                f"{mode}_lambda",
                lam,
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )

        self.log(
            f"{mode}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            f"{mode}_logEw",
            log_Ew,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        return loss
