import torch
from nflows import distributions, flows, transforms
from torch import nn


class LocationScaleNet(nn.Module):
    def __init__(
        self,
        x_dim: int,
        num_hidden_channels: int,
        z_dim: int = 1,
    ):
        super().__init__()
        self.z_dim = z_dim

        self.net = nn.Sequential(
            nn.Linear(x_dim, num_hidden_channels),
            nn.SiLU(),
            nn.Linear(num_hidden_channels, num_hidden_channels),
            nn.SiLU(),
            nn.Linear(num_hidden_channels, num_hidden_channels),
            nn.SiLU(),
            nn.Linear(num_hidden_channels, 2 * z_dim),
        )

    def forward(self, x):
        params = self.net(x)
        mean = params[:, : self.z_dim]
        log_std = params[:, self.z_dim :].clamp(-10.0, 10.0)
        if self.z_dim == 1:
            mean = mean.squeeze(-1)
            log_std = log_std.squeeze(-1)
        return mean, log_std


class ConditionalSplineFlow(nn.Module):
    """Conditional Neural Spline Flow for density estimation.

    Given conditioning variable x, models the distribution q(z|x) using
    a normalizing flow with rational quadratic spline transforms.
    """

    def __init__(
        self,
        z_dim: int,
        x_dim: int,
        num_hidden_channels: int = 50,
        num_transforms: int = 8,
        num_bins: int = 8,
        num_blocks: int = 1,
        tail_bound: float = 25.0,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim

        transform_list = []
        for _ in range(num_transforms):
            transform_list.append(
                transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=z_dim,
                    hidden_features=num_hidden_channels,
                    context_features=x_dim,
                    num_bins=num_bins,
                    tails="linear",
                    tail_bound=tail_bound,
                    num_blocks=num_blocks,
                    use_residual_blocks=False,
                    random_mask=False,
                    activation=torch.nn.functional.relu,
                    use_batch_norm=False,
                )
            )

        self.transform = transforms.CompositeTransform(transform_list)
        self.base_distribution = distributions.StandardNormal(shape=[z_dim])
        self.flow = flows.Flow(
            transform=self.transform, distribution=self.base_distribution
        )

    def log_prob(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute log probability of z given x.

        Args:
            z: Target variable, shape [batch_size] or [batch_size, z_dim]
            x: Conditioning variable, shape [batch_size, x_dim]

        Returns:
            log_prob: Log probability, shape [batch_size]
        """
        if z.dim() == 1:
            z = z.unsqueeze(-1)
        return self.flow.log_prob(z, context=x)

    def sample(self, num_samples: int, x: torch.Tensor) -> torch.Tensor:
        """Sample from the conditional distribution q(z|x).

        Args:
            num_samples: Number of samples per conditioning value
            x: Conditioning variable, shape [batch_size, x_dim]

        Returns:
            samples: Shape [batch_size, num_samples, z_dim]
        """
        return self.flow.sample(num_samples, context=x)
