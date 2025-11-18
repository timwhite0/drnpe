from torch import nn


class LocationScaleNet(nn.Module):
    def __init__(
        self,
        x_dim: int,
        num_hidden_channels: int,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(x_dim, num_hidden_channels),
            nn.ReLU(),
            nn.Linear(num_hidden_channels, num_hidden_channels),
            nn.ReLU(),
            nn.Linear(num_hidden_channels, 2),
        )

    def forward(self, x):
        params = self.net(x)
        mean = params[:, 0]
        log_std = params[:, 1].clamp(-8.0, 8.0)
        return mean, log_std
