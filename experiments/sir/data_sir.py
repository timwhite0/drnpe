import torch
from data import BaseDataModule
from torch import Tensor
from torch.utils.data import TensorDataset


def _simulate_sir_batch(
    beta,
    gamma,
    population_size,
    num_days,
    eta,
    sigma,
    dt,
    device="cpu",
):
    """Simulate stochastic SIR with time-varying R0 (Ward et al., 2022).

    Vectorized over the batch dimension using PyTorch for GPU acceleration.

    The infection rate β̃(t) is derived from a stochastic R0(t) process:
        dR0(t) = η(β/γ - R0(t))dt + σ√R0(t)dW(t)

    The SIR ODEs are integrated with Euler steps:
        dS/dt = -β̃(t)*S*I
        dI/dt = β̃(t)*S*I - γ*I
        dR/dt = γ*I

    Args:
        beta: Tensor of shape [batch_size], infection rate parameters
        gamma: Tensor of shape [batch_size], recovery rate parameters
        population_size: Total population N
        num_days: Number of days to simulate
        eta: Mean reversion strength for R0
        sigma: Volatility of R0
        dt: Euler-Maruyama time step
        device: torch device

    Returns:
        daily_infections: Tensor of shape [batch_size, num_days]
    """
    batch_size = beta.shape[0]
    pop = population_size
    num_steps = int(num_days / dt)
    steps_per_day = int(1.0 / dt)
    sqrt_dt = dt**0.5

    beta = beta.to(device)
    gamma = gamma.to(device)

    sus = torch.full((batch_size,), 1.0 - 1.0 / pop, device=device)
    inf = torch.full((batch_size,), 1.0 / pop, device=device)
    r0_t = beta / gamma
    r0_mean = beta / gamma  # mean-reversion target

    daily_infections = torch.zeros(batch_size, num_days, device=device)
    daily_new_infections = torch.zeros(batch_size, device=device)
    day = 0

    for step in range(num_steps):
        # SDE for R0(t): mean-reverting CIR-like process
        dw = torch.randn(batch_size, device=device) * sqrt_dt
        r0_t = (
            r0_t
            + eta * (r0_mean - r0_t) * dt
            + sigma * torch.sqrt(r0_t.clamp(min=0.0)) * dw
        )
        r0_t = r0_t.clamp(min=0.0)

        # Time-varying infection rate
        beta_t = r0_t * gamma

        # SIR ODE step
        new_infections = beta_t * sus * inf * dt
        new_recoveries = gamma * inf * dt

        sus = (sus - new_infections).clamp(min=0.0)
        inf = (inf + new_infections - new_recoveries).clamp(min=0.0)

        daily_new_infections = daily_new_infections + new_infections * pop

        # Record daily counts
        if (step + 1) % steps_per_day == 0 and day < num_days:
            daily_infections[:, day] = daily_new_infections
            daily_new_infections = torch.zeros(batch_size, device=device)
            day += 1

    return daily_infections


def _apply_reporting_delays(daily_infections, delay_fraction):
    """Apply weekend reporting delays (Ward et al. misspecification).

    Weekend (Sat=day 5, Sun=day 6 in 0-indexed week) counts are reduced
    by delay_fraction, and the deficit is recouped on Monday (day 0).
    First day is Monday. Vectorized over batch dimension.

    Args:
        daily_infections: Tensor [batch_size, num_days]
        delay_fraction: fraction of weekend counts to delay (e.g. 0.05)

    Returns:
        Modified daily infection counts [batch_size, num_days]
    """
    y = daily_infections.clone()
    num_days = y.shape[1]

    for d in range(num_days):
        day_of_week = d % 7  # 0=Mon, 5=Sat, 6=Sun
        if day_of_week in (5, 6):
            deficit = y[:, d] * delay_fraction
            y[:, d] -= deficit
            next_monday = d + (7 - day_of_week)
            if next_monday < num_days:
                y[:, next_monday] += deficit

    return y


def _compute_summary_stats(y):
    """Compute 6 summary statistics from daily infection counts (Ward et al.).

    Stats:
        1. Mean daily infections
        2. Median daily infections
        3. Max daily infections
        4. Max Day (day of max infections)
        5. Half Day (day cumulative infections reach half of total)
        6. Autocorrelation at lag 1

    Args:
        y: Tensor of shape [batch_size, num_days]

    Returns:
        Tensor of shape [batch_size, 6], standardized.
    """
    mean = y.mean(dim=1)
    median = y.median(dim=1).values
    peak = y.max(dim=1).values
    max_day = y.argmax(dim=1).float()

    # Half Day: day at which cumulative infections reach half of total
    cumsum = y.cumsum(dim=1)
    half_total = cumsum[:, -1:] / 2.0
    half_day = (cumsum >= half_total).int().argmax(dim=1).float()

    # Autocorrelation at lag 1
    y_centered = y - y.mean(dim=1, keepdim=True)
    autocov = (y_centered[:, :-1] * y_centered[:, 1:]).mean(dim=1)
    variance = y_centered.pow(2).mean(dim=1)
    autocorr = autocov / (variance + 1e-8)

    x = torch.stack([mean, median, peak, max_day, half_day, autocorr], dim=1)

    # Standardize
    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0).clamp(min=1e-8)
    x = (x - x_mean) / x_std

    return x


class SIRDataModule(BaseDataModule):
    def __init__(
        self,
        # Simulator params
        population_size: int = 100_000,
        num_days: int = 365,
        eta: float = 0.05,
        sigma: float = 0.05,
        dt: float = 0.1,
        # Prior params
        beta_min: float = 0.0,
        beta_max: float = 0.5,
        gamma_min: float = 0.0,
        gamma_max: float = 0.5,
        reject_gamma_gt_beta: bool = True,
        # DataModule params
        num_batches: int = 200,
        batch_size: int = 256,
        train_split: float = 0.8,
        val_split: float = 0.1,
        # Device
        simulator_device: str = "cpu",
    ):
        super().__init__(
            num_batches=num_batches,
            batch_size=batch_size,
            train_split=train_split,
            val_split=val_split,
        )
        self.population_size = population_size
        self.num_days = num_days
        self.eta = eta
        self.sigma = sigma
        self.dt = dt
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.reject_gamma_gt_beta = reject_gamma_gt_beta
        self.simulator_device = simulator_device

    def _sample_params(self, num_samples):
        """Sample (beta, gamma) pairs from the prior with optional rejection."""
        if not self.reject_gamma_gt_beta:
            beta = torch.empty(num_samples).uniform_(self.beta_min, self.beta_max)
            gamma = torch.empty(num_samples).uniform_(self.gamma_min, self.gamma_max)
            return beta, gamma

        # Rejection sampling: oversample then filter
        betas = []
        gammas = []
        while len(betas) < num_samples:
            n = (num_samples - len(betas)) * 3
            b = torch.empty(n).uniform_(self.beta_min, self.beta_max)
            g = torch.empty(n).uniform_(self.gamma_min, self.gamma_max)
            mask = g < b
            betas.append(b[mask])
            gammas.append(g[mask])
        beta = torch.cat(betas)[:num_samples]
        gamma = torch.cat(gammas)[:num_samples]
        return beta, gamma

    def _simulate(self, beta, gamma, sigma_override=None):
        """Run batched simulation, returns daily infections on CPU."""
        sigma = sigma_override if sigma_override is not None else self.sigma
        y = _simulate_sir_batch(
            beta,
            gamma,
            self.population_size,
            self.num_days,
            self.eta,
            sigma,
            self.dt,
            device=self.simulator_device,
        )
        return y.cpu()

    def generate_dataset(self, num_samples):
        beta, gamma = self._sample_params(num_samples)
        z = torch.stack([beta, gamma], dim=1)

        y = self._simulate(beta, gamma)
        x = _compute_summary_stats(y)

        return TensorDataset(Tensor(z), Tensor(x))

    def generate_misspecified_data(self, num_samples, misspec_type, misspec_param):
        """Generate data under model misspecification.

        Args:
            num_samples: Number of samples to generate
            misspec_type: One of:
                - "reporting_delay": Weekend counts reduced by misspec_param fraction,
                  recouped on Monday (Ward et al.)
                - "increased_volatility": R0 volatility set to misspec_param
                  instead of default sigma
                - "overdispersed_noise": NegBinomial observation noise with
                  dispersion = misspec_param
            misspec_param: Misspecification strength parameter

        Returns:
            TensorDataset(z, x) where z: [num_samples, 2], x: [num_samples, 6]
        """
        beta, gamma = self._sample_params(num_samples)
        z = torch.stack([beta, gamma], dim=1)

        if misspec_type == "reporting_delay":
            y = self._simulate(beta, gamma)
            y = _apply_reporting_delays(y, misspec_param)
        elif misspec_type == "increased_volatility":
            y = self._simulate(beta, gamma, sigma_override=misspec_param)
        elif misspec_type == "overdispersed_noise":
            y = self._simulate(beta, gamma)
            r = misspec_param
            mu = y.clamp(min=1.0)
            rate = torch.distributions.Gamma(r, r / mu).sample()
            y = torch.poisson(rate)
        else:
            raise ValueError(f"Unknown misspec_type: {misspec_type}")

        x = _compute_summary_stats(y)
        return TensorDataset(Tensor(z), Tensor(x))
