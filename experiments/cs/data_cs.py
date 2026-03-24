import torch
from data import BaseDataModule
from torch import Tensor
from torch.utils.data import TensorDataset


def _simulate_cs(lambda_c, lambda_p, lambda_d, num_stromal_samples, rng):
    """Simulate one Cancer & Stromal point process (Ward et al., 2022).

    Args:
        lambda_c: Poisson rate for total cells
        lambda_p: Poisson rate for parent points
        lambda_d: Poisson rate for daughter cells per parent
        num_stromal_samples: Number of stromal cells to sample for distance metrics
        rng: numpy random generator

    Returns:
        n_cancer: number of cancer cells
        n_stromal: number of stromal cells
        mean_min_dist: mean distance from sampled stromal cells to nearest cancer cell
        max_min_dist: max distance from sampled stromal cells to nearest cancer cell
    """
    # Sample counts
    n_cells = max(rng.poisson(lambda_c), 1)
    n_parents = max(rng.poisson(lambda_p), 1)

    # Generate positions uniformly on [0,1]^2
    cell_pos = torch.tensor(rng.uniform(0, 1, (n_cells, 2)), dtype=torch.float32)
    parent_pos = torch.tensor(rng.uniform(0, 1, (n_parents, 2)), dtype=torch.float32)

    # For each parent, determine affected radius
    # r_i = distance to N_d_i-th nearest cell
    # Compute distances from each parent to all cells: [n_parents, n_cells]
    dists_to_cells = torch.cdist(parent_pos, cell_pos)

    is_cancer = torch.zeros(n_cells, dtype=torch.bool)

    for p in range(n_parents):
        n_daughter = max(rng.poisson(lambda_d), 1)
        # r_i = distance to n_daughter-th nearest cell (0-indexed: n_daughter-1)
        k = min(n_daughter, n_cells) - 1
        sorted_dists = dists_to_cells[p].sort().values
        r_i = sorted_dists[k].item()

        # Mark cells within r_i as cancer
        within = dists_to_cells[p] <= r_i
        is_cancer = is_cancer | within

    n_cancer = int(is_cancer.sum().item())
    n_stromal = n_cells - n_cancer

    # Compute distance metrics
    if n_cancer == 0 or n_stromal == 0:
        return n_cancer, n_stromal, 0.0, 0.0

    cancer_pos = cell_pos[is_cancer]
    stromal_pos = cell_pos[~is_cancer]

    # Sample stromal cells for efficiency (Ward et al. use 50)
    if n_stromal > num_stromal_samples:
        indices = torch.randperm(n_stromal)[:num_stromal_samples]
        stromal_sample = stromal_pos[indices]
    else:
        stromal_sample = stromal_pos

    # Min distance from each sampled stromal cell to nearest cancer cell
    dists = torch.cdist(stromal_sample, cancer_pos)  # [n_sampled, n_cancer]
    min_dists = dists.min(dim=1).values

    mean_min_dist = min_dists.mean().item()
    max_min_dist = min_dists.max().item()

    return n_cancer, n_stromal, mean_min_dist, max_min_dist


def _simulate_cs_with_necrosis(
    lambda_c, lambda_p, lambda_d, num_stromal_samples, necrosis_prob, rng
):
    """Simulate CS with necrosis misspecification (Ward et al., 2022).

    For each parent, with probability necrosis_prob, remove cancer cells
    within 0.8 * r_i of the parent (simulating cell death in tumor cores).
    """
    n_cells = max(rng.poisson(lambda_c), 1)
    n_parents = max(rng.poisson(lambda_p), 1)

    cell_pos = torch.tensor(rng.uniform(0, 1, (n_cells, 2)), dtype=torch.float32)
    parent_pos = torch.tensor(rng.uniform(0, 1, (n_parents, 2)), dtype=torch.float32)

    dists_to_cells = torch.cdist(parent_pos, cell_pos)

    is_cancer = torch.zeros(n_cells, dtype=torch.bool)
    radii = torch.zeros(n_parents)

    for p in range(n_parents):
        n_daughter = max(rng.poisson(lambda_d), 1)
        k = min(n_daughter, n_cells) - 1
        sorted_dists = dists_to_cells[p].sort().values
        r_i = sorted_dists[k].item()
        radii[p] = r_i

        within = dists_to_cells[p] <= r_i
        is_cancer = is_cancer | within

    # Apply necrosis: remove cancer cells in core regions
    for p in range(n_parents):
        if rng.random() < necrosis_prob:
            core_radius = 0.8 * radii[p].item()
            in_core = dists_to_cells[p] <= core_radius
            # Only remove cells that are cancer
            is_cancer = is_cancer & ~in_core

    n_cancer = int(is_cancer.sum().item())
    n_stromal = n_cells - n_cancer

    if n_cancer == 0 or n_stromal == 0:
        return n_cancer, n_stromal, 0.0, 0.0

    cancer_pos = cell_pos[is_cancer]
    stromal_pos = cell_pos[~is_cancer]

    if n_stromal > num_stromal_samples:
        indices = torch.randperm(n_stromal)[:num_stromal_samples]
        stromal_sample = stromal_pos[indices]
    else:
        stromal_sample = stromal_pos

    dists = torch.cdist(stromal_sample, cancer_pos)
    min_dists = dists.min(dim=1).values

    mean_min_dist = min_dists.mean().item()
    max_min_dist = min_dists.max().item()

    return n_cancer, n_stromal, mean_min_dist, max_min_dist


def _compute_summary_stats(stats_list):
    """Standardize summary statistics.

    Args:
        stats_list: list of (n_cancer, n_stromal, mean_min_dist, max_min_dist) tuples

    Returns:
        Tensor of shape [num_samples, 4], standardized.
    """
    x = torch.tensor(stats_list, dtype=torch.float32)  # [num_samples, 4]

    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0).clamp(min=1e-8)
    x = (x - x_mean) / x_std

    return x


class CSDataModule(BaseDataModule):
    def __init__(
        self,
        # Prior params
        lambda_c_min: float = 200.0,
        lambda_c_max: float = 1500.0,
        lambda_p_min: float = 3.0,
        lambda_p_max: float = 20.0,
        lambda_d_min: float = 10.0,
        lambda_d_max: float = 20.0,
        # Simulator params
        num_stromal_samples: int = 50,
        # DataModule params
        num_batches: int = 200,
        batch_size: int = 256,
        train_split: float = 0.8,
        val_split: float = 0.1,
    ):
        super().__init__(
            num_batches=num_batches,
            batch_size=batch_size,
            train_split=train_split,
            val_split=val_split,
        )
        self.lambda_c_min = lambda_c_min
        self.lambda_c_max = lambda_c_max
        self.lambda_p_min = lambda_p_min
        self.lambda_p_max = lambda_p_max
        self.lambda_d_min = lambda_d_min
        self.lambda_d_max = lambda_d_max
        self.num_stromal_samples = num_stromal_samples

    def _sample_params(self, num_samples):
        """Sample (lambda_c, lambda_p, lambda_d) from uniform priors."""
        lc = torch.empty(num_samples).uniform_(self.lambda_c_min, self.lambda_c_max)
        lp = torch.empty(num_samples).uniform_(self.lambda_p_min, self.lambda_p_max)
        ld = torch.empty(num_samples).uniform_(self.lambda_d_min, self.lambda_d_max)
        return lc, lp, ld

    def generate_dataset(self, num_samples):
        import numpy as np

        rng = np.random.default_rng()
        lc, lp, ld = self._sample_params(num_samples)
        z = torch.stack([lc, lp, ld], dim=1)  # [num_samples, 3]

        stats_list = []
        for i in range(num_samples):
            stats = _simulate_cs(
                lc[i].item(),
                lp[i].item(),
                ld[i].item(),
                self.num_stromal_samples,
                rng,
            )
            stats_list.append(stats)

        x = _compute_summary_stats(stats_list)
        return TensorDataset(Tensor(z), Tensor(x))

    def generate_misspecified_data(self, num_samples, misspec_type, misspec_param):
        """Generate data under model misspecification.

        Args:
            num_samples: Number of samples to generate
            misspec_type: One of:
                - "necrosis": Remove cancer cells in core regions with
                  probability misspec_param per parent
            misspec_param: Misspecification strength parameter

        Returns:
            TensorDataset(z, x) where z: [num_samples, 3], x: [num_samples, 4]
        """
        import numpy as np

        rng = np.random.default_rng()
        lc, lp, ld = self._sample_params(num_samples)
        z = torch.stack([lc, lp, ld], dim=1)

        stats_list = []
        for i in range(num_samples):
            if misspec_type == "necrosis":
                stats = _simulate_cs_with_necrosis(
                    lc[i].item(),
                    lp[i].item(),
                    ld[i].item(),
                    self.num_stromal_samples,
                    necrosis_prob=misspec_param,
                    rng=rng,
                )
            else:
                raise ValueError(f"Unknown misspec_type: {misspec_type}")
            stats_list.append(stats)

        x = _compute_summary_stats(stats_list)
        return TensorDataset(Tensor(z), Tensor(x))
