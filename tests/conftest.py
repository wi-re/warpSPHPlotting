import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/warpsph-mpl-tests")
import pytest
import torch
import warp as wp
from warpSPHCore import DomainDescription, ParticleState


@pytest.fixture(scope="session", autouse=True)
def warp_runtime():
    wp.config.kernel_cache_dir = "/tmp/warpsph-warp-cache"
    wp.init()


@pytest.fixture
def domain():
    return DomainDescription(
        torch.zeros(2), torch.ones(2), torch.zeros(2, dtype=torch.bool), 2
    )


@pytest.fixture
def particles():
    x = torch.linspace(0.125, 0.875, 4)
    pos = torch.stack(torch.meshgrid(x, x, indexing="ij"), -1).reshape(-1, 2)
    return ParticleState(
        positions=pos,
        supports=torch.full((16,), 0.7),
        masses=torch.full((16,), 1 / 16),
        densities=torch.ones(16),
        kinds=torch.zeros(16, dtype=torch.int32),
    )


@pytest.fixture(autouse=True)
def close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
