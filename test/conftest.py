"""
Pytest configuration and shared fixtures for OpenManus-RL tests.

Sets up a minimal ``verl`` stub so the openmanus_rl package can be imported
in CI environments where the full verl stack is not available.
"""

import sys
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Minimal verl stub
# ---------------------------------------------------------------------------

class _DataProtoItem:
    """Single-item view returned by DataProto.__getitem__."""

    def __init__(self, batch: Dict[str, Any], non_tensor_batch: Dict[str, Any]):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch


class DataProto:
    """Minimal stand-in for verl.DataProto sufficient for openmanus_rl tests."""

    def __init__(
        self,
        batch: Optional[Dict[str, Any]] = None,
        non_tensor_batch: Optional[Dict[str, Any]] = None,
    ):
        self.batch = batch or {}
        self.non_tensor_batch = non_tensor_batch or {}

    def __len__(self) -> int:
        if self.batch:
            return len(next(iter(self.batch.values())))
        if self.non_tensor_batch:
            return len(next(iter(self.non_tensor_batch.values())))
        return 0

    def __getitem__(self, idx: int) -> _DataProtoItem:
        item_batch = {k: v[idx] for k, v in self.batch.items()}
        item_non_tensor = {}
        for k, v in self.non_tensor_batch.items():
            try:
                item_non_tensor[k] = v[idx]
            except (TypeError, KeyError):
                item_non_tensor[k] = v
        return _DataProtoItem(item_batch, item_non_tensor)


def _install_verl_stub() -> None:
    """Install the verl stub into sys.modules if verl.DataProto is unavailable."""
    try:
        from verl import DataProto as _real  # noqa: F401
        return  # Real verl is installed – nothing to do
    except (ImportError, AttributeError):
        pass

    verl_mod = MagicMock()
    verl_mod.DataProto = DataProto
    # Forcibly replace the verl entry in sys.modules.  When the verl directory
    # exists as a git sub-module (but is not initialised / installed), Python
    # will have already created a bare module object for it without DataProto.
    # Using sys.modules["verl"] = ... ensures our stub always wins.
    sys.modules["verl"] = verl_mod
    # Sub-modules referenced by openmanus_rl – register each level explicitly
    # so Python's import machinery can find them via sys.modules lookups.
    for sub in (
        "verl.utils",
        "verl.utils.tracking",
        "verl.utils.dataset",
        "verl.utils.dataset.rl_dataset",
    ):
        sys.modules[sub] = MagicMock()

    # Other heavy optional dependencies – only register if not already available
    for dep in ("omegaconf", "transformers", "agentenv", "agentenv.envs"):
        sys.modules.setdefault(dep, MagicMock())


_install_verl_stub()


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_data_proto():
    """Return a minimal DataProto with tensor and non-tensor data."""
    import torch

    return DataProto(
        batch={"input_ids": torch.zeros(3, 10)},
        non_tensor_batch={
            "rewards": np.array([1.0, 2.0, 3.0]),
            "traj_uid": np.array([0, 0, 0]),
            "active_masks": np.ones(3),
        },
    )


@pytest.fixture
def multi_traj_data_proto():
    """Return a DataProto with two distinct trajectories."""
    import torch

    return DataProto(
        batch={"input_ids": torch.zeros(4, 10)},
        non_tensor_batch={
            "rewards": np.array([1.0, 2.0, 3.0, 4.0]),
            "traj_uid": np.array([0, 0, 1, 1]),
            "active_masks": np.ones(4),
        },
    )
