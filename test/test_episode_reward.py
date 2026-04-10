"""
Exhaustive unit tests for EpisodeRewardManager.

Covers:
- openmanus_rl/reward_manager/episode.py  (EpisodeRewardManager)
"""

import pytest
import numpy as np
import torch

from openmanus_rl.reward_manager.episode import EpisodeRewardManager

# conftest.py has already installed the verl stub and DataProto
from conftest import DataProto


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockTokenizer:
    """Minimal tokenizer stub."""

    def decode(self, token_ids, skip_special_tokens=True):
        return "decoded text"


def _make_item(
    prompt_len: int = 5,
    response_len: int = 10,
    episode_rewards: float = 1.0,
    episode_lengths: int = 5,
    data_source: str = "test",
    pad_token_id: int = 0,
    device: str = "cpu",
):
    """Return a DataProto that mimics a single-item rollout batch."""
    total = prompt_len + response_len

    # Construct attention mask: first pad_prompt tokens are padding (0), rest are 1
    attention_mask = torch.ones(total, dtype=torch.long)
    prompts = torch.randint(1, 100, (total,))
    responses = torch.randint(1, 100, (response_len,))

    return DataProto(
        batch={
            "prompts": prompts,
            "responses": responses,
            "attention_mask": attention_mask,
        },
        non_tensor_batch={
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "data_source": data_source,
        },
    )


def _make_batch(n_items: int = 4, **kwargs):
    """Return a DataProto batch of n_items."""
    prompt_len = kwargs.get("prompt_len", 5)
    response_len = kwargs.get("response_len", 10)
    total = prompt_len + response_len

    attention_mask = torch.ones(n_items, total, dtype=torch.long)
    prompts = torch.randint(1, 100, (n_items, total))
    responses = torch.randint(1, 100, (n_items, response_len))

    batch = DataProto(
        batch={
            "prompts": prompts,
            "responses": responses,
            "attention_mask": attention_mask,
        },
        non_tensor_batch={
            "episode_rewards": np.array([kwargs.get("episode_rewards", 1.0)] * n_items),
            "episode_lengths": np.array([kwargs.get("episode_lengths", 5)] * n_items),
            "data_source": np.array(["test"] * n_items),
        },
    )
    return batch


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEpisodeRewardManagerInit:
    def test_basic_init(self):
        tok = _MockTokenizer()
        mgr = EpisodeRewardManager(tokenizer=tok, num_examine=1)
        assert mgr.tokenizer is tok
        assert mgr.num_examine == 1
        assert mgr.normalize_by_length is False

    def test_normalize_flag(self):
        mgr = EpisodeRewardManager(_MockTokenizer(), 0, normalize_by_length=True)
        assert mgr.normalize_by_length is True


class TestEpisodeRewardManagerCall:
    @pytest.fixture
    def manager(self):
        return EpisodeRewardManager(_MockTokenizer(), num_examine=0)

    def test_returns_tensor(self, manager):
        batch = _make_batch(n_items=2)
        out = manager(batch)
        assert isinstance(out, torch.Tensor)

    def test_output_shape_matches_responses(self, manager):
        batch = _make_batch(n_items=3, response_len=8)
        out = manager(batch)
        assert out.shape == (3, 8)

    def test_reward_placed_at_last_valid_response_token(self, manager):
        """The scalar episode reward should appear at the last valid response position."""
        batch = _make_batch(n_items=1, prompt_len=3, response_len=6, episode_rewards=2.5)
        out = manager(batch)
        # Last valid response token index = valid_response_length - 1
        # With all-ones attention mask and prompt_len=3, response_len=6, valid_response=6
        assert out[0, 5].item() == pytest.approx(2.5, abs=1e-4)

    def test_zero_reward_episode(self, manager):
        batch = _make_batch(n_items=2, episode_rewards=0.0)
        out = manager(batch)
        assert (out == 0).all()

    def test_negative_reward_episode(self, manager):
        batch = _make_batch(n_items=2, episode_rewards=-1.0)
        out = manager(batch)
        # All non-last-position tokens should be 0
        assert (out[:, :-1] == 0).all()
        # Last token per sample should be -1
        assert out[0, -1].item() == pytest.approx(-1.0, abs=1e-4)

    def test_return_dict_mode(self, manager):
        batch = _make_batch(n_items=2)
        out = manager(batch, return_dict=True)
        assert isinstance(out, dict)
        assert "reward_tensor" in out
        assert "reward_extra_info" in out
        assert isinstance(out["reward_tensor"], torch.Tensor)

    def test_return_dict_false_returns_tensor(self, manager):
        batch = _make_batch(n_items=2)
        out = manager(batch, return_dict=False)
        assert isinstance(out, torch.Tensor)

    def test_rm_scores_passthrough(self, manager):
        """When 'rm_scores' already in batch, they should be returned directly."""
        rm = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        batch = DataProto(
            batch={"rm_scores": rm},
            non_tensor_batch={},
        )
        out = manager(batch)
        assert torch.equal(out, rm)

    def test_rm_scores_passthrough_return_dict(self, manager):
        rm = torch.tensor([[1.0, 2.0]])
        batch = DataProto(batch={"rm_scores": rm}, non_tensor_batch={})
        out = manager(batch, return_dict=True)
        assert torch.equal(out["reward_tensor"], rm)


class TestEpisodeRewardManagerNormalization:
    def test_normalize_by_length_divides(self):
        mgr = EpisodeRewardManager(_MockTokenizer(), num_examine=0, normalize_by_length=True)
        batch = _make_batch(n_items=1, episode_rewards=6.0, episode_lengths=3)
        out = mgr(batch)
        # Score = 6.0 / 3 = 2.0
        assert out[0, -1].item() == pytest.approx(2.0, abs=1e-4)

    def test_no_normalize_by_length(self):
        mgr = EpisodeRewardManager(_MockTokenizer(), num_examine=0, normalize_by_length=False)
        batch = _make_batch(n_items=1, episode_rewards=6.0, episode_lengths=3)
        out = mgr(batch)
        # Score = 6.0 (unchanged)
        assert out[0, -1].item() == pytest.approx(6.0, abs=1e-4)


class TestEpisodeRewardManagerEdgeCases:
    @pytest.fixture
    def manager(self):
        return EpisodeRewardManager(_MockTokenizer(), num_examine=0)

    def test_single_item_batch(self, manager):
        batch = _make_batch(n_items=1)
        out = manager(batch)
        assert out.shape[0] == 1

    def test_large_batch(self, manager):
        batch = _make_batch(n_items=32)
        out = manager(batch)
        assert out.shape[0] == 32

    def test_reward_tensor_dtype_float32(self, manager):
        batch = _make_batch(n_items=2)
        out = manager(batch)
        assert out.dtype == torch.float32

    def test_only_last_position_nonzero(self, manager):
        """All tokens except the last valid response token should be zero."""
        response_len = 8
        batch = _make_batch(n_items=2, response_len=response_len, episode_rewards=3.0)
        out = manager(batch)
        # For each sample the last token carries the reward; all others are 0
        for i in range(2):
            assert out[i, response_len - 1].item() == pytest.approx(3.0, abs=1e-4)
            assert (out[i, : response_len - 1] == 0).all()

    def test_multiple_data_sources(self, manager):
        """Multiple different data_source values should work without error."""
        batch = DataProto(
            batch={
                "prompts": torch.randint(1, 100, (3, 8)),
                "responses": torch.randint(1, 100, (3, 5)),
                "attention_mask": torch.ones(3, 13, dtype=torch.long),
            },
            non_tensor_batch={
                "episode_rewards": np.array([1.0, 0.5, -0.5]),
                "episode_lengths": np.array([5, 5, 5]),
                "data_source": np.array(["env_a", "env_b", "env_c"]),
            },
        )
        out = manager(batch)
        assert out.shape == (3, 5)
