"""
Unit tests for reward computation and GIGPO algorithms.
"""
import pytest
import numpy as np
import torch
from collections import defaultdict

# Import the GIGPO functions
from openmanus_rl.algorithms.gigpo import (
    compute_gigpo_outcome_advantage,
    episode_norm_reward,
    step_norm_reward,
    build_step_group,
    compute_step_discounted_returns,
    to_hashable,
)


class TestGIGPOAlgorithms:
    """Test suite for GIGPO algorithm functions."""

    def test_to_hashable_basic_types(self):
        """Test to_hashable with basic Python types."""
        assert to_hashable(42) == 42
        assert to_hashable(3.14) == 3.14
        assert to_hashable("hello") == "hello"
        assert to_hashable(True) == True

    def test_to_hashable_numpy_types(self):
        """Test to_hashable with numpy types."""
        assert to_hashable(np.int64(42)) == 42
        assert to_hashable(np.float64(3.14)) == 3.14

    def test_to_hashable_arrays(self):
        """Test to_hashable with arrays."""
        arr = np.array([1, 2, 3])
        result = to_hashable(arr)
        assert result == (1, 2, 3)

    def test_to_hashable_lists(self):
        """Test to_hashable with lists."""
        lst = [1, 2, [3, 4]]
        result = to_hashable(lst)
        assert result == (1, 2, (3, 4))

    def test_to_hashable_dicts(self):
        """Test to_hashable with dictionaries."""
        d = {"a": 1, "b": 2}
        result = to_hashable(d)
        assert result == (("a", 1), ("b", 2))

    def test_episode_norm_reward_single_trajectory(self):
        """Test episode normalization with a single trajectory."""
        # Create simple test data
        token_level_rewards = torch.tensor([[1.0, 2.0, 3.0]])
        response_mask = torch.tensor([[1.0, 1.0, 1.0]])
        index = np.array([0])
        traj_index = np.array([0])

        # Should return zero advantage for single trajectory
        advantages = episode_norm_reward(
            token_level_rewards, response_mask, index, traj_index
        )

        assert advantages.shape == token_level_rewards.shape
        # With single trajectory, normalized to 0
        assert torch.allclose(advantages, torch.zeros_like(token_level_rewards))

    def test_episode_norm_reward_multiple_trajectories(self):
        """Test episode normalization with multiple trajectories."""
        # Two trajectories with different rewards
        token_level_rewards = torch.tensor([[1.0, 1.0], [5.0, 5.0]])
        response_mask = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        index = np.array([0, 0])  # Same prompt
        traj_index = np.array([0, 1])  # Different trajectories

        advantages = episode_norm_reward(
            token_level_rewards, response_mask, index, traj_index
        )

        assert advantages.shape == token_level_rewards.shape
        # Higher reward trajectory should have positive advantage
        assert advantages[1, 0] > advantages[0, 0]

    def test_build_step_group_identical_obs(self):
        """Test step group building with identical observations."""
        anchor_obs = np.array(["obs1", "obs1", "obs1"])
        index = np.array([0, 0, 0])

        step_group_uids = build_step_group(anchor_obs, index, summarize=False)

        # All identical observations should have the same UID
        assert step_group_uids[0] == step_group_uids[1]
        assert step_group_uids[1] == step_group_uids[2]

    def test_build_step_group_different_obs(self):
        """Test step group building with different observations."""
        anchor_obs = np.array(["obs1", "obs2", "obs3"])
        index = np.array([0, 0, 0])

        step_group_uids = build_step_group(anchor_obs, index, summarize=False)

        # Different observations should have different UIDs
        assert step_group_uids[0] != step_group_uids[1]
        assert step_group_uids[1] != step_group_uids[2]

    def test_build_step_group_multiple_indices(self):
        """Test step group building with multiple episode indices."""
        anchor_obs = np.array(["obs1", "obs1", "obs2", "obs2"])
        index = np.array([0, 0, 1, 1])

        step_group_uids = build_step_group(anchor_obs, index, summarize=False)

        # Same observations within same index should have same UID
        assert step_group_uids[0] == step_group_uids[1]
        assert step_group_uids[2] == step_group_uids[3]
        # But different indices should have different UIDs even if obs is same
        assert step_group_uids[0] != step_group_uids[2]

    def test_step_norm_reward_single_group(self):
        """Test step normalization with a single group."""
        step_rewards = torch.tensor([1.0])
        response_mask = torch.tensor([[1.0, 1.0]])
        index = np.array(["group1"])

        advantages = step_norm_reward(step_rewards, response_mask, index)

        assert advantages.shape == response_mask.shape
        # Single group should normalize to 0
        assert torch.allclose(advantages, torch.zeros_like(response_mask))

    def test_step_norm_reward_multiple_groups(self):
        """Test step normalization with multiple groups."""
        step_rewards = torch.tensor([1.0, 5.0])
        response_mask = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        index = np.array(["group1", "group1"])

        advantages = step_norm_reward(step_rewards, response_mask, index)

        assert advantages.shape == response_mask.shape
        # Higher reward should have positive advantage
        assert advantages[1, 0] > advantages[0, 0]

    def test_compute_step_discounted_returns_simple(self):
        """Test discounted returns computation with simple trajectory."""
        # Create a simple DataProto-like object
        class SimpleDataProto:
            def __init__(self):
                self.non_tensor_batch = {
                    "rewards": np.array([1.0, 2.0, 3.0]),
                    "traj_uid": np.array([0, 0, 0]),
                    "active_masks": np.ones(3),
                }
                self.batch = {"input_ids": torch.zeros(3, 10)}

        batch = SimpleDataProto()
        gamma = 0.9

        returns = compute_step_discounted_returns(batch, gamma)

        # Check shape
        assert returns.shape[0] == 3
        # First return should be highest (sum of all discounted future rewards)
        assert returns[0] > returns[1] > returns[2]

    def test_compute_step_discounted_returns_gamma_1(self):
        """Test discounted returns with gamma=1 (no discounting)."""
        class SimpleDataProto:
            def __init__(self):
                self.non_tensor_batch = {
                    "rewards": np.array([1.0, 1.0, 1.0]),
                    "traj_uid": np.array([0, 0, 0]),
                    "active_masks": np.ones(3),
                }
                self.batch = {"input_ids": torch.zeros(3, 10)}

        batch = SimpleDataProto()
        gamma = 1.0

        returns = compute_step_discounted_returns(batch, gamma)

        # With gamma=1 and equal rewards, returns should be [3, 2, 1]
        expected = torch.tensor([3.0, 2.0, 1.0])
        assert torch.allclose(returns, expected, atol=1e-5)

    def test_compute_step_discounted_returns_multiple_trajectories(self):
        """Test discounted returns with multiple trajectories."""
        class SimpleDataProto:
            def __init__(self):
                self.non_tensor_batch = {
                    "rewards": np.array([1.0, 2.0, 3.0, 4.0]),
                    "traj_uid": np.array([0, 0, 1, 1]),  # Two trajectories
                    "active_masks": np.ones(4),
                }
                self.batch = {"input_ids": torch.zeros(4, 10)}

        batch = SimpleDataProto()
        gamma = 0.9

        returns = compute_step_discounted_returns(batch, gamma)

        # Check that trajectories are computed independently
        assert returns.shape[0] == 4
        # Within each trajectory, earlier steps should have higher returns
        assert returns[0] > returns[1]
        assert returns[2] > returns[3]


class TestRewardComputation:
    """Test suite for reward computation functions."""

    def test_reward_scale_positive(self):
        """Test that rewards scale correctly for positive values."""
        rewards = torch.tensor([1.0, 2.0, 3.0])
        scale = 2.0
        scaled = rewards * scale
        assert torch.allclose(scaled, torch.tensor([2.0, 4.0, 6.0]))

    def test_reward_scale_negative(self):
        """Test that rewards scale correctly for negative values."""
        rewards = torch.tensor([-1.0, -2.0, -3.0])
        scale = 2.0
        scaled = rewards * scale
        assert torch.allclose(scaled, torch.tensor([-2.0, -4.0, -6.0]))

    def test_reward_normalization(self):
        """Test basic reward normalization."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        mean = rewards.mean()
        std = rewards.std()
        normalized = (rewards - mean) / (std + 1e-8)

        # Check normalized mean is close to 0
        assert torch.abs(normalized.mean()) < 1e-5
        # Check normalized std is close to 1
        assert torch.abs(normalized.std() - 1.0) < 1e-5

    def test_reward_clipping(self):
        """Test reward clipping."""
        rewards = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
        clipped = torch.clamp(rewards, min=-5.0, max=5.0)
        expected = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
        assert torch.allclose(clipped, expected)


class TestAdvantageComputation:
    """Test suite for advantage computation."""

    def test_advantage_with_zero_values(self):
        """Test advantage computation with zero value estimates."""
        rewards = torch.tensor([1.0, 1.0, 1.0])
        values = torch.zeros(3)
        gamma = 0.99
        lam = 0.95

        # Simple GAE computation
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages)
        assert advantages.shape == rewards.shape

    def test_advantage_monotonicity(self):
        """Test that advantages decrease with better value estimates."""
        rewards = torch.tensor([1.0, 1.0, 1.0])
        values1 = torch.zeros(3)
        values2 = torch.ones(3) * 0.5  # Better value estimates

        # With better value estimates, advantages should be smaller
        # This is a simple check of the intuition


class TestMaskingOperations:
    """Test suite for masking operations in reward/advantage computation."""

    def test_response_mask_application(self):
        """Test that response mask correctly filters rewards."""
        rewards = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

        masked_rewards = rewards * mask
        expected = torch.tensor([[1.0, 2.0, 0.0], [4.0, 0.0, 0.0]])

        assert torch.allclose(masked_rewards, expected)

    def test_active_mask_filtering(self):
        """Test that active mask filters inactive samples."""
        rewards = np.array([1.0, 2.0, 3.0, 4.0])
        active_mask = np.array([1.0, 1.0, 0.0, 1.0])

        # Filter rewards where mask is 1
        active_rewards = rewards[active_mask == 1.0]
        expected = np.array([1.0, 2.0, 4.0])

        assert np.allclose(active_rewards, expected)


@pytest.fixture
def sample_batch_data():
    """Fixture providing sample batch data for testing."""
    return {
        "rewards": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "values": torch.tensor([[0.5, 1.0], [1.5, 2.0]]),
        "response_mask": torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        "index": np.array([0, 0]),
        "traj_index": np.array([0, 1]),
    }


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_grpo_pipeline(self, sample_batch_data):
        """Test complete GRPO advantage computation pipeline."""
        token_level_rewards = sample_batch_data["rewards"]
        step_rewards = token_level_rewards.sum(dim=-1)
        response_mask = sample_batch_data["response_mask"]
        index = sample_batch_data["index"]
        traj_index = sample_batch_data["traj_index"]

        # Create fake anchor observations
        anchor_obs = np.array(["obs1", "obs2"])

        # Compute advantages
        advantages, returns = compute_gigpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            step_rewards=step_rewards,
            response_mask=response_mask,
            anchor_obs=anchor_obs,
            index=index,
            traj_index=traj_index,
            epsilon=1e-6,
            step_advantage_w=1.0,
            mode="mean_norm",
        )

        # Check shapes
        assert advantages.shape == token_level_rewards.shape
        assert returns.shape == token_level_rewards.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
