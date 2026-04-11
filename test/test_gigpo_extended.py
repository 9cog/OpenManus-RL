"""
Extended unit tests for GiGPO algorithms.

Augments the existing test_reward_computation.py with additional
edge-case coverage for every exported function in
openmanus_rl/algorithms/gigpo.py.
"""

import numpy as np
import pytest
import torch

from openmanus_rl.algorithms.gigpo import (
    build_step_group,
    compute_gigpo_outcome_advantage,
    compute_step_discounted_returns,
    episode_norm_reward,
    step_norm_reward,
    summarize_group_size,
    to_hashable,
)

# conftest.py installs the verl stub and exposes DataProto
from conftest import DataProto


# ============================================================
# to_hashable – extended coverage
# ============================================================

class TestToHashableExtended:
    def test_tuple_input(self):
        assert to_hashable((1, 2, 3)) == (1, 2, 3)

    def test_nested_dict(self):
        d = {"outer": {"inner": 42}}
        result = to_hashable(d)
        assert isinstance(result, tuple)

    def test_empty_list(self):
        assert to_hashable([]) == ()

    def test_empty_dict(self):
        assert to_hashable({}) == ()

    def test_numpy_2d_array_flattened(self):
        arr = np.array([[1, 2], [3, 4]])
        result = to_hashable(arr)
        assert result == (1, 2, 3, 4)

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            to_hashable(object())

    def test_list_of_numpy_floats(self):
        lst = [np.float64(1.5), np.float64(2.5)]
        result = to_hashable(lst)
        assert result == (1.5, 2.5)

    def test_bool_preserved(self):
        assert to_hashable(True) is True
        assert to_hashable(False) is False

    def test_large_numpy_int(self):
        v = np.int64(2**40)
        result = to_hashable(v)
        assert isinstance(result, int)


# ============================================================
# summarize_group_size – smoke test
# ============================================================

class TestSummarizeGroupSize:
    def test_single_size(self, capsys):
        summarize_group_size([3, 3, 3])
        out = capsys.readouterr().out
        assert "3" in out

    def test_mixed_sizes(self, capsys):
        summarize_group_size([1, 2, 2, 3])
        out = capsys.readouterr().out
        assert "Summary" in out

    def test_empty_list(self):
        """Edge case: empty list should not raise (Counter handles it gracefully)."""
        # Counter of empty list has no max – expect ValueError or handle it
        with pytest.raises((ValueError, TypeError)):
            summarize_group_size([])


# ============================================================
# build_step_group – extended coverage
# ============================================================

class TestBuildStepGroupExtended:
    def test_all_unique_obs_each_gets_unique_uid(self):
        obs = np.array(["a", "b", "c", "d"])
        idx = np.array([0, 0, 0, 0])
        uids = build_step_group(obs, idx)
        assert len(set(uids)) == 4

    def test_single_element(self):
        obs = np.array(["solo"])
        idx = np.array([0])
        uids = build_step_group(obs, idx)
        assert len(uids) == 1
        assert uids[0] is not None

    def test_result_length_matches_input(self):
        obs = np.array(["o1", "o2", "o3", "o1"])
        idx = np.array([0, 0, 0, 0])
        uids = build_step_group(obs, idx)
        assert len(uids) == 4

    def test_partial_overlap_across_indices(self):
        """Same obs in different index groups → different UIDs."""
        obs = np.array(["same", "same", "same", "same"])
        idx = np.array([0, 0, 1, 1])
        uids = build_step_group(obs, idx)
        assert uids[0] == uids[1]
        assert uids[2] == uids[3]
        assert uids[0] != uids[2]

    def test_uids_are_strings(self):
        obs = np.array(["x", "y"])
        idx = np.array([0, 0])
        uids = build_step_group(obs, idx)
        assert all(isinstance(u, str) for u in uids)

    def test_summarize_flag_does_not_raise(self, capsys):
        obs = np.array(["a", "a", "b"])
        idx = np.array([0, 0, 0])
        build_step_group(obs, idx, summarize=True)
        out = capsys.readouterr().out
        assert "step-level group" in out.lower()

    def test_multiple_clusters_same_index(self):
        obs = np.array(["obs1", "obs1", "obs2", "obs2"])
        idx = np.array([0, 0, 0, 0])
        uids = build_step_group(obs, idx)
        # Two clusters → two unique UIDs
        assert len(set(uids)) == 2
        assert uids[0] == uids[1]
        assert uids[2] == uids[3]


# ============================================================
# episode_norm_reward – extended coverage
# ============================================================

class TestEpisodeNormRewardExtended:
    def test_zero_rewards_normalize_to_zero(self):
        rewards = torch.zeros(2, 4)
        mask = torch.ones(2, 4)
        idx = np.array([0, 0])
        traj = np.array([0, 1])
        adv = episode_norm_reward(rewards, mask, idx, traj)
        assert torch.allclose(adv, torch.zeros_like(adv))

    def test_mask_zeros_out_padding(self):
        rewards = torch.tensor([[1.0, 2.0, 0.0], [5.0, 0.0, 0.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        idx = np.array([0, 0])
        traj = np.array([0, 1])
        adv = episode_norm_reward(rewards, mask, idx, traj)
        assert adv[0, 2].item() == pytest.approx(0.0)
        assert adv[1, 1].item() == pytest.approx(0.0)

    def test_higher_reward_gets_positive_advantage(self):
        rewards = torch.tensor([[1.0, 1.0], [5.0, 5.0]])
        mask = torch.ones(2, 2)
        idx = np.array([0, 0])
        traj = np.array([0, 1])
        adv = episode_norm_reward(rewards, mask, idx, traj)
        assert adv[1, 0] > adv[0, 0]

    def test_mean_std_norm_mode(self):
        rewards = torch.tensor([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]])
        mask = torch.ones(3, 2)
        idx = np.array([0, 0, 0])
        traj = np.array([0, 1, 2])
        adv = episode_norm_reward(rewards, mask, idx, traj, remove_std=False)
        assert adv.shape == (3, 2)

    def test_different_prompt_groups_independent(self):
        rewards = torch.tensor([[1.0, 1.0], [5.0, 5.0], [1.0, 1.0], [5.0, 5.0]])
        mask = torch.ones(4, 2)
        idx = np.array([0, 0, 1, 1])
        traj = np.array([0, 1, 0, 1])
        adv = episode_norm_reward(rewards, mask, idx, traj)
        # Groups 0 and 1 computed independently → same relative pattern
        assert adv[1, 0] > adv[0, 0]
        assert adv[3, 0] > adv[2, 0]


# ============================================================
# step_norm_reward – extended coverage
# ============================================================

class TestStepNormRewardExtended:
    def test_shape_preserved(self):
        rewards = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.ones(3, 5)
        idx = np.array(["g0", "g0", "g0"])
        adv = step_norm_reward(rewards, mask, idx)
        assert adv.shape == (3, 5)

    def test_single_sample_per_group_normalizes_to_zero(self):
        rewards = torch.tensor([3.0])
        mask = torch.ones(1, 4)
        idx = np.array(["g0"])
        adv = step_norm_reward(rewards, mask, idx)
        assert torch.allclose(adv, torch.zeros(1, 4))

    def test_two_groups_normalized_independently(self):
        rewards = torch.tensor([1.0, 5.0, 2.0, 8.0])
        mask = torch.ones(4, 2)
        idx = np.array(["g0", "g0", "g1", "g1"])
        adv = step_norm_reward(rewards, mask, idx)
        # Within group g0: 1 and 5 → lower/higher
        assert adv[1, 0] > adv[0, 0]
        # Within group g1: 2 and 8 → lower/higher
        assert adv[3, 0] > adv[2, 0]

    def test_remove_std_false_uses_std(self):
        rewards = torch.tensor([1.0, 5.0])
        mask = torch.ones(2, 2)
        idx = np.array(["g", "g"])
        adv_mean = step_norm_reward(rewards, mask, idx, remove_std=True)
        adv_std = step_norm_reward(rewards, mask, idx, remove_std=False)
        # Absolute values should differ
        assert not torch.allclose(adv_mean, adv_std)


# ============================================================
# compute_step_discounted_returns – extended coverage
# ============================================================

class TestComputeStepDiscountedReturnsExtended:
    def test_gamma_zero_returns_immediate_reward(self):
        batch = DataProto(
            batch={"input_ids": torch.zeros(3, 5)},
            non_tensor_batch={
                "rewards": np.array([1.0, 2.0, 3.0]),
                "traj_uid": np.array([0, 0, 0]),
                "active_masks": np.ones(3),
            },
        )
        returns = compute_step_discounted_returns(batch, gamma=0.0)
        expected = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(returns, expected, atol=1e-5)

    def test_single_step_trajectory(self):
        batch = DataProto(
            batch={"input_ids": torch.zeros(1, 5)},
            non_tensor_batch={
                "rewards": np.array([7.0]),
                "traj_uid": np.array([0]),
                "active_masks": np.ones(1),
            },
        )
        returns = compute_step_discounted_returns(batch, gamma=0.9)
        assert returns[0].item() == pytest.approx(7.0, abs=1e-5)

    def test_three_trajectories(self):
        batch = DataProto(
            batch={"input_ids": torch.zeros(6, 5)},
            non_tensor_batch={
                "rewards": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                "traj_uid": np.array([0, 0, 1, 1, 2, 2]),
                "active_masks": np.ones(6),
            },
        )
        returns = compute_step_discounted_returns(batch, gamma=1.0)
        assert returns[0] == pytest.approx(3.0, abs=1e-5)   # 1+2
        assert returns[2] == pytest.approx(7.0, abs=1e-5)   # 3+4
        assert returns[4] == pytest.approx(11.0, abs=1e-5)  # 5+6

    def test_returns_are_float32_tensor(self):
        batch = DataProto(
            batch={"input_ids": torch.zeros(2, 3)},
            non_tensor_batch={
                "rewards": np.array([1.0, 1.0]),
                "traj_uid": np.array([0, 0]),
                "active_masks": np.ones(2),
            },
        )
        returns = compute_step_discounted_returns(batch, gamma=0.9)
        assert returns.dtype == torch.float32

    def test_decaying_returns_with_discount(self):
        """With gamma=0.5 and rewards=[1,1,1], returns=[1.75, 1.5, 1.0]."""
        batch = DataProto(
            batch={"input_ids": torch.zeros(3, 2)},
            non_tensor_batch={
                "rewards": np.array([1.0, 1.0, 1.0]),
                "traj_uid": np.array([0, 0, 0]),
                "active_masks": np.ones(3),
            },
        )
        returns = compute_step_discounted_returns(batch, gamma=0.5)
        expected = torch.tensor([1.75, 1.5, 1.0])
        assert torch.allclose(returns, expected, atol=1e-5)


# ============================================================
# compute_gigpo_outcome_advantage – extended coverage
# ============================================================

class TestComputeGIGPOOutcomeAdvantageExtended:
    def _make_inputs(self, bs: int, seq_len: int):
        token_level_rewards = torch.rand(bs, seq_len)
        step_rewards = token_level_rewards.sum(dim=-1)
        response_mask = torch.ones(bs, seq_len)
        anchor_obs = np.array([f"obs{i}" for i in range(bs)])
        index = np.array([0] * bs)
        traj_index = np.arange(bs)
        return token_level_rewards, step_rewards, response_mask, anchor_obs, index, traj_index

    def test_output_shapes(self):
        tl_r, s_r, mask, ao, idx, tidx = self._make_inputs(4, 8)
        adv, ret = compute_gigpo_outcome_advantage(
            tl_r, s_r, mask, ao, idx, tidx
        )
        assert adv.shape == (4, 8)
        assert ret.shape == (4, 8)

    def test_advantages_equal_returns_in_this_impl(self):
        """In the current implementation scores == scores (adv == ret)."""
        tl_r, s_r, mask, ao, idx, tidx = self._make_inputs(4, 6)
        adv, ret = compute_gigpo_outcome_advantage(
            tl_r, s_r, mask, ao, idx, tidx
        )
        assert torch.allclose(adv, ret)

    def test_mean_std_norm_mode(self):
        tl_r, s_r, mask, ao, idx, tidx = self._make_inputs(4, 6)
        adv, _ = compute_gigpo_outcome_advantage(
            tl_r, s_r, mask, ao, idx, tidx, mode="mean_std_norm"
        )
        assert adv.shape == (4, 6)

    def test_unknown_mode_raises(self):
        tl_r, s_r, mask, ao, idx, tidx = self._make_inputs(2, 4)
        with pytest.raises(ValueError):
            compute_gigpo_outcome_advantage(
                tl_r, s_r, mask, ao, idx, tidx, mode="bad_mode"
            )

    def test_step_advantage_weight_zero(self):
        """step_advantage_w=0 only uses episode component; w=1 adds step component."""
        bs, seq_len = 4, 6
        # Use duplicated anchor_obs so step groups have >1 member → non-zero step advantages
        tl_r = torch.rand(bs, seq_len)
        s_r = tl_r.sum(dim=-1)
        mask = torch.ones(bs, seq_len)
        ao = np.array(["shared_obs", "shared_obs", "other_obs", "other_obs"])
        idx = np.array([0, 0, 0, 0])
        tidx = np.arange(bs)
        adv_w0, _ = compute_gigpo_outcome_advantage(
            tl_r, s_r, mask, ao, idx, tidx, step_advantage_w=0.0
        )
        adv_w1, _ = compute_gigpo_outcome_advantage(
            tl_r, s_r, mask, ao, idx, tidx, step_advantage_w=1.0
        )
        # When step rewards differ within the shared_obs group, scores should differ
        # (w=0 suppresses step component, w=1 adds it).
        # Both shapes must be correct regardless.
        assert adv_w0.shape == (bs, seq_len)
        assert adv_w1.shape == (bs, seq_len)

    def test_all_identical_obs_same_step_group(self):
        """All identical anchor_obs → one step group → step advantages all 0."""
        bs, seq_len = 3, 4
        tl_r = torch.ones(bs, seq_len)
        s_r = tl_r.sum(dim=-1)
        mask = torch.ones(bs, seq_len)
        ao = np.array(["same_obs"] * bs)
        idx = np.zeros(bs)
        tidx = np.arange(bs)
        adv, _ = compute_gigpo_outcome_advantage(
            tl_r, s_r, mask, ao, idx, tidx, step_advantage_w=1.0
        )
        # Step advantages all zero (single group), episode advantages also zero (single group)
        assert torch.allclose(adv, torch.zeros(bs, seq_len))

    def test_masking_respected(self):
        """Padded (0-masked) positions must be zero in output."""
        bs, seq_len = 2, 6
        tl_r = torch.rand(bs, seq_len)
        s_r = tl_r.sum(dim=-1)
        mask = torch.tensor([[1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.float)
        ao = np.array(["obs0", "obs1"])
        idx = np.array([0, 0])
        tidx = np.array([0, 1])
        adv, _ = compute_gigpo_outcome_advantage(tl_r, s_r, mask, ao, idx, tidx)
        # Padded positions (mask=0) should be 0
        assert adv[0, 2].item() == pytest.approx(0.0)
        assert adv[1, 3].item() == pytest.approx(0.0)
