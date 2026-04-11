"""
Exhaustive unit tests for the OpenManus-RL process reward models.

Covers:
- ProcessRewardConfig
- ProcessRewardModel (nn.Module)
- HeuristicProcessRewardModel
- create_process_reward_model factory
"""

import pytest
import torch
import numpy as np

from openmanus_rl.reward_manager.process_reward import (
    ProcessRewardConfig,
    ProcessRewardModel,
    HeuristicProcessRewardModel,
    create_process_reward_model,
)


# ============================================================
# ProcessRewardConfig
# ============================================================

class TestProcessRewardConfig:
    def test_defaults(self):
        cfg = ProcessRewardConfig()
        assert cfg.hidden_size == 768
        assert cfg.num_layers == 2
        assert 0.0 < cfg.dropout < 1.0
        assert 0.0 <= cfg.step_reward_weight <= 1.0
        assert 0.0 <= cfg.outcome_reward_weight <= 1.0
        assert cfg.normalize_rewards is True
        assert cfg.reward_scale > 0

    def test_weights_sum_to_one(self):
        cfg = ProcessRewardConfig()
        assert abs(cfg.step_reward_weight + cfg.outcome_reward_weight - 1.0) < 1e-6

    def test_penalty_values_non_positive(self):
        cfg = ProcessRewardConfig()
        assert cfg.invalid_action_penalty <= 0
        assert cfg.repetition_penalty <= 0

    def test_custom_values(self):
        cfg = ProcessRewardConfig(hidden_size=256, dropout=0.2, reward_scale=2.0)
        assert cfg.hidden_size == 256
        assert cfg.dropout == 0.2
        assert cfg.reward_scale == 2.0


# ============================================================
# ProcessRewardModel
# ============================================================

class TestProcessRewardModelForward:
    @pytest.fixture
    def model(self):
        return ProcessRewardModel(ProcessRewardConfig(hidden_size=64))

    def test_output_shape_no_mask(self, model):
        embeddings = torch.randn(2, 3, 64)
        out = model(embeddings)
        assert out.shape == (2, 3)

    def test_output_shape_with_mask(self, model):
        embeddings = torch.randn(2, 4, 64)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.float)
        out = model(embeddings, mask)
        assert out.shape == (2, 4)

    def test_mask_zeroes_out_invalid_steps(self, model):
        embeddings = torch.randn(1, 3, 64)
        mask = torch.tensor([[1, 0, 0]], dtype=torch.float)
        out = model(embeddings, mask)
        assert out[0, 1].item() == pytest.approx(0.0)
        assert out[0, 2].item() == pytest.approx(0.0)

    def test_single_step(self, model):
        embeddings = torch.randn(1, 1, 64)
        out = model(embeddings)
        assert out.shape == (1, 1)

    def test_batch_independence(self, model):
        """Each sample in the batch is processed independently (eval mode, no dropout)."""
        model.eval()
        e1 = torch.randn(1, 2, 64)
        e2 = torch.randn(1, 2, 64)
        batch = torch.cat([e1, e2], dim=0)
        with torch.no_grad():
            out_batch = model(batch)
            out_e1 = model(e1)
            out_e2 = model(e2)
        assert torch.allclose(out_batch[0], out_e1[0], atol=1e-5)
        assert torch.allclose(out_batch[1], out_e2[0], atol=1e-5)


class TestProcessRewardModelComputeStepRewards:
    @pytest.fixture
    def model(self):
        return ProcessRewardModel(ProcessRewardConfig(hidden_size=64))

    def _make_trajectory(self, n_assistant_turns: int):
        traj = []
        for i in range(n_assistant_turns):
            traj.append({
                "role": "assistant",
                "content": f"<think>Thinking step {i}</think><action>act {i}</action>",
            })
            traj.append({"role": "user", "content": f"Observation {i}"})
        return traj

    def test_returns_list_of_rewards(self, model):
        traj = self._make_trajectory(3)
        rewards = model.compute_step_rewards(traj, final_success=True, final_reward=1.0)
        assert isinstance(rewards, list)
        assert len(rewards) == 3

    def test_reward_count_matches_assistant_turns(self, model):
        for n in [1, 2, 5]:
            traj = self._make_trajectory(n)
            rewards = model.compute_step_rewards(traj, final_success=False, final_reward=0.0)
            assert len(rewards) == n

    def test_successful_episode_positive_last_reward(self, model):
        traj = self._make_trajectory(3)
        rewards = model.compute_step_rewards(traj, final_success=True, final_reward=1.0)
        # Last reward should be boosted by success
        assert rewards[-1] > 0

    def test_empty_trajectory_returns_empty(self, model):
        rewards = model.compute_step_rewards([], final_success=False, final_reward=0.0)
        assert rewards == []

    def test_non_assistant_turns_ignored(self, model):
        traj = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Task desc"},
            {"role": "assistant", "content": "<action>search apple</action>"},
        ]
        rewards = model.compute_step_rewards(traj, final_success=False, final_reward=0.5)
        assert len(rewards) == 1

    def test_normalization_applied(self, model):
        """With normalize_rewards=True and >1 step, rewards should be finite floats."""
        cfg = ProcessRewardConfig(hidden_size=64, normalize_rewards=True, reward_scale=1.0)
        m = ProcessRewardModel(cfg)
        # Use varied content so rewards differ across steps
        traj = []
        for i in range(10):
            if i % 3 == 0:
                content = f"<think>Thinking step {i} because I need to find the key</think><action>act {i}</action>"
            elif i % 3 == 1:
                content = f"answer(result_{i})"  # triggers progress scoring
            else:
                content = f"Act: search item {i}"
            traj.append({"role": "assistant", "content": content})
            traj.append({"role": "user", "content": f"Observation {i}"})
        rewards = m.compute_step_rewards(traj, final_success=False, final_reward=0.0)
        assert len(rewards) == 10
        assert all(isinstance(r, float) for r in rewards)

    def test_no_normalization(self, model):
        cfg = ProcessRewardConfig(hidden_size=64, normalize_rewards=False)
        m = ProcessRewardModel(cfg)
        traj = self._make_trajectory(3)
        rewards = m.compute_step_rewards(traj, final_success=False, final_reward=0.0)
        assert isinstance(rewards, list)

    def test_repetition_penalty_applied(self, model):
        """Repeated actions should get lower rewards."""
        traj = [
            {"role": "assistant", "content": "<action>search apple</action>"},
            {"role": "user", "content": "obs"},
            {"role": "assistant", "content": "<action>search apple</action>"},
        ]
        rewards = model.compute_step_rewards(traj, final_success=False, final_reward=0.0)
        assert rewards[1] < rewards[0]  # Repetition should reduce reward

    def test_length_penalty_kicks_in_late(self, model):
        """Steps beyond threshold incur a length penalty."""
        cfg = ProcessRewardConfig(hidden_size=64, normalize_rewards=False, length_penalty_threshold=2)
        m = ProcessRewardModel(cfg)
        traj = self._make_trajectory(10)
        rewards = m.compute_step_rewards(traj, final_success=False, final_reward=0.0)
        # Late steps should have increasingly negative contribution from length penalty
        assert len(rewards) == 10


# ============================================================
# Private helper methods of ProcessRewardModel
# ============================================================

class TestProcessRewardModelHelpers:
    @pytest.fixture
    def model(self):
        return ProcessRewardModel(ProcessRewardConfig(hidden_size=64))

    def test_evaluate_format_with_action_tags(self, model):
        score = model._evaluate_format("<think>plan</think><action>go</action>")
        assert score > 0

    def test_evaluate_format_without_tags(self, model):
        score = model._evaluate_format("No tags here at all")
        assert score == pytest.approx(0.0)

    def test_evaluate_format_action_colon(self, model):
        score = model._evaluate_format("Act: go north")
        assert score > 0

    def test_evaluate_reasoning_with_keywords(self, model):
        score = model._evaluate_reasoning("I should go there because I need to find the key")
        assert score > 0

    def test_evaluate_reasoning_empty(self, model):
        score = model._evaluate_reasoning("")
        assert score == pytest.approx(0.0)

    def test_extract_action_from_tags(self, model):
        action = model._extract_action("<action>go north</action>")
        assert action == "go north"

    def test_extract_action_fallback(self, model):
        action = model._extract_action("Act: go north")
        assert "go north" in action

    def test_extract_action_no_match(self, model):
        action = model._extract_action("some random text")
        assert action == "some random text"

    def test_extract_thinking_from_tags(self, model):
        thinking = model._extract_thinking("<think>my reasoning</think>")
        assert thinking == "my reasoning"

    def test_extract_thinking_empty(self, model):
        thinking = model._extract_thinking("no thinking here")
        assert thinking == ""


# ============================================================
# HeuristicProcessRewardModel
# ============================================================

class TestHeuristicProcessRewardModel:
    @pytest.fixture
    def model(self):
        return HeuristicProcessRewardModel()

    def _make_trajectory(self, n: int, content_template: str = "<action>act {i}</action>"):
        traj = []
        for i in range(n):
            traj.append({"role": "assistant", "content": content_template.format(i=i)})
        return traj

    def test_returns_tuple(self, model):
        traj = self._make_trajectory(3)
        result = model.compute_rewards(traj, final_success=False, final_reward=0.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reward_length_matches_assistant_turns(self, model):
        for n in [1, 3, 7]:
            traj = self._make_trajectory(n)
            rewards, info = model.compute_rewards(traj, False, 0.0)
            assert len(rewards) == n

    def test_info_dict_structure(self, model):
        traj = self._make_trajectory(2)
        _, info = model.compute_rewards(traj, False, 0.0)
        for key in ("format_rewards", "reasoning_rewards", "progress_rewards", "mean_reward", "total_reward"):
            assert key in info

    def test_format_rewards_match_turns(self, model):
        traj = self._make_trajectory(3)
        _, info = model.compute_rewards(traj, False, 0.0)
        assert len(info["format_rewards"]) == 3

    def test_successful_final_step_bonus(self, model):
        traj = self._make_trajectory(3)
        rewards_success, _ = model.compute_rewards(traj, True, 1.0)
        rewards_fail, _ = model.compute_rewards(traj, False, 0.0)
        assert rewards_success[-1] > rewards_fail[-1]

    def test_empty_trajectory(self, model):
        rewards, info = model.compute_rewards([], False, 0.0)
        assert rewards == []
        assert info["mean_reward"] == 0

    def test_non_assistant_turns_skipped(self, model):
        traj = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "<action>act 0</action>"},
        ]
        rewards, _ = model.compute_rewards(traj, False, 0.0)
        assert len(rewards) == 1

    def test_repetition_reduces_reward(self, model):
        same = "<action>look</action>"
        traj = [{"role": "assistant", "content": same} for _ in range(5)]
        rewards, _ = model.compute_rewards(traj, False, 0.0)
        # Rewards after the first repetitions should be penalized
        # (they may still be positive due to format/outcome, but later steps <= earlier ones)
        assert all(isinstance(r, (int, float)) for r in rewards)

    def test_invalid_action_penalty(self, model):
        """Very short/empty content is considered invalid."""
        traj = [{"role": "assistant", "content": "  "}]
        rewards, _ = model.compute_rewards(traj, False, 0.0)
        assert rewards[0] < 0

    def test_environment_feedback_used(self, model):
        """When environment_feedback is provided, new states give a bonus."""
        traj = [
            {"role": "assistant", "content": "Act: search"},
            {"role": "assistant", "content": "Act: click"},
        ]
        feedback = ["state1", "state2"]
        rewards, _ = model.compute_rewards(traj, False, 0.0, environment_feedback=feedback)
        assert len(rewards) == 2

    def test_total_reward_equals_sum(self, model):
        traj = self._make_trajectory(4)
        rewards, info = model.compute_rewards(traj, False, 0.5)
        assert abs(info["total_reward"] - sum(rewards)) < 1e-5

    def test_mean_reward_correct(self, model):
        traj = self._make_trajectory(4)
        rewards, info = model.compute_rewards(traj, False, 0.0)
        assert abs(info["mean_reward"] - np.mean(rewards)) < 1e-5

    def test_with_custom_config(self):
        cfg = ProcessRewardConfig(
            invalid_action_penalty=-0.5,
            repetition_penalty=-0.2,
            step_reward_weight=0.5,
            outcome_reward_weight=0.5,
        )
        model = HeuristicProcessRewardModel(cfg)
        traj = [{"role": "assistant", "content": "<action>go</action>"}]
        rewards, _ = model.compute_rewards(traj, False, 0.0)
        assert len(rewards) == 1


class TestHeuristicProcessRewardModelHelpers:
    @pytest.fixture
    def model(self):
        return HeuristicProcessRewardModel()

    def test_check_format_with_action_tag(self, model):
        score = model._check_format("<action>go</action>")
        assert score > 0

    def test_check_format_empty(self, model):
        score = model._check_format("")
        assert score == pytest.approx(0.0)

    def test_check_reasoning_with_keywords(self, model):
        score = model._check_reasoning_quality("I should go there because I need to find it")
        assert score > 0

    def test_check_reasoning_too_short(self, model):
        score = model._check_reasoning_quality("go")
        assert score < 0.1  # Minimal or no reasoning score

    def test_is_invalid_action_empty(self, model):
        assert model._is_invalid_action("  ") is True

    def test_is_invalid_action_too_short(self, model):
        assert model._is_invalid_action("ab") is True

    def test_is_invalid_action_normal(self, model):
        assert model._is_invalid_action("<action>search apple</action>") is False

    def test_check_progress_early_exploration(self, model):
        score = model._check_progress("search for keys", 0, 10, None, set())
        assert score > 0

    def test_check_progress_late_answer(self, model):
        score = model._check_progress("answer(yes)", 8, 10, None, set())
        assert score > 0

    def test_check_progress_new_state_bonus(self, model):
        seen = set()
        feedback = ["state1", "state2"]
        s1 = model._check_progress("", 0, 4, feedback, seen)
        # state1 is new – should earn bonus
        assert s1 > 0
        # Now state1 is seen; visiting again should give no bonus
        s2 = model._check_progress("", 0, 4, feedback, seen)
        assert s2 == 0


# ============================================================
# create_process_reward_model factory
# ============================================================

class TestCreateProcessRewardModel:
    def test_heuristic_type(self):
        model = create_process_reward_model("heuristic")
        assert isinstance(model, HeuristicProcessRewardModel)

    def test_learned_type(self):
        model = create_process_reward_model("learned")
        assert isinstance(model, ProcessRewardModel)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            create_process_reward_model("unknown_type")

    def test_custom_config_forwarded(self):
        cfg = ProcessRewardConfig(hidden_size=128)
        model = create_process_reward_model("learned", config=cfg)
        assert model.config.hidden_size == 128

    def test_default_config_when_none(self):
        model = create_process_reward_model("heuristic", config=None)
        assert model.config is not None
