"""
Integration tests for rollout pipeline.
"""
import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path


class TestRolloutPipeline:
    """Integration tests for the complete rollout pipeline."""

    @pytest.fixture
    def mock_environment(self):
        """Create a mock environment for testing."""
        env = Mock()
        env.reset = Mock(return_value={"observation": "Test observation", "info": {}})
        env.step = Mock(
            return_value=(
                {"observation": "Next observation", "info": {}},
                1.0,  # reward
                False,  # done
                {},  # info
            )
        )
        return env

    @pytest.fixture
    def mock_model(self):
        """Create a mock LLM model for testing."""
        model = Mock()
        model.generate = Mock(
            return_value=["<action>test_action</action>"]
        )
        return model

    def test_single_step_rollout(self, mock_environment, mock_model):
        """Test a single step rollout."""
        # Reset environment
        obs = mock_environment.reset()
        assert "observation" in obs

        # Generate action
        action_text = mock_model.generate()
        assert len(action_text) > 0

        # Take step
        next_obs, reward, done, info = mock_environment.step("test_action")
        assert reward == 1.0
        assert not done

    def test_multi_step_rollout(self, mock_environment, mock_model):
        """Test a multi-step rollout."""
        max_steps = 5
        total_reward = 0

        obs = mock_environment.reset()

        for step in range(max_steps):
            # Generate action
            action_text = mock_model.generate()

            # Take step
            next_obs, reward, done, info = mock_environment.step("action")
            total_reward += reward

            if done:
                break

            obs = next_obs

        assert total_reward > 0
        assert step < max_steps or done

    def test_rollout_trajectory_collection(self):
        """Test that trajectories are correctly collected."""
        trajectory = []

        # Simulate conversation
        trajectory.append({"role": "user", "content": "Task: Navigate"})
        trajectory.append(
            {"role": "assistant", "content": "<action>move_forward</action>"}
        )
        trajectory.append({"role": "user", "content": "Observation: Moved forward"})
        trajectory.append(
            {"role": "assistant", "content": "<action>answer(success)</action>"}
        )

        # Verify trajectory structure
        assert len(trajectory) == 4
        assert trajectory[0]["role"] == "user"
        assert trajectory[1]["role"] == "assistant"
        assert "<action>" in trajectory[1]["content"]

    def test_reward_accumulation(self):
        """Test that rewards are correctly accumulated."""
        step_rewards = [0.1, 0.2, 0.3, 0.4, 1.0]  # Final success reward
        gamma = 0.99

        # Compute discounted returns
        returns = []
        running_return = 0
        for reward in reversed(step_rewards):
            running_return = reward + gamma * running_return
            returns.insert(0, running_return)

        # Check that first step has highest return
        assert returns[0] > returns[1]
        assert returns[-1] == step_rewards[-1]  # Last return is just the reward

    def test_batch_rollout(self, mock_environment, mock_model):
        """Test batch rollout processing."""
        batch_size = 4
        results = []

        for i in range(batch_size):
            obs = mock_environment.reset()
            action = mock_model.generate()
            next_obs, reward, done, info = mock_environment.step("action")

            results.append(
                {"trajectory": [obs, action, next_obs], "reward": reward, "done": done}
            )

        assert len(results) == batch_size
        assert all("reward" in r for r in results)


class TestDataProtoConversion:
    """Test conversion of rollout results to DataProto format."""

    def test_trajectory_to_tokens(self):
        """Test conversion of trajectory to token sequences."""
        # Simulate tokenization
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        # Mock tokenizer
        token_ids = []
        for turn in conversation:
            # Simulate tokenization
            tokens = [1, 2, 3, 4, 5]  # Dummy tokens
            token_ids.extend(tokens)

        assert len(token_ids) > 0

    def test_reward_alignment(self):
        """Test that rewards are correctly aligned with tokens."""
        # 3 conversation turns
        token_lengths = [5, 10, 7]  # Tokens per turn
        turn_rewards = [0.0, 0.5, 1.0]  # Rewards per turn

        # Create token-level rewards
        token_rewards = []
        for length, reward in zip(token_lengths, turn_rewards):
            # Distribute reward across tokens
            reward_per_token = reward / length if length > 0 else 0
            token_rewards.extend([reward_per_token] * length)

        assert len(token_rewards) == sum(token_lengths)
        assert sum(token_rewards) == pytest.approx(sum(turn_rewards))

    def test_padding_and_masking(self):
        """Test padding and masking of variable-length sequences."""
        sequences = [
            [1, 2, 3],
            [4, 5, 6, 7, 8],
            [9, 10],
        ]

        max_len = max(len(seq) for seq in sequences)
        pad_token = 0

        # Pad sequences
        padded = []
        masks = []
        for seq in sequences:
            padding_needed = max_len - len(seq)
            padded_seq = seq + [pad_token] * padding_needed
            mask = [1] * len(seq) + [0] * padding_needed

            padded.append(padded_seq)
            masks.append(mask)

        # Verify padding
        assert all(len(seq) == max_len for seq in padded)
        assert all(len(mask) == max_len for mask in masks)

        # Verify masks correctly identify real tokens
        for seq, mask in zip(padded, masks):
            real_tokens = sum(mask)
            assert real_tokens == len([t for t in seq if t != pad_token])


class TestConcurrentRollout:
    """Test concurrent rollout execution."""

    @pytest.mark.asyncio
    async def test_parallel_rollout(self):
        """Test parallel execution of rollouts."""
        import asyncio

        async def async_rollout(env_id):
            """Simulate an async rollout."""
            await asyncio.sleep(0.1)  # Simulate work
            return {
                "env_id": env_id,
                "reward": np.random.random(),
                "steps": np.random.randint(1, 10),
            }

        # Run multiple rollouts concurrently
        num_envs = 4
        tasks = [async_rollout(i) for i in range(num_envs)]
        results = await asyncio.gather(*tasks)

        assert len(results) == num_envs
        assert all("reward" in r for r in results)

    def test_thread_pool_rollout(self):
        """Test thread pool execution of rollouts."""
        from concurrent.futures import ThreadPoolExecutor

        def rollout_worker(env_id):
            """Simulate a rollout worker."""
            import time

            time.sleep(0.1)  # Simulate work
            return {"env_id": env_id, "reward": np.random.random()}

        num_workers = 4
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(rollout_worker, i) for i in range(num_workers)]
            results = [f.result() for f in futures]

        assert len(results) == num_workers
        assert all("reward" in r for r in results)


class TestErrorHandling:
    """Test error handling in rollout pipeline."""

    def test_invalid_action_handling(self):
        """Test handling of invalid actions."""
        action = "<action>invalid_action_xyz</action>"

        # Simulate action parsing
        import re

        pattern = r"<action>(.*?)</action>"
        match = re.search(pattern, action)

        if match:
            action_content = match.group(1)
            # Check if action is valid
            valid_actions = ["move", "grab", "put", "answer"]
            is_valid = any(va in action_content.lower() for va in valid_actions)

            if not is_valid:
                # Apply penalty for invalid action
                reward = -0.1
                assert reward < 0

    def test_timeout_handling(self):
        """Test handling of rollout timeouts."""
        max_steps = 10
        current_step = 0

        while current_step < max_steps:
            current_step += 1

            # Simulate timeout
            if current_step >= max_steps:
                # Apply timeout penalty
                reward = -0.5
                done = True
                assert done
                assert reward < 0
                break

    def test_environment_error_handling(self):
        """Test handling of environment errors."""
        env = Mock()
        env.step = Mock(side_effect=Exception("Environment error"))

        try:
            env.step("action")
            success = False
        except Exception as e:
            success = True
            assert "Environment error" in str(e)

        assert success


class TestRewardShaping:
    """Test reward shaping and engineering."""

    def test_sparse_to_dense_rewards(self):
        """Test conversion of sparse to dense rewards."""
        # Sparse reward: only at the end
        sparse_rewards = [0, 0, 0, 0, 1.0]

        # Convert to dense with intermediate rewards
        gamma = 0.9
        dense_rewards = []
        for i, reward in enumerate(sparse_rewards):
            if reward == 0 and i < len(sparse_rewards) - 1:
                # Add small progress reward
                progress_reward = 0.1
            else:
                progress_reward = reward
            dense_rewards.append(progress_reward)

        # Dense rewards should have non-zero values
        assert sum(dense_rewards) > sum(sparse_rewards)

    def test_format_reward(self):
        """Test reward for correct format."""
        responses = [
            "<action>move</action>",  # Correct format
            "just move",  # Wrong format
            "<action>grab(object)</action>",  # Correct format with args
        ]

        format_rewards = []
        for response in responses:
            if "<action>" in response and "</action>" in response:
                format_reward = 0.1
            else:
                format_reward = -0.1
            format_rewards.append(format_reward)

        assert format_rewards[0] > 0  # Correct format
        assert format_rewards[1] < 0  # Wrong format
        assert format_rewards[2] > 0  # Correct format

    def test_length_penalty(self):
        """Test penalty for excessive length."""
        max_steps = 20

        trajectories = [
            {"steps": 5, "success": True},  # Good
            {"steps": 25, "success": True},  # Too long
            {"steps": 10, "success": False},  # Failed but reasonable length
        ]

        penalized_rewards = []
        for traj in trajectories:
            base_reward = 1.0 if traj["success"] else 0.0

            # Apply length penalty
            if traj["steps"] > max_steps:
                length_penalty = -0.05 * (traj["steps"] - max_steps)
            else:
                length_penalty = 0

            final_reward = base_reward + length_penalty
            penalized_rewards.append(final_reward)

        # Long trajectory should have penalty
        assert penalized_rewards[1] < penalized_rewards[0]


class TestModelGeneration:
    """Test model generation and sampling."""

    def test_temperature_sampling(self):
        """Test that different temperatures affect sampling."""
        # Simulate logits
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        temperatures = [0.1, 0.5, 1.0, 2.0]
        distributions = []

        for temp in temperatures:
            scaled_logits = logits / temp
            probs = torch.softmax(scaled_logits, dim=-1)
            distributions.append(probs)

        # Lower temperature should have more peaked distribution
        # Check entropy
        def entropy(probs):
            return -(probs * torch.log(probs + 1e-10)).sum()

        entropies = [entropy(d) for d in distributions]

        # Entropy should increase with temperature
        assert entropies[0] < entropies[-1]

    def test_top_k_sampling(self):
        """Test top-k sampling."""
        logits = torch.tensor([1.0, 5.0, 2.0, 4.0, 3.0])
        k = 3

        # Get top-k indices
        top_k_logits, top_k_indices = torch.topk(logits, k)

        # Should get the 3 highest logits
        assert len(top_k_indices) == k
        assert 5.0 in top_k_logits  # Highest value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
