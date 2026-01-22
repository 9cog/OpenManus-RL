# Copyright 2025 OpenManus-RL Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Process Reward Model (PRM) for evaluating intermediate reasoning steps.

Unlike outcome reward models that only evaluate final results, process reward
models evaluate each intermediate step in the reasoning process, providing
finer-grained feedback for training.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class ProcessRewardConfig:
    """Configuration for Process Reward Model."""

    # Model architecture
    hidden_size: int = 768
    num_layers: int = 2
    dropout: float = 0.1

    # Reward computation
    step_reward_weight: float = 0.3  # Weight for intermediate steps
    outcome_reward_weight: float = 0.7  # Weight for final outcome
    normalize_rewards: bool = True
    reward_scale: float = 1.0

    # Step evaluation criteria
    evaluate_format: bool = True  # Check if format is correct
    evaluate_reasoning: bool = True  # Check reasoning quality
    evaluate_progress: bool = True  # Check if making progress

    # Penalties
    invalid_action_penalty: float = -0.1
    repetition_penalty: float = -0.05
    length_penalty_threshold: int = 50  # Steps before applying penalty


class ProcessRewardModel(nn.Module):
    """
    Process Reward Model that evaluates each step in agent trajectories.

    This model learns to assign rewards to individual reasoning and action steps,
    enabling more fine-grained training signals compared to outcome-only rewards.
    """

    def __init__(self, config: ProcessRewardConfig, tokenizer=None):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        # Simple MLP for step scoring (can be replaced with a transformer)
        self.step_scorer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1),  # Single score per step
        )

    def forward(
        self,
        step_embeddings: torch.Tensor,
        step_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass to compute step-level rewards.

        Args:
            step_embeddings: (batch_size, num_steps, hidden_size)
            step_masks: (batch_size, num_steps) - mask for valid steps

        Returns:
            step_rewards: (batch_size, num_steps)
        """
        # Score each step
        step_scores = self.step_scorer(step_embeddings).squeeze(-1)

        # Apply mask if provided
        if step_masks is not None:
            step_scores = step_scores * step_masks

        return step_scores

    def compute_step_rewards(
        self,
        trajectory: List[Dict[str, Any]],
        final_success: bool,
        final_reward: float,
    ) -> List[float]:
        """
        Compute rewards for each step in a trajectory.

        Args:
            trajectory: List of conversation turns with actions
            final_success: Whether the episode was successful
            final_reward: Final outcome reward

        Returns:
            List of rewards for each step
        """
        step_rewards = []
        seen_actions = set()

        for i, turn in enumerate(trajectory):
            if turn.get("role") != "assistant":
                continue

            content = turn.get("content", "")

            # Base reward
            reward = 0.0

            # Format reward
            if self.config.evaluate_format:
                reward += self._evaluate_format(content)

            # Reasoning reward
            if self.config.evaluate_reasoning:
                reward += self._evaluate_reasoning(content)

            # Progress reward
            if self.config.evaluate_progress:
                reward += self._evaluate_progress(content, i, len(trajectory))

            # Repetition penalty
            action = self._extract_action(content)
            if action in seen_actions:
                reward += self.config.repetition_penalty
            seen_actions.add(action)

            # Length penalty
            if i > self.config.length_penalty_threshold:
                reward += self.config.length_penalty_threshold * -0.01

            step_rewards.append(reward)

        # Combine with outcome reward
        if len(step_rewards) > 0:
            # Distribute some outcome reward to steps
            outcome_bonus = (
                final_reward * self.config.outcome_reward_weight / len(step_rewards)
            )
            step_rewards = [r + outcome_bonus for r in step_rewards]

            # Give extra bonus to last step if successful
            if final_success:
                step_rewards[-1] += final_reward * self.config.outcome_reward_weight

        # Normalize if configured
        if self.config.normalize_rewards and len(step_rewards) > 1:
            mean = np.mean(step_rewards)
            std = np.std(step_rewards) + 1e-8
            step_rewards = [(r - mean) / std for r in step_rewards]

        # Scale rewards
        step_rewards = [r * self.config.reward_scale for r in step_rewards]

        return step_rewards

    def _evaluate_format(self, content: str) -> float:
        """Evaluate if the format is correct (Think/Act structure)."""
        reward = 0.0

        # Check for action tags
        if "<action>" in content and "</action>" in content:
            reward += 0.1
        elif "Act:" in content or "Action:" in content:
            reward += 0.05

        # Check for thinking tags
        if "<think>" in content and "</think>" in content:
            reward += 0.05
        elif "Think:" in content or "Reasoning:" in content:
            reward += 0.02

        return reward

    def _evaluate_reasoning(self, content: str) -> float:
        """Evaluate reasoning quality (simple heuristics)."""
        reward = 0.0

        # Length of thinking (more detailed is better, up to a point)
        thinking_section = self._extract_thinking(content)
        if thinking_section:
            word_count = len(thinking_section.split())
            if 10 <= word_count <= 100:  # Sweet spot
                reward += 0.1
            elif word_count > 0:
                reward += 0.05

        # Keywords indicating good reasoning
        reasoning_keywords = [
            "because",
            "therefore",
            "since",
            "need to",
            "should",
            "will",
            "can",
        ]
        keyword_count = sum(
            1 for keyword in reasoning_keywords if keyword in content.lower()
        )
        reward += min(keyword_count * 0.02, 0.1)

        return reward

    def _evaluate_progress(self, content: str, step: int, total_steps: int) -> float:
        """Evaluate if the action shows progress toward the goal."""
        reward = 0.0

        # Later steps should be more decisive
        if "answer(" in content.lower():
            # Answering is good if we're far enough into the episode
            if step / total_steps > 0.3:  # At least 30% through
                reward += 0.2
            else:
                reward -= 0.1  # Too early to answer

        # Exploration in early steps is good
        if step / total_steps < 0.5:
            exploratory_actions = ["search", "look", "examine", "find", "navigate"]
            if any(action in content.lower() for action in exploratory_actions):
                reward += 0.05

        return reward

    def _extract_action(self, content: str) -> str:
        """Extract action from content."""
        import re

        # Try to extract from <action> tags
        match = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try to extract from Act: format
        match = re.search(r"Act[ion]*:\s*(.+?)(?:\n|$)", content)
        if match:
            return match.group(1).strip()

        return content

    def _extract_thinking(self, content: str) -> str:
        """Extract thinking/reasoning from content."""
        import re

        # Try to extract from <think> tags
        match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try to extract from Think: format
        match = re.search(r"Think[ing]*:\s*(.+?)(?:\nAct|$)", content, re.DOTALL)
        if match:
            return match.group(1).strip()

        return ""


class HeuristicProcessRewardModel:
    """
    A simpler heuristic-based process reward model that doesn't require training.

    This can be used as a baseline or when you don't have labeled step-level data.
    """

    def __init__(self, config: Optional[ProcessRewardConfig] = None):
        self.config = config or ProcessRewardConfig()

    def compute_rewards(
        self,
        trajectory: List[Dict[str, Any]],
        final_success: bool,
        final_reward: float,
        environment_feedback: Optional[List[str]] = None,
    ) -> Tuple[List[float], Dict[str, Any]]:
        """
        Compute heuristic rewards for each step.

        Args:
            trajectory: List of conversation turns
            final_success: Whether episode succeeded
            final_reward: Final outcome reward
            environment_feedback: Optional list of environment observations

        Returns:
            Tuple of (step_rewards, info_dict)
        """
        step_rewards = []
        info = {"format_rewards": [], "reasoning_rewards": [], "progress_rewards": []}

        seen_states = set()
        action_history = []

        for i, turn in enumerate(trajectory):
            if turn.get("role") != "assistant":
                continue

            content = turn.get("content", "")
            reward = 0.0

            # Format reward
            format_reward = self._check_format(content)
            reward += format_reward
            info["format_rewards"].append(format_reward)

            # Reasoning quality
            reasoning_reward = self._check_reasoning_quality(content)
            reward += reasoning_reward
            info["reasoning_rewards"].append(reasoning_reward)

            # Progress reward
            progress_reward = self._check_progress(
                content, i, len(trajectory), environment_feedback, seen_states
            )
            reward += progress_reward
            info["progress_rewards"].append(progress_reward)

            # Invalid action penalty
            if self._is_invalid_action(content):
                reward += self.config.invalid_action_penalty

            # Repetition penalty
            if content in action_history[-3:]:  # Check last 3 actions
                reward += self.config.repetition_penalty

            action_history.append(content)
            step_rewards.append(reward)

        # Redistribute outcome reward
        if len(step_rewards) > 0:
            outcome_per_step = final_reward * 0.3 / len(step_rewards)
            step_rewards = [r + outcome_per_step for r in step_rewards]

            # Bonus to final step
            if final_success:
                step_rewards[-1] += final_reward * 0.7

        info["mean_reward"] = np.mean(step_rewards) if step_rewards else 0
        info["total_reward"] = sum(step_rewards)

        return step_rewards, info

    def _check_format(self, content: str) -> float:
        """Check if content follows expected format."""
        score = 0.0
        if "<action>" in content or "Act:" in content:
            score += 0.05
        if "<think>" in content or "Think:" in content:
            score += 0.05
        return score

    def _check_reasoning_quality(self, content: str) -> float:
        """Simple heuristics for reasoning quality."""
        score = 0.0

        # Presence of reasoning keywords
        if any(
            word in content.lower()
            for word in ["because", "need", "should", "therefore", "since"]
        ):
            score += 0.05

        # Not too short, not too long
        word_count = len(content.split())
        if 20 <= word_count <= 200:
            score += 0.05

        return score

    def _check_progress(
        self, content: str, step: int, total: int, feedback: List[str], seen_states: set
    ) -> float:
        """Check if making progress."""
        score = 0.0

        # Early exploration is good
        if step / total < 0.5 and any(
            w in content.lower() for w in ["search", "look", "explore"]
        ):
            score += 0.05

        # Late decision-making is good
        if step / total > 0.5 and "answer(" in content.lower():
            score += 0.1

        # Check if we're in a new state (progress)
        if feedback and step < len(feedback):
            state = feedback[step]
            if state not in seen_states:
                score += 0.05
                seen_states.add(state)

        return score

    def _is_invalid_action(self, content: str) -> bool:
        """Check if action is likely invalid."""
        # Very simple check - could be enhanced with environment-specific rules
        return len(content.strip()) < 3 or content.strip() == ""


# Utility function to integrate with training
def create_process_reward_model(
    model_type: str = "heuristic", config: Optional[ProcessRewardConfig] = None
) -> Any:
    """
    Factory function to create a process reward model.

    Args:
        model_type: "heuristic" or "learned"
        config: Configuration for the model

    Returns:
        ProcessRewardModel instance
    """
    if config is None:
        config = ProcessRewardConfig()

    if model_type == "heuristic":
        return HeuristicProcessRewardModel(config)
    elif model_type == "learned":
        return ProcessRewardModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
