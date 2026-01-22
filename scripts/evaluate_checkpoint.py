"""
Model Checkpoint Evaluation Tool

This script evaluates model checkpoints across multiple environments and provides
comprehensive performance metrics.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime


class CheckpointEvaluator:
    """Evaluate model checkpoints on multiple environments."""

    def __init__(
        self,
        checkpoint_path: str,
        environments: List[str],
        output_dir: str = "evaluation_results",
        num_tasks: int = 50,
        max_steps: int = 50,
        batch_size: int = 4,
    ):
        self.checkpoint_path = checkpoint_path
        self.environments = environments
        self.output_dir = Path(output_dir)
        self.num_tasks = num_tasks
        self.max_steps = max_steps
        self.batch_size = batch_size

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_on_environment(
        self, env_name: str, model_path: str
    ) -> Dict[str, Any]:
        """
        Run evaluation on a specific environment.

        Args:
            env_name: Name of environment (alfworld, webshop, gaia)
            model_path: Path to model checkpoint

        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n{'='*80}")
        print(f"Evaluating on {env_name}")
        print(f"{'='*80}\n")

        # Prepare output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"{env_name}_{timestamp}.jsonl"

        # Build command
        cmd = [
            "python",
            "scripts/rollout/openmanus_rollout.py",
            "--env",
            env_name,
            "--model",
            model_path,
            "--batch_size",
            str(self.batch_size),
            "--total_envs",
            str(self.num_tasks),
            "--max_steps",
            str(self.max_steps),
            "--dump_path",
            str(output_file),
        ]

        # Add environment-specific parameters
        if env_name == "alfworld":
            cmd.extend(["--unique_envs"])
        elif env_name == "gaia":
            cmd.extend(["--gaia_tools", "python_code_generator"])

        # Run evaluation
        try:
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True, timeout=3600
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error running evaluation: {e}")
            print(f"Output: {e.output}")
            return {"error": str(e)}
        except subprocess.TimeoutExpired:
            print(f"Evaluation timed out after 1 hour")
            return {"error": "Timeout"}

        # Parse results
        try:
            results = self._parse_results(output_file)
            return results
        except Exception as e:
            print(f"Error parsing results: {e}")
            return {"error": str(e)}

    def _parse_results(self, results_file: Path) -> Dict[str, Any]:
        """Parse evaluation results from JSONL file."""
        df = pd.read_json(results_file, lines=True)

        metrics = {
            "num_tasks": len(df),
            "success_rate": df["success"].mean() if "success" in df.columns else 0,
            "avg_reward": df["total_reward"].mean()
            if "total_reward" in df.columns
            else 0,
            "avg_steps": df["num_steps"].mean() if "num_steps" in df.columns else 0,
            "min_reward": df["total_reward"].min()
            if "total_reward" in df.columns
            else 0,
            "max_reward": df["total_reward"].max()
            if "total_reward" in df.columns
            else 0,
            "std_reward": df["total_reward"].std()
            if "total_reward" in df.columns
            else 0,
        }

        # Add distribution of steps
        if "num_steps" in df.columns:
            metrics["steps_distribution"] = {
                "25th": df["num_steps"].quantile(0.25),
                "50th": df["num_steps"].quantile(0.50),
                "75th": df["num_steps"].quantile(0.75),
            }

        return metrics

    def evaluate_all(self) -> Dict[str, Dict[str, Any]]:
        """Evaluate checkpoint on all environments."""
        all_results = {}

        for env in self.environments:
            results = self.evaluate_on_environment(env, self.checkpoint_path)
            all_results[env] = results

        # Save combined results
        self._save_summary(all_results)

        return all_results

    def _save_summary(self, results: Dict[str, Dict[str, Any]]):
        """Save evaluation summary to JSON."""
        summary = {
            "checkpoint": self.checkpoint_path,
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }

        summary_file = self.output_dir / "evaluation_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*80}")
        print("Evaluation Summary")
        print(f"{'='*80}")
        print(f"\nCheckpoint: {self.checkpoint_path}\n")

        for env, metrics in results.items():
            if "error" in metrics:
                print(f"{env}: ERROR - {metrics['error']}")
                continue

            print(f"{env}:")
            print(f"  Success Rate: {metrics['success_rate']:.2%}")
            print(f"  Avg Reward: {metrics['avg_reward']:.2f}")
            print(f"  Avg Steps: {metrics['avg_steps']:.1f}")
            print()

        print(f"Full results saved to: {summary_file}")


def compare_checkpoints(
    checkpoint_paths: List[str],
    environments: List[str],
    output_dir: str = "comparison_results",
) -> pd.DataFrame:
    """
    Compare multiple checkpoints across environments.

    Args:
        checkpoint_paths: List of checkpoint paths to compare
        environments: List of environments to evaluate on
        output_dir: Directory to save comparison results

    Returns:
        DataFrame with comparison results
    """
    results = []

    for checkpoint in checkpoint_paths:
        evaluator = CheckpointEvaluator(
            checkpoint_path=checkpoint,
            environments=environments,
            output_dir=f"{output_dir}/{Path(checkpoint).name}",
        )

        checkpoint_results = evaluator.evaluate_all()

        for env, metrics in checkpoint_results.items():
            if "error" not in metrics:
                results.append(
                    {
                        "checkpoint": Path(checkpoint).name,
                        "environment": env,
                        **metrics,
                    }
                )

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)

    # Save to CSV
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_file = output_path / "checkpoint_comparison.csv"
    comparison_df.to_csv(comparison_file, index=False)

    print(f"\nComparison results saved to: {comparison_file}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("Checkpoint Comparison")
    print("=" * 80)
    print(
        comparison_df.pivot_table(
            values="success_rate",
            index="checkpoint",
            columns="environment",
            aggfunc="mean",
        )
    )

    return comparison_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate model checkpoints")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--environments",
        nargs="+",
        default=["alfworld", "webshop"],
        help="Environments to evaluate on",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--num_tasks", type=int, default=50, help="Number of tasks per environment"
    )
    parser.add_argument(
        "--max_steps", type=int, default=50, help="Max steps per episode"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--compare",
        nargs="+",
        help="Compare multiple checkpoints (provide multiple paths)",
    )

    args = parser.parse_args()

    if args.compare:
        # Compare mode
        compare_checkpoints(
            checkpoint_paths=args.compare,
            environments=args.environments,
            output_dir=args.output_dir,
        )
    else:
        # Single checkpoint evaluation
        evaluator = CheckpointEvaluator(
            checkpoint_path=args.checkpoint,
            environments=args.environments,
            output_dir=args.output_dir,
            num_tasks=args.num_tasks,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
        )

        evaluator.evaluate_all()


if __name__ == "__main__":
    main()
