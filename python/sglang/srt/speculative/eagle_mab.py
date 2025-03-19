"""Multi-Armed Bandit implementation for adaptive speculative decoding."""

import math
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class MetricsEntry:
    """A single entry of metrics data for one step."""

    reward: Optional[float] = None
    goodput: Optional[float] = None
    accept_length: Optional[float] = None
    draft_time: Optional[float] = None
    verify_time: Optional[float] = None
    draft_extend_time: Optional[float] = None
    other_time: Optional[float] = None
    total_time: Optional[float] = None

    # Dictionary for any additional metrics that might be added
    additional_metrics: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, metrics_dict: Dict[str, float]) -> "MetricsEntry":
        """Create a MetricsEntry from a dictionary."""
        # Create a clean entry
        entry = cls()

        # Standard metrics mapping
        standard_metrics = {
            "reward": "reward",
            "goodput": "goodput",
            "accept_length": "accept_length",
            "draft_time": "draft_time",
            "verify_time": "verify_time",
            "draft_extend_time": "draft_extend_time",
            "other_time": "other_time",
            "total_time": "total_time",
        }

        # Process each metric in the dictionary
        for key, value in metrics_dict.items():
            if key in standard_metrics:
                # Set attribute directly using setattr
                setattr(entry, standard_metrics[key], value)
            else:
                # Store in additional_metrics
                entry.additional_metrics[key] = value

        return entry


@dataclass
class MABMetricsManager:
    """Metrics data container and service class.

    Stores time-series data for each metric type with a sliding window and
    provides methods for metrics calculations and management.
    """

    def __init__(self, window_size: int):
        """Initialize deques with proper maxlen."""
        self.window_size = window_size
        self.steps: Deque[int] = deque(maxlen=self.window_size)
        self.metric_values: Dict[str, Deque[float]] = {}

    def add_metric_value(self, metric_name: str, value: Optional[float]):
        """Add a value for a specific metric."""
        if value is None:
            return

        if metric_name not in self.metric_values:
            self.metric_values[metric_name] = deque(maxlen=self.window_size)

        self.metric_values[metric_name].append(value)

    def get_all_metric_names(self) -> List[str]:
        """Get all metric names that have data."""
        return list(self.metric_values.keys())

    def expire_old_data(self, current_step: int):
        """Remove data points older than window_size steps."""
        cutoff = current_step - self.window_size
        while self.steps and self.steps[0] < cutoff:
            self.steps.popleft()

            if "reward" in self.metric_values:
                self.metric_values["reward"].popleft()

    def has_metric_data(self, metric_name: str) -> bool:
        """Check if a metric exists and has data.

        Args:
            metric_name: Name of the metric to check

        Returns:
            True if the metric exists and has data, False otherwise
        """
        return (
            metric_name in self.metric_values
            and len(self.metric_values[metric_name]) > 0
        )

    def trimmed_mean(
        self, metric_name: str, trim_percent: float = 0.1
    ) -> Optional[float]:
        """Calculate trimmed mean for the specified metric."""
        if not self.has_metric_data(metric_name):
            return None

        values = self.metric_values[metric_name]
        sorted_vals = sorted(values)
        n_trim = int(len(sorted_vals) * trim_percent)
        return (
            np.mean(sorted_vals[n_trim:-n_trim])
            if len(sorted_vals) > 2 * n_trim
            else np.mean(sorted_vals)
        )

    def get_stats(self, metric_name: str) -> Optional[Dict[str, float]]:
        """Calculate statistics for the specified metric.

        Returns:
            Dictionary with statistics (mean, std, median, 25th, 75th percentiles)
            or None if no data available
        """
        if not self.has_metric_data(metric_name):
            return None

        values = np.array(self.metric_values[metric_name])
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)) if len(values) > 1 else 0.0,
            "median": float(np.median(values)),
            "25th": float(np.percentile(values, 25)),
            "75th": float(np.percentile(values, 75)),
        }

    def median(self, metric_name: str) -> Optional[float]:
        """Calculate median for the specified metric."""
        if not self.has_metric_data(metric_name):
            return None

        return float(np.median(self.metric_values[metric_name]))

    def add_single_step_metrics(self, metrics_entry: MetricsEntry, current_step: int):
        """Add metrics for a single step in the sliding window.

        Args:
            metrics_entry: MetricsEntry object containing the metrics for this step
            current_step: Current step number for sliding window
        """
        # Record the step
        self.steps.append(current_step)

        # Add standard metrics from the MetricsEntry object
        for field_name, field_value in vars(metrics_entry).items():
            # Skip the additional_metrics field as we'll handle it separately
            if field_name == "additional_metrics" or field_value is None:
                continue
            self.add_metric_value(field_name, field_value)

        # Add any additional metrics
        for name, value in metrics_entry.additional_metrics.items():
            self.add_metric_value(name, value)

        # Expire old data if needed
        self.expire_old_data(current_step)


class BaseMAB:
    """Base class for Multi-Armed Bandit implementations.

    Each MAB instance manages strategy selection within a specific group (e.g., batch size group).
    Each instance maintains its own metrics for each strategy and handles its own time window.
    """

    def __init__(self, strategies: List[str], window_size: int = 1000):
        self.strategies = strategies
        self.window_size = window_size
        self.current_step = 0  # Track steps per MAB instance
        # Initialize metrics for each strategy
        self.strategy_metrics = {s: MABMetricsManager(window_size) for s in strategies}

    def add_single_step_metrics(self, strategy: str, metrics_entry: MetricsEntry):
        """Add metrics for a single step and advance the step counter atomically.

        Args:
            strategy: The strategy for which to record metrics
            metrics_entry: MetricsEntry object containing metrics to record
        """
        self.strategy_metrics[strategy].add_single_step_metrics(
            metrics_entry, self.current_step
        )
        self.current_step += 1

    def _expire_old_data(self):
        """Expire old data across all metrics."""
        for metrics_data in self.strategy_metrics.values():
            metrics_data.expire_old_data(self.current_step)

    def select_strategy(self, valid_strategies: List[str]) -> str:
        """Choose a strategy based on the algorithm and filtering rules.

        Args:
            valid_strategies: List of valid strategies to choose from

        Returns:
            Selected strategy
        """
        raise NotImplementedError


class EpsilonGreedyMAB(BaseMAB):
    """Epsilon-Greedy implementation of MAB."""

    def __init__(self, strategies: List[str], epsilon: float = 0.1, **kwargs):
        super().__init__(strategies, **kwargs)
        self.epsilon = epsilon

    def select_strategy(self, valid_strategies: List[str]) -> str:
        """Choose strategy using epsilon-greedy approach."""
        # First expire old data for all metrics
        self._expire_old_data()

        if np.random.random() < self.epsilon:
            return np.random.choice(valid_strategies)

        # Get median reward for each strategy. If None, force exploration with 'inf'
        strategy_scores = {
            s: self.strategy_metrics[s].median("reward") or float("inf")
            for s in valid_strategies
        }
        return max(strategy_scores.items(), key=lambda x: x[1])[0]


class UCB1MAB(BaseMAB):
    """UCB1 implementation of MAB."""

    def __init__(self, strategies: List[str], **kwargs):
        super().__init__(strategies, **kwargs)

    def select_strategy(self, valid_strategies: List[str]) -> str:
        """Choose strategy using UCB1 algorithm."""
        # First expire old data for all metrics
        self._expire_old_data()

        # Calculate total pulls based on current window sizes
        total_pulls = sum(
            len(metrics.metric_values.get("reward", []))
            for metrics in self.strategy_metrics.values()
        )

        # Calculate UCB scores for each strategy
        strategy_scores = {
            s: self._ucb_score(self.strategy_metrics[s], total_pulls)
            for s in valid_strategies
        }
        return max(strategy_scores.items(), key=lambda x: x[1])[0]

    def _ucb_score(self, metrics: MABMetricsManager, total_pulls: int) -> float:
        """Calculate UCB score for a strategy."""
        n_i = len(metrics.metric_values.get("reward", []))
        if n_i == 0:  # force exploration if no recent data
            return float("inf")

        # if trimmed mean is None, use inf to force exploration
        r_hat = metrics.trimmed_mean("reward") or float("inf")
        return r_hat + math.sqrt(2 * math.log(total_pulls) / n_i)


class MABGroupManager:
    """Manages groups and their associated MAB instances.

    Each group (e.g., batch size group) has its own MAB instance that learns
    independently which strategies work best for that group.
    """

    # Map algorithm names to their factory functions
    ALGORITHM_FACTORIES = {
        "EG": lambda strategies, window_size: EpsilonGreedyMAB(
            strategies=strategies, window_size=window_size
        ),
        "UCB1": lambda strategies, window_size: UCB1MAB(
            strategies=strategies, window_size=window_size
        ),
    }

    def __init__(
        self,
        groups: List[int],
        strategies: List[str],
        algorithm: str = "EG",
        window_size: int = 1000,
        output_dir: str = "mab_plots",
    ):
        self.groups = sorted(groups)

        # get 'MAB_RESULTS_DIR' from environment variable
        self.output_dir = Path(os.getenv("MAB_RESULTS_DIR", ".")) / output_dir
        self.strategies = strategies

        # Validate algorithm type
        algorithm = algorithm.upper()
        if algorithm not in self.ALGORITHM_FACTORIES:
            raise ValueError(
                f"Unsupported MAB algorithm: {algorithm}. "
                f"Must be one of: {', '.join(sorted(self.ALGORITHM_FACTORIES.keys()))}"
            )
        self.algorithm = algorithm

        # Generate a descriptive name based on algorithm and strategies
        self.name = f"{algorithm.lower()},{','.join(strategies)}"

        self.window_size = window_size
        self.output_dir = output_dir

        # Initialize MAB instances
        self.mabs: Dict[int, BaseMAB] = {}
        self._init_mabs()
        self.metrics_stats = None
        self.global_step = 0
        self.accept_lengths = {}

    def _init_mabs(self):
        """Initialize MAB instances for each group.

        Creates a MAB instance for each group based on the specified algorithm (EG or UCB1).
        Each MAB instance manages its own metrics."""
        factory = self.ALGORITHM_FACTORIES[self.algorithm]
        self.mabs = {
            group: factory(strategies=self.strategies, window_size=self.window_size)
            for group in self.groups
        }

    def get_valid_strategies(self, batch_size: int) -> List[str]:
        """Get valid strategies for a given batch size to prevent OOM.

        This method filters strategies based on memory constraints to avoid OOM errors.

        Args:
            batch_size: Current batch size for OOM checking

        Returns:
            List of valid strategy names that won't cause OOM
        """
        get_draft_tokens = lambda strategy: int(strategy.split("_")[-1])

        # Check heuristic OOM condition
        valid_strategies = [
            s for s in self.strategies if get_draft_tokens(s) * batch_size <= 2048
        ]
        # If all strategies would cause OOM, return the safest one - the one with least tokens
        if not valid_strategies:
            return [min(self.strategies, key=lambda s: get_draft_tokens(s))]

        return valid_strategies

    def get_group(self, batch_size: int) -> int:
        """Find the largest group <= batch_size."""
        for group in reversed(self.groups):
            if group <= batch_size:
                return group
        return self.groups[0]  # Fallback to smallest group

    def select_strategy(self, batch_size: int) -> str:
        """Select strategy for given batch size using appropriate MAB.

        This method handles both filtering strategies to prevent OOM and selecting
        the best strategy using the MAB algorithm.
        """
        # Fast path for single strategy
        if len(self.strategies) == 1:
            return self.strategies[0]

        # Get appropriate group and valid strategies
        group = self.get_group(batch_size)
        valid_strategy_names = self.get_valid_strategies(batch_size)

        # Select strategy using the MAB algorithm
        return self.mabs[group].select_strategy(valid_strategy_names)

    def add_single_step_metrics(
        self, batch_size: int, strategy_name: str, metrics: MetricsEntry
    ):
        """Add metrics for a strategy in appropriate group.

        Args:
            batch_size: Size of the batch for group assignment
            strategy_name: Name of the strategy to record metrics for
            metrics: MetricsEntry object containing the metrics for this step
        """
        group = self.get_group(batch_size)
        self.mabs[group].add_single_step_metrics(strategy_name, metrics)

        self.global_step += 1
        if self.global_step % 30_000 == 0 or self.global_step == 1000:
            self.plot_mab_metrics()
            self.plot_accept_length_stats()

    def get_stable_accept_length(self, strategy_name: str) -> float:
        """Get stable accept_length across all groups.

        This method calculates a batch-size-independent metric by looking at the
        median accept_length across all batch size groups. This provides a more
        stable metric since accept_length is primarily determined by the strategy
        itself rather than the batch size.

        Args:
            strategy_name: Name of the strategy to get median accept_length for

        Returns:
            Median accept_length across all groups, or 0.0 if no data is available
        """
        # Only recalculate 10% of the time to reduce overhead
        if np.random.random() < 0.1 or not self.accept_lengths.get(strategy_name, None):
            accept_lengths = np.concatenate(
                [
                    self.mabs[group]
                    .strategy_metrics[strategy_name]
                    .metric_values.get("accept_length", [])
                    for group in self.groups
                ]
            )
            self.accept_lengths[strategy_name] = (
                np.median(accept_lengths) if len(accept_lengths) > 0 else 0.0
            )

        return self.accept_lengths.get(strategy_name, 0.0)

    def calculate_mab_metrics(
        self,
    ) -> Dict[str, Dict[int, Dict[str, Dict[str, float]]]]:
        """Get comprehensive statistics for all metrics across all groups and strategies.

        Returns:
            Nested dictionary with structure:
            metrics[stat_type][group][strategy][metric_name] = value
            where stat_type is one of: mean, std, median, 25th, 75th
        """
        # Initialize the structure
        stats = {
            stat: {group: {s: {} for s in self.strategies} for group in self.groups}
            for stat in ["mean", "std", "median", "25th", "75th"]
        }

        # Calculate stable accept length for each strategy
        stable_accept_lengths = {}
        for strategy in self.strategies:
            stable_accept_lengths[strategy] = self.get_stable_accept_length(strategy)
        pulls = {}
        for group in self.groups:
            pulls[group] = sum(
                [
                    len(self.mabs[group].strategy_metrics[strategy].steps)
                    for strategy in self.strategies
                ]
            )

        # Calculate statistics for each group and strategy
        for group in self.groups:
            for strategy in self.strategies:
                metrics_obj = self.mabs[group].strategy_metrics[strategy]
                metric_names = metrics_obj.metric_values.keys()
                # metric_names = ['reward', 'total_time', 'verify_time', 'mab_time']
                for metric_name in metric_names:
                    metric_stats = metrics_obj.get_stats(metric_name)
                    if metric_stats is not None:
                        for stat_type, value in metric_stats.items():
                            stats[stat_type][group][strategy]["pulls"] = len(
                                metrics_obj.steps
                            )
                            stats[stat_type][group][strategy]["pulls_norm"] = (
                                len(metrics_obj.steps) / pulls[group]
                            )
                            stats[stat_type][group][strategy][metric_name] = round(
                                value, 4
                            )
                            if (
                                metric_name.endswith("time")
                                and metric_name != "total_time"
                            ):
                                stats[stat_type][group][strategy][
                                    metric_name + "_perc"
                                ] = round(
                                    value
                                    / max(
                                        1e-8,
                                        stats[stat_type][group][strategy]["total_time"],
                                    )
                                    * 100,
                                    2,
                                )
                            stats[stat_type][group][strategy][
                                "stable_accept_length"
                            ] = round(stable_accept_lengths[strategy], 4)
                    else:
                        # Fill with None for missing metrics
                        for stat_type in stats:
                            stats[stat_type][group][strategy][metric_name] = None

        self.metrics_stats = stats
        return stats

    def _compute_per_strategy_stats(
        self,
        metrics_obj: MABMetricsManager,
        strategy_name: str,
        pulls_group,
        stable_accept_lengths,
    ):
        """Compute all necessary stats for a given strategy within a group."""
        stats_per_strategy = {
            "pulls": {"value": len(metrics_obj.steps)},
            "pulls_norm": {
                "value": len(metrics_obj.steps) / pulls_group if pulls_group > 0 else 0
            },
            "stable_accept_length": {
                "value": (
                    round(stable_accept_lengths[strategy_name], 4)
                    if stable_accept_lengths[strategy_name] is not None
                    else None
                )
            },
        }

        # Get total_time stats once for percentage calculations
        total_time_stats = metrics_obj.get_stats("total_time") or {}

        # Process all metrics
        for metric_name in metrics_obj.get_all_metric_names():
            metric_stats = metrics_obj.get_stats(metric_name)

            if not metric_stats:
                continue

            # Initialize the metric entry
            stats_per_strategy[metric_name] = {}

            # Store all stat types for the metric and compute percentages for time metrics
            if metric_name.endswith("time") and metric_name != "total_time":
                perc_metric_name = f"{metric_name}_perc"
                stats_per_strategy[perc_metric_name] = {}

            for stat_type, value in metric_stats.items():
                stats_per_strategy[metric_name][stat_type] = (
                    round(value, 4) if value is not None else None
                )

                # Calculate percentage for time metrics
                if (
                    metric_name.endswith("time")
                    and metric_name != "total_time"
                    and stat_type in total_time_stats
                ):
                    total_time = max(1e-8, total_time_stats[stat_type])

                    perc_value = round(value / total_time * 100, 2)
                    stats_per_strategy[perc_metric_name][stat_type] = perc_value

        return stats_per_strategy

    def plot_mab_metrics(self, output_dir: str = None):
        """Plot performance metrics for different groups and strategies.

        Generates subplots for various metrics showing performance across different groups.
        For time-related metrics, uses absolute values and shares y-axis range with total_time.

        Args:
            output_dir: Directory to save the plot files
        """
        # Ensure output directory exists
        output_path = Path(output_dir or self.output_dir)
        output_path.mkdir(exist_ok=True)

        # Get fresh metrics
        self.calculate_mab_metrics()

        # Get available metrics from the data
        metrics = [
            "pulls",
            "pulls_norm",
            "reward",
            "total_time",
            "stable_accept_length",
        ]
        n_columns = len(metrics)

        available_metrics = set()
        for group in self.groups:
            for strategy in self.strategies:
                available_metrics.update(
                    self.metrics_stats["median"][group][strategy].keys()
                )

        metrics.extend(
            sorted(
                [
                    x
                    for x in available_metrics
                    if x.endswith("_time") and x != "total_time"
                ]
            )
        )
        metrics.extend(sorted([x for x in available_metrics if x.endswith("_perc")]))

        # Create figure with appropriate number of subplots
        n_metrics = len(metrics)
        n_rows = (n_metrics + n_columns - 1) // n_columns  # Ensure at least 5 columns
        fig, axs = plt.subplots(
            n_rows, n_columns, figsize=(9 * n_columns, 6 * n_rows), squeeze=False
        )
        fig.suptitle(
            "Performance Metrics Across Batch Sizes and Strategies", fontsize=14, y=1.02
        )

        # Use Set3 colormap - provides distinct and visually appealing pastel colors
        colors = [plt.cm.tab10(i % 10) for i in range(len(self.strategies))]

        time_ylim = None  # Will be set after plotting total_time

        # Plot each metric
        for idx, metric_name in enumerate(metrics):
            row = idx // n_columns
            col = idx % n_columns
            ax = axs[row, col]

            for strategy, color in zip(self.strategies, colors):
                groups = []
                values = []

                for group in self.groups:
                    val = self.metrics_stats["median"][group][strategy].get(metric_name)
                    if metric_name.endswith("_time") and val is not None and val > 1:
                        val = None  # Exclude values > 1 for total_time

                    if val is not None:
                        groups.append(group)
                        values.append(val)

                if groups:
                    # Plot median values with connecting lines
                    ax.plot(
                        groups,
                        values,
                        "-",
                        color=color,
                        label=strategy,
                        linewidth=1,
                    )
                    ax.plot(groups, values, "o", color=color, markersize=4)

            ax.set_xlabel("Batch Size")
            display_name = " ".join(word.title() for word in metric_name.split("_"))
            ax.set_ylabel(display_name)

            # In case of percentage, set yliit to be [0, 1]
            # if metric_name.endswith('_perc'):
            #     ax.set_ylim([0, 100])
            if metric_name.endswith("_time"):
                ax.set_yscale("log")
                if metric_name == "total_time":
                    time_ylim = ax.get_ylim()
                else:
                    ax.set_ylim(top=time_ylim[1])

            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            ax.legend(ncol=2)

            # Remove title from subplot to match original style
            ax.set_title("")

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(
            output_path / f"mab_metrics_{self.name}_{self.global_step}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_accept_length_stats(self, output_dir: str = None):
        """Plot the distribution of acceptance counts for each strategy.

        Shows median with 25th-75th percentile bands for accept_length across groups.
        Each strategy gets its own subplot with batch size groups on x-axis.

        Args:
            output_dir: Directory to save the plot files
        """
        # Ensure output directory exists
        output_path = Path(output_dir or self.output_dir)
        output_path.mkdir(exist_ok=True)

        # Determine grid layout based on number of strategies
        n_rows = 2 if len(self.strategies) >= 4 else 1
        n_cols = np.ceil(len(self.strategies) / n_rows).astype(int)

        # Create figure
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False
        )
        fig.suptitle("Accept Length Distribution by Strategy", fontsize=14, y=1.02)

        # Use Set3 colormap - provides distinct and visually appealing pastel colors
        colors = [plt.cm.tab10(i % 10) for i in range(len(self.strategies))]

        # Plot each strategy
        for idx, (strategy, color) in enumerate(zip(self.strategies, colors)):
            row = idx // n_cols
            col = idx % n_cols
            ax = axs[row, col]

            groups = []
            medians = []
            percentile_25 = []
            percentile_75 = []
            all_values = []
            all_groups = []

            for group in self.groups:
                metrics_obj = self.mabs[group].strategy_metrics[strategy]
                if "accept_length" not in metrics_obj.get_all_metric_names():
                    continue

                stats = metrics_obj.get_stats("accept_length")
                if stats is None:
                    continue

                groups.append(group)
                medians.append(stats["median"])
                percentile_25.append(stats["25th"])
                percentile_75.append(stats["75th"])

                values = metrics_obj.metric_values["accept_length"]
                # Store individual points for scatter
                all_values.extend(values)
                all_groups.extend([group] * len(values))

            if not groups:
                continue

            # Plot individual points with low opacity
            ax.plot(all_groups, all_values, "o", color=color, alpha=0.1, markersize=2)

            # Plot median line
            ax.plot(groups, medians, "-", color=color, label="Median", linewidth=2)

            # Plot percentile band
            ax.fill_between(
                groups,
                percentile_25,
                percentile_75,
                color=color,
                alpha=0.2,
                label="25-75th Percentile",
            )

            ax.set_xlabel("Batch Size")
            ax.set_ylabel("Accept Length")
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"Strategy: {strategy}")

            ax.legend()

        plt.tight_layout()
        plt.savefig(
            output_path / f"mab_accept_length_stats_{self.name}_{self.global_step}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
