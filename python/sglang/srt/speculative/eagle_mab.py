"""Multi-Armed Bandit implementation for adaptive speculative decoding."""

from collections import deque
from dataclasses import dataclass, field
import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Deque
import os

@dataclass
class MABStrategyMetrics:
    """Container for sliding window metrics per (group, mab_strategy).
    
    Uses a centralized step tracking deque and a dictionary of metric deques
    to store time-series data for each metric type.
    
    Expected metrics:
    - reward: A more stable version of goodput, by using the median accept_length across batch sizes
    - goodput: Tokens accepted per second (accept_length * batch_size / total_time)
    - accept_length: Average number of tokens accepted per request
    - draft_time: Time spent in draft generation
    - verify_time: Time spent in verification
    - draft_extend_time: Time spent in draft extension
    - other_time: Time spent in other operations
    - total_time: Total time spent in all operations
    """
    window_size: int
    steps: Deque[int] = field(default_factory=deque)
    metrics: Dict[str, Deque[float]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize step tracking and metrics deques with maxlen."""
        self.steps = deque(maxlen=self.window_size)
        self.metrics = {}
        
    def _expire_old_data(self, current_step: int):
        """Remove data points older than window_size steps."""
        cutoff = current_step - self.window_size
        while self.steps and self.steps[0] < cutoff:
            self.steps.popleft()
            self.metrics['reward'].popleft()

    def trimmed_mean(self, metric_name: str, trim_percent: float = 0.1) -> float:
        """Calculate trimmed mean for the specified metric."""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        values = self.metrics[metric_name]
        sorted_vals = sorted(values)
        n_trim = int(len(sorted_vals) * trim_percent)
        return np.mean(sorted_vals[n_trim:-n_trim]) if len(sorted_vals) > 2*n_trim else np.mean(sorted_vals)

    def get_stats(self, metric_name: str) -> Optional[Dict[str, float]]:
        """Calculate statistics for the specified metric.
        
        Returns:
            Dictionary with statistics (mean, std, median, 25th, 75th percentiles)
            or None if no data available
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        values = np.array(self.metrics[metric_name])
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)) if len(values) > 1 else 0.0,
            'median': float(np.median(values)),
            '25th': float(np.percentile(values, 25)),
            '75th': float(np.percentile(values, 75))
        }

    def median(self, metric_name: str) -> float:
        """Calculate median for the specified metric."""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        return float(np.median(self.metrics[metric_name]))

    def record_metrics(self, metrics: dict, current_step: int):
        """Record new metrics in the deques with shared step tracking.
        
        Args:
            metrics: Dictionary of metric values to record
            current_step: Current step number for sliding window
        """
        # Initialize any new metrics we haven't seen before
        for metric_name in metrics:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = deque(maxlen=self.window_size)
        
        # Record step and all metric values
        self.steps.append(current_step)
        for metric_name, value in metrics.items():
            self.metrics[metric_name].append(value)
        
        self._expire_old_data(current_step)

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
        self.metrics = {s: MABStrategyMetrics(window_size) for s in strategies}
    
    def record_metrics(self, strategy: str, metrics: dict):
        """Record metrics for a strategy and advance the step counter atomically.
        
        Args:
            strategy: The strategy for which to record metrics
            metrics: Dictionary of metrics to record
        """
        self.metrics[strategy].record_metrics(metrics, self.current_step)
        self.current_step += 1
    
    def _expire_old_data(self):
        """Expire old data across all metrics."""
        for metric in self.metrics.values():
            metric._expire_old_data(self.current_step)
    
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
            s: self.metrics[s].median("reward") or float('inf')
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
        total_pulls = sum(len(m.metrics.get("reward", [])) for m in self.metrics.values())
        
        # Calculate UCB scores for each strategy
        strategy_scores = {
            s: self._ucb_score(self.metrics[s], total_pulls)
            for s in valid_strategies
        }
        return max(strategy_scores.items(), key=lambda x: x[1])[0]
    
    def _ucb_score(self, metric: MABStrategyMetrics, total_pulls: int) -> float:
        """Calculate UCB score for a strategy."""
        n_i = len(metric.metrics.get("reward", []))
        if n_i == 0:  # force exploration if no recent data
            return float('inf')
        
        # if trimmed mean is None, use inf to force exploration
        r_hat = metric.trimmed_mean("reward") or float('inf')
        return r_hat + math.sqrt(2 * math.log(total_pulls) / n_i)

class MABGroupManager:
    """Manages groups and their associated MAB instances.
    
    Each group (e.g., batch size group) has its own MAB instance that learns
    independently which strategies work best for that group.
    """
    def __init__(self, name: str, strategies: List[str], 
                 algorithm: str = "EG", window_size: int = 1000,
                 output_dir = "mab_plots"):
        self.name = name
        self.strategies = strategies
        self.algorithm = algorithm.upper()
        self.window_size = window_size

        # get 'MAB_RESULTS_DIR' from environment variable
        self.output_dir = Path(os.getenv('MAB_RESULTS_DIR', '.')) / output_dir
        print(f"SGLang MAB Results will be saved to: {self.output_dir} for {self.strategies}")

        # Initialize groups
        self.groups = list(range(1,32)) + list(range(32, 128, 8)) + list(range(128, 256, 32))

        # Initialize MAB instances
        self.mabs: Dict[int, BaseMAB] = {}
        self._init_mabs()
        self.metrics_stats = None
        self.global_step = 0
        self.accept_lengths = {}
        
    def _init_mabs(self):
        """Initialize MAB instances for each group.
        
        Creates a MAB instance for each group based on the specified algorithm (EG or UCB1).
        Each MAB instance manages its own metrics.
        """
        for group in self.groups:
            # Create MAB instance based on algorithm type
            if self.algorithm == "EG":
                self.mabs[group] = EpsilonGreedyMAB(
                    strategies=self.strategies,
                    window_size=self.window_size
                )
            elif self.algorithm == "UCB1":
                self.mabs[group] = UCB1MAB(
                    strategies=self.strategies,
                    window_size=self.window_size
                )
            else:
                raise ValueError(f"Unsupported MAB algorithm: {self.algorithm}")
        
    def _filter_strategies(self, batch_size: int) -> List[str]:
        """Filter strategies based on OOM risk.
        
        Args:
            batch_size: Current batch size for OOM checking
        Returns:
            List of valid strategies that won't cause OOM
        """
        valid_strategies = []
        for s in self.strategies:
            # Check heuristic OOM condition
            tokens_per_ps = int(s.split('_')[-1])
            if tokens_per_ps * batch_size <= 2048:
                valid_strategies.append(s)
        
        # If all strategies would cause OOM, return the safest one - the one with least tokens per ps
        if not valid_strategies:
            return [min(self.strategies, key=lambda x: int(x.split('_')[-1]))]
                        
        return valid_strategies
                
    def get_group(self, batch_size: int) -> int:
        """Find the largest group <= batch_size."""
        for group in reversed(self.groups):
            if group <= batch_size:
                return group
        return self.groups[0]  # Fallback to smallest group
    
    def select_strategy(self, batch_size: int) -> str:
        """Select strategy for given batch size using appropriate MAB.
        
        First filters strategies based on OOM risk, then uses the MAB
        algorithm to select from valid strategies.
        """
        if len(self.strategies) == 1:
            return self.strategies[0]

        group = self.get_group(batch_size)
        valid_strategies = self._filter_strategies(batch_size)
        return self.mabs[group].select_strategy(valid_strategies)
    
    def record_metrics(self, batch_size: int, strategy: str, metrics: dict):
        """Record metrics for a strategy in appropriate group."""
        group = self.get_group(batch_size)
        self.mabs[group].record_metrics(strategy, metrics)

        self.global_step += 1
        if self.global_step % 30_000 == 0 or self.global_step == 1000:
            self.plot_mab_metrics()
            self.plot_accept_length_stats()
    
    def get_stable_accept_length(self, strategy: str) -> float:
        """Get median accept_length across all groups, because accept_length is independent of batch size."""
        # No need to update every time
        if np.random.random() < 0.1 or not self.accept_lengths.get(strategy, None):
            accept_lengths = np.concatenate([self.mabs[group].metrics[strategy].metrics.get('accept_length', []) for group in self.groups])
            self.accept_lengths[strategy] = np.median(accept_lengths)

        return self.accept_lengths[strategy]
    
    def calculate_mab_metrics(self) -> Dict[str, Dict[int, Dict[str, Dict[str, float]]]]:
        """Get comprehensive statistics for all metrics across all groups and strategies.
        
        Returns:
            Nested dictionary with structure:
            metrics[stat_type][group][strategy][metric_name] = value
            where stat_type is one of: mean, std, median, 25th, 75th
        """
        # Initialize the structure
        stats = {
            stat: {group: {s: {} for s in self.strategies} 
                   for group in self.groups}
            for stat in ['mean', 'std', 'median', '25th', '75th']
        }
        
        # Calculate stable accept length for each strategy
        stable_accept_lengths = {}
        for strategy in self.strategies:
            stable_accept_lengths[strategy] = self.get_stable_accept_length(strategy)
        pulls = {}
        for group in self.groups:
            pulls[group] = sum([len(self.mabs[group].metrics[strategy].steps) for strategy in self.strategies])

        # Calculate statistics for each group and strategy
        for group in self.groups:
            for strategy in self.strategies:
                metrics_obj = self.mabs[group].metrics[strategy]
                metric_names = metrics_obj.metrics.keys()
                # metric_names = ['reward', 'total_time', 'verify_time', 'mab_time']
                for metric_name in metric_names:
                    metric_stats = metrics_obj.get_stats(metric_name)
                    if metric_stats is not None:
                        for stat_type, value in metric_stats.items():
                            stats[stat_type][group][strategy]['pulls'] = len(metrics_obj.steps)
                            stats[stat_type][group][strategy]['pulls_norm'] = len(metrics_obj.steps) / pulls[group]
                            stats[stat_type][group][strategy][metric_name] = round(value, 4)
                            if metric_name.endswith('time') and metric_name != 'total_time':
                                stats[stat_type][group][strategy][metric_name+"_perc"] = round(value / max(1e-8, stats[stat_type][group][strategy]['total_time']) * 100, 2)
                            stats[stat_type][group][strategy]['stable_accept_length'] = round(stable_accept_lengths[strategy], 4)
                    else:
                        # Fill with None for missing metrics
                        for stat_type in stats:
                            stats[stat_type][group][strategy][metric_name] = None
        
        self.metrics_stats = stats
        return stats
    
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
        metrics = ['pulls', 'pulls_norm', 'reward', 'total_time', 'stable_accept_length']
        n_columns = len(metrics)

        available_metrics = set()
        for group in self.groups:
            for strategy in self.strategies:
                available_metrics.update(self.metrics_stats['median'][group][strategy].keys())

        metrics.extend(sorted([x for x in available_metrics if x.endswith('_time') and x != 'total_time']))
        metrics.extend(sorted([x for x in available_metrics if x.endswith('_perc')]))
        
        # Create figure with appropriate number of subplots
        n_metrics = len(metrics)
        n_rows = (n_metrics + n_columns - 1) // n_columns  # Ensure at least 5 columns
        fig, axs = plt.subplots(n_rows, n_columns, figsize=(9*n_columns, 6*n_rows), squeeze=False)
        fig.suptitle('Performance Metrics Across Batch Sizes and Strategies', fontsize=14, y=1.02)
        
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
                    val = self.metrics_stats['median'][group][strategy].get(metric_name)
                    if metric_name.endswith('_time') and val is not None and val > 1:
                        val = None  # Exclude values > 1 for total_time
                    
                    if val is not None:
                        groups.append(group)
                        values.append(val)
                
                if groups:
                    # Plot median values with connecting lines
                    ax.plot(groups, values, '-', color=color, label=strategy, linewidth=1)
                    ax.plot(groups, values, 'o', color=color, markersize=4)
            
            ax.set_xlabel('Batch Size')
            display_name = ' '.join(word.title() for word in metric_name.split('_'))
            ax.set_ylabel(display_name)
            
            # In case of percentage, set yliit to be [0, 1]
            # if metric_name.endswith('_perc'):
            #     ax.set_ylim([0, 100])
            if metric_name.endswith('_time'):
                ax.set_yscale('log')
                if metric_name == 'total_time':
                    time_ylim = ax.get_ylim()
                else:
                    ax.set_ylim(top=time_ylim[1])
            
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(ncol=2)
            
            # Remove title from subplot to match original style
            ax.set_title('')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path / f'mab_metrics_{self.name}_{self.global_step}.png', dpi=300, bbox_inches='tight')
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
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), squeeze=False)
        fig.suptitle('Accept Length Distribution by Strategy', fontsize=14, y=1.02)
        
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
                metrics_obj = self.mabs[group].metrics[strategy]
                if 'accept_length' not in metrics_obj.metrics:
                    continue

                stats = metrics_obj.get_stats('accept_length')
                if stats is None:
                    continue

                groups.append(group)
                medians.append(stats['median'])
                percentile_25.append(stats['25th'])
                percentile_75.append(stats['75th'])
                    
                values = list(metrics_obj.metrics['accept_length'])
                # Store individual points for scatter
                all_values.extend(values)
                all_groups.extend([group] * len(values))
            
            if not groups:
                continue

            # Plot individual points with low opacity
            ax.plot(all_groups, all_values, 'o', 
                    color=color, alpha=0.1, markersize=2)
            
            # Plot median line
            ax.plot(groups, medians, '-', color=color, 
                    label='Median', linewidth=2)
            
            # Plot percentile band
            ax.fill_between(groups, percentile_25, percentile_75, 
                            color=color, alpha=0.2, 
                            label='25-75th Percentile')
            
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Accept Length')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Strategy: {strategy}')
            
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / f'mab_accept_length_stats_{self.name}_{self.global_step}.png', dpi=300, bbox_inches='tight')
        plt.close()
        