"""
IntrospectionNode — Recursive Self-Monitoring

Implements the Deep Tree Echo introspection system: a node that observes
its own reservoir state and produces meta-cognitive signals. Supports
recursive depth (introspection of introspection) up to a configurable
maximum depth.

The introspection node computes:
  - State statistics (mean, variance, entropy proxy)
  - Lyapunov-like divergence estimate (chaos detection)
  - Activation distribution analysis
  - Recursive self-observation at multiple depths

This corresponds to the Introspection Membrane in the DTE architecture.
"""

import numpy as np


class IntrospectionNode:
    """Recursive self-monitoring node.

    Parameters
    ----------
    max_depth : int
        Maximum recursion depth for introspection.
    window_size : int
        Number of past states to keep for temporal analysis.
    name : str or None
        Node name.
    """

    def __init__(
        self,
        max_depth: int = 3,
        window_size: int = 50,
        name: str = None,
    ):
        self.max_depth = max_depth
        self.window_size = window_size
        self.name = name or "IntrospectionNode"

        # Output: per-depth metrics (5 metrics per depth level)
        # [mean, var, entropy_proxy, divergence, activation_ratio]
        self.metrics_per_depth = 5
        self.output_dim = max_depth * self.metrics_per_depth
        self.input_dim = None
        self.initialized = False
        self.state = {}

    def initialize(self, x):
        """Initialize introspection buffers."""
        if x.ndim == 1:
            self.input_dim = x.shape[0]
        else:
            self.input_dim = x.shape[-1]

        self.state = {
            "out": np.zeros(self.output_dim),
            "history": np.zeros((self.window_size, self.input_dim)),
            "position": 0,
            "prev_state": np.zeros(self.input_dim),
            "divergence_history": np.zeros(self.window_size),
        }
        self.initialized = True

    def _compute_metrics(self, x, history, position):
        """Compute introspection metrics for a single depth level.

        Returns
        -------
        metrics : np.ndarray, shape (5,)
            [mean, variance, entropy_proxy, divergence, activation_ratio]
        """
        # Mean activation
        mean_act = np.mean(x)

        # Variance of activations
        var_act = np.var(x)

        # Entropy proxy: normalized histogram entropy
        hist, _ = np.histogram(x, bins=20, range=(-1, 1))
        hist = hist / (hist.sum() + 1e-10)
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        entropy_norm = entropy / np.log(20)  # Normalize to [0, 1]

        # Lyapunov-like divergence: rate of change
        valid_len = min(position, self.window_size)
        if valid_len > 1:
            recent = history[:valid_len]
            diffs = np.diff(recent, axis=0)
            divergence = np.mean(np.linalg.norm(diffs, axis=1))
        else:
            divergence = 0.0

        # Activation ratio: fraction of units above threshold
        activation_ratio = np.mean(np.abs(x) > 0.5)

        return np.array([mean_act, var_act, entropy_norm, divergence, activation_ratio])

    def _step(self, state, x):
        """Single introspection step with recursive depth."""
        history = state["history"].copy()
        position = state["position"]

        # Update history buffer (circular)
        idx = position % self.window_size
        history[idx] = x
        new_position = position + 1

        # Compute metrics at each depth
        all_metrics = []
        current_signal = x

        for depth in range(self.max_depth):
            metrics = self._compute_metrics(current_signal, history, new_position)
            all_metrics.append(metrics)

            # For next depth: introspect on the metrics themselves
            # Pad or project metrics to match input dim for recursive analysis
            if depth < self.max_depth - 1:
                # Use metrics as the signal for next depth
                # Tile to create a richer signal
                current_signal = np.tile(metrics, (self.input_dim // len(metrics) + 1))[:self.input_dim]

        out = np.concatenate(all_metrics)

        return {
            "out": out,
            "history": history,
            "position": new_position,
            "prev_state": x.copy(),
            "divergence_history": state["divergence_history"],
        }

    def step(self, x):
        """Run one introspection step."""
        if not self.initialized:
            self.initialize(x)
        self.state = self._step(self.state, x)
        return self.state["out"]

    def run(self, X):
        """Run introspection on full timeseries."""
        if not self.initialized:
            self.initialize(X[0])

        T = X.shape[0]
        outputs = np.zeros((T, self.output_dim))
        state = {k: v.copy() if isinstance(v, np.ndarray) else v
                 for k, v in self.state.items()}

        for t in range(T):
            state = self._step(state, X[t])
            outputs[t] = state["out"]

        self.state = state
        return outputs

    def get_depth_metrics(self, depth: int = 0):
        """Return metrics for a specific depth level.

        Returns dict with keys: mean, variance, entropy, divergence, activation_ratio
        """
        if depth >= self.max_depth:
            raise ValueError(f"Depth {depth} exceeds max_depth {self.max_depth}")

        out = self.state.get("out", np.zeros(self.output_dim))
        start = depth * self.metrics_per_depth
        end = start + self.metrics_per_depth
        metrics = out[start:end]

        return {
            "mean": metrics[0],
            "variance": metrics[1],
            "entropy": metrics[2],
            "divergence": metrics[3],
            "activation_ratio": metrics[4],
        }

    def reset(self):
        """Reset introspection state."""
        if self.input_dim:
            self.state = {
                "out": np.zeros(self.output_dim),
                "history": np.zeros((self.window_size, self.input_dim)),
                "position": 0,
                "prev_state": np.zeros(self.input_dim),
                "divergence_history": np.zeros(self.window_size),
            }
