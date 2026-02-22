"""
EchoReservoir — The Arena (need-to-be / state space)

A multi-scale Echo State Network reservoir implementing the Deep Tree Echo
cognitive architecture's arena concept. Features:
  - Fast dynamics (perception) with high leak rate
  - Slow dynamics (memory) with low leak rate
  - Spectral radius control for edge-of-chaos computation
  - Sparse random connectivity with configurable density
  - Input scaling for signal conditioning

The reservoir is the "need-to-be" — the base manifold in which cognitive
states evolve. It does not learn; it provides the rich dynamical substrate
from which the Agent (readout) extracts meaning.

AAR Mapping: Arena = Reservoir
"""

import numpy as np


class EchoReservoir:
    """Multi-scale Echo State Network reservoir (Arena).

    Parameters
    ----------
    units : int
        Total reservoir size. Split equally between fast and slow pools.
    spectral_radius : float
        Spectral radius of the recurrent weight matrix. Controls the
        echo state property. Values near 1.0 give edge-of-chaos dynamics.
    input_scaling : float
        Scaling factor for input weights.
    leak_rate_fast : float
        Leak rate for the fast (perception) pool. Higher = more responsive.
    leak_rate_slow : float
        Leak rate for the slow (memory) pool. Lower = more persistent.
    density : float
        Connection density of the recurrent weight matrix.
    seed : int or None
        Random seed for reproducibility.
    name : str or None
        Node name.
    """

    def __init__(
        self,
        units: int = 256,
        spectral_radius: float = 0.95,
        input_scaling: float = 0.1,
        leak_rate_fast: float = 0.8,
        leak_rate_slow: float = 0.1,
        density: float = 0.1,
        seed: int = None,
        name: str = None,
    ):
        self.units = units
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leak_rate_fast = leak_rate_fast
        self.leak_rate_slow = leak_rate_slow
        self.density = density
        self.seed = seed
        self.name = name or "EchoReservoir"

        # Derived
        self.fast_units = units // 2
        self.slow_units = units - self.fast_units
        self.output_dim = units
        self.input_dim = None
        self.initialized = False

        # Weights (set during initialize)
        self.Win = None
        self.W = None
        self.state = {}

    def initialize(self, x):
        """Initialize reservoir weights from first input."""
        if x.ndim == 1:
            self.input_dim = x.shape[0]
        else:
            self.input_dim = x.shape[-1]

        rng = np.random.RandomState(self.seed)

        # Input weights: dense, scaled
        self.Win = rng.randn(self.units, self.input_dim) * self.input_scaling

        # Recurrent weights: sparse, scaled to spectral radius
        W = rng.randn(self.units, self.units)
        mask = rng.rand(self.units, self.units) < self.density
        W *= mask
        # Scale to desired spectral radius
        eigenvalues = np.abs(np.linalg.eigvals(W))
        max_eig = np.max(eigenvalues) if len(eigenvalues) > 0 else 1.0
        if max_eig > 0:
            W = W * (self.spectral_radius / max_eig)
        self.W = W

        # Cross-coupling between fast and slow pools
        coupling_strength = 0.05
        # Fast -> Slow (top-down modulation)
        self.W[self.fast_units:, :self.fast_units] *= (1 + coupling_strength)
        # Slow -> Fast (bottom-up context)
        self.W[:self.fast_units, self.fast_units:] *= (1 + coupling_strength)

        # Initial state
        self.state = {
            "out": np.zeros(self.units),
            "fast": np.zeros(self.fast_units),
            "slow": np.zeros(self.slow_units),
        }
        self.initialized = True

    def _step(self, state, x):
        """Single timestep: pure function, no side effects.

        Fast pool: high leak rate, responds quickly to input.
        Slow pool: low leak rate, maintains persistent memory.
        """
        fast = state["fast"]
        slow = state["slow"]
        h = np.concatenate([fast, slow])

        # Pre-activation: input + recurrent
        pre = self.Win @ x + self.W @ h

        # Split into fast and slow pools
        pre_fast = pre[:self.fast_units]
        pre_slow = pre[self.fast_units:]

        # Leaky integration with different time constants
        fast_new = (1 - self.leak_rate_fast) * fast + self.leak_rate_fast * np.tanh(pre_fast)
        slow_new = (1 - self.leak_rate_slow) * slow + self.leak_rate_slow * np.tanh(pre_slow)

        out = np.concatenate([fast_new, slow_new])
        return {"out": out, "fast": fast_new, "slow": slow_new}

    def step(self, x):
        """Run one timestep and update internal state."""
        if not self.initialized:
            self.initialize(x)
        self.state = self._step(self.state, x)
        return self.state["out"]

    def run(self, X):
        """Run the reservoir on a full timeseries.

        Parameters
        ----------
        X : np.ndarray, shape (T, input_dim)
            Input timeseries.

        Returns
        -------
        states : np.ndarray, shape (T, units)
            Reservoir states at each timestep.
        """
        if not self.initialized:
            self.initialize(X[0])

        T = X.shape[0]
        states = np.zeros((T, self.units))
        state = {k: v.copy() for k, v in self.state.items()}

        for t in range(T):
            state = self._step(state, X[t])
            states[t] = state["out"]

        self.state = state
        return states

    def reset(self):
        """Reset reservoir state to zeros."""
        self.state = {
            "out": np.zeros(self.units),
            "fast": np.zeros(self.fast_units),
            "slow": np.zeros(self.slow_units),
        }

    def get_echo_state_property(self, X, n_washout=100):
        """Verify the echo state property by running from two random ICs.

        Returns the max absolute difference after washout — should be ~0.
        """
        if not self.initialized:
            self.initialize(X[0])

        rng = np.random.RandomState(42)

        # Run 1: from zeros
        self.reset()
        states1 = self.run(X)

        # Run 2: from random IC
        self.state = {
            "out": rng.randn(self.units) * 0.5,
            "fast": rng.randn(self.fast_units) * 0.5,
            "slow": rng.randn(self.slow_units) * 0.5,
        }
        states2 = self.run(X)

        # After washout, states should converge
        diff = np.max(np.abs(states1[n_washout:] - states2[n_washout:]))
        return diff
