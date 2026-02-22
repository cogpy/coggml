"""
EchobeatNode — 9-Step Cognitive Cycle Orchestrator

Implements the Deep Tree Echo echobeat cycle: a 9-step process that
orchestrates perception, action, and simulation across 3 concurrent
streams phased 120 degrees apart over a 12-step cycle.

The 9 steps correspond to the 9 terms of 4 nestings (OEIS A000081):
  N=1: 1 term  (ground state)
  N=2: 2 terms (dyadic distinction)
  N=3: 4 terms (tetradic structure)
  N=4: 9 terms (full echobeat)

The 3 streams are interleaved 4 steps apart:
  Stream A: steps {1, 4, 7}  — Perception
  Stream B: steps {2, 5, 8}  — Action
  Stream C: steps {3, 6, 9}  — Simulation

Each stream cycles through: perceive → act → simulate
"""

import numpy as np


class EchobeatNode:
    """9-step cognitive cycle orchestrator.

    Parameters
    ----------
    units : int
        State dimensionality per stream.
    n_streams : int
        Number of concurrent cognitive streams (default 3).
    cycle_length : int
        Steps per full echobeat cycle (default 9).
    coupling : float
        Inter-stream coupling strength.
    name : str or None
        Node name.
    """

    # Stream phase offsets (120 degrees apart in 12-step cycle)
    STREAM_PHASES = {
        0: [0, 3, 6],  # Stream A: perceive, act, simulate
        1: [1, 4, 7],  # Stream B: perceive, act, simulate
        2: [2, 5, 8],  # Stream C: perceive, act, simulate
    }

    # Cognitive modes per step within a stream
    MODES = ["perceive", "act", "simulate"]

    def __init__(
        self,
        units: int = 64,
        n_streams: int = 3,
        cycle_length: int = 9,
        coupling: float = 0.1,
        seed: int = None,
        name: str = None,
    ):
        self.units = units
        self.n_streams = n_streams
        self.cycle_length = cycle_length
        self.coupling = coupling
        self.seed = seed
        self.name = name or "EchobeatNode"

        self.output_dim = units * n_streams
        self.input_dim = None
        self.initialized = False
        self.state = {}

    def initialize(self, x):
        """Initialize echobeat weights and state."""
        if x.ndim == 1:
            self.input_dim = x.shape[0]
        else:
            self.input_dim = x.shape[-1]

        rng = np.random.RandomState(self.seed)

        # Per-stream, per-mode weight matrices
        # 3 streams x 3 modes x (units, input_dim)
        self.W_mode = {}
        for s in range(self.n_streams):
            for m, mode in enumerate(self.MODES):
                key = (s, mode)
                self.W_mode[key] = rng.randn(self.units, self.input_dim) * 0.1

        # Inter-stream coupling matrices
        self.W_couple = {}
        for s1 in range(self.n_streams):
            for s2 in range(self.n_streams):
                if s1 != s2:
                    self.W_couple[(s1, s2)] = (
                        rng.randn(self.units, self.units) * self.coupling
                    )

        # Recurrent per-stream weights
        self.W_rec = {}
        for s in range(self.n_streams):
            W = rng.randn(self.units, self.units) * 0.1
            # Scale to spectral radius < 1
            eigs = np.abs(np.linalg.eigvals(W))
            max_eig = np.max(eigs) if len(eigs) > 0 else 1.0
            if max_eig > 0:
                W = W * (0.9 / max_eig)
            self.W_rec[s] = W

        # State: per-stream hidden states + global step counter
        self.state = {
            "out": np.zeros(self.output_dim),
            "step": 0,
        }
        for s in range(self.n_streams):
            self.state[f"stream_{s}"] = np.zeros(self.units)

        self.initialized = True

    def _get_active_mode(self, step, stream_id):
        """Determine which cognitive mode is active for a stream at a step."""
        phases = self.STREAM_PHASES[stream_id]
        step_in_cycle = step % self.cycle_length
        for mode_idx, phase_step in enumerate(phases):
            if step_in_cycle == phase_step:
                return self.MODES[mode_idx]
        # Between explicit phase steps, use the most recent mode
        for mode_idx in range(len(phases) - 1, -1, -1):
            if step_in_cycle > phases[mode_idx]:
                return self.MODES[mode_idx]
        return self.MODES[-1]  # Default to simulate

    def _step(self, state, x):
        """Single echobeat step across all streams."""
        step = state["step"]
        streams = {}
        for s in range(self.n_streams):
            streams[s] = state[f"stream_{s}"].copy()

        new_streams = {}
        for s in range(self.n_streams):
            mode = self._get_active_mode(step, s)
            h = streams[s]

            # Mode-specific input transformation
            W = self.W_mode[(s, mode)]
            driven = W @ x

            # Recurrent dynamics
            recurrent = self.W_rec[s] @ h

            # Inter-stream coupling
            coupled = np.zeros(self.units)
            for s2 in range(self.n_streams):
                if s2 != s:
                    coupled += self.W_couple[(s, s2)] @ streams[s2]

            # Update with leaky integration
            h_new = 0.7 * h + 0.3 * np.tanh(driven + recurrent + coupled)
            new_streams[s] = h_new

        # Concatenate all streams for output
        out = np.concatenate([new_streams[s] for s in range(self.n_streams)])

        new_state = {
            "out": out,
            "step": step + 1,
        }
        for s in range(self.n_streams):
            new_state[f"stream_{s}"] = new_streams[s]

        return new_state

    def step(self, x):
        """Run one echobeat step."""
        if not self.initialized:
            self.initialize(x)
        self.state = self._step(self.state, x)
        return self.state["out"]

    def run(self, X):
        """Run echobeat on full timeseries."""
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

    def get_stream_states(self):
        """Return individual stream states."""
        return {
            s: self.state.get(f"stream_{s}", None)
            for s in range(self.n_streams)
        }

    def get_current_modes(self):
        """Return the active cognitive mode for each stream."""
        step = self.state.get("step", 0)
        return {
            s: self._get_active_mode(step, s)
            for s in range(self.n_streams)
        }

    def reset(self):
        """Reset echobeat state."""
        self.state = {
            "out": np.zeros(self.output_dim),
            "step": 0,
        }
        for s in range(self.n_streams):
            self.state[f"stream_{s}"] = np.zeros(self.units)
