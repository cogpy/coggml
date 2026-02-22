"""
MembraneNode — Hierarchical Membrane Boundary

Implements the P-system membrane architecture from Deep Tree Echo.
A membrane acts as a selective boundary that gates information flow
between nested cognitive regions. It controls what enters and exits
each processing region, implementing the nested shells structure:

  ( ( pro ) org ) glo

Where:
  - pro = project (innermost, local context)
  - org = organization (middle, team context)
  - glo = global (outermost, world context)

The membrane uses learned gates to selectively pass, block, or
transform signals crossing the boundary.
"""

import numpy as np


class MembraneNode:
    """Hierarchical membrane boundary with selective gating.

    Parameters
    ----------
    units : int
        Dimensionality of the membrane state.
    permeability : float
        Base permeability (0 = fully closed, 1 = fully open).
    n_layers : int
        Number of nested membrane layers.
    name : str or None
        Node name.
    """

    def __init__(
        self,
        units: int = 64,
        permeability: float = 0.5,
        n_layers: int = 3,
        seed: int = None,
        name: str = None,
    ):
        self.units = units
        self.permeability = permeability
        self.n_layers = n_layers
        self.seed = seed
        self.name = name or "MembraneNode"

        self.output_dim = units
        self.input_dim = None
        self.initialized = False
        self.state = {}

    def initialize(self, x):
        """Initialize membrane gates and projections."""
        if x.ndim == 1:
            self.input_dim = x.shape[0]
        else:
            self.input_dim = x.shape[-1]

        rng = np.random.RandomState(self.seed)

        # Per-layer gate weights
        self.W_gate = []
        self.W_transform = []
        self.bias_gate = []

        prev_dim = self.input_dim
        for layer in range(self.n_layers):
            # Gate: sigmoid determines what passes through
            self.W_gate.append(rng.randn(self.units, prev_dim) * 0.1)
            self.bias_gate.append(
                np.ones(self.units) * np.log(self.permeability / (1 - self.permeability + 1e-8))
            )
            # Transform: what the membrane does to the signal
            self.W_transform.append(rng.randn(self.units, prev_dim) * 0.1)
            prev_dim = self.units

        # State: per-layer membrane potentials
        self.state = {
            "out": np.zeros(self.units),
        }
        for layer in range(self.n_layers):
            self.state[f"layer_{layer}"] = np.zeros(self.units)

        self.initialized = True

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _step(self, state, x):
        """Pass signal through nested membrane layers."""
        signal = x.copy()
        new_state = {"step": state.get("step", 0) + 1}
        layer_states = {}

        for layer in range(self.n_layers):
            prev_layer = state.get(f"layer_{layer}", np.zeros(self.units))

            # Gate: what fraction of signal passes
            gate = self._sigmoid(self.W_gate[layer] @ signal + self.bias_gate[layer])

            # Transform: how the signal is modified
            transformed = np.tanh(self.W_transform[layer] @ signal)

            # Membrane dynamics: leaky integration with gating
            layer_new = 0.8 * prev_layer + 0.2 * (gate * transformed)
            layer_states[layer] = layer_new

            # Output of this layer becomes input to next
            signal = layer_new

        out = layer_states[self.n_layers - 1]
        new_state["out"] = out
        for layer in range(self.n_layers):
            new_state[f"layer_{layer}"] = layer_states[layer]

        return new_state

    def step(self, x):
        """Run one membrane step."""
        if not self.initialized:
            self.initialize(x)
        self.state = self._step(self.state, x)
        return self.state["out"]

    def run(self, X):
        """Run membrane on full timeseries."""
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

    def get_layer_states(self):
        """Return per-layer membrane states."""
        return {
            layer: self.state.get(f"layer_{layer}", None)
            for layer in range(self.n_layers)
        }

    def get_permeability_profile(self):
        """Return effective permeability at each layer."""
        # Approximate from bias terms
        perms = []
        for layer in range(self.n_layers):
            bias = self.bias_gate[layer]
            perm = self._sigmoid(bias).mean()
            perms.append(float(perm))
        return perms

    def reset(self):
        """Reset membrane state."""
        self.state = {"out": np.zeros(self.units)}
        for layer in range(self.n_layers):
            self.state[f"layer_{layer}"] = np.zeros(self.units)
