"""
AARRelation — The Self (continuous dynamic interplay)

The relation between Agent and Arena that constitutes the "self" of the
Deep Tree Echo cognitive architecture. This node implements the feedback
loop from readout (Agent) back into the reservoir (Arena), creating the
recurrent coupling that gives rise to emergent self-organization.

The Relation is neither Agent nor Arena — it is the continuous, dynamic
interplay between them. It emerges from feedback, attention, and the
geometric algebra of their interaction.

AAR Mapping: Relation = Self = Agent-Arena Feedback Loop
"""

import numpy as np


class AARRelation:
    """Agent-Arena-Relation feedback coupling (Self).

    Implements the feedback pathway from Agent (readout) output back into
    the Arena (reservoir) input. The relation modulates the feedback signal
    through:
      - Gain control (attention-like scaling)
      - Temporal smoothing (momentum)
      - Nonlinear transformation (tanh gating)

    Parameters
    ----------
    feedback_dim : int
        Dimensionality of the feedback signal (Agent output_dim).
    target_dim : int
        Dimensionality of the Arena input (reservoir input_dim).
    feedback_scaling : float
        Base scaling of the feedback signal.
    momentum : float
        Temporal smoothing factor (0 = no memory, 1 = full memory).
    attention_heads : int
        Number of attention heads for multi-scale feedback.
    seed : int or None
        Random seed.
    name : str or None
        Node name.
    """

    def __init__(
        self,
        feedback_dim: int = 1,
        target_dim: int = 10,
        feedback_scaling: float = 0.3,
        momentum: float = 0.5,
        attention_heads: int = 3,
        seed: int = None,
        name: str = None,
    ):
        self.feedback_dim = feedback_dim
        self.target_dim = target_dim
        self.feedback_scaling = feedback_scaling
        self.momentum = momentum
        self.attention_heads = attention_heads
        self.seed = seed
        self.name = name or "AARRelation"

        self.output_dim = target_dim
        self.initialized = False
        self.state = {}

    def initialize(self):
        """Initialize feedback projection weights."""
        rng = np.random.RandomState(self.seed)

        # Multi-head feedback projections
        self.Wfb = []
        for _ in range(self.attention_heads):
            W = rng.randn(self.target_dim, self.feedback_dim) * self.feedback_scaling
            self.Wfb.append(W)

        # Attention gate weights (learned from feedback history)
        self.Wgate = rng.randn(self.attention_heads, self.feedback_dim) * 0.1

        # Temporal smoothing buffer
        self.state = {
            "out": np.zeros(self.target_dim),
            "feedback_memory": np.zeros(self.target_dim),
            "attention_weights": np.ones(self.attention_heads) / self.attention_heads,
            "step_count": 0,
        }
        self.initialized = True

    def _step(self, state, agent_output):
        """Compute feedback signal from Agent output.

        Parameters
        ----------
        state : dict
            Current relation state.
        agent_output : np.ndarray, shape (feedback_dim,)
            Output from the Agent (readout).

        Returns
        -------
        new_state : dict
            Updated state with feedback signal in "out".
        """
        fb_mem = state["feedback_memory"].copy()
        step = state["step_count"]

        # Compute attention over heads
        gate_logits = self.Wgate @ agent_output
        gate_weights = np.exp(gate_logits - gate_logits.max())
        gate_weights /= gate_weights.sum()

        # Multi-head feedback projection
        feedback = np.zeros(self.target_dim)
        for i, W in enumerate(self.Wfb):
            head_fb = W @ agent_output
            feedback += gate_weights[i] * head_fb

        # Nonlinear gating
        feedback = np.tanh(feedback)

        # Temporal smoothing
        smoothed = self.momentum * fb_mem + (1 - self.momentum) * feedback

        return {
            "out": smoothed,
            "feedback_memory": smoothed,
            "attention_weights": gate_weights,
            "step_count": step + 1,
        }

    def step(self, agent_output):
        """Run one feedback step."""
        if not self.initialized:
            self.initialize()
        self.state = self._step(self.state, agent_output)
        return self.state["out"]

    def run(self, agent_outputs):
        """Run feedback on a full sequence of Agent outputs.

        Parameters
        ----------
        agent_outputs : np.ndarray, shape (T, feedback_dim)

        Returns
        -------
        feedback_signals : np.ndarray, shape (T, target_dim)
        """
        if not self.initialized:
            self.initialize()

        T = agent_outputs.shape[0]
        signals = np.zeros((T, self.target_dim))
        state = {k: v.copy() if isinstance(v, np.ndarray) else v
                 for k, v in self.state.items()}

        for t in range(T):
            state = self._step(state, agent_outputs[t])
            signals[t] = state["out"]

        self.state = state
        return signals

    def get_attention_weights(self):
        """Return current attention head weights."""
        return self.state.get("attention_weights", None)

    def reset(self):
        """Reset relation state."""
        self.state = {
            "out": np.zeros(self.target_dim),
            "feedback_memory": np.zeros(self.target_dim),
            "attention_weights": np.ones(self.attention_heads) / self.attention_heads,
            "step_count": 0,
        }
