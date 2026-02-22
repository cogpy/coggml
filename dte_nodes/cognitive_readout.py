"""
CognitiveReadout — The Agent (urge-to-act / dynamic tensor transformations)

A trainable readout layer that extracts meaning from the reservoir's state
space. Supports both offline (ridge regression) and online (FORCE/RLS)
learning modes.

The readout is the "urge-to-act" — the dynamic tensor transformation that
maps the arena's high-dimensional state into actionable outputs.

AAR Mapping: Agent = Readout
"""

import numpy as np


class CognitiveReadout:
    """Trainable readout with offline and online learning (Agent).

    Parameters
    ----------
    output_dim : int
        Dimensionality of the output.
    ridge : float
        Ridge (L2) regularization parameter for offline training.
    alpha : float
        Forgetting factor for online RLS learning (0 < alpha <= 1).
    mode : str
        Learning mode: 'offline' (ridge regression) or 'online' (RLS).
    name : str or None
        Node name.
    """

    def __init__(
        self,
        output_dim: int = 1,
        ridge: float = 1e-6,
        alpha: float = 1.0,
        mode: str = "offline",
        name: str = None,
    ):
        self.output_dim = output_dim
        self.ridge = ridge
        self.alpha = alpha
        self.mode = mode
        self.name = name or "CognitiveReadout"

        self.input_dim = None
        self.Wout = None
        self.bias = None
        self.initialized = False

        # Online learning state (RLS)
        self.P = None  # Inverse correlation matrix

        self.state = {}

    def initialize(self, x, y=None):
        """Initialize readout weights."""
        if x.ndim == 1:
            self.input_dim = x.shape[0]
        else:
            self.input_dim = x.shape[-1]

        self.Wout = np.zeros((self.input_dim, self.output_dim))
        self.bias = np.zeros(self.output_dim)

        if self.mode == "online":
            self.P = np.eye(self.input_dim) / self.ridge

        self.state = {"out": np.zeros(self.output_dim)}
        self.initialized = True

    def _step(self, state, x):
        """Single timestep: linear readout."""
        y = x @ self.Wout + self.bias
        return {"out": y}

    def step(self, x):
        """Run one timestep."""
        if not self.initialized:
            self.initialize(x)
        self.state = self._step(self.state, x)
        return self.state["out"]

    def run(self, X):
        """Run readout on full timeseries."""
        if not self.initialized:
            self.initialize(X[0])
        Y = X @ self.Wout + self.bias
        self.state = {"out": Y[-1]}
        return Y

    def fit(self, X, Y, warmup: int = 0):
        """Offline training via ridge regression.

        Parameters
        ----------
        X : np.ndarray, shape (T, input_dim)
            Reservoir states (arena output).
        Y : np.ndarray, shape (T, output_dim)
            Target outputs.
        warmup : int
            Number of initial timesteps to discard (transient washout).

        Returns
        -------
        self
        """
        if not self.initialized:
            self.initialize(X[0], Y[0] if Y.ndim > 1 else Y)

        X_train = X[warmup:]
        Y_train = Y[warmup:]

        if Y_train.ndim == 1:
            Y_train = Y_train.reshape(-1, 1)

        # Ridge regression: Wout = (X^T X + ridge * I)^{-1} X^T Y
        XTX = X_train.T @ X_train
        XTY = X_train.T @ Y_train
        ridge_I = self.ridge * np.eye(self.input_dim)
        self.Wout = np.linalg.solve(XTX + ridge_I, XTY)

        # Bias from residual mean
        residuals = Y_train - X_train @ self.Wout
        self.bias = residuals.mean(axis=0)

        return self

    def partial_fit(self, x, y):
        """Online learning step via Recursive Least Squares (RLS).

        Parameters
        ----------
        x : np.ndarray, shape (input_dim,)
            Current reservoir state.
        y : np.ndarray, shape (output_dim,)
            Current target.

        Returns
        -------
        prediction : np.ndarray
            Prediction before update.
        """
        if not self.initialized:
            self.initialize(x, y)

        if y.ndim == 0:
            y = y.reshape(1)

        # Prediction before update
        prediction = x @ self.Wout + self.bias

        # RLS update
        Px = self.P @ x
        denom = self.alpha + x @ Px
        k = Px / denom  # Kalman gain

        error = y - prediction
        self.Wout += np.outer(k, error)
        self.P = (self.P - np.outer(k, Px)) / self.alpha

        return prediction

    def get_weights_norm(self):
        """Return the Frobenius norm of the readout weights."""
        if self.Wout is not None:
            return np.linalg.norm(self.Wout)
        return 0.0

    def reset(self):
        """Reset state (not weights)."""
        self.state = {"out": np.zeros(self.output_dim)}
        if self.mode == "online" and self.input_dim is not None:
            self.P = np.eye(self.input_dim) / self.ridge
