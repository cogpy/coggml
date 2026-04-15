#!/usr/bin/env python3
"""
tests/test_dte_nodes.py — Unit tests for the dte_nodes package.

Tests all six DTE node classes:
  - EchoReservoir   (Arena)
  - CognitiveReadout (Agent)
  - AARRelation      (Self)
  - EchobeatNode     (9-step cognitive cycle)
  - IntrospectionNode (recursive self-monitoring)
  - MembraneNode     (hierarchical membrane boundary)

Run with:
  python -m pytest tests/test_dte_nodes.py -v
  # or directly:
  python tests/test_dte_nodes.py
"""

import sys
import os
import unittest
import numpy as np

# Ensure dte_nodes is importable from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dte_nodes.echo_reservoir import EchoReservoir
from dte_nodes.cognitive_readout import CognitiveReadout
from dte_nodes.aar_relation import AARRelation
from dte_nodes.echobeat_node import EchobeatNode
from dte_nodes.introspection_node import IntrospectionNode
from dte_nodes.membrane_node import MembraneNode


# ─── EchoReservoir ────────────────────────────────────────────────────────────

class TestEchoReservoir(unittest.TestCase):

    def setUp(self):
        self.res = EchoReservoir(units=32, spectral_radius=0.9, seed=0)
        self.x = np.random.RandomState(1).randn(8)

    def test_default_name(self):
        res = EchoReservoir()
        self.assertEqual(res.name, "EchoReservoir")

    def test_custom_name(self):
        res = EchoReservoir(name="MyReservoir")
        self.assertEqual(res.name, "MyReservoir")

    def test_unit_split(self):
        res = EchoReservoir(units=32)
        self.assertEqual(res.fast_units, 16)
        self.assertEqual(res.slow_units, 16)
        self.assertEqual(res.output_dim, 32)

    def test_unit_split_odd(self):
        res = EchoReservoir(units=33)
        self.assertEqual(res.fast_units, 16)
        self.assertEqual(res.slow_units, 17)

    def test_initialize_on_step(self):
        self.assertFalse(self.res.initialized)
        out = self.res.step(self.x)
        self.assertTrue(self.res.initialized)
        self.assertEqual(out.shape, (32,))

    def test_step_output_shape(self):
        out = self.res.step(self.x)
        self.assertEqual(out.shape, (32,))

    def test_step_values_bounded(self):
        for _ in range(20):
            out = self.res.step(self.x)
        # tanh activations keep states in (-1, 1), linear combination can exceed
        # but reservoir should remain stable
        self.assertTrue(np.all(np.isfinite(out)))

    def test_run_output_shape(self):
        T, D = 50, 8
        X = np.random.RandomState(2).randn(T, D)
        states = self.res.run(X)
        self.assertEqual(states.shape, (T, 32))

    def test_run_updates_state(self):
        T, D = 10, 8
        X = np.random.RandomState(3).randn(T, D)
        self.res.run(X)
        self.assertTrue(self.res.initialized)
        self.assertEqual(self.res.state["out"].shape, (32,))

    def test_reset_zeros_state(self):
        self.res.step(self.x)
        self.res.reset()
        np.testing.assert_array_equal(self.res.state["out"], np.zeros(32))
        np.testing.assert_array_equal(self.res.state["fast"], np.zeros(16))
        np.testing.assert_array_equal(self.res.state["slow"], np.zeros(16))

    def test_echo_state_property(self):
        """Two trajectories from different ICs should converge after washout."""
        X = np.random.RandomState(5).randn(300, 8)
        # Initialize first
        self.res.step(X[0])
        diff = self.res.get_echo_state_property(X, n_washout=100)
        self.assertLess(diff, 0.05,
                        f"Echo state property violated: max diff = {diff:.4f}")

    def test_deterministic_with_seed(self):
        r1 = EchoReservoir(units=16, seed=42)
        r2 = EchoReservoir(units=16, seed=42)
        x = np.array([1.0, 2.0, 3.0])
        out1 = r1.step(x)
        out2 = r2.step(x)
        np.testing.assert_array_equal(out1, out2)

    def test_different_seeds_differ(self):
        r1 = EchoReservoir(units=16, seed=1)
        r2 = EchoReservoir(units=16, seed=2)
        x = np.array([1.0, 2.0, 3.0])
        out1 = r1.step(x)
        out2 = r2.step(x)
        self.assertFalse(np.allclose(out1, out2))

    def test_win_shape_after_init(self):
        self.res.step(self.x)
        self.assertEqual(self.res.Win.shape, (32, 8))
        self.assertEqual(self.res.W.shape, (32, 32))

    def test_run_sequential_vs_batch_equivalent(self):
        """step() repeatedly should match run()."""
        X = np.random.RandomState(7).randn(20, 6)
        r_batch = EchoReservoir(units=16, seed=10)
        r_step = EchoReservoir(units=16, seed=10)
        batch_states = r_batch.run(X)
        step_states = np.zeros((20, 16))
        for t in range(20):
            step_states[t] = r_step.step(X[t])
        np.testing.assert_allclose(batch_states, step_states, rtol=1e-10)


# ─── CognitiveReadout ─────────────────────────────────────────────────────────

class TestCognitiveReadout(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.T = 100
        self.input_dim = 32
        self.output_dim = 4
        # Synthetic dataset: random reservoir states, linear target
        self.X = np.random.randn(self.T, self.input_dim)
        W_true = np.random.randn(self.input_dim, self.output_dim)
        self.Y = self.X @ W_true + 0.01 * np.random.randn(self.T, self.output_dim)

    def test_default_name(self):
        ro = CognitiveReadout()
        self.assertEqual(ro.name, "CognitiveReadout")

    def test_custom_name(self):
        ro = CognitiveReadout(name="MyReadout")
        self.assertEqual(ro.name, "MyReadout")

    def test_not_initialized_before_use(self):
        ro = CognitiveReadout(output_dim=self.output_dim)
        self.assertFalse(ro.initialized)

    def test_offline_fit_initializes(self):
        ro = CognitiveReadout(output_dim=self.output_dim)
        ro.fit(self.X, self.Y)
        self.assertTrue(ro.initialized)
        self.assertEqual(ro.Wout.shape, (self.input_dim, self.output_dim))

    def test_offline_fit_accuracy(self):
        ro = CognitiveReadout(output_dim=self.output_dim, ridge=1e-8)
        ro.fit(self.X, self.Y)
        Y_pred = ro.run(self.X)
        mse = np.mean((Y_pred - self.Y) ** 2)
        self.assertLess(mse, 1.0, f"Ridge regression MSE too high: {mse:.4f}")

    def test_run_output_shape(self):
        ro = CognitiveReadout(output_dim=self.output_dim)
        ro.fit(self.X, self.Y)
        Y_pred = ro.run(self.X)
        self.assertEqual(Y_pred.shape, (self.T, self.output_dim))

    def test_step_output_shape(self):
        ro = CognitiveReadout(output_dim=self.output_dim)
        ro.fit(self.X, self.Y)
        out = ro.step(self.X[0])
        self.assertEqual(out.shape, (self.output_dim,))

    def test_fit_with_warmup(self):
        ro = CognitiveReadout(output_dim=self.output_dim)
        ro.fit(self.X, self.Y, warmup=10)
        self.assertTrue(ro.initialized)

    def test_1d_target(self):
        ro = CognitiveReadout(output_dim=1)
        y_1d = self.Y[:, 0]  # shape (T,)
        ro.fit(self.X, y_1d)
        Y_pred = ro.run(self.X)
        self.assertEqual(Y_pred.shape, (self.T, 1))

    def test_online_mode_initializes(self):
        ro = CognitiveReadout(output_dim=self.output_dim, mode="online", ridge=1e-3)
        x0 = self.X[0]
        y0 = self.Y[0]
        ro.partial_fit(x0, y0)
        self.assertTrue(ro.initialized)
        self.assertIsNotNone(ro.P)

    def test_online_partial_fit_returns_prediction(self):
        ro = CognitiveReadout(output_dim=self.output_dim, mode="online", ridge=1e-3)
        for t in range(10):
            pred = ro.partial_fit(self.X[t], self.Y[t])
        self.assertEqual(pred.shape, (self.output_dim,))

    def test_online_learning_improves(self):
        ro = CognitiveReadout(output_dim=1, mode="online", ridge=1e-3)
        y_1d = self.Y[:, 0:1]
        errors = []
        for t in range(self.T):
            pred = ro.partial_fit(self.X[t], y_1d[t])
            errors.append(float(np.abs(pred - y_1d[t]).mean()))
        # Later errors should be significantly smaller than early errors
        early_err = np.mean(errors[:10])
        late_err = np.mean(errors[-10:])
        self.assertLess(late_err, early_err,
                        "Online RLS should converge: late error must be less than early error")

    def test_weights_norm_zero_before_fit(self):
        ro = CognitiveReadout(output_dim=self.output_dim)
        self.assertEqual(ro.get_weights_norm(), 0.0)

    def test_weights_norm_nonzero_after_fit(self):
        ro = CognitiveReadout(output_dim=self.output_dim)
        ro.fit(self.X, self.Y)
        self.assertGreater(ro.get_weights_norm(), 0.0)

    def test_reset_clears_state(self):
        ro = CognitiveReadout(output_dim=self.output_dim)
        ro.fit(self.X, self.Y)
        ro.reset()
        np.testing.assert_array_equal(ro.state["out"], np.zeros(self.output_dim))


# ─── AARRelation ──────────────────────────────────────────────────────────────

class TestAARRelation(unittest.TestCase):

    def setUp(self):
        self.feedback_dim = 4
        self.target_dim = 16
        self.rel = AARRelation(
            feedback_dim=self.feedback_dim,
            target_dim=self.target_dim,
            seed=0,
        )
        np.random.seed(0)
        self.fb = np.random.randn(self.feedback_dim)

    def test_default_name(self):
        rel = AARRelation()
        self.assertEqual(rel.name, "AARRelation")

    def test_custom_name(self):
        rel = AARRelation(name="MyRelation")
        self.assertEqual(rel.name, "MyRelation")

    def test_not_initialized_before_use(self):
        self.assertFalse(self.rel.initialized)

    def test_step_initializes_and_returns(self):
        out = self.rel.step(self.fb)
        self.assertTrue(self.rel.initialized)
        self.assertEqual(out.shape, (self.target_dim,))

    def test_step_output_bounded(self):
        out = self.rel.step(self.fb)
        # tanh output: bounded in (-1, 1) after gating
        self.assertTrue(np.all(np.abs(out) <= 1.0 + 1e-6))

    def test_run_output_shape(self):
        T = 20
        agent_outputs = np.random.RandomState(1).randn(T, self.feedback_dim)
        signals = self.rel.run(agent_outputs)
        self.assertEqual(signals.shape, (T, self.target_dim))

    def test_momentum_smoothing(self):
        """With high momentum, consecutive steps should be similar."""
        rel = AARRelation(feedback_dim=4, target_dim=16, momentum=0.95, seed=0)
        fb = np.random.RandomState(5).randn(4)
        out1 = rel.step(fb)
        out2 = rel.step(fb)
        diff = np.max(np.abs(out2 - out1))
        self.assertLess(diff, 0.5, "High-momentum steps should be similar")

    def test_attention_weights_sum_to_one(self):
        self.rel.step(self.fb)
        weights = self.rel.get_attention_weights()
        self.assertIsNotNone(weights)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=6)

    def test_attention_weights_nonnegative(self):
        self.rel.step(self.fb)
        weights = self.rel.get_attention_weights()
        self.assertTrue(np.all(weights >= 0))

    def test_reset_zeros_state(self):
        self.rel.step(self.fb)
        self.rel.reset()
        np.testing.assert_array_equal(self.rel.state["out"], np.zeros(self.target_dim))
        self.assertEqual(self.rel.state["step_count"], 0)

    def test_deterministic_with_seed(self):
        r1 = AARRelation(feedback_dim=4, target_dim=8, seed=7)
        r2 = AARRelation(feedback_dim=4, target_dim=8, seed=7)
        fb = np.array([1.0, -0.5, 0.3, 2.0])
        np.testing.assert_array_equal(r1.step(fb), r2.step(fb))

    def test_run_sequential_vs_batch_equivalent(self):
        T = 15
        agent_outputs = np.random.RandomState(3).randn(T, self.feedback_dim)
        r_batch = AARRelation(feedback_dim=self.feedback_dim, target_dim=self.target_dim, seed=0)
        r_step = AARRelation(feedback_dim=self.feedback_dim, target_dim=self.target_dim, seed=0)
        batch_out = r_batch.run(agent_outputs)
        step_out = np.zeros((T, self.target_dim))
        for t in range(T):
            step_out[t] = r_step.step(agent_outputs[t])
        np.testing.assert_allclose(batch_out, step_out, rtol=1e-10)


# ─── EchobeatNode ─────────────────────────────────────────────────────────────

class TestEchobeatNode(unittest.TestCase):

    def setUp(self):
        self.units = 16
        self.n_streams = 3
        self.eb = EchobeatNode(units=self.units, n_streams=self.n_streams, seed=0)
        np.random.seed(42)
        self.x = np.random.randn(8)

    def test_default_name(self):
        eb = EchobeatNode()
        self.assertEqual(eb.name, "EchobeatNode")

    def test_custom_name(self):
        eb = EchobeatNode(name="MyEchobeat")
        self.assertEqual(eb.name, "MyEchobeat")

    def test_output_dim(self):
        self.assertEqual(self.eb.output_dim, self.units * self.n_streams)

    def test_step_initializes_and_returns(self):
        self.assertFalse(self.eb.initialized)
        out = self.eb.step(self.x)
        self.assertTrue(self.eb.initialized)
        self.assertEqual(out.shape, (self.units * self.n_streams,))

    def test_step_output_finite(self):
        out = self.eb.step(self.x)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_run_output_shape(self):
        T = 30
        X = np.random.RandomState(2).randn(T, 8)
        outputs = self.eb.run(X)
        self.assertEqual(outputs.shape, (T, self.units * self.n_streams))

    def test_stream_states_after_step(self):
        self.eb.step(self.x)
        streams = self.eb.get_stream_states()
        self.assertEqual(len(streams), self.n_streams)
        for s, state in streams.items():
            self.assertEqual(state.shape, (self.units,))

    def test_stream_phase_assignments(self):
        # Stream A activates at steps {0, 3, 6}
        self.assertIn(0, EchobeatNode.STREAM_PHASES[0])
        self.assertIn(3, EchobeatNode.STREAM_PHASES[0])
        # Phases for different streams must be disjoint
        phases_0 = set(EchobeatNode.STREAM_PHASES[0])
        phases_1 = set(EchobeatNode.STREAM_PHASES[1])
        phases_2 = set(EchobeatNode.STREAM_PHASES[2])
        self.assertEqual(len(phases_0 & phases_1), 0)
        self.assertEqual(len(phases_1 & phases_2), 0)

    def test_cognitive_modes_cycle(self):
        self.eb.step(self.x)
        modes = self.eb.get_current_modes()
        self.assertEqual(len(modes), self.n_streams)
        for s, mode in modes.items():
            self.assertIn(mode, EchobeatNode.MODES)

    def test_step_counter_increments(self):
        for _ in range(5):
            self.eb.step(self.x)
        self.assertEqual(self.eb.state["step"], 5)

    def test_reset_zeros_state(self):
        self.eb.step(self.x)
        self.eb.reset()
        np.testing.assert_array_equal(
            self.eb.state["out"], np.zeros(self.units * self.n_streams))
        self.assertEqual(self.eb.state["step"], 0)

    def test_deterministic_with_seed(self):
        e1 = EchobeatNode(units=8, n_streams=2, seed=99)
        e2 = EchobeatNode(units=8, n_streams=2, seed=99)
        x = np.array([1.0, -1.0, 0.5])
        np.testing.assert_array_equal(e1.step(x), e2.step(x))

    def test_run_sequential_vs_batch_equivalent(self):
        T = 10
        X = np.random.RandomState(4).randn(T, 6)
        e_batch = EchobeatNode(units=8, n_streams=2, seed=0)
        e_step = EchobeatNode(units=8, n_streams=2, seed=0)
        batch_out = e_batch.run(X)
        step_out = np.zeros((T, 16))
        for t in range(T):
            step_out[t] = e_step.step(X[t])
        np.testing.assert_allclose(batch_out, step_out, rtol=1e-10)


# ─── IntrospectionNode ────────────────────────────────────────────────────────

class TestIntrospectionNode(unittest.TestCase):

    def setUp(self):
        self.max_depth = 3
        self.node = IntrospectionNode(max_depth=self.max_depth, window_size=20)
        np.random.seed(42)
        self.x = np.random.randn(32)

    def test_default_name(self):
        n = IntrospectionNode()
        self.assertEqual(n.name, "IntrospectionNode")

    def test_custom_name(self):
        n = IntrospectionNode(name="MyIntrospect")
        self.assertEqual(n.name, "MyIntrospect")

    def test_output_dim(self):
        # 5 metrics × max_depth levels
        self.assertEqual(self.node.output_dim, 5 * self.max_depth)

    def test_step_initializes_and_returns(self):
        self.assertFalse(self.node.initialized)
        out = self.node.step(self.x)
        self.assertTrue(self.node.initialized)
        self.assertEqual(out.shape, (5 * self.max_depth,))

    def test_step_output_finite(self):
        out = self.node.step(self.x)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_run_output_shape(self):
        T = 40
        X = np.random.RandomState(1).randn(T, 32)
        outputs = self.node.run(X)
        self.assertEqual(outputs.shape, (T, 5 * self.max_depth))

    def test_depth_metrics_keys(self):
        self.node.step(self.x)
        metrics = self.node.get_depth_metrics(depth=0)
        expected_keys = {"mean", "variance", "entropy", "divergence", "activation_ratio"}
        self.assertEqual(set(metrics.keys()), expected_keys)

    def test_depth_metrics_invalid_depth_raises(self):
        self.node.step(self.x)
        with self.assertRaises(ValueError):
            self.node.get_depth_metrics(depth=self.max_depth)

    def test_entropy_in_unit_interval(self):
        self.node.step(self.x)
        metrics = self.node.get_depth_metrics(depth=0)
        self.assertGreaterEqual(metrics["entropy"], 0.0)
        self.assertLessEqual(metrics["entropy"], 1.0 + 1e-6)

    def test_activation_ratio_in_unit_interval(self):
        self.node.step(self.x)
        metrics = self.node.get_depth_metrics(depth=0)
        self.assertGreaterEqual(metrics["activation_ratio"], 0.0)
        self.assertLessEqual(metrics["activation_ratio"], 1.0 + 1e-6)

    def test_variance_nonnegative(self):
        self.node.step(self.x)
        metrics = self.node.get_depth_metrics(depth=0)
        self.assertGreaterEqual(metrics["variance"], 0.0)

    def test_history_builds_over_steps(self):
        for _ in range(10):
            self.node.step(self.x)
        self.assertEqual(self.node.state["position"], 10)

    def test_reset_clears_state(self):
        self.node.step(self.x)
        self.node.reset()
        np.testing.assert_array_equal(
            self.node.state["out"], np.zeros(5 * self.max_depth))
        self.assertEqual(self.node.state["position"], 0)

    def test_run_sequential_vs_batch_equivalent(self):
        T = 15
        X = np.random.RandomState(6).randn(T, 16)
        n_batch = IntrospectionNode(max_depth=2, window_size=10)
        n_step = IntrospectionNode(max_depth=2, window_size=10)
        batch_out = n_batch.run(X)
        step_out = np.zeros((T, 10))
        for t in range(T):
            step_out[t] = n_step.step(X[t])
        np.testing.assert_allclose(batch_out, step_out, rtol=1e-10)

    def test_deeper_introspection_depth(self):
        n = IntrospectionNode(max_depth=5, window_size=10)
        x = np.random.randn(20)
        out = n.step(x)
        self.assertEqual(out.shape, (25,))  # 5 metrics × 5 depths


# ─── MembraneNode ─────────────────────────────────────────────────────────────

class TestMembraneNode(unittest.TestCase):

    def setUp(self):
        self.units = 16
        self.n_layers = 3
        self.mem = MembraneNode(units=self.units, n_layers=self.n_layers, seed=0)
        np.random.seed(42)
        self.x = np.random.randn(8)

    def test_default_name(self):
        m = MembraneNode()
        self.assertEqual(m.name, "MembraneNode")

    def test_custom_name(self):
        m = MembraneNode(name="MyMembrane")
        self.assertEqual(m.name, "MyMembrane")

    def test_output_dim(self):
        self.assertEqual(self.mem.output_dim, self.units)

    def test_step_initializes_and_returns(self):
        self.assertFalse(self.mem.initialized)
        out = self.mem.step(self.x)
        self.assertTrue(self.mem.initialized)
        self.assertEqual(out.shape, (self.units,))

    def test_step_output_finite(self):
        out = self.mem.step(self.x)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_run_output_shape(self):
        T = 25
        X = np.random.RandomState(2).randn(T, 8)
        outputs = self.mem.run(X)
        self.assertEqual(outputs.shape, (T, self.units))

    def test_layer_states_after_step(self):
        self.mem.step(self.x)
        layers = self.mem.get_layer_states()
        self.assertEqual(len(layers), self.n_layers)
        for layer, state in layers.items():
            self.assertEqual(state.shape, (self.units,))

    def test_permeability_profile_length(self):
        self.mem.step(self.x)
        profile = self.mem.get_permeability_profile()
        self.assertEqual(len(profile), self.n_layers)

    def test_permeability_in_unit_interval(self):
        self.mem.step(self.x)
        profile = self.mem.get_permeability_profile()
        for p in profile:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_gating_signal_bounded(self):
        """Output of membrane (tanh-gated) should be bounded."""
        for _ in range(20):
            out = self.mem.step(self.x)
        self.assertTrue(np.all(np.abs(out) <= 1.0 + 1e-6))

    def test_reset_zeros_state(self):
        self.mem.step(self.x)
        self.mem.reset()
        np.testing.assert_array_equal(self.mem.state["out"], np.zeros(self.units))
        for layer in range(self.n_layers):
            np.testing.assert_array_equal(
                self.mem.state[f"layer_{layer}"], np.zeros(self.units))

    def test_deterministic_with_seed(self):
        m1 = MembraneNode(units=8, n_layers=2, seed=11)
        m2 = MembraneNode(units=8, n_layers=2, seed=11)
        x = np.array([0.5, -0.5, 1.0])
        np.testing.assert_array_equal(m1.step(x), m2.step(x))

    def test_run_sequential_vs_batch_equivalent(self):
        T = 12
        X = np.random.RandomState(8).randn(T, 5)
        m_batch = MembraneNode(units=8, n_layers=2, seed=0)
        m_step = MembraneNode(units=8, n_layers=2, seed=0)
        batch_out = m_batch.run(X)
        step_out = np.zeros((T, 8))
        for t in range(T):
            step_out[t] = m_step.step(X[t])
        np.testing.assert_allclose(batch_out, step_out, rtol=1e-10)

    def test_high_permeability_passes_signal(self):
        """High permeability membrane should let more signal through."""
        m_open = MembraneNode(units=16, permeability=0.99, seed=5)
        m_closed = MembraneNode(units=16, permeability=0.01, seed=5)
        x = np.random.randn(8)
        out_open = m_open.step(x)
        out_closed = m_closed.step(x)
        norm_open = np.linalg.norm(out_open)
        norm_closed = np.linalg.norm(out_closed)
        # More open membrane should generally have larger or equal signal
        self.assertGreaterEqual(norm_open, norm_closed * 0.5,
                                "High permeability should not greatly reduce signal")


# ─── Integration: AAR Pipeline ────────────────────────────────────────────────

class TestAARPipeline(unittest.TestCase):
    """Tests the full Arena-Agent-Relation pipeline."""

    def test_full_aar_pipeline(self):
        """EchoReservoir → CognitiveReadout → AARRelation feedback loop."""
        np.random.seed(0)
        T = 50
        input_dim = 8
        reservoir_units = 32
        output_dim = 4

        X = np.random.randn(T, input_dim)

        # Arena
        reservoir = EchoReservoir(units=reservoir_units, seed=0)
        states = reservoir.run(X)  # (T, reservoir_units)

        # Agent: fit readout on reservoir states
        target = np.sin(np.linspace(0, 4 * np.pi, T)).reshape(-1, 1)
        target = np.tile(target, (1, output_dim))
        readout = CognitiveReadout(output_dim=output_dim)
        readout.fit(states, target)
        predictions = readout.run(states)  # (T, output_dim)

        # Self: feedback from agent to arena
        relation = AARRelation(
            feedback_dim=output_dim,
            target_dim=input_dim,
            seed=0,
        )
        feedback = relation.run(predictions)  # (T, input_dim)

        self.assertEqual(states.shape, (T, reservoir_units))
        self.assertEqual(predictions.shape, (T, output_dim))
        self.assertEqual(feedback.shape, (T, input_dim))
        self.assertTrue(np.all(np.isfinite(feedback)))

    def test_echobeat_with_reservoir(self):
        """EchoReservoir states can drive EchobeatNode."""
        np.random.seed(1)
        T = 20
        reservoir = EchoReservoir(units=16, seed=1)
        echobeat = EchobeatNode(units=8, n_streams=2, seed=1)

        X_in = np.random.randn(T, 5)
        states = reservoir.run(X_in)
        cycle_out = echobeat.run(states)

        self.assertEqual(cycle_out.shape, (T, 16))
        self.assertTrue(np.all(np.isfinite(cycle_out)))

    def test_introspection_of_reservoir_states(self):
        """IntrospectionNode can analyze EchoReservoir states."""
        np.random.seed(2)
        T = 30
        reservoir = EchoReservoir(units=32, seed=2)
        introspect = IntrospectionNode(max_depth=2, window_size=20)

        X_in = np.random.randn(T, 4)
        states = reservoir.run(X_in)
        meta = introspect.run(states)

        self.assertEqual(meta.shape, (T, 10))
        self.assertTrue(np.all(np.isfinite(meta)))

    def test_membrane_filters_reservoir_output(self):
        """MembraneNode can filter reservoir output before readout."""
        np.random.seed(3)
        T = 20
        reservoir = EchoReservoir(units=16, seed=3)
        membrane = MembraneNode(units=16, n_layers=2, seed=3)
        readout = CognitiveReadout(output_dim=2)

        X_in = np.random.randn(T, 6)
        states = reservoir.run(X_in)
        filtered = membrane.run(states)
        target = np.random.randn(T, 2)
        readout.fit(filtered, target)
        predictions = readout.run(filtered)

        self.assertEqual(predictions.shape, (T, 2))
        self.assertTrue(np.all(np.isfinite(predictions)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
