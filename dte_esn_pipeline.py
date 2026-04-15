#!/usr/bin/env python3
"""
Deep Tree Echo — Full ESN Pipeline

Demonstrates the complete AAR (Agent-Arena-Relation) reservoir computing
pipeline using the custom DTE nodes, trained on the DTE JSONL datasets.

Pipeline:
  1. Load and tokenize DTE training data into numeric sequences
  2. Create feature embeddings from token sequences
  3. Run EchoReservoir (Arena) to generate rich state representations
  4. Train CognitiveReadout (Agent) via ridge regression
  5. Wire AARRelation (Self) feedback loop
  6. Run EchobeatNode for cognitive cycle analysis
  7. Run IntrospectionNode for self-monitoring
  8. Evaluate and visualize results
"""

import json
import sys
import os
import numpy as np

# Add the nodes directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dte_nodes.echo_reservoir import EchoReservoir
from dte_nodes.cognitive_readout import CognitiveReadout
from dte_nodes.aar_relation import AARRelation
from dte_nodes.echobeat_node import EchobeatNode
from dte_nodes.introspection_node import IntrospectionNode
from dte_nodes.membrane_node import MembraneNode


# ─── 1. Data Loading & Tokenization ───────────────────────────────────────

def load_dte_data(filepath):
    """Load JSONL training data and extract text."""
    texts = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Handle messages format
            if "messages" in record:
                for msg in record["messages"]:
                    content = msg.get("content", "")
                    if content:
                        texts.append(content)
            elif "text" in record:
                texts.append(record["text"])
    return texts


def texts_to_char_embeddings(texts, embed_dim=32, max_len=200):
    """Convert texts to character-level embeddings for reservoir input.

    Uses a simple hash-based embedding: each character maps to a
    pseudo-random vector in R^embed_dim.
    """
    rng = np.random.RandomState(42)
    # Create character embedding table (256 ASCII chars)
    char_table = rng.randn(256, embed_dim) * 0.1

    all_embeddings = []
    for text in texts:
        chars = [ord(c) % 256 for c in text[:max_len]]
        embeddings = np.array([char_table[c] for c in chars])
        all_embeddings.append(embeddings)

    return all_embeddings


# ─── 2. Next-Character Prediction Task ────────────────────────────────────

def create_prediction_task(embeddings_list, n_samples=500):
    """Create a next-step prediction task from embedded sequences.

    Input X[t] = embedding[t], Target Y[t] = embedding[t+1]
    """
    X_all, Y_all = [], []
    for emb in embeddings_list:
        if len(emb) < 3:
            continue
        X_all.append(emb[:-1])
        Y_all.append(emb[1:])
        if sum(len(x) for x in X_all) >= n_samples:
            break

    X = np.concatenate(X_all, axis=0)[:n_samples]
    Y = np.concatenate(Y_all, axis=0)[:n_samples]
    return X, Y


# ─── 3. Pipeline Execution ────────────────────────────────────────────────

def run_pipeline():
    """Execute the full DTE ESN pipeline."""
    results = {}

    print("=" * 70)
    print("  Deep Tree Echo — ESN Pipeline (AAR Architecture)")
    print("=" * 70)

    # ── Load Data ──
    print("\n[1/8] Loading DTE training data...")
    data_file = "/home/ubuntu/upload/training_dataset_dtesnn.jsonl"
    texts = load_dte_data(data_file)
    print(f"  Loaded {len(texts)} text segments")

    # ── Embed ──
    print("\n[2/8] Creating character embeddings (dim=32)...")
    embed_dim = 32
    embeddings = texts_to_char_embeddings(texts, embed_dim=embed_dim)
    print(f"  Created {len(embeddings)} embedded sequences")
    print(f"  Total timesteps: {sum(len(e) for e in embeddings)}")

    # ── Prediction Task ──
    print("\n[3/8] Creating next-step prediction task...")
    X, Y = create_prediction_task(embeddings, n_samples=1000)
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # ── Arena (EchoReservoir) ──
    print("\n[4/8] Running EchoReservoir (Arena)...")
    arena = EchoReservoir(
        units=128,
        spectral_radius=0.95,
        input_scaling=0.1,
        leak_rate_fast=0.8,
        leak_rate_slow=0.1,
        density=0.1,
        seed=42,
    )
    states_train = arena.run(X_train)
    arena.reset()
    states_test = arena.run(X_test)
    print(f"  Reservoir states: train={states_train.shape}, test={states_test.shape}")

    # Verify echo state property
    esp_diff = arena.get_echo_state_property(X_train[:200])
    print(f"  Echo state property check: max diff after washout = {esp_diff:.6f}")
    results["echo_state_property"] = float(esp_diff)

    # ── Agent (CognitiveReadout) — Offline Training ──
    print("\n[5/8] Training CognitiveReadout (Agent) via ridge regression...")
    agent = CognitiveReadout(
        output_dim=embed_dim,
        ridge=1e-4,
        mode="offline",
    )
    agent.fit(states_train, Y_train, warmup=50)

    # Evaluate
    Y_pred_train = agent.run(states_train)
    Y_pred_test = agent.run(states_test)

    mse_train = np.mean((Y_pred_train[50:] - Y_train[50:]) ** 2)
    mse_test = np.mean((Y_pred_test - Y_test) ** 2)
    baseline_mse = np.mean((Y_test - Y_test.mean(axis=0)) ** 2)
    nrmse = np.sqrt(mse_test / baseline_mse) if baseline_mse > 0 else float("inf")

    print(f"  Train MSE: {mse_train:.6f}")
    print(f"  Test MSE:  {mse_test:.6f}")
    print(f"  NRMSE:     {nrmse:.4f}")
    print(f"  Wout norm: {agent.get_weights_norm():.4f}")
    results["mse_train"] = float(mse_train)
    results["mse_test"] = float(mse_test)
    results["nrmse"] = float(nrmse)

    # ── Agent (Online Learning) ──
    print("\n[5b/8] Testing online learning (RLS)...")
    agent_online = CognitiveReadout(
        output_dim=embed_dim,
        ridge=1e-3,
        alpha=0.99,
        mode="online",
    )
    online_errors = []
    for t in range(min(200, len(states_train))):
        pred = agent_online.partial_fit(states_train[t], Y_train[t])
        err = np.mean((pred - Y_train[t]) ** 2)
        online_errors.append(err)

    print(f"  Online MSE (first 50):  {np.mean(online_errors[:50]):.6f}")
    print(f"  Online MSE (last 50):   {np.mean(online_errors[-50:]):.6f}")
    print(f"  Learning improvement:   {np.mean(online_errors[:50]) / (np.mean(online_errors[-50:]) + 1e-10):.2f}x")
    results["online_improvement"] = float(np.mean(online_errors[:50]) / (np.mean(online_errors[-50:]) + 1e-10))

    # ── Relation (AARRelation) — Feedback Loop ──
    print("\n[6/8] Running AARRelation (Self) feedback loop...")
    relation = AARRelation(
        feedback_dim=embed_dim,
        target_dim=embed_dim,
        feedback_scaling=0.3,
        momentum=0.5,
        attention_heads=3,
        seed=42,
    )

    # Run closed-loop: Arena -> Agent -> Relation -> Arena
    n_closed = 100
    closed_states = np.zeros((n_closed, 128))
    closed_outputs = np.zeros((n_closed, embed_dim))
    feedback_signals = np.zeros((n_closed, embed_dim))

    arena.reset()
    x = X_train[0]
    for t in range(n_closed):
        # Arena processes input + feedback
        if t > 0:
            x_with_fb = x + 0.1 * relation.step(closed_outputs[t - 1])
        else:
            x_with_fb = x
        state = arena.step(x_with_fb)
        closed_states[t] = state
        output = agent.step(state)
        closed_outputs[t] = output
        if t < len(X_train) - 1:
            x = X_train[t + 1]

    attn = relation.get_attention_weights()
    print(f"  Closed-loop steps: {n_closed}")
    print(f"  Attention weights: {attn}")
    print(f"  Feedback signal norm (mean): {np.mean(np.linalg.norm(feedback_signals, axis=1)):.4f}")
    results["attention_weights"] = attn.tolist() if attn is not None else []

    # ── Echobeat (9-step cycle) ──
    print("\n[7/8] Running EchobeatNode (9-step cognitive cycle)...")
    echobeat = EchobeatNode(
        units=32,
        n_streams=3,
        cycle_length=9,
        coupling=0.1,
        seed=42,
    )
    eb_outputs = echobeat.run(X_train[:27])  # 3 full cycles
    stream_states = echobeat.get_stream_states()
    modes = echobeat.get_current_modes()

    print(f"  Echobeat output shape: {eb_outputs.shape}")
    print(f"  Current modes: {modes}")
    for s, state in stream_states.items():
        if state is not None:
            print(f"  Stream {s} norm: {np.linalg.norm(state):.4f}")
    results["echobeat_modes"] = modes

    # ── Introspection ──
    print("\n[8/8] Running IntrospectionNode (recursive self-monitoring)...")
    introspect = IntrospectionNode(max_depth=3, window_size=50)
    intro_outputs = introspect.run(states_train[:100])

    for depth in range(3):
        metrics = introspect.get_depth_metrics(depth)
        print(f"  Depth {depth}: mean={metrics['mean']:.4f}, "
              f"var={metrics['variance']:.4f}, "
              f"entropy={metrics['entropy']:.4f}, "
              f"divergence={metrics['divergence']:.6f}, "
              f"activation={metrics['activation_ratio']:.4f}")
    results["introspection_depth_0"] = introspect.get_depth_metrics(0)
    results["introspection_depth_1"] = introspect.get_depth_metrics(1)
    results["introspection_depth_2"] = introspect.get_depth_metrics(2)

    # ── Membrane ──
    print("\n[Bonus] Running MembraneNode (3-layer nested boundary)...")
    membrane = MembraneNode(units=32, permeability=0.5, n_layers=3, seed=42)
    mb_outputs = membrane.run(X_train[:50])
    perms = membrane.get_permeability_profile()
    print(f"  Membrane output shape: {mb_outputs.shape}")
    print(f"  Permeability profile: {[f'{p:.3f}' for p in perms]}")
    results["membrane_permeability"] = perms

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  Pipeline Summary")
    print("=" * 70)
    print(f"  Arena (EchoReservoir):  128 units, SR=0.95, fast/slow dual-pool")
    print(f"  Agent (CognitiveReadout): ridge={agent.ridge}, NRMSE={nrmse:.4f}")
    print(f"  Relation (AARRelation): 3-head attention feedback, momentum=0.5")
    print(f"  Echobeat: 3 streams x 32 units, 9-step cycle")
    print(f"  Introspection: 3-depth recursive monitoring")
    print(f"  Membrane: 3-layer nested boundary, permeability={perms}")
    print("=" * 70)

    # Save results
    results_file = "/home/ubuntu/nanecho-demo/esn_pipeline_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    return results


# ─── 4. Visualization ─────────────────────────────────────────────────────

def create_visualizations(results_file="/home/ubuntu/nanecho-demo/esn_pipeline_results.json"):
    """Create visualizations of the pipeline results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Re-run a smaller version to get data for plots
    texts = load_dte_data("/home/ubuntu/upload/training_dataset_dtesnn.jsonl")
    embeddings = texts_to_char_embeddings(texts, embed_dim=32)
    X, Y = create_prediction_task(embeddings, n_samples=500)

    arena = EchoReservoir(units=128, spectral_radius=0.95, seed=42)
    states = arena.run(X)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Deep Tree Echo — ESN Pipeline (AAR Architecture)", fontsize=16, fontweight="bold")

    # 1. Reservoir state heatmap (fast pool)
    ax = axes[0, 0]
    im = ax.imshow(states[:100, :64].T, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    ax.set_title("Arena: Fast Pool States (first 100 steps)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Neuron")
    plt.colorbar(im, ax=ax)

    # 2. Reservoir state heatmap (slow pool)
    ax = axes[0, 1]
    im = ax.imshow(states[:100, 64:].T, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    ax.set_title("Arena: Slow Pool States (first 100 steps)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Neuron")
    plt.colorbar(im, ax=ax)

    # 3. State PCA
    ax = axes[0, 2]
    from numpy.linalg import svd
    states_centered = states - states.mean(axis=0)
    U, S, Vt = svd(states_centered, full_matrices=False)
    pc = states_centered @ Vt[:2].T
    scatter = ax.scatter(pc[:, 0], pc[:, 1], c=np.arange(len(pc)), cmap="viridis", s=5, alpha=0.7)
    ax.set_title("Arena: PCA of Reservoir States")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(scatter, ax=ax, label="Timestep")

    # 4. Online learning curve
    agent_online = CognitiveReadout(output_dim=32, ridge=1e-3, alpha=0.99, mode="online")
    online_errors = []
    for t in range(min(300, len(states))):
        pred = agent_online.partial_fit(states[t], Y[t] if t < len(Y) else np.zeros(32))
        if t < len(Y):
            err = np.mean((pred - Y[t]) ** 2)
            online_errors.append(err)

    ax = axes[1, 0]
    ax.plot(online_errors, color="#2196F3", alpha=0.3, linewidth=0.5)
    # Smoothed
    window = 20
    if len(online_errors) > window:
        smoothed = np.convolve(online_errors, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(online_errors)), smoothed, color="#1565C0", linewidth=2)
    ax.set_title("Agent: Online Learning Curve (RLS)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")

    # 5. Echobeat stream dynamics
    echobeat = EchobeatNode(units=32, n_streams=3, cycle_length=9, coupling=0.1, seed=42)
    eb_out = echobeat.run(X[:36])  # 4 full cycles

    ax = axes[1, 1]
    colors = ["#E53935", "#43A047", "#1E88E5"]
    labels = ["Stream A (Perceive)", "Stream B (Act)", "Stream C (Simulate)"]
    for s in range(3):
        stream_data = eb_out[:, s * 32:(s + 1) * 32]
        stream_norm = np.linalg.norm(stream_data, axis=1)
        ax.plot(stream_norm, color=colors[s], label=labels[s], linewidth=1.5)
    # Mark cycle boundaries
    for c in range(0, 36, 9):
        ax.axvline(c, color="gray", linestyle="--", alpha=0.3)
    ax.set_title("Echobeat: 3-Stream Cognitive Cycle")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Stream Norm")
    ax.legend(fontsize=8)

    # 6. Introspection depth metrics
    introspect = IntrospectionNode(max_depth=3, window_size=50)
    intro_out = introspect.run(states[:100])

    ax = axes[1, 2]
    metric_names = ["Mean", "Variance", "Entropy", "Divergence", "Activation"]
    x_pos = np.arange(len(metric_names))
    width = 0.25
    depth_colors = ["#7B1FA2", "#AB47BC", "#CE93D8"]

    for depth in range(3):
        metrics = introspect.get_depth_metrics(depth)
        values = [metrics["mean"], metrics["variance"], metrics["entropy"],
                  metrics["divergence"], metrics["activation_ratio"]]
        ax.bar(x_pos + depth * width, values, width, label=f"Depth {depth}",
               color=depth_colors[depth], alpha=0.8)

    ax.set_title("Introspection: Recursive Self-Monitoring")
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(metric_names, rotation=15)
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = "/home/ubuntu/nanecho-demo/esn_pipeline_visualization.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to {fig_path}")
    return fig_path


if __name__ == "__main__":
    results = run_pipeline()
    fig_path = create_visualizations()
