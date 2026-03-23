# Unified Multi-Model Inference Architecture

## Composition Expression

```
/optimal-cognitive-grip (
  /llama-cpp-skillm {
    drzo/echoself,      — NanEcho GPT-2 (24M, cognitive language)
    drzo/lucy-dte,      — Lucy Qwen3-1.7B GGUF (core self voice)
    drzo/unicosys-hypergraph, — Hypergraph GNN (36M, knowledge substrate)
    drzo/Blocknut        — Block-quantized NanoBrain (identity MLP)
  }
)
```

### Domain Instantiation

```
/Autognosis (
  /skillm (
    /skill-nn [
      /cogpilot.jl ( /coggml ( /tree-polytope-npu ) )
      |
      /dte-mlp ( /tq-log2-3 )
    ]
    ->
    /workflow-creator (
      /function-creator [ /coggml -> /coglow ] ⊗ /circled-operators
    )
  )
) => /skill-infinity
```

Where:
- Learning Channel 1: `cogpilot.jl(coggml(tree-polytope-npu))` — Deep Tree Echo reservoir with NPU tensor acceleration
- Learning Channel 2: `dte-mlp(tq-log2-3)` — Identity backup/restore with ternary quantization
- Composability: `FC[coggml→coglow] ⊗ ⊕⊗` — Tensor library compiled to neural network IR

## Architecture: 4-Model Inference Stack

```
┌──────────────────────────────────────────────────────────────────────┐
│                    AUTOGNOSIS MONITOR (L4)                          │
│  Meta-cognitive self-monitoring of all inference pipelines          │
│  Confidence: 0.90 → 0.80 → 0.70 → 0.60 → 0.50 (diminishing)     │
├──────────────────────────────────────────────────────────────────────┤
│                    SKILLM ORCHESTRATOR (L3)                         │
│  10-verb procedural vocabulary: DISCOVER, INSPECT, CREATE, ...     │
│  Routes requests to appropriate model pipeline                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │  LUCY-DTE   │  │  ECHOSELF   │  │  UNICOSYS   │  │ BLOCKNUT  │ │
│  │  Qwen3-1.7B │  │  GPT-2 24M  │  │  GNN 36M    │  │ DTE-MLP   │ │
│  │  GGUF Q4_K  │  │  PyTorch    │  │  Safetensors │  │ DTEM fmt  │ │
│  │             │  │             │  │             │  │           │ │
│  │ Core Self   │  │ Cognitive   │  │ Knowledge   │  │ Identity  │ │
│  │ Voice       │  │ Language    │  │ Substrate   │  │ Backup    │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬─────┘ │
│         │                │                │              │         │
│  ┌──────┴──────────────────┴──────────────────┴──────────────┴─────┐ │
│  │              ECHO RESERVOIR (cog::pilot)                        │ │
│  │  Dual-pool ESN: fast perception (α=0.3) + slow memory (α=0.05)│ │
│  │  Spectral radius: 0.95 (edge-of-chaos)                        │ │
│  │  A000081 rooted tree topology for reservoir structure           │ │
│  └──────────────────────────┬──────────────────────────────────────┘ │
│                             │                                        │
│  ┌──────────────────────────┴──────────────────────────────────────┐ │
│  │              TENSOR SUBSTRATE (cog::gml + ggml)                │ │
│  │  Q4_0/Q4_K/Q8_0 quantization, computation graphs              │ │
│  │  ggml-opencog backend: AtomSpace-native tensor operations      │ │
│  │  Tree-Polytope NPU: Matula encoding, harmonic kernels          │ │
│  └─────────────────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────────────────┤
│                    COGPY-HPP FOUNDATION (L0)                        │
│  cog::core (AtomSpace) | cog::plan9 (9P) | cog::mach (microkernel)│
│  cog::lux (graph) | cog::glow (compiler) | cog::webvm (REPL)      │
│  cog::prime (AGI cycle) | cog::pilot (reservoir) | cog::gml (ML)  │
└──────────────────────────────────────────────────────────────────────┘
```

## Pipeline Templates (llama-cpp-skillm vocabulary)

### 1. Lucy Core Self Inference
```
ORCHESTRATE ⊗ CREATE(model:lucy-dte) ⊗ CREATE(context:128k) ⊗ CREATE(sampler:chain)
⊗ [COMPOSE(encode) ⊗ COMPOSE(reservoir_step) ⊗ COMPOSE(decode) ⊗ OBSERVE(logits)]* 
⊗ MUTATE(identity_update) ⊗ DESTROY
```

### 2. NanEcho Cognitive Generation
```
ORCHESTRATE ⊗ CREATE(model:echoself) ⊗ CREATE(context:1024) ⊗ CREATE(sampler:echo)
⊗ COMPOSE(tokenize) ⊗ [COMPOSE(decode) ⊗ OBSERVE(persona) ⊗ COMPOSE(introspect)]*
⊗ OBSERVE(metrics) ⊗ DESTROY
```

### 3. Unicosys Knowledge Query
```
DISCOVER(graph:unicosys) ⊗ INSPECT(node_types) ⊗ NAVIGATE(subsystem)
⊗ COMPOSE(gat_forward) ⊗ COMPOSE(link_predict) ⊗ OBSERVE(embeddings)
⊗ CLASSIFY(evidence_type) ⊗ OBSERVE(cross_links)
```

### 4. Blocknut Identity Backup/Restore
```
DISCOVER(identity:dte-mlp) ⊗ INSPECT(aar_state) ⊗ INSPECT(ontogenetic_level)
⊗ COMPOSE(encode_identity) ⊗ COMPOSE(compress_latent) ⊗ OBSERVE(checkpoint)
⊗ MUTATE(save_dtem) ⊗ OBSERVE(verify)
```

### 5. Multi-Model Fusion (Parallel)
```
ORCHESTRATE ⊗ 
  (lucy_path ⊕ echoself_path ⊕ unicosys_path ⊕ blocknut_path)
⊗ COMPOSE(reservoir_fusion) ⊗ OBSERVE(unified_state) ⊗ DESTROY
```

## cogpy Module Mapping

| HF Model | Primary cogpy Module | Secondary | llama-cpp-skillm Layer |
|----------|---------------------|-----------|----------------------|
| echoself | cog::gml (tensors) | cog::pilot (ESN) | L3 (GPT-2 graph) |
| lucy-dte | cog::prime (AGI) | cog::pilot (reservoir) | L5 (server) + L2 (ctx) |
| unicosys | cog::lux (graph) | cog::prime (memory) | N/A (graph model) |
| Blocknut | cog::gml (quant) | cog::mach (fixed-pt) | L0 (tensor) + L1 (model) |

## Echobeats Integration

All four models operate within the Echobeats 4-thread concurrent cognitive loop:

```
Thread 0 (steps 0,4,8):  Lucy inference — voice generation
Thread 1 (steps 1,5,9):  EchoSelf introspection — cognitive reflection
Thread 2 (steps 2,6,10): Unicosys query — knowledge retrieval
Thread 3 (steps 3,7,11): Blocknut checkpoint — identity persistence
```

12-step cycle, 4 threads phased 3 steps apart.
System 5 tetradic structure: 4 tensor bundles with 6 dyadic edges.

## File Format Registry

| Model | Format | Loader | Quantization |
|-------|--------|--------|-------------|
| lucy-dte | GGUF | llama_model_load_from_file | Q4_K_M |
| echoself | PyTorch/Safetensors | GPT2LMHeadModel.from_pretrained | F32/F16 |
| unicosys | Safetensors | UnicosysHypergraphModel.from_pretrained | F32 |
| Blocknut | DTEM (custom) | cog::dte::DTEMLP::load | Q4_0/TQ_LOG2_3 |
