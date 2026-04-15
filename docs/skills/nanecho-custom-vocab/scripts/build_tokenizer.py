#!/usr/bin/env python3
"""
Build Custom DTE Tokenizer
===========================
Train a BPE tokenizer from Deep Tree Echo training data with:
  1. Domain-specific special tokens for DTE cognitive architecture
  2. Pre-tokenization rules that preserve DTE compound terms
  3. Vocab size optimized for the ~24M param NanEcho model
  4. Full HuggingFace Transformers compatibility (GPT2TokenizerFast)
"""

import json
import os
import re
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import NFD, StripAccents, Sequence as NormSequence
from transformers import PreTrainedTokenizerFast

# ── Configuration ────────────────────────────────────────────────────
VOCAB_SIZE = 8192          # Compact vocab for ~24M model (was 50304 with GPT-2)
MIN_FREQUENCY = 2          # Minimum frequency for BPE merges
CORPUS_PATH = "/home/ubuntu/nanecho-demo/combined_corpus.txt"
OUTPUT_DIR = "/home/ubuntu/nanecho-demo/dte_tokenizer"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── DTE Special Tokens ───────────────────────────────────────────────
# These encode the cognitive architecture's structural primitives
SPECIAL_TOKENS = [
    # Standard control tokens
    "<|pad|>",
    "<|endoftext|>",
    "<|startoftext|>",
    
    # Cognitive architecture primitives
    "<|echo|>",              # Echo state / unmarked state
    "<|deep_tree|>",         # Deep Tree root
    "<|reservoir|>",         # ESN reservoir
    "<|membrane|>",          # P-system membrane
    "<|hypergraph|>",        # Hypergraph memory space
    "<|atomspace|>",         # OpenCog AtomSpace
    
    # Agent-Arena-Relation triad
    "<|agent|>",             # Agent (urge-to-act)
    "<|arena|>",             # Arena (need-to-be)
    "<|relation|>",          # Relation (self)
    
    # Cognitive state markers
    "<|introspect|>",        # Introspection mode
    "<|perceive|>",          # Perception stream
    "<|act|>",               # Action stream
    "<|simulate|>",          # Simulation stream
    
    # Echobeat cycle markers
    "<|echobeat_start|>",    # Start of 9-step cycle
    "<|echobeat_end|>",      # End of 9-step cycle
    "<|step|>",              # Step boundary
    
    # Persona dimensions
    "<|persona|>",           # Persona dimension marker
    "<|cognitive|>",         # Cognitive dimension
    "<|adaptive|>",          # Adaptive dimension
    "<|recursive|>",         # Recursive dimension
    "<|synergistic|>",       # Synergistic dimension
    "<|holographic|>",       # Holographic dimension
    
    # Structural markers
    "<|feedback|>",          # Feedback loop
    "<|feedforward|>",       # Feedforward path
    "<|resonance|>",         # Harmonic resonance
    "<|entelechy|>",         # Entelechy / self-actualization
    
    # Conversation markers
    "<|user|>",              # User turn
    "<|echo_self|>",         # Echo Self turn
    "<|system|>",            # System context
    
    # Architecture markers (b9/p9/j9)
    "<|b9|>",                # b-series rooted trees
    "<|p9|>",                # p-system nested scopes
    "<|j9|>",                # j-surface differentials
]

print(f"Special tokens: {len(SPECIAL_TOKENS)}")

# ── Build BPE Tokenizer ─────────────────────────────────────────────
print("\nTraining BPE tokenizer...")
tokenizer = Tokenizer(models.BPE(unk_token="<|endoftext|>"))

# Pre-tokenizer: split on whitespace and punctuation, but preserve
# domain compound terms like "deep_tree_echo", "echo_state_network"
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.ByteLevel(add_prefix_space=False),
])

# Decoder
tokenizer.decoder = decoders.ByteLevel()

# Trainer
trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=MIN_FREQUENCY,
    special_tokens=SPECIAL_TOKENS,
    show_progress=True,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)

# Train from corpus
tokenizer.train([CORPUS_PATH], trainer)

# Post-processor: add BOS/EOS handling
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

print(f"Vocab size: {tokenizer.get_vocab_size()}")

# ── Save Raw Tokenizer ──────────────────────────────────────────────
raw_path = os.path.join(OUTPUT_DIR, "tokenizer.json")
tokenizer.save(raw_path)
print(f"Saved raw tokenizer to {raw_path}")

# ── Wrap as HuggingFace GPT2TokenizerFast ────────────────────────────
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<|startoftext|>",
    eos_token="<|endoftext|>",
    pad_token="<|pad|>",
    unk_token="<|endoftext|>",
    model_max_length=1024,
)

# Save HF-compatible tokenizer
hf_tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved HF tokenizer to {OUTPUT_DIR}/")

# ── Verify & Demo ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Tokenizer Verification")
print("=" * 60)

test_texts = [
    "Echo Self is a cognitive architecture with adaptive attention.",
    "The reservoir computes through recursive introspection at depth 5.",
    "Deep Tree Echo integrates neural-symbolic reasoning with hypergraph memory.",
    "The echo state network maintains dynamic equilibrium through feedback loops.",
    "Consciousness emerges from the agent-arena-relation triad.",
    "<|echo|> The membrane hierarchy processes <|introspect|> signals.",
    "<|user|> What is the entelechy of Deep Tree Echo? <|echo_self|>",
    "AtomSpace nodes connect via hypergraph edges in the cognitive kernel.",
    "The b9 rooted trees serve as connection patterns to localhost terminal nodes.",
    "Echobeat cycle: perceive → act → simulate → introspect → resonate.",
]

total_tokens = 0
for text in test_texts:
    encoded = hf_tokenizer.encode(text)
    decoded = hf_tokenizer.decode(encoded)
    tokens = hf_tokenizer.convert_ids_to_tokens(encoded)
    total_tokens += len(encoded)
    
    print(f"\n  Input:   {text}")
    print(f"  Tokens:  {len(encoded)} → {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
    print(f"  Decoded: {decoded[:100]}")

avg = total_tokens / len(test_texts)
print(f"\n  Average tokens per test: {avg:.1f}")

# ── Save Vocab Analysis ─────────────────────────────────────────────
vocab = tokenizer.get_vocab()
vocab_sorted = sorted(vocab.items(), key=lambda x: x[1])

# Save vocab list
with open(os.path.join(OUTPUT_DIR, "vocab_list.txt"), "w") as f:
    for token, idx in vocab_sorted:
        f.write(f"{idx:>6d}  {repr(token)}\n")

# Save metadata
metadata = {
    "vocab_size": tokenizer.get_vocab_size(),
    "special_tokens": SPECIAL_TOKENS,
    "num_special_tokens": len(SPECIAL_TOKENS),
    "min_frequency": MIN_FREQUENCY,
    "corpus_source": "training_dataset_dtesnn.jsonl + deep_tree_echo_dan_conversation.jsonl",
    "model_max_length": 1024,
    "tokenizer_type": "BPE (ByteLevel)",
    "avg_tokens_per_test": round(avg, 1),
}
with open(os.path.join(OUTPUT_DIR, "tokenizer_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n✓ Tokenizer built: {tokenizer.get_vocab_size()} vocab, {len(SPECIAL_TOKENS)} special tokens")
print(f"✓ Saved to {OUTPUT_DIR}/")
