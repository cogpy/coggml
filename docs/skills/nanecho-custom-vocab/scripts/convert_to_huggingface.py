#!/usr/bin/env python3
"""
Convert DTE NanEcho Checkpoint to HuggingFace Format
=====================================================
Converts a NanEcho checkpoint trained with the custom DTE tokenizer
to a HuggingFace-compatible model that bundles the DTE tokenizer.

Usage:
  python NanEcho/convert_dte_to_huggingface.py \
    --checkpoint ckpt.pt \
    --tokenizer-dir NanEcho/dte_tokenizer \
    --output-dir hf-model-dte
"""

import argparse
import json
import os
import sys
import shutil

import torch
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast


def convert_state_dict(nanoGPT_state: dict) -> dict:
    """Convert nanoGPT state dict keys to HuggingFace GPT-2 format.
    
    nanoGPT uses:  transformer.wte.weight, transformer.h.{i}.attn.c_attn.weight, ...
    HF GPT-2 uses: transformer.wte.weight, transformer.h.{i}.attn.c_attn.weight, ...
    
    The naming is largely compatible, but we need to handle the lm_head weight tying.
    """
    hf_state = {}
    for key, value in nanoGPT_state.items():
        # Remove '_orig_mod.' prefix if present (from torch.compile)
        clean_key = key.replace('_orig_mod.', '')
        hf_state[clean_key] = value
    return hf_state


def main():
    parser = argparse.ArgumentParser(description="Convert DTE NanEcho checkpoint to HF format")
    parser.add_argument("--checkpoint", required=True, help="Path to NanEcho checkpoint (.pt)")
    parser.add_argument("--tokenizer-dir", required=True, help="Path to DTE tokenizer directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for HF model")
    parser.add_argument("--repo-id", default=None, help="HuggingFace repo ID for model card")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    
    model_args = ckpt.get("model_args", {})
    config_data = ckpt.get("config", {})
    iter_num = ckpt.get("iter_num", 0)
    val_loss = ckpt.get("best_val_loss", float("inf"))
    
    print(f"  Iteration: {iter_num}, Val Loss: {val_loss:.4f}")
    print(f"  Model args: {model_args}")
    
    # Load DTE tokenizer to get vocab_size
    print(f"Loading DTE tokenizer: {args.tokenizer_dir}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    vocab_size = tokenizer.vocab_size
    print(f"  Vocab size: {vocab_size}")
    
    # Create HF config
    hf_config = GPT2Config(
        vocab_size=vocab_size,
        n_embd=model_args.get("n_embd", 256),
        n_head=model_args.get("n_head", 4),
        n_layer=model_args.get("n_layer", 4),
        n_positions=model_args.get("block_size", 1024),
        resid_pdrop=model_args.get("dropout", 0.1),
        embd_pdrop=model_args.get("dropout", 0.1),
        attn_pdrop=model_args.get("dropout", 0.1),
        bos_token_id=tokenizer.convert_tokens_to_ids("<|startoftext|>"),
        eos_token_id=tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        pad_token_id=tokenizer.convert_tokens_to_ids("<|pad|>"),
    )
    
    # Create HF model and load weights
    print("Creating HF GPT-2 model...")
    model = GPT2LMHeadModel(hf_config)
    
    # Convert and load state dict
    nanoGPT_state = ckpt["model"]
    hf_state = convert_state_dict(nanoGPT_state)
    
    # Handle potential size mismatches (vocab_size padding)
    model_state = model.state_dict()
    for key in list(hf_state.keys()):
        if key in model_state:
            if hf_state[key].shape != model_state[key].shape:
                print(f"  Shape mismatch for {key}: ckpt={hf_state[key].shape} vs model={model_state[key].shape}")
                # Resize embedding/lm_head if vocab changed
                if 'wte' in key or 'lm_head' in key:
                    old_tensor = hf_state[key]
                    new_tensor = model_state[key].clone()
                    min_size = min(old_tensor.shape[0], new_tensor.shape[0])
                    new_tensor[:min_size] = old_tensor[:min_size]
                    hf_state[key] = new_tensor
                    print(f"    Resized {key} from {old_tensor.shape} to {new_tensor.shape}")
    
    model.load_state_dict(hf_state, strict=False)
    
    # Save model and tokenizer
    print(f"Saving to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save conversion metadata
    meta = {
        "source_checkpoint": os.path.basename(args.checkpoint),
        "source_iter": iter_num,
        "source_val_loss": val_loss,
        "tokenizer_type": "dte_bpe",
        "vocab_size": vocab_size,
        "model_args": model_args,
        "conversion_info": "NanEcho DTE checkpoint -> HuggingFace GPT-2",
    }
    with open(os.path.join(args.output_dir, "dte_conversion_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    # Create model card
    repo_id = args.repo_id or "drzo/echoself-dte"
    model_card = f"""---
license: agpl-3.0
tags:
  - echo-self
  - deep-tree-echo
  - cognitive-architecture
  - nanecho
  - dte-tokenizer
  - custom-vocab
language:
  - en
pipeline_tag: text-generation
---

# NanEcho DTE — Deep Tree Echo with Custom Vocabulary

NanEcho model trained with a **custom DTE BPE tokenizer** (vocab_size={vocab_size})
built from Deep Tree Echo cognitive architecture training data.

## Model Details

| Parameter | Value |
|:---|:---|
| Architecture | GPT-2 (causal LM) |
| Vocab Size | {vocab_size} |
| Layers | {model_args.get('n_layer', 4)} |
| Heads | {model_args.get('n_head', 4)} |
| Embedding | {model_args.get('n_embd', 256)} |
| Context | {model_args.get('block_size', 1024)} |
| Tokenizer | Custom DTE BPE |
| Training Iter | {iter_num} |
| Val Loss | {val_loss:.4f} |

## Custom Tokenizer

The DTE tokenizer includes 35 special tokens encoding cognitive architecture primitives:
- **AAR Triad**: `<|agent|>`, `<|arena|>`, `<|relation|>`
- **Cognitive State**: `<|echo|>`, `<|introspect|>`, `<|perceive|>`, `<|act|>`, `<|simulate|>`
- **Architecture**: `<|b9|>`, `<|p9|>`, `<|j9|>`, `<|membrane|>`, `<|hypergraph|>`
- **Echobeat**: `<|echobeat_start|>`, `<|echobeat_end|>`, `<|step|>`

## Usage

```python
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

model = GPT2LMHeadModel.from_pretrained("{repo_id}")
tokenizer = PreTrainedTokenizerFast.from_pretrained("{repo_id}")

inputs = tokenizer("Echo Self is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Source

- Repository: [9cog/echoself](https://github.com/9cog/echoself)
- Tokenizer: Custom DTE BPE trained on DTE corpus data
"""
    with open(os.path.join(args.output_dir, "README.md"), "w") as f:
        f.write(model_card)
    
    print(f"\nDone. Model saved to {args.output_dir}/")
    print(f"  Config: {args.output_dir}/config.json")
    print(f"  Weights: {args.output_dir}/model.safetensors")
    print(f"  Tokenizer: {args.output_dir}/tokenizer.json")
    print(f"  Model card: {args.output_dir}/README.md")


if __name__ == "__main__":
    main()
