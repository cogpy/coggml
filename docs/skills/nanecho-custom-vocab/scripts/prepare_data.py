#!/usr/bin/env python3
"""
DTE Custom Tokenizer Data Preparation
=======================================
Prepare training data for NanEcho using the custom DTE BPE tokenizer.

This script replaces the character-level tokenization fallback with a
domain-specific BPE tokenizer trained on Deep Tree Echo corpus data.

Usage:
  python NanEcho/prepare_dte_data.py \
    --tokenizer-dir NanEcho/dte_tokenizer \
    --data-files data/training_dataset_dtesnn.jsonl data/deep_tree_echo_dan_conversation.jsonl \
    --output-dir data/nanecho_dte \
    --val-split 0.1

The output train.bin / val.bin are uint16 numpy arrays compatible with
the nanoGPT data loading format used by train_cached.py and train_nanecho.py.
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from typing import List


def load_jsonl_texts(path: str) -> List[str]:
    """Extract all text content from a JSONL file, handling multiple formats.
    
    Supports:
      - Single-line JSONL with {"messages": [...]} format
      - Multi-line JSON conversation objects with {"mapping": {...}} format
    """
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    lines = content.split('\n')
    
    if lines[0].strip() == '{':
        # Multi-line JSON objects (conversation format)
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(content):
            content_slice = content[idx:].lstrip()
            if not content_slice:
                break
            try:
                obj, end = decoder.raw_decode(content_slice)
                idx += len(content) - len(content_slice) + end
                texts.extend(_extract_text(obj))
            except json.JSONDecodeError:
                idx += 1
    else:
        # Standard single-line JSONL
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                texts.extend(_extract_text(obj))
            except json.JSONDecodeError:
                continue
    return texts


def _extract_text(obj) -> List[str]:
    """Recursively extract text from a JSON object."""
    texts = []
    if isinstance(obj, str):
        texts.append(obj)
    elif isinstance(obj, dict):
        for key in ['content', 'text', 'title', 'value', 'message']:
            if key in obj and isinstance(obj[key], str):
                texts.append(obj[key])
        for key in ['messages', 'parts', 'children']:
            if key in obj:
                texts.extend(_extract_text(obj[key]))
        if 'mapping' in obj and isinstance(obj['mapping'], dict):
            for node_id, node in obj['mapping'].items():
                if isinstance(node, dict) and 'message' in node and node['message']:
                    msg = node['message']
                    if isinstance(msg, dict) and 'content' in msg:
                        ct = msg['content']
                        if isinstance(ct, dict) and 'parts' in ct:
                            for part in ct['parts']:
                                if isinstance(part, str):
                                    texts.append(part)
    elif isinstance(obj, list):
        for item in obj:
            texts.extend(_extract_text(item))
    return texts


def main():
    parser = argparse.ArgumentParser(description="Prepare DTE training data for NanEcho")
    parser.add_argument("--tokenizer-dir", required=True,
                        help="Path to DTE tokenizer directory")
    parser.add_argument("--data-files", nargs="+", required=True,
                        help="JSONL data files")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for train.bin/val.bin")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split ratio (default: 0.1)")
    parser.add_argument("--block-size", type=int, default=1024,
                        help="Context length / block size (default: 1024)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    try:
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
        print(f"Loaded DTE tokenizer: vocab_size={tokenizer.vocab_size}")
    except ImportError:
        print("ERROR: transformers package required. Install with: pip install transformers")
        return
    except Exception as e:
        print(f"ERROR loading tokenizer from {args.tokenizer_dir}: {e}")
        return
    
    # Load all texts
    all_texts = []
    for data_file in args.data_files:
        if not os.path.exists(data_file):
            print(f"WARNING: {data_file} not found, skipping")
            continue
        texts = load_jsonl_texts(data_file)
        print(f"  {data_file}: {len(texts)} segments")
        all_texts.extend(texts)
    
    if not all_texts:
        print("ERROR: No text data found")
        return
    
    print(f"Total text segments: {len(all_texts)}")
    
    # Format and tokenize
    all_token_ids = []
    bos_id = tokenizer.encode("<|startoftext|>", add_special_tokens=False)
    eos_id = tokenizer.encode("<|endoftext|>", add_special_tokens=False)
    
    for text in all_texts:
        text = text.strip()
        if not text:
            continue
        formatted = f"<|startoftext|> {text} <|endoftext|>"
        ids = tokenizer.encode(formatted)
        all_token_ids.extend(ids)
    
    total_tokens = len(all_token_ids)
    max_id = max(all_token_ids)
    print(f"Total tokens: {total_tokens:,}")
    print(f"Max token ID: {max_id} (uint16 max: 65535)")
    assert max_id < 65536, f"Token ID {max_id} exceeds uint16 range!"
    
    # Convert and split
    token_array = np.array(all_token_ids, dtype=np.uint16)
    n_val = int(total_tokens * args.val_split)
    n_train = total_tokens - n_val
    
    train_data = token_array[:n_train]
    val_data = token_array[n_train:]
    
    # Save binary files
    train_path = os.path.join(args.output_dir, "train.bin")
    val_path = os.path.join(args.output_dir, "val.bin")
    train_data.tofile(train_path)
    val_data.tofile(val_path)
    
    print(f"Train: {n_train:,} tokens ({os.path.getsize(train_path):,} bytes)")
    print(f"Val:   {n_val:,} tokens ({os.path.getsize(val_path):,} bytes)")
    
    # Save metadata
    metadata = {
        "tokenizer_type": "dte_bpe",
        "tokenizer_dir": args.tokenizer_dir,
        "vocab_size": tokenizer.vocab_size,
        "total_tokens": total_tokens,
        "train_tokens": n_train,
        "val_tokens": n_val,
        "val_split": args.val_split,
        "block_size": args.block_size,
        "num_documents": len([t for t in all_texts if t.strip()]),
        "source_files": [os.path.basename(f) for f in args.data_files],
        "max_token_id": int(max_id),
        "dtype": "uint16",
    }
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved metadata to {meta_path}")
    print(f"Done. Use --data_dir {args.output_dir} --vocab_size {tokenizer.vocab_size} with train_cached.py")


if __name__ == "__main__":
    main()
