---
name: nanecho-custom-vocab
description: "Build custom BPE vocabularies and tokenizers for the NanEcho model (drzo/echoself) from domain-specific JSONL training data. Covers data analysis, tokenizer training, nanoGPT binary preparation, 9cog/echoself repo integration, and triggering the netrain-cached GitHub Actions training pipeline. Use when adapting NanEcho to a new domain with specialized terminology."
---

# NanEcho Custom Vocabulary Training

## Triggers

Use this skill when the user wants to:
- Train the NanEcho model with a custom vocabulary.
- Build a custom tokenizer for NanEcho from their own data.
- Fine-tune NanEcho on a new domain with specialized terminology.
- Create a more efficient version of NanEcho for a specific task.

**Example prompts**:
- "I want to train NanEcho on my own dataset with a custom vocabulary."
- "How can I build a custom tokenizer for the echoself model?"
- "Fine-tune nanecho with a new vocabulary based on these documents."

## Workflow

The end-to-end process is broken down into the following phases:

### Phase 1: Data Analysis & Preparation

1.  **Receive Training Data**: The user must provide one or more training data files in JSONL format. These files should contain the text corpus for the new domain.
2.  **Analyze Data**: Before proceeding, analyze the provided JSONL files to understand their structure, content, and size. Use a script to count documents, analyze text length distribution, and identify the keys containing the text content.

### Phase 2: Custom Tokenizer Training

1.  **Train BPE Tokenizer**: Use the `scripts/build_tokenizer.py` script to train a custom Byte-Pair Encoding (BPE) tokenizer from the training corpus.
    - This script takes the raw text from the JSONL files and trains a `tokenizers` model.
    - It defines special tokens crucial for the Deep Tree Echo cognitive architecture (e.g., `<|echo|>`, `<|agent|>`, `<|b9|>`).
    - The output is a directory (e.g., `dte_tokenizer/`) containing `tokenizer.json` and other configuration files.

    ```bash
    python3 scripts/build_tokenizer.py \
      --data-files /path/to/your/data.jsonl \
      --output-dir dte_tokenizer \
      --vocab-size 8192
    ```

### Phase 3: Data Tokenization for nanoGPT

1.  **Prepare Binary Data**: Use the `scripts/prepare_data.py` script to convert the raw JSONL data into tokenized `train.bin` and `val.bin` files using the newly trained tokenizer.
    - This script reads the JSONL files, formats the text with special tokens (`<|startoftext|>`, `<|endoftext|>`), and uses the custom tokenizer to encode the text into integer token IDs.
    - The output is a pair of binary files containing `uint16` numpy arrays, which is the format expected by the `nanoGPT` training pipeline in the `9cog/echoself` repository.

    ```bash
    python3 scripts/prepare_data.py \
      --tokenizer-dir dte_tokenizer \
      --data-files /path/to/your/data.jsonl \
      --output-dir data/nanecho_dte
    ```

### Phase 4: Repository Integration

1.  **Clone Repository**: Clone the `9cog/echoself` repository.
2.  **Copy Artifacts**: Copy the generated tokenizer directory (`dte_tokenizer/`) and the prepared data directory (`data/nanecho_dte/`) into the cloned repository. Place them in `NanEcho/dte_tokenizer/` and `data/nanecho_dte/` respectively.
3.  **Add Integration Scripts**: Copy the integration scripts from this skill (`scripts/`) into the `NanEcho/` directory of the repository.
4.  **Commit and Push**: Commit all the new files (tokenizer, data, scripts) to a new feature branch and push it to the `9cog/echoself` repository.

### Phase 5: Trigger Training

1.  **Dispatch Workflow**: Trigger the `netrain-cached.yml` GitHub Actions workflow.
    - **Crucially**, you must override the default parameters to point to the new data directory and specify the new vocabulary size.
    - This is done by modifying the `train_cached.py` command in the workflow or by passing inputs if the workflow is configured to accept them.

    Example invocation (if using command-line overrides):
    ```bash
    python train_cached.py \
      --data_dir data/nanecho_dte \
      --vocab_size 8192 \
      --n_layer 4 --n_head 4 --n_embd 256 # Match model size to vocab size
    ```

### Phase 6: Model Conversion & Deployment

1.  **Download Checkpoint**: Once training is complete, download the best checkpoint (`ckpt.pt`) from the workflow artifacts.
2.  **Convert to HuggingFace**: Use the `scripts/convert_to_huggingface.py` script to convert the `nanoGPT`-style checkpoint into a standard HuggingFace model format. This script bundles the custom tokenizer with the model weights.
3.  **Upload to Hub**: Upload the converted model directory to the HuggingFace Hub.

## Resources

- **Scripts**: See the `scripts/` directory for the Python scripts required for this workflow.
- **References**: See the `references/` directory for documentation on the tokenizer configuration and special tokens.
- **Templates**: See the `templates/` directory for a template of the `dte_tokenizer_config.json` file.
