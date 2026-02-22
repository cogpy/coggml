---
name: dte-reservoir-llm
description: A meta-skill to create and train Reservoir-Augmented Transformers (RATs) for the Deep Tree Echo cognitive architecture. This skill orchestrates nanecho-custom-vocab, reservoirpy-nodes, echo-train, and echo-deploy to build stateful LLMs with custom architectures. Use for building LLMs with integrated reservoir computing, creating custom transformer layers, or training stateful language models.
---

# DTE Reservoir-LLM: Building Stateful Transformers with Reservoir Computing

This meta-skill provides a complete workflow for creating, training, and deploying a novel type of language model: the **Reservoir-Augmented Transformer (RAT)**. The RAT integrates a stateful **Echo State Network (ESN)** directly into the transformer architecture, replacing the standard feed-forward layers with a dynamic, recurrent reservoir system. This allows the LLM to maintain a rich internal state, enabling more sophisticated temporal reasoning and cognitive modeling.

This skill unifies the capabilities of several other skills into a single, coherent pipeline:

-   `nanecho-custom-vocab`: For building the domain-specific tokenizer.
-   `reservoirpy-nodes`: For defining the custom reservoir nodes.
-   `echo-train`: For managing the CI/CD training pipeline.
-   `echo-deploy`: For deploying the final model to the HuggingFace Hub.
-   `echo-introspect`, `unreal-echo`, `meta-echo-dna`: For integrating the trained model into the full Deep Tree Echo cognitive architecture.

## Core Architecture: The Reservoir-Augmented Transformer (RAT)

The key innovation of the RAT is the replacement of the standard feed-forward network (FFN) in each transformer block with a `CognitiveReadout` module. This module contains a fixed `EchoReservoir` and a trainable linear readout layer.

For a detailed architectural breakdown, see `references/architecture.md`.

## The End-to-End Workflow

### Phase 1: Data and Vocabulary Preparation

1.  **Create a Custom Vocabulary**: Start by using the `nanecho-custom-vocab` skill to build a BPE tokenizer from your domain-specific training data. This is a critical first step to ensure efficient tokenization.

2.  **Prepare Tokenized Data**: Once the tokenizer is built, use the `prepare_data.py` script from the `nanecho-custom-vocab` skill to convert your raw text into the `train.bin` and `val.bin` files required by the `nanoGPT` training pipeline.

    For more details, see `references/data_preparation.md`.

### Phase 2: Model and Workflow Modification

1.  **Clone the `echoself` Repository**: All modifications will be made to a local clone of the `9cog/echoself` repository.

    ```bash
    gh repo clone 9cog/echoself ~/echoself
    ```

2.  **Patch the Model File**: Run the `modify_model.py` script to inject the `RATBlock` and `CognitiveReadout` class definitions into the `NanEcho/nanecho_model.py` file.

    ```bash
    python3 /home/ubuntu/skills/dte-reservoir-llm/scripts/modify_model.py
    ```

3.  **Patch the Workflow File**: Run the `modify_workflow.py` script to add a `model_type` input to the `netrain-cached.yml` workflow, allowing you to select between `NanEcho` and `RAT` models during training.

    ```bash
    python3 /home/ubuntu/skills/dte-reservoir-llm/scripts/modify_workflow.py
    ```

4.  **Commit and Push Changes**: Commit the patched files to a new branch and push them to the `9cog/echoself` repository.

### Phase 3: Training the RAT

1.  **Trigger the Training Workflow**: Use the `echo-train` skill to dispatch the `netrain-cached.yml` workflow. Crucially, set the `model_type` input to `RAT`.

    ```bash
    gh workflow run netrain-cached.yml --repo 9cog/echoself \
      -f training_type=full \
      -f model_type=RAT \
      -f data_dir=data/nanecho_dte # Use your custom data directory
    ```

2.  **Monitor Training**: Use the `echo-train` skill's monitoring commands to track the progress of the training run.

### Phase 4: Deployment

1.  **Deploy the Trained Model**: Once training is complete, use the `echo-deploy` skill to deploy the new RAT model to the HuggingFace Hub. The deployment workflow will automatically use the correct conversion script for the RAT architecture.

### Phase 5: Cognitive Integration

With the trained RAT model deployed, you can now integrate it into the full Deep Tree Echo cognitive architecture:

-   **Introspection (`echo-introspect`)**: Analyze the internal state of the reservoir during inference to gain insights into the model's "cognitive state."
-   **Avatar Embodiment (`unreal-echo` & `meta-echo-dna`)**: Use the model's output to drive the expression and behavior of a MetaHuman avatar.
-   **Harmonic Analysis (`harmonic-llm`)**: Apply frequency-domain analysis to the reservoir's dynamics to explore new forms of cognitive modeling.

## Bundled Resources

-   **`references/`**: Contains detailed documentation on the RAT architecture, data preparation, and model implementation.
-   **`scripts/`**: Contains the Python scripts for patching the `echoself` repository.
