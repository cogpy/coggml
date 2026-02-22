# Data Preparation for Reservoir-Augmented Transformer

Preparing data for the RAT involves the same initial steps as the standard NanEcho model, but with additional considerations for the reservoir.

## 1. Custom Vocabulary

Follow the `nanecho-custom-vocab` skill to create a domain-specific BPE tokenizer. This is the first and most critical step.

## 2. Data Tokenization

Use the `prepare_data.py` script from the `nanecho-custom-vocab` skill to convert your raw text data into `train.bin` and `val.bin` files. These files contain the token IDs that will be fed into the model.

## 3. Reservoir-Specific Preprocessing (Optional)

For certain tasks, it may be beneficial to preprocess the input data in a way that is specifically tailored to the reservoir. For example, you could add a "warm-up" sequence of tokens to the beginning of each training sample to allow the reservoir to settle into a stable state before processing the actual data.

This can be done by modifying the `prepare_data.py` script to prepend a fixed sequence of tokens to each document.
