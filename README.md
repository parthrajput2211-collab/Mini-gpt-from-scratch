# GPT-Style Transformer Language Model From Scratch

A compact decoder-only Transformer language model built and trained from scratch using **PyTorch**. The project implements the core ideas behind GPT-style language models manually, including causal self-attention, multi-head attention, positional embeddings, residual connections, feed-forward networks, and autoregressive text generation.

## Highlights

- Built the Transformer architecture from scratch instead of using a pre-trained language model
- Trained on the Tiny Shakespeare corpus using character-level tokenization
- Implemented a Bigram language model as a baseline for comparison
- Tracked training and validation loss throughout training
- Saved the trained checkpoint for direct inference without retraining

### Final Training Results

| Metric | Initial | Final |
|---|---:|---:|
| Training Loss | 4.3086 | 2.0549 |
| Validation Loss | 4.3154 | 2.0933 |

**Trainable parameters:** 824,897  
**Vocabulary size:** 65 characters  
**Context length:** 128 characters

## Architecture

The model is a decoder-only Transformer with:

- Character-level token embeddings
- Learnable positional embeddings
- Causal scaled dot-product self-attention
- 4 attention heads
- 4 Transformer decoder blocks
- Feed-forward networks with ReLU activation
- Residual connections
- Layer normalization
- Dropout regularization
- Linear language-modeling head

### Model Configuration

| Parameter | Value |
|---|---:|
| Embedding dimension | 128 |
| Attention heads | 4 |
| Transformer layers | 4 |
| Context length | 128 |
| Batch size | 64 |
| Dropout | 0.1 |
| Learning rate | 0.0003 |
| Training iterations | 1000 |
| Optimizer | AdamW |

## Project Structure

```text
gpt-transformer-from-scratch-pytorch/
├── MiniGPT_From_Scratch.ipynb   # Complete implementation, training and analysis
├── inference.py                  # Generate text from the saved model
├── input.txt                     # Tiny Shakespeare dataset
├── minigpt_model.pth             # Trained model checkpoint
├── minigpt_config.json           # Model and training configuration
├── results.txt                   # Final metrics and sample generation
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

## Dataset

The model is trained on the **Tiny Shakespeare** dataset, included as `input.txt`.

- Approximately 1.1 million characters
- 65 unique characters
- 90% training / 10% validation split
- Character-level tokenization

The notebook uses the local dataset when it is already available, so it does not require an internet connection to run.

## Setup

Python 3.12 was used for the original experiment.

```bash
pip install -r requirements.txt
```

## Run the Notebook

Open:

```text
MiniGPT_From_Scratch.ipynb
```

and run the cells in order.

The notebook walks through the project step by step: dataset exploration, tokenization, Bigram baseline, attention mechanics, full Transformer construction, training, evaluation, and text generation.

The recorded CPU training run took roughly 22 minutes for 1,000 iterations. Runtime will vary by hardware.

## Generate Text Without Retraining

The repository includes a saved checkpoint, so the model can be used immediately:

```bash
python inference.py --prompt "The " --tokens 200
```

The inference script:

1. Reads the character vocabulary from `input.txt`
2. Reconstructs the architecture from `minigpt_config.json`
3. Loads `minigpt_model.pth`
4. Generates text autoregressively from the supplied prompt

## Bigram Baseline

A Bigram language model is included as a simple baseline. It predicts the next character using only the current character, while the Transformer can use a much longer preceding context through masked self-attention.

This comparison helped me see clearly why contextual attention produces more structured output than a simple next-character lookup model.

## What I Learned

The most useful part of this project was implementing the Transformer components manually instead of treating the architecture as a black box. In particular, working through the Query/Key/Value projections, causal masking, tensor dimensions, multi-head attention, and residual connections made the internal flow of a GPT-style model much clearer.

I also learned how important validation loss is when training a language model. The model's validation loss decreased from **4.3154 to 2.0933**, which showed that it was learning useful patterns rather than only reducing training loss.

## Limitations

- Character-level tokenization is less efficient than modern subword tokenization
- The model is intentionally small compared with production LLMs
- Training was limited to 1,000 iterations
- Generated text can contain spelling and grammatical errors
- The recorded experiment was trained on CPU

## Possible Improvements

- BPE or other subword tokenization
- Longer GPU-based training
- Larger embedding size and deeper Transformer
- Temperature, top-k, and top-p sampling
- Training checkpoints and resume support
- Fine-tuning on a domain-specific dataset

## Project Context

I originally developed this project while completing my **Data Science Internship at Celebal Technologies**. This standalone repository is the cleaned and documented portfolio version of that work.

## Author

**Parth Rohilla**
