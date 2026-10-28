# CS336: Language Modeling from Scratch

English | [简体中文](README.zh-CN.md)

This repository contains my coursework and implementation notes for
[Stanford CS336](https://cs336.stanford.edu/).

## Repository Layout

```text
cs336/
├── assignment1-basics/     # Tokenizer, Transformer LM, optimizer, and training utilities
├── assignment2-systems/    # Placeholder for future coursework
├── assignment3-scaling/    # Placeholder for future coursework
├── assignment4-data/       # Placeholder for future coursework
└── assignment5-alignment/  # Placeholder for future coursework
```

## Assignment 1: Basics

`assignment1-basics/` contains the Assignment 1 implementation, including:

- Byte-level BPE training and pre-tokenization
- A tokenizer with special-token handling and streaming encoding
- Linear, embedding, RMSNorm, SwiGLU, RoPE, and attention modules
- A pre-norm Transformer block and Transformer language model
- Cross-entropy loss, AdamW, learning-rate scheduling, and gradient clipping
- Batch sampling and model/optimizer checkpointing

The implementation is exercised by the course-provided unit tests.

### Setup and Tests

The project uses [`uv`](https://docs.astral.sh/uv/):

```bash
cd assignment1-basics
uv run pytest
```

### BPE Training CLI

`assignment1-basics/scripts/train_bpe_cli.py` exposes the BPE training code as a
small command-line program.

Example using the bundled debug corpus:

```bash
cd assignment1-basics
uv run python scripts/train_bpe_cli.py \
  tests/fixtures/tinystories_sample_5M.txt \
  --vocab-size 500 \
  --special-token '<|endoftext|>' \
  --output-dir runs/debug-ts
```

The command writes:

- `vocab.json`: token IDs mapped to Base64-encoded byte strings
- `merges.txt`: ordered BPE merges, with each byte string represented in Base64

## Notes

Implementation notes for BPE training and pair-index optimization are available
in [`assignment1-basics/notes/bpe.md`](assignment1-basics/notes/bpe.md).

## Data and Generated Artifacts

Datasets, virtual environments, checkpoints, tokenized arrays, logs, and
experiment outputs are intentionally excluded from version control because of
their size.

## Acknowledgements

The assignment specifications, starter code, and test fixtures are provided by
the Stanford CS336 course staff.
