import argparse
import time
import resource
from pathlib import Path
import json
import base64

from cs336_basics.pretokenization import train_bpe

DEFAULT_NUM_CHUNKS = 4
DEFAULT_SPECIAL_TOKEN = "<|endoftext|>"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")

    parser.add_argument("input_path", type=Path, help="Path to corpus file")
    parser.add_argument("--vocab-size", type=int, required=True, help="Target vocabulary size")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save vocab & merges")

    parser.add_argument(
        "--special-token",
        action="append",
        default=None,
        help="Special token, can pass multiple times. Default: <|endoftext|>"
    )

    parser.add_argument("--num-chunks", type=int, default=DEFAULT_NUM_CHUNKS, help="Chunk count, default 4")
    parser.add_argument("--profile", action="store_true", help="Enable memory/time profiling")

    args = parser.parse_args()

    if args.special_token is None:
        args.special_tokens = [DEFAULT_SPECIAL_TOKEN]
    else:
        args.special_tokens = args.special_token

    return args


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_path.exists():
        raise SystemExit(f"Error: input file {args.input_path} does not exist.")
    if not args.input_path.is_file():
        raise SystemExit(f"Error: {args.input_path} is not a regular file.")

    if args.vocab_size <= 0:
        raise SystemExit(f"Error: vocab-size must be positive, got {args.vocab_size}")
    if args.num_chunks <= 0:
        raise SystemExit(f"Error: num-chunks must be positive, got {args.num_chunks}")

    if not isinstance(args.special_tokens, list) or len(args.special_tokens) == 0:
        raise SystemExit("Error: special tokens list cannot be empty.")

    min_required_vocab = 256 + len(args.special_tokens)
    if args.vocab_size <= min_required_vocab:
        raise SystemExit(
            f"Error: vocab-size must > {min_required_vocab} "
            f"(256 base bytes + {len(args.special_tokens)} special tokens)."
        )

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise SystemExit(f"Error: cannot create output dir {args.output_dir}, {e}")


def main():
    args = parse_args()
    validate_args(args)

    vocab, merges = train_bpe(args.input_path, args.vocab_size, args.special_tokens)


    vocab_save = {}
    for token_id, token_bytes in vocab.items():
        vocab_save[token_id] = base64.b64encode(token_bytes).decode("utf-8")
    vocab_path = args.output_dir / "vocab.json"
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(vocab_save, f, ensure_ascii=False, indent=2)

    merges_path = args.output_dir / "merges.txt"
    with merges_path.open("w", encoding="utf-8") as f:
        for a, b in merges:
            line_a = base64.b64encode(a).decode("utf-8")
            line_b = base64.b64encode(b).decode("utf-8")
            f.write(f"{line_a} {line_b}\n")


if __name__ == "__main__":
    main()
