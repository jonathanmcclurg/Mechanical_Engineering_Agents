"""Download and cache a local sentence-transformers embedding model."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a sentence-transformers model to a local directory.",
    )
    parser.add_argument(
        "--model-id",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face model id to download once.",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/models/all-MiniLM-L6-v2",
        help="Directory where the local model is saved.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if output directory already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        print(f"Local embedding model already exists at: {output_dir}")
        print("Use --force to re-download.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import SentenceTransformer

    print(f"Downloading model: {args.model_id}")
    model = SentenceTransformer(args.model_id)
    model.save(str(output_dir))
    print(f"Saved local embedding model to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
