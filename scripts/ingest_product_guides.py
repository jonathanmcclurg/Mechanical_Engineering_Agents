#!/usr/bin/env python3
"""Ingest product guide documents into the RAG system."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from src.tools.rag_tool import RAGTool


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest product guide documents.")
    parser.add_argument(
        "--dir",
        default=None,
        help="Directory containing product guides (defaults to settings.product_guide_dir)",
    )
    parser.add_argument(
        "--store",
        default=None,
        help="RAG storage dir for persisted chunks (defaults to settings.rag_store_dir)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Clear the existing store before ingesting",
    )
    args = parser.parse_args()

    settings = get_settings()
    guides_dir = Path(args.dir or settings.product_guide_dir).resolve()
    store_dir = Path(args.store or settings.rag_store_dir).resolve()

    if not guides_dir.exists():
        print(f"Product guide directory not found: {guides_dir}")
        return 1

    if args.rebuild and store_dir.exists():
        for child in store_dir.glob("*"):
            if child.is_file():
                child.unlink()

    rag_tool = RAGTool(storage_dir=store_dir)
    ingested = rag_tool.ingest_directory(guides_dir, rebuild=args.rebuild)

    if not ingested:
        print(f"No product guides found in {guides_dir}")
        return 1

    print(f"\nTotal documents ingested: {len(ingested)}")
    for doc in ingested:
        print(f"  - {doc.title} ({doc.chunk_count} chunks)")

    print(f"\nRAG store location: {store_dir}")
    return 0 if len(ingested) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
