#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.semantic_search import embed_text, verify_embeddings, verify_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify if the model is loaded.")
    subparsers.add_parser("verify_embeddings", help="Verify if the embeddings are created.")

    embed_parser = subparsers.add_parser(
        "embed", help="Generate a text embedding"
    )
    embed_parser.add_argument("text", type=str, help="The text to embed.")

    args = parser.parse_args()
    match args.command:
        case "verify":
            verify_model()

        case "embed":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
