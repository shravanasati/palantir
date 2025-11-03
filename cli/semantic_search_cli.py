#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.load_dataset import load_data
from src.semantic_search import (
    SemanticSearch,
    embed_text,
    verify_embeddings,
    verify_model,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify if the model is loaded.")
    subparsers.add_parser(
        "verify_embeddings", help="Verify if the embeddings are created."
    )

    embed_parser = subparsers.add_parser("embed", help="Generate a text embedding")
    embed_parser.add_argument("text", type=str, help="The text to embed.")

    search_parser = subparsers.add_parser(
        "search", help="Search movies using semantic search"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "limit", type=int, default=5, nargs="?", help="Number of search items"
    )

    args = parser.parse_args()
    match args.command:
        case "verify":
            verify_model()

        case "embed":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "search":
            ss = SemanticSearch.load_or_create_embeddings(load_data())
            results = ss.search(args.query, args.limit)
            for i, movie in enumerate(results):
                print(
                    f"{i + 1}. ({movie['id']}) {movie['title']} - Score: {movie['score']:.4f}"
                )
                print(movie["description"], "\n")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
