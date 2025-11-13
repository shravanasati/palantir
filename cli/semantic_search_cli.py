#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.load_dataset import load_data
from src.semantic_search import (
    ChunkedSemanticSearch,
    SemanticSearch,
    fixed_size_chunk_text,
    embed_text,
    semantic_chunk_text,
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
    subparsers.add_parser(
        "embed_chunks", help="Generate chunked embeddings for all documents."
    )

    embed_parser = subparsers.add_parser("embed", help="Generate a text embedding")
    embed_parser.add_argument("text", type=str, help="The text to embed.")

    semantic_chunk_parser = subparsers.add_parser("chunk", help="Chunk the given text.")
    semantic_chunk_parser.add_argument("text", type=str, help="The text to chunk.")
    semantic_chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, nargs="?", help="Chunk size"
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=-1,
        nargs="?",
        help="Overlap (default 20% of chunk size)",
    )

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Semantically chunk the given text.")
    semantic_chunk_parser.add_argument("text", type=str, help="The text to chunk.")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size", type=int, default=4, nargs="?", help="Chunk size"
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        nargs="?",
        help="Overlap",
    )

    search_parser = subparsers.add_parser(
        "search", help="Search movies using semantic search"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "limit", type=int, default=5, nargs="?", help="Number of search items"
    )

    chunked_search_parser = subparsers.add_parser(
        "search_chunked", help="Search movies using chunked semantic search"
    )
    chunked_search_parser.add_argument("query", type=str, help="Search query")
    chunked_search_parser.add_argument(
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

        case "chunk":
            overlap = args.overlap
            if overlap == -1:
                overlap = round(0.2 * args.chunk_size)
            else:
                overlap = max(0, overlap)
            chunks = fixed_size_chunk_text(args.text, args.chunk_size, overlap)
            for i, chunk in enumerate(chunks):
                print(f"{i + 1}. {chunk}")

        case "semantic_chunk":
            overlap = args.overlap
            chunks = semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
            for i, chunk in enumerate(chunks):
                print(f"{i + 1}. {chunk}")
        
        case "embed_chunks":
            css = ChunkedSemanticSearch.load_or_create_embeddings(load_data())
            print(f"Generated {len(css.chunk_embeddings)} chunked embeddings")

        case "search_chunked":
            ss = ChunkedSemanticSearch.load_or_create_embeddings(load_data())
            results = ss.search_chunks(args.query, args.limit)
            for i, movie in enumerate(results):
                print(
                    f"{i + 1}. ({movie['id']}) {movie['title']} - Score: {movie['score']:.4f}"
                )
                print(movie["description"], "\n")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
