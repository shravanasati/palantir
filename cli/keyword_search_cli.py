#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path is set
from src.keyword_search import InvertedIndex  # , search_keyword


def main() -> None:

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the inverted index")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            try:
                ii = InvertedIndex.from_cache()
            except FileNotFoundError:
                print("index doesn't exist, run build command first")
                exit(1)
            # for i, result in enumerate(search_keyword(args.query)):
            for i, result in enumerate(ii.get_documents(args.query)):
                print(f"{i+1}. {result['title']}")
                if i == 4:
                    break

        case "build":
            ii = InvertedIndex()
            ii.build()
            ii.save()
            result = ii.get_documents("merida")
            if len(result) > 0:
                print("first document ID matching merida", result[0]["id"])
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
