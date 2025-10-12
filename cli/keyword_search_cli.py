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

    idf_parser = subparsers.add_parser("idf", help="Calculate the IDF score of a term")
    idf_parser.add_argument("term", type=str, help="Term")

    tf_parser = subparsers.add_parser(
        "tf", help="Get term frequencies for a term in a document"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Get the TF-IDF for a term in a document"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term")

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

        case "tf":
            try:
                ii = InvertedIndex.from_cache()
            except FileNotFoundError:
                print("index doesn't exist, run build command first")
                exit(1)

            print(ii.get_tf(args.doc_id, args.term))

        case "idf":
            try:
                ii = InvertedIndex.from_cache()
            except FileNotFoundError:
                print("index doesn't exist, run build command first")
                exit(1)

            idf = ii.get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            try:
                ii = InvertedIndex.from_cache()
            except FileNotFoundError:
                print("index doesn't exist, run build command first")
                exit(1)

            tf = ii.get_tf(args.doc_id, args.term)
            idf = ii.get_idf(args.term)
            tf_idf = tf * idf
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
