#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path is set
from src.keyword_search import BM25_B, BM25_K1, InvertedIndex  # , search_keyword  # noqa: E402


def main() -> None:

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using TFIDF")
    search_parser.add_argument("query", type=str, help="Search query")

    idf_parser = subparsers.add_parser("idf", help="Calculate the IDF score of a term")
    idf_parser.add_argument("term", type=str, help="Term")

    tf_parser = subparsers.add_parser(
        "tf", help="Get term frequencies for a term in a document"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf",
        help="Get term frequencies for a term in a document using BM25 saturation",
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Get the TF-IDF for a term in a document"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    bm25_search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using BM25"
    )
    bm25_search_parser.add_argument("query", type=str, help="Search query")
    bm25_search_parser.add_argument(
        "limit", type=int, default=5, nargs="?", help="Number of search items"
    )

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
                # if i == 4:
                #     break

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

            bm25_idf = ii.get_tfidf(args.doc_id, args.term)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {bm25_idf:.2f}"
            )

        case "bm25idf":
            try:
                ii = InvertedIndex.from_cache()
            except FileNotFoundError:
                print("index doesn't exist, run build command first")
                exit(1)

            bm25_idf = ii.get_bm25_idf(args.term)
            print(f"BM25-IDF score of '{args.term}': {bm25_idf:.2f}")

        case "bm25tf":
            try:
                ii = InvertedIndex.from_cache()
            except FileNotFoundError:
                print("index doesn't exist, run build command first")
                exit(1)

            bm25tf = ii.get_bm25_tf(args.doc_id, args.term, args.k1, args.b)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
            )

        case "bm25search":
            try:
                ii = InvertedIndex.from_cache()
            except FileNotFoundError:
                print("index doesn't exist, run build command first")
                exit(1)

            results = ii.bm25_search(args.query, args.limit)
            for i, movie in enumerate(results):
                print(
                    f"{i + 1}. ({movie['id']}) {movie['title']} - Score: {movie['score']:.2f}"
                )

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
