import argparse
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from src.reranker import Reranker
from src.hybrid_search import HybridSearch
from src.load_dataset import ScoredMovie, load_data
from src.query_enhancer import QueryEnhancer, EnhancementMethod




def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    weighted_search_parser = subparsers.add_parser(
        "weighted_search", help="Search movies using weighted search"
    )
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument(
        "--limit", type=int, default=5, nargs="?", help="Number of search items"
    )
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        nargs="?",
        help="Weightage for keyword search",
    )

    rrf_search_parser = subparsers.add_parser(
        "rrf_search", help="Search movies using RRF search"
    )
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument(
        "--limit", type=int, default=5, nargs="?", help="Number of search items"
    )
    rrf_search_parser.add_argument(
        "--k", type=int, default=60, nargs="?", help="K parameter for RRF"
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=[m.value for m in EnhancementMethod],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to rerank documents.",
    )

    args = parser.parse_args()
    match args.command:
        case "weighted_search":
            hs = HybridSearch(load_data())
            results = hs.weighted_search(args.query, args.alpha, args.limit)
            for i, movie in enumerate(results):
                print(
                    f"{i + 1}. ({movie['id']}) {movie['title']} - Score: {movie['score']:.4f}"
                )
                print(movie["description"][:100], "\n")

        case "rrf_search":
            query = args.query
            if args.enhance:
                method = EnhancementMethod(args.enhance)
                qe = QueryEnhancer()
                query = qe.enhance(method, query)
                print(f"Enhanced query ({method.value}): '{args.query}' -> '{query}'\n")

            limit = args.limit
            if args.rerank:
                limit *= 5

            hs = HybridSearch(load_data())
            results = hs.rrf_search(query, args.k, limit)
            if args.rerank:
                reranker = Reranker()
                # results = reranker.rerank_llm(
                #     query, results
                # )
                results = reranker.rerank_cross_encoder(query, results)
                results = results[:int(limit / 5)]

            for i, movie in enumerate(results):
                print(
                    f"{i + 1}. ({movie['id']}) {movie['title']} - Score: {movie['score']:.4f}"
                )
                print(movie["description"][:100], "\n")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
