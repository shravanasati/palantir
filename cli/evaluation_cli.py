import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Set

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hybrid_search import HybridSearch
from src.keyword_search import InvertedIndex
from src.load_dataset import Movie, ScoredMovie, load_data
from src.query_enhancer import EnhancementMethod, QueryEnhancer
from src.reranker import Reranker
from src.semantic_search import ChunkedSemanticSearch


# todo llm evaluator

def load_golden_dataset():
    with open(project_root / "data" / "golden_dataset.json") as f:
        return json.load(f)


def precision_at_k(retrieved_docs: Set[int], relevant_docs: Set[int], k: int) -> float:
    retrieved_k = list(retrieved_docs)[:k]
    true_positives = len(set(retrieved_k) & relevant_docs)
    return true_positives / k if k > 0 else 0.0


def recall_at_k(retrieved_docs: Set[int], relevant_docs: Set[int], k: int) -> float:
    retrieved_k = list(retrieved_docs)[:k]
    true_positives = len(set(retrieved_k) & relevant_docs)
    return true_positives / len(relevant_docs) if relevant_docs else 0.0


def f1_score_at_k(precision: float, recall: float) -> float:
    return (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

def evaluate(
    search_fn: Callable[[str, int], List[Movie]],
    golden_dataset: dict[str, Any],
    k: int,
):
    total_precision = 0
    total_recall = 0
    total_f1_score = 0

    test_cases = golden_dataset["test_cases"]

    for item in test_cases:
        query = item["query"]
        relevant_docs = set(item["relevant_docs"])

        results = search_fn(query, k)
        retrieved_docs = {result["title"] for result in results}

        precision = precision_at_k(retrieved_docs, relevant_docs, k)
        recall = recall_at_k(retrieved_docs, relevant_docs, k)
        f1_score = f1_score_at_k(precision, recall)

        total_precision += precision
        total_recall += recall
        total_f1_score += f1_score

    num_queries = len(test_cases)
    avg_precision = total_precision / num_queries
    avg_recall = total_recall / num_queries
    avg_f1_score = total_f1_score / num_queries

    return avg_precision, avg_recall, avg_f1_score


def main():
    parser = argparse.ArgumentParser(description="Evaluation CLI")
    parser.add_argument(
        "-k", type=int, default=5, help="Value of k for precision@k and recall@k"
    )
    args = parser.parse_args()

    golden_dataset = load_golden_dataset()
    documents = load_data()

    # Initialize searchers
    try:
        keyword_searcher = InvertedIndex.from_cache()
    except FileNotFoundError:
        keyword_searcher = InvertedIndex()
        keyword_searcher.build()
        keyword_searcher.save()
    semantic_searcher = ChunkedSemanticSearch.load_or_create_embeddings(documents)
    hybrid_searcher = HybridSearch(documents)
    query_enhancer = QueryEnhancer()
    reranker = Reranker()

    def search_with_query_enhancement(query: str, limit: int) -> List[ScoredMovie]:
        enhanced_query = query_enhancer.enhance(EnhancementMethod.REWRITE, query)
        return hybrid_searcher.rrf_search(enhanced_query, 60, limit)

    def search_with_reranking(query: str, limit: int) -> List[ScoredMovie]:
        # Fetch more results initially for the reranker to work with
        initial_results = hybrid_searcher.rrf_search(query, 60, limit * 5)
        reranked_results = reranker.rerank_cross_encoder(query, initial_results)
        return reranked_results[:limit]

    def search_with_enhancement_and_reranking(query: str, limit: int) -> List[ScoredMovie]:
        enhanced_query = query_enhancer.enhance(EnhancementMethod.REWRITE, query)
        initial_results = hybrid_searcher.rrf_search(enhanced_query, 60, limit * 5)
        reranked_results = reranker.rerank_cross_encoder(
            enhanced_query, initial_results
        )
        return reranked_results[:limit]

    search_functions = {
        "Keyword Search": lambda q, l: keyword_searcher.bm25_search(q, l),
        "Semantic Search": lambda q, l: semantic_searcher.search_chunks(q, l),
        "Hybrid Search (Weighted)": lambda q, l: hybrid_searcher.weighted_search(
            q, 0.6, l
        ),
        "Hybrid Search (RRF)": lambda q, l: hybrid_searcher.rrf_search(q, 60, l),
        "Hybrid Search (RRF) + Query Enhancement": search_with_query_enhancement,
        "Hybrid Search (RRF) + Reranking": search_with_reranking,
        "Hybrid Search (RRF) + Enhancement + Reranking": search_with_enhancement_and_reranking,
    }

    print(f"Evaluating at k={args.k}")
    print("-" * 50)

    for name, search_fn in search_functions.items():
        print(f"Search Type: {name}")
        precision, recall, f1_score = evaluate(search_fn, golden_dataset, args.k)
        print(f"  Precision@{args.k}: {precision:.4f}")
        print(f"  Recall@{args.k}:    {recall:.4f}")
        print(f"  F1-score@{args.k}:  {f1_score:.4f}")
        print("-" * 50)


if __name__ == "__main__":
    main()
