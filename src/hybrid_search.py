from typing import Collection

from .load_dataset import Movie, ScoredMovie
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


def min_max_normalize(l: Collection[int | float]):
    if len(l) <= 1:
        return l

    min_, max_ = min(l), max(l)
    return [(i - min_) / (max_ - min_) for i in l]


def rrf_score(rank: int, k: int = 60):
    return 1 / (k + rank)


class HybridSearch:
    def __init__(self, documents: list[Movie]):
        self.documents = documents
        self.document_map = {d["id"]: d for d in self.documents}
        self.semantic_search = ChunkedSemanticSearch.load_or_create_embeddings(
            documents
        )

        try:
            self.idx = InvertedIndex.from_cache()
        except FileNotFoundError:
            self.idx = InvertedIndex()
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int):
        return self.idx.bm25_search(query, limit)

    def _semantic_search(self, query: str, limit: int):
        return self.semantic_search.search_chunks(query, limit)

    @staticmethod
    def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5):
        return alpha * bm25_score + (1 - alpha) * semantic_score

    def rrf_search(self, query: str, k: int, limit: int = 5) -> list[ScoredMovie]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self._semantic_search(query, limit * 500)

        # map document IDs to their scores
        bm25_result_map: dict[int, float] = {}
        for i, r in enumerate(bm25_results):
            bm25_result_map[r["id"]] = rrf_score(i + 1)

        semantic_result_map: dict[int, float] = {}
        for i, r in enumerate(semantic_results):
            semantic_result_map[r["id"]] = rrf_score(i + 1)

        unique_doc_ids = set(bm25_result_map.keys())
        unique_doc_ids = unique_doc_ids.union(semantic_result_map.keys())

        doc_hybrid_scores: dict[int, float] = {}
        for doc_id in unique_doc_ids:
            doc_hybrid_scores[doc_id] = bm25_result_map.get(
                doc_id, 0
            ) + semantic_result_map.get(doc_id, 0)

        doc_hybrid_scores_sorted = sorted(
            doc_hybrid_scores.items(), key=lambda x: x[1], reverse=True
        )[:limit]

        return [
            ScoredMovie(self.document_map[d] | {"score": s})
            for d, s in doc_hybrid_scores_sorted
        ]

    def weighted_search(self, query: str, alpha: float, limit: int = 10):
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self._semantic_search(query, limit * 500)

        bm25_scores_norm = min_max_normalize([r["score"] for r in bm25_results])
        # map document IDs to their scores
        bm25_result_map: dict[int, float] = {}
        for r, norm_score in zip(bm25_results, bm25_scores_norm):
            bm25_result_map[r["id"]] = norm_score

        semantic_scores_norm = min_max_normalize([r["score"] for r in semantic_results])
        semantic_result_map: dict[int, float] = {}
        for r, norm_score in zip(semantic_results, semantic_scores_norm):
            semantic_result_map[r["id"]] = norm_score

        unique_doc_ids = set(bm25_result_map.keys())
        unique_doc_ids = unique_doc_ids.union(semantic_result_map.keys())

        doc_hybrid_scores: dict[int, float] = {}
        for doc_id in unique_doc_ids:
            doc_hybrid_scores[doc_id] = self.hybrid_score(
                bm25_result_map.get(doc_id, 0),
                semantic_result_map.get(doc_id, 0),
                alpha,
            )

        doc_hybrid_scores_sorted = sorted(
            doc_hybrid_scores.items(), key=lambda x: x[1], reverse=True
        )[:limit]

        return [
            ScoredMovie(self.document_map[d] | {"score": s})
            for d, s in doc_hybrid_scores_sorted
        ]
