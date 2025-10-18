from collections import Counter, defaultdict
from functools import cache
import math
from pathlib import Path
import pickle
from src.load_dataset import get_stopwords, load_data, Movie

from nltk import word_tokenize
from nltk.stem import PorterStemmer

STOPWORDS = set(get_stopwords())
BM25_K1 = 1.5
BM25_B = 0.75


class BM25SearchResult(Movie):
    score: float


def preprocess_text(query: str) -> list[str]:
    """
    Performs basic text pre-processing on the query: case-insensitivity,
    punctuation, tokenization, stop words, stemming.
    """
    query = query.lower()

    stemmer = PorterStemmer()
    # tokenize, remove stopwords and stemming
    query_tokens = [
        stemmer.stem(q) for q in word_tokenize(query) if q and q not in STOPWORDS
    ]

    return query_tokens


class InvertedIndex:
    def __init__(self) -> None:
        self.documents = load_data()
        self.index: dict[str, list[int]] = defaultdict(list)
        self.docmap: dict[int, Movie] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.doc_lengths: dict[int, int] = {}

    @staticmethod
    def get_cache_paths():
        """
        Returns the filepaths to the cached index file, docmap,
        term frequency and doc lengths files.
        """
        cache_dir = Path(__file__).parent.parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        index_file = cache_dir / "index.pkl"
        docmap_file = cache_dir / "docmap.pkl"
        tf_file = cache_dir / "term_frequencies.pkl"
        doc_lengths_file = cache_dir / "doc_lengths.pkl"
        return index_file, docmap_file, tf_file, doc_lengths_file

    def _add_document(self, doc: Movie):
        self.docmap[doc["id"]] = doc
        tokens = preprocess_text(f"{doc['title']} {doc['description']}")
        for tok in tokens:
            self.index[tok].append(doc["id"])
        self.term_frequencies[doc["id"]] = Counter(tokens)
        self.doc_lengths[doc["id"]] = len(tokens)

    def build(self):
        for doc in self.documents:
            self._add_document(doc)

    def get_documents(self, query: str):
        query_tokens = preprocess_text(query)
        doc_ids = set(
            sorted(
                [(i, qtok) for qtok in query_tokens for i in self.index.get(qtok, [])],
                # key=lambda r: self.get_tfidf(*r),
                # reverse=True
            )
        )

        return [self.docmap[i] for i, _ in doc_ids]

    @cache
    def get_tf(self, doc_id: int, term: str):
        query_tokens = preprocess_text(term)
        if len(query_tokens) > 1 or len(query_tokens) == 0:
            raise ValueError(f"{term=} has invalid length for get_tf")
        if not self.term_frequencies.get(doc_id):
            return 0
        tf_counter = self.term_frequencies[doc_id]
        return tf_counter.get(query_tokens[0], 0)

    @cache
    def get_idf(self, term: str):
        N = len(self.docmap)
        # this is the number of documents term appears in, not total number
        # of times the term appears in all documents
        term_doc_count = sum((bool(self.get_tf(i, term)) for i in self.docmap))
        return math.log((N + 1) / (term_doc_count + 1))

    def get_tfidf(self, doc_id: int, term: str):
        return self.get_tf(doc_id, term) * self.get_idf(term)

    @cache
    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)
        # document frequency
        df = sum(bool(self.get_tf(i, term)) for i in self.docmap)
        return math.log(1 + (N - df + 0.5) / (df + 0.5))

    @cache
    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        tf = self.get_tf(doc_id, term)
        length_norm = (
            1 - b + b * (self.doc_lengths[doc_id] / self._get_avg_doc_length())
        )
        saturation = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return saturation

    def _get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0

        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def bm25(self, doc_id: int, term: str):
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int) -> list[BM25SearchResult]:
        query_tokens = preprocess_text(query)
        scores: dict[int, float] = {}
        for doc_id in self.docmap:
            scores[doc_id] = 0
            for tok in query_tokens:
                scores[doc_id] += self.bm25(doc_id, tok)

        sorted_doc_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            BM25SearchResult(self.docmap[i] | {"score": score})  # type: ignore
            for i, score in sorted_doc_scores[:limit]
        ]

    def save(self):
        index_file, docmap_file, tf_file, doc_lengths_file = self.get_cache_paths()
        with open(index_file, "wb") as f:
            pickle.dump(self.index, f)
        with open(docmap_file, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(tf_file, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(doc_lengths_file, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    @classmethod
    def from_cache(cls):
        ii = cls()
        index_file, docmap_file, tf_file, doc_lengths_file = ii.get_cache_paths()
        if (
            index_file.exists()
            and docmap_file.exists()
            and tf_file.exists()
            and doc_lengths_file.exists()
        ):
            with open(index_file, "rb") as iff:
                ii.index = pickle.load(iff)
            with open(docmap_file, "rb") as df:
                ii.docmap = pickle.load(df)
            with open(tf_file, "rb") as tf:
                ii.term_frequencies = pickle.load(tf)
            with open(doc_lengths_file, "rb") as dlf:
                ii.doc_lengths = pickle.load(dlf)

        else:
            raise FileNotFoundError(
                "cache/index.pkl or cache/docmap.pkl or cache/term_frequencies.pkl or cache/doc_lengths.pkl files don't exist"
            )

        return ii


def search_keyword(query: str):
    data = load_data()
    query_tokenized = preprocess_text(query)
    for movie in data:
        title_tokenized = preprocess_text(movie["title"])
        found_match = False
        for qt in query_tokenized:
            for tt in title_tokenized:
                if qt in tt:
                    yield movie["title"]
                    found_match = True
                    break
            if found_match:
                break
