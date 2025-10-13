from collections import Counter, defaultdict
import math
from pathlib import Path
import pickle
from src.load_dataset import get_stopwords, load_data, Movie

from nltk import word_tokenize
from nltk.stem import PorterStemmer

STOPWORDS = set(get_stopwords())


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

    @staticmethod
    def get_cache_paths():
        """
        Returns the filepaths to the cached index file, docmap and
        term frequency files.
        """
        cache_dir = Path(__file__).parent.parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        index_file = cache_dir / "index.pkl"
        docmap_file = cache_dir / "docmap.pkl"
        tf_file = cache_dir / "term_frequencies.pkl"
        return index_file, docmap_file, tf_file

    def _add_document(self, doc: Movie):
        self.docmap[doc["id"]] = doc
        tokens = preprocess_text(f"{doc['title']} {doc['description']}")
        for tok in tokens:
            self.index[tok].append(doc["id"])
        self.term_frequencies[doc["id"]] = Counter(tokens)

    def build(self):
        for doc in self.documents:
            self._add_document(doc)

    def get_documents(self, query: str):
        query_tokens = preprocess_text(query)
        doc_ids = sorted([i for qtok in query_tokens for i in self.index.get(qtok, [])])
        return [self.docmap[i] for i in doc_ids]

    def get_tf(self, doc_id: int, term: str):
        query_tokens = preprocess_text(term)
        if len(query_tokens) > 1 or len(query_tokens) == 0:
            raise ValueError(f"{term=} has invalid length for get_tf")
        if not self.term_frequencies.get(doc_id):
            return 0
        tf_counter = self.term_frequencies[doc_id]
        return tf_counter.get(query_tokens[0], 0)

    def get_idf(self, term: str):
        N = len(self.docmap)
        # this is the number of documents term appears in, not total number
        # of times the term appears in all documents
        term_doc_count = sum((bool(self.get_tf(i, term)) for i in self.docmap))
        return math.log((N + 1) / (term_doc_count + 1))

    def get_tfidf(self, doc_id: int, term: str):
        return self.get_tf(doc_id, term) * self.get_idf(term)

    def save(self):
        index_file, docmap_file, tf_file = self.get_cache_paths()
        with open(index_file, "wb") as f:
            pickle.dump(self.index, f)
        with open(docmap_file, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(tf_file, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    @classmethod
    def from_cache(cls):
        ii = cls()
        index_file, docmap_file, tf_file = ii.get_cache_paths()
        if index_file.exists() and docmap_file.exists() and tf_file.exists():
            with open(index_file, "rb") as iff:
                ii.index = pickle.load(iff)
            with open(docmap_file, "rb") as df:
                ii.docmap = pickle.load(df)
            with open(tf_file, "rb") as tf:
                ii.term_frequencies = pickle.load(tf)

        else:
            raise FileNotFoundError(
                "cache/index.pkl or cache/docmap.pkl or cache/term_frequencies.pkl files don't exist"
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
