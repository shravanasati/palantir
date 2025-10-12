from collections import defaultdict
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

    @staticmethod
    def get_cache_paths():
        """
        Returns the filepaths to the cached index file and docmap files.
        """
        cache_dir = Path(__file__).parent.parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        index_file = cache_dir / "index.pkl"
        docmap_file = cache_dir / "docmap.pkl"
        return index_file, docmap_file

    def _add_document(self, doc: Movie):
        self.docmap[doc["id"]] = doc
        tokens = preprocess_text(f"{doc['title']} {doc['description']}")
        for tok in tokens:
            self.index[tok].append(doc["id"])

    def build(self):
        for doc in self.documents:
            self._add_document(doc)

    def get_documents(self, query: str):
        query_tokens = preprocess_text(query)
        doc_ids = sorted([i for qtok in query_tokens for i in self.index.get(qtok, [])])
        return [self.docmap[i] for i in doc_ids]

    def save(self):
        index_file, docmap_file = self.get_cache_paths()
        with open(index_file, "wb") as f:
            pickle.dump(self.index, f)
        with open(docmap_file, "wb") as f:
            pickle.dump(self.docmap, f)

    @classmethod
    def from_cache(cls):
        ii = cls()
        index_file, docmap_file = ii.get_cache_paths()
        if index_file.exists() and docmap_file.exists():
            with open(index_file, "rb") as iff:
                ii.index = pickle.load(iff)
            with open(docmap_file, "rb") as df:
                ii.docmap = pickle.load(df)

        else:
            raise FileNotFoundError(
                "cache/index.pkl or cache/docmap.pkl files don't exist"
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
