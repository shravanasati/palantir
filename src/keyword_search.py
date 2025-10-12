from src.load_dataset import get_stopwords, load_data

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
    query_tokens = [stemmer.stem(q) for q in word_tokenize(query) if q and q not in STOPWORDS]

    return query_tokens


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
