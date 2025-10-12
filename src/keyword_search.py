from src.load_dataset import get_stopwords, load_data
import string


STOPWORDS = set(get_stopwords())


def preprocess_text(query: str) -> list[str]:
    """
    Performs basic text pre-processing on the query: case-insensitivity,
    punctuation, tokenization, stop words, stemming.
    """
    query = query.lower()

    # remove punctuation
    ptt = str.maketrans({k: "" for k in string.punctuation})
    query = query.translate(ptt)

    # tokenize and remove stopwords
    query_tokens = [q for q in query.split() if q and q not in STOPWORDS]

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
