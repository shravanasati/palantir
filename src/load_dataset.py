import json
from pathlib import Path
from typing import TypedDict

_DATA_PATH = Path(__file__).parent.parent / "data"
MOVIES_DATA_PATH = _DATA_PATH / "movies.json"
STOPWORDS_PATH = _DATA_PATH / "stopwords.txt"


class Movie(TypedDict):
    id: int
    title: str
    description: str


class ScoredMovie(Movie):
    score: float


def load_data() -> list[Movie]:
    with open(str(MOVIES_DATA_PATH.absolute())) as f:
        return json.load(f)["movies"]


def get_stopwords() -> list[str]:
    with open(str(STOPWORDS_PATH)) as f:
        return f.read().splitlines()
