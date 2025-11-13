from pathlib import Path
import re
import numpy as np
from sentence_transformers import SentenceTransformer

from src.load_dataset import Movie, load_data


def fixed_size_chunk_text(text: str, chunk_size: int = 200, overlap: int = 0):
    words = text.split()
    chunks: list[str] = []
    i = 0
    while i < len(words):
        skip = 0
        current_chunk = []
        current_chunk.extend(words[max(i - overlap, 0) : i])
        skip = len(current_chunk)
        rem = chunk_size - skip
        current_chunk.extend(words[i : i + rem])

        chunks.append(" ".join(current_chunk))
        i += rem

    return chunks


def semantic_chunk_text(text: str, max_chunk_size: int = 4, overlap: int = 0):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    i = 0
    while i < len(sentences):
        skip = 0
        current_chunk = []
        current_chunk.extend(sentences[max(i - overlap, 0) : i])
        skip = len(current_chunk)
        rem = max_chunk_size - skip
        current_chunk.extend(sentences[i : i + rem])

        chunks.append(" ".join(current_chunk))
        i += rem

    return chunks


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class SemanticSearchResult(Movie):
    score: float


class SemanticSearch:
    def __init__(self):
        # Load the model (downloads automatically the first time)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        # model.encode(text)

        self.embeddings = np.array([])
        self.documents = []
        self.document_map: dict[int, Movie] = {}

    def generate_embedding(self, text: str):
        text = text.strip()
        if not text:
            raise ValueError("empty text")

        return self.model.encode([text])[0]

    @staticmethod
    def get_cache_paths():
        cache_dir = Path(__file__).parent.parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        embeddings_file = cache_dir / "movie_embeddings.npy"
        return embeddings_file

    def build_embeddings(self, documents: list[Movie]):
        self.documents = documents
        self.document_map = {d["id"]: d for d in self.documents}
        movies = [f"{d['title']}: {d['description']}" for d in documents]
        self.embeddings = self.model.encode(movies, show_progress_bar=True)

        embeddings_file_path = self.get_cache_paths()
        np.save(embeddings_file_path, self.embeddings)

        return self.embeddings

    @classmethod
    def from_cache(cls, documents: list):
        ss = cls()
        embeddings_file_path = ss.get_cache_paths()
        if embeddings_file_path.exists():
            ss.embeddings = np.load(embeddings_file_path)
            if len(ss.embeddings) != len(documents):
                raise ValueError("length mismatch")
            ss.documents = documents
            ss.document_map = {d["id"]: d for d in ss.documents}

        else:
            raise FileNotFoundError(
                "cache/index.pkl or cache/docmap.pkl or cache/term_frequencies.pkl or cache/doc_lengths.pkl files don't exist"
            )

        return ss

    @classmethod
    def load_or_create_embeddings(cls, docs: list):
        try:
            ss = SemanticSearch.from_cache(docs)
        except FileNotFoundError:
            ss = SemanticSearch()
            ss.build_embeddings(docs)

        return ss

    def search(self, query: str, limit: int) -> list[SemanticSearchResult]:
        if len(self.embeddings) == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embed = self.generate_embedding(query)
        # document, similarity_score
        results: list[tuple[int, float]] = []
        for i, e in enumerate(self.embeddings):
            results.append((i + 1, cosine_similarity(query_embed, e)))

        results.sort(key=lambda x: x[1], reverse=True)
        return [
            SemanticSearchResult(self.document_map[i] | {"score": s})  # type: ignore
            for i, s in results[:limit]
        ]


def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def verify_embeddings():
    ss = SemanticSearch.load_or_create_embeddings(load_data())
    print(f"Number of docs: {len(ss.documents)}")
    print(
        f"Embeddings shape: {ss.embeddings.shape[0]} vectors in {ss.embeddings.shape[1]} dimensions"
    )


def embed_text(text: str):
    embedding = SemanticSearch().generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 4 dimensions: {embedding[:4]}")
    print(f"Dimensions: {embedding.shape[0]}")
