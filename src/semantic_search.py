import json
from pathlib import Path
import re
from typing import TypedDict
import numpy as np
from sentence_transformers import SentenceTransformer

from src.load_dataset import Movie, ScoredMovie, load_data


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
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1:
        return sentences

    chunks: list[str] = []
    i = 0
    while i < len(sentences):
        skip = 0
        current_chunk = []
        current_chunk.extend(sentences[max(i - overlap, 0) : i])
        skip = len(current_chunk)
        rem = max_chunk_size - skip
        current_chunk.extend(sentences[i : i + rem])

        chunk_sentence = " ".join(current_chunk).strip()
        if chunk_sentence:
            chunks.append(chunk_sentence)
        i += rem

    return chunks


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class ChunkMetadata(TypedDict):
    chunk_idx: int
    movie_idx: int
    total_chunks: int


class ChunkMetadataScored(ChunkMetadata):
    score: float


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Load the model (downloads automatically the first time)
        self.model = SentenceTransformer(model_name)
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
            raise FileNotFoundError("cache/movie_embeddings.npy files don't exist")

        return ss

    @classmethod
    def load_or_create_embeddings(cls, docs: list):
        try:
            ss = cls.from_cache(docs)
        except FileNotFoundError:
            ss = cls()
            ss.build_embeddings(docs)

        return ss

    def search(self, query: str, limit: int) -> list[ScoredMovie]:
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
            ScoredMovie(self.document_map[i] | {"score": s})  # type: ignore
            for i, s in results[:limit]
        ]


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = np.array([])
        self.chunk_metadata: list[ChunkMetadata] = []

    @staticmethod
    def get_cache_paths():
        cache_dir = Path(__file__).parent.parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        embeddings_file = cache_dir / "chunk_embeddings.npy"
        metadata_file = cache_dir / "chunk_metadata.json"
        return embeddings_file, metadata_file

    def build_embeddings(self, documents: list[Movie]):
        self.documents = documents
        self.document_map = {d["id"]: d for d in self.documents}
        chunks = []
        chunk_metadatas: list[ChunkMetadata] = []
        for i, doc in enumerate(self.documents):
            if not doc["description"]:
                continue

            desc_chunks = semantic_chunk_text(doc["description"], 4, 1)
            chunks.extend(desc_chunks)
            for ci, c in enumerate(desc_chunks):
                chunk_metadatas.append(
                    {"movie_idx": doc["id"], "chunk_idx": ci, "total_chunks": len(desc_chunks)}
                )

        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadatas

        embeddings_file_path, metadata_file_path = self.get_cache_paths()
        np.save(embeddings_file_path, self.chunk_embeddings)
        with metadata_file_path.open("w") as f:
            json.dump(self.chunk_metadata, f)

        return self.embeddings

    @classmethod
    def from_cache(cls, documents: list):
        ss = cls()
        embeddings_file_path, metadata_file_path = ss.get_cache_paths()
        if embeddings_file_path.exists() and metadata_file_path.exists():
            ss.chunk_embeddings = np.load(embeddings_file_path)
            ss.documents = documents
            ss.document_map = {d["id"]: d for d in ss.documents}

            with metadata_file_path.open() as f:
                ss.chunk_metadata = json.load(f)

        else:
            raise FileNotFoundError(
                "cache/chunk_embeddings.npy or cache/chunk_metadata.json files don't exist"
            )

        return ss

    def search_chunks(self, query: str, limit: int = 10):
        query_embed = self.generate_embedding(query)
        chunk_scores: list[ChunkMetadataScored] = []
        for i, e in enumerate(self.chunk_embeddings):
            score = cosine_similarity(e, query_embed)
            metadata = self.chunk_metadata[i]
            chunk_scores.append(
                {
                    "movie_idx": metadata["movie_idx"],
                    "chunk_idx": metadata["chunk_idx"],
                    "total_chunks": metadata["total_chunks"],
                    "score": float(score),
                }
            )

        # stores aggregated scores per movie
        doc_scores: dict[int, float] = {}
        for cs in chunk_scores:
            curr_score = doc_scores.get(cs["movie_idx"], 0)
            curr_score = max(cs["score"], curr_score)
            doc_scores[cs["movie_idx"]] = curr_score

        results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:limit]

        return [ScoredMovie(self.document_map[i] | {"score": s}) for i, s in results]


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
