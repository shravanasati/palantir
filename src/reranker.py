import json
from google import genai
import os

from sentence_transformers import CrossEncoder

from src.load_dataset import ScoredMovie


RANKING_SYSTEM_PROMPT = """
Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first).
"""


def movie_str_no_desc(r: ScoredMovie):
    return str({"id": r["id"], "title": r["title"]})


def movie_str(r: ScoredMovie):
    return str({"id": r["id"], "title": r["title"], "description": r["description"]})


class Reranker:
    def __init__(
        self,
        llm: str = "gemini-2.0-flash-001",
        cross_encoder: str = "cross-encoder/ms-marco-TinyBert-L2-v2",
    ) -> None:
        # llm reranker
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = llm

        # cross encoder reranker
        self.cross_encoder = CrossEncoder(cross_encoder)

    def rerank_llm(self, query: str, docs: list[ScoredMovie]) -> list[ScoredMovie]:
        prompt = RANKING_SYSTEM_PROMPT.format(
            query=query, doc_list_str="\n".join([movie_str_no_desc(d) for d in docs])
        )
        print(f"{prompt=}")
        result = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": {"type": "array", "items": {"type": "integer"}},
            },
        ).text
        ranks = json.loads(result)
        order_index = {id_: index for index, id_ in enumerate(ranks)}
        results = sorted(docs, key=lambda x: order_index.get(x["id"], float("inf")))

        return results

    def rerank_cross_encoder(
        self, query: str, docs: list[ScoredMovie]
    ) -> list[ScoredMovie]:
        docmap = {d["id"]: d for d in docs}
        pairs = [(query, movie_str(d)) for d in docs]
        scores = self.cross_encoder.predict(pairs)
        doc_score_map = {d["id"]: s for d, s in zip(docs, scores)}
        doc_score_sorted = sorted(
            doc_score_map.items(), reverse=True, key=lambda x: x[1]
        )
        return [docmap[i] for i, _ in doc_score_sorted]
