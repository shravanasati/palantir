import json
from google import genai
import os


RANKING_SYSTEM_PROMPT = """
Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first).
"""


class Reranker:
    def __init__(self, model: str = "gemini-2.0-flash-001") -> None:
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model

    def rerank_llm(self, query: str, docs: list[str]):
        prompt = RANKING_SYSTEM_PROMPT.format(query=query, doc_list_str=docs)
        print(f"{prompt=}")
        result = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": {"type": "array", "items": {"type": "integer"}},
            },
        ).text
        rank_list = json.loads(result)
        return rank_list
