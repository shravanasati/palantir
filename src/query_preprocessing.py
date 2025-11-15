from enum import StrEnum, auto
import os
from pathlib import Path
from google import genai
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


class EnhancementMethod(StrEnum):
    SPELL = auto()
    REWRITE = auto()
    EXPAND = auto()


SPELL_SYSTEM_PROMPT = """
Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query without any extra quotation marks.
Corrected:
"""

REWRITE_SYSTEM_PROMPT = """Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> The Revenant Leonardo DiCaprio bear attack
- "movie about bear in london with marmalade" -> Paddington London marmalade
- "scary movie with bear from few years ago" -> bear horror movie 2015-2025

Return the modified query without any quotation marks.

Rewritten query:"""

EXPAND_SYSTEM_PROMPT = """Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> scary horror grizzly bear movie terrifying film
- "action movie with bear" -> action thriller bear chase fight adventure
- "comedy with bear" -> comedy funny bear humor lighthearted

Return the modified query without any quotation marks.

Query: "{query}"
"""


class QueryEnhancer:
    def __init__(self, model: str = "gemini-2.0-flash-001") -> None:
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model

    def enhance(self, method: EnhancementMethod, query: str):
        prompt = ""
        match method:
            case EnhancementMethod.SPELL:
                prompt = SPELL_SYSTEM_PROMPT.format(query=query)
            case EnhancementMethod.REWRITE:
                prompt = REWRITE_SYSTEM_PROMPT.format(query=query)
            case EnhancementMethod.EXPAND:
                prompt = EXPAND_SYSTEM_PROMPT.format(query=query)
            case _:
                raise ValueError(f"unknown enhancement {method=}")

        return self.client.models.generate_content(
            model=self.model, contents=prompt
        ).text.strip()
