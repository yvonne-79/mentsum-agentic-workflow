from __future__ import annotations

from ..llm import LLMClient
from ..prompts import SUMMARIZER_SYSTEM_PROMPT, summarizer_user_prompt
from ..schemas import GuidanceItem
from .base import Summarizer


class LLMSummarizer(Summarizer):
    def __init__(self, llm: LLMClient, max_summary_words: int):
        self.llm = llm
        self.max_summary_words = max_summary_words

    def summarize(self, source: str, guidance: list[GuidanceItem]) -> str:
        return self.llm.complete_text(
            SUMMARIZER_SYSTEM_PROMPT,
            summarizer_user_prompt(
                source=source,
                guidance=guidance,
                max_summary_words=self.max_summary_words,
            ),
        )
