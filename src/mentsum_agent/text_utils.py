from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SentenceSpan:
    sentence_id: int
    text: str
    start: int
    end: int


_SENTENCE_PATTERN = re.compile(r"[^.!?\n]+(?:[.!?]+|(?=\n)|$)", re.MULTILINE)


def split_sentences_with_offsets(text: str) -> list[SentenceSpan]:
    sentences: list[SentenceSpan] = []
    for match in _SENTENCE_PATTERN.finditer(text):
        raw = match.group(0)
        left_trimmed = len(raw) - len(raw.lstrip())
        right_trimmed = len(raw.rstrip())
        start = match.start() + left_trimmed
        end = match.start() + right_trimmed
        if start >= end:
            continue
        sentence_text = text[start:end]
        sentences.append(
            SentenceSpan(
                sentence_id=len(sentences),
                text=sentence_text,
                start=start,
                end=end,
            )
        )
    if not sentences and text.strip():
        start = len(text) - len(text.lstrip())
        end = len(text.rstrip())
        sentences.append(SentenceSpan(0, text[start:end], start, end))
    return sentences


def find_sentence_for_span(
    sentences: list[SentenceSpan], char_start: int, char_end: int
) -> SentenceSpan | None:
    for sentence in sentences:
        if sentence.start <= char_start and char_end <= sentence.end:
            return sentence
    return None


def exact_span_offsets(container: str, exact_span: str) -> tuple[int, int] | None:
    start = container.find(exact_span)
    if start < 0:
        return None
    return start, start + len(exact_span)


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+(?:['’-]\w+)?\b", text))


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
