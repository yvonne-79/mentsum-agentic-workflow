from __future__ import annotations

import json
import re
from pathlib import Path

from ..schemas import CandidateOrigin, ConceptCandidate, LexiconEntry
from ..text_utils import SentenceSpan, find_sentence_for_span
from .base import TermMatcher


class LexiconMatcher(TermMatcher):
    def __init__(self, lexicon_path: Path):
        if not lexicon_path.exists():
            raise FileNotFoundError(f"Lexicon file not found: {lexicon_path}")
        raw = json.loads(lexicon_path.read_text(encoding="utf-8"))
        self.entries = [LexiconEntry.model_validate(item) for item in raw]

    def match(
        self,
        source: str,
        sentences: list[SentenceSpan],
    ) -> list[ConceptCandidate]:
        candidates: list[ConceptCandidate] = []
        seen: set[tuple[int, int, str]] = set()

        for entry in self.entries:
            for surface in entry.surface_forms:
                pattern = re.compile(rf"(?<!\w){re.escape(surface)}(?!\w)", re.IGNORECASE)
                for match in pattern.finditer(source):
                    key = (match.start(), match.end(), entry.preferred_term.lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    sentence = find_sentence_for_span(sentences, match.start(), match.end())
                    if sentence is None:
                        continue
                    candidates.append(
                        ConceptCandidate(
                            candidate_id=f"lex-{len(candidates):04d}",
                            sentence_id=sentence.sentence_id,
                            sentence_text=sentence.text,
                            span_text=source[match.start() : match.end()],
                            char_start=match.start(),
                            char_end=match.end(),
                            preferred_term=entry.preferred_term,
                            cui=entry.cui,
                            semantic_types=entry.semantic_types,
                            match_score=1.0,
                            origin=CandidateOrigin.LEXICON,
                        )
                    )

        candidates.sort(key=lambda item: (item.char_start, -item.match_score))
        for index, candidate in enumerate(candidates):
            candidate.candidate_id = f"lex-{index:04d}"
        return candidates
