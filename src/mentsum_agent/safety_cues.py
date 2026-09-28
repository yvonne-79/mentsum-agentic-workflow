from __future__ import annotations

import json
import re
from pathlib import Path

from .schemas import RiskCandidateSentence, RiskCueEntry, RiskCueHit
from .text_utils import find_sentence_for_span, split_sentences_with_offsets


_HYPHEN_CHARS = "-‐‑‒–—−"
_HORIZONTAL_SPACE = r"[^\S\r\n]"
_SEPARATOR_RE = re.compile(
    rf"(?:{_HORIZONTAL_SPACE}+|[{re.escape(_HYPHEN_CHARS)}])+"
)
_FLEXIBLE_HYPHEN_SEPARATOR = (
    rf"(?:{_HORIZONTAL_SPACE}*[{re.escape(_HYPHEN_CHARS)}]"
    rf"{_HORIZONTAL_SPACE}*|{_HORIZONTAL_SPACE}+)"
)


def _compile_surface_form(surface: str) -> re.Pattern[str]:
    canonical = surface.strip()
    if not canonical:
        raise ValueError("Risk cue surface forms must be non-empty.")

    fragments: list[str] = []
    cursor = 0
    for separator in _SEPARATOR_RE.finditer(canonical):
        fragments.append(re.escape(canonical[cursor : separator.start()]))
        separator_text = separator.group(0)
        if any(char in _HYPHEN_CHARS for char in separator_text):
            fragments.append(_FLEXIBLE_HYPHEN_SEPARATOR)
        else:
            fragments.append(rf"{_HORIZONTAL_SPACE}+")
        cursor = separator.end()
    fragments.append(re.escape(canonical[cursor:]))

    return re.compile(
        rf"(?<!\w){''.join(fragments)}(?!\w)",
        re.IGNORECASE,
    )


class RiskCueScanner:
    """Generate sentence-level risk candidates without assigning issue labels."""

    def __init__(self, cue_path: Path | None):
        self.entries: list[RiskCueEntry] = []
        self.patterns: list[tuple[str, re.Pattern[str]]] = []
        if cue_path is None:
            return
        if not cue_path.exists():
            raise FileNotFoundError(f"Risk cue file not found: {cue_path}")
        raw = json.loads(cue_path.read_text(encoding="utf-8"))
        self.entries = [RiskCueEntry.model_validate(item) for item in raw]
        for entry in self.entries:
            self.patterns.extend(
                (entry.category, _compile_surface_form(surface))
                for surface in entry.surface_forms
            )

    def scan(self, text: str) -> list[RiskCueHit]:
        hits: list[RiskCueHit] = []
        seen: set[tuple[int, int, str]] = set()
        for category, pattern in self.patterns:
            for match in pattern.finditer(text):
                key = (match.start(), match.end(), category)
                if key in seen:
                    continue
                seen.add(key)
                hits.append(
                    RiskCueHit(
                        category=category,
                        span_text=text[match.start() : match.end()],
                        char_start=match.start(),
                        char_end=match.end(),
                    )
                )
        return sorted(hits, key=lambda item: (item.char_start, item.char_end))

    def candidate_sentences(self, text: str) -> list[RiskCandidateSentence]:
        sentences = split_sentences_with_offsets(text)
        cues_by_sentence: dict[int, list[RiskCueHit]] = {}
        sentence_by_id = {sentence.sentence_id: sentence for sentence in sentences}

        for cue in self.scan(text):
            sentence = find_sentence_for_span(
                sentences,
                cue.char_start,
                cue.char_end,
            )
            if sentence is None:
                continue
            cues_by_sentence.setdefault(sentence.sentence_id, []).append(cue)

        return [
            RiskCandidateSentence(
                sentence_id=sentence_id,
                sentence_text=sentence_by_id[sentence_id].text,
                sentence_start=sentence_by_id[sentence_id].start,
                sentence_end=sentence_by_id[sentence_id].end,
                risk_cues=cues,
            )
            for sentence_id, cues in sorted(cues_by_sentence.items())
        ]
