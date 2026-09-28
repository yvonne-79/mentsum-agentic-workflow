from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from ..schemas import CandidateOrigin, ConceptCandidate
from ..text_utils import SentenceSpan, find_sentence_for_span
from .base import TermMatcher


_CUI_PATTERN = re.compile(r"C\d{7}")


def _load_allowed_cuis(path: Path) -> tuple[set[str], dict[str, str]]:
    """Load a plain-text CUI list or a CSV with CUI and preferred-term columns."""
    if not path.exists():
        raise FileNotFoundError(f"QuickUMLS CUI allow-list not found: {path}")

    cuis: set[str] = set()
    labels: dict[str, str] = {}

    if path.suffix.casefold() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            if "cui" not in fieldnames:
                raise ValueError(
                    "QuickUMLS CUI allow-list CSV must contain a 'cui' column."
                )
            for line_number, row in enumerate(reader, start=2):
                cui = str(row.get("cui") or "").strip().upper()
                if not _CUI_PATTERN.fullmatch(cui):
                    raise ValueError(
                        f"Invalid UMLS CUI in {path} at line {line_number}: {cui!r}"
                    )
                cuis.add(cui)
                label = str(row.get("preferred_term") or "").strip()
                if label:
                    previous = labels.get(cui)
                    if previous is not None and previous != label:
                        raise ValueError(
                            f"Conflicting preferred terms for {cui} in {path}."
                        )
                    labels[cui] = label
    else:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            cui = line.strip().upper()
            if not cui or cui.startswith("#"):
                continue
            if not _CUI_PATTERN.fullmatch(cui):
                raise ValueError(
                    f"Invalid UMLS CUI in {path} at line {line_number}: {cui!r}"
                )
            cuis.add(cui)

    if not cuis:
        raise ValueError(f"QuickUMLS CUI allow-list is empty: {path}")
    return cuis, labels


class QuickUMLSMatcher(TermMatcher):
    def __init__(
        self,
        quickumls_path: Path,
        threshold: float = 0.85,
        accepted_semtypes: set[str] | None = None,
        allowed_cuis_path: Path | None = None,
        cui_labels_path: Path | None = None,
    ):
        if not quickumls_path.is_dir():
            raise FileNotFoundError(
                f"QuickUMLS installation directory not found: {quickumls_path}"
            )
        required_paths = (
            quickumls_path / "umls-simstring.db",
            quickumls_path / "cui-semtypes.db",
        )
        missing_paths = [str(path) for path in required_paths if not path.exists()]
        if missing_paths:
            raise FileNotFoundError(
                "QuickUMLS installation is incomplete; missing: "
                + ", ".join(missing_paths)
            )

        try:
            from quickumls import QuickUMLS
        except ImportError as exc:
            raise ImportError(
                "Install the optional QuickUMLS dependencies before using this matcher."
            ) from exc

        kwargs: dict = {
            "threshold": threshold,
            "overlapping_criteria": "score",
            "similarity_name": "jaccard",
            "window": 5,
            # None disables QuickUMLS' built-in semantic-type filter. The CUI
            # allow-list remains the authoritative domain boundary in that mode.
            "accepted_semtypes": accepted_semtypes,
        }

        self.matcher = QuickUMLS(str(quickumls_path), **kwargs)
        self.allowed_cuis: set[str] | None = None
        self.cui_labels: dict[str, str] = {}
        if allowed_cuis_path:
            self.allowed_cuis, allowlist_labels = _load_allowed_cuis(
                allowed_cuis_path
            )
            self.cui_labels.update(allowlist_labels)
        if cui_labels_path:
            raw = json.loads(cui_labels_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("QuickUMLS CUI label file must be a JSON object.")
            explicit_labels = {
                str(cui): str(label)
                for cui, label in raw.items()
                if str(cui).strip() and str(label).strip()
            }
            self.cui_labels.update(explicit_labels)

    def match(
        self,
        source: str,
        sentences: list[SentenceSpan],
    ) -> list[ConceptCandidate]:
        groups = self.matcher.match(source, best_match=True, ignore_syntax=False)
        candidates: list[ConceptCandidate] = []

        for group in groups:
            if not group:
                continue
            eligible = [
                item
                for item in group
                if self.allowed_cuis is None
                or str(item.get("cui", "")) in self.allowed_cuis
            ]
            if not eligible:
                continue
            top = eligible[0]
            cui = str(top.get("cui", "")) or None

            char_start = int(top["start"])
            char_end = int(top["end"])
            sentence = find_sentence_for_span(sentences, char_start, char_end)
            if sentence is None:
                continue

            semtypes = top.get("semtypes", [])
            if isinstance(semtypes, str):
                semantic_types = [semtypes]
            else:
                semantic_types = sorted(str(item) for item in semtypes)

            candidates.append(
                ConceptCandidate(
                    candidate_id=f"qumls-{len(candidates):04d}",
                    sentence_id=sentence.sentence_id,
                    sentence_text=sentence.text,
                    span_text=source[char_start:char_end],
                    char_start=char_start,
                    char_end=char_end,
                    preferred_term=self.cui_labels.get(
                        cui or "", str(top.get("term") or top.get("ngram") or "")
                    ),
                    cui=cui,
                    semantic_types=semantic_types,
                    match_score=float(top.get("similarity", 0.0)),
                    origin=CandidateOrigin.QUICKUMLS,
                )
            )

        candidates.sort(key=lambda item: (item.char_start, -item.match_score))
        for index, candidate in enumerate(candidates):
            candidate.candidate_id = f"qumls-{index:04d}"
        return candidates
