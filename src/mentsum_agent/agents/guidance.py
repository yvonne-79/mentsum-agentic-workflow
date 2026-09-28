from __future__ import annotations

from collections import defaultdict

from ..llm import LLMClient
from ..matchers import TermMatcher
from ..prompts import GUIDANCE_SYSTEM_PROMPT, guidance_user_prompt
from ..schemas import (
    CandidateOrigin,
    ConceptCandidate,
    GuidanceControllerOutput,
    GuidanceItem,
    GuidanceTerm,
)
from ..state import WorkflowState
from ..text_utils import SentenceSpan, split_sentences_with_offsets


class GuidanceAgent:
    def __init__(
        self,
        matcher: TermMatcher,
        llm: LLMClient,
        max_guidance_items: int = 10,
        max_match_candidates: int = 30,
    ):
        self.matcher = matcher
        self.llm = llm
        self.max_guidance_items = max_guidance_items
        self.max_match_candidates = max_match_candidates

    def __call__(self, state: WorkflowState) -> dict:
        source = state["source"]
        sentences = split_sentences_with_offsets(source)
        candidates = self.matcher.match(source, sentences)
        candidates = self._cap_candidates(candidates)

        sentence_payload = [
            {
                "sentence_id": item.sentence_id,
                "text": item.text,
                "char_start": item.start,
                "char_end": item.end,
            }
            for item in sentences
        ]
        candidate_payload = [item.model_dump(mode="json") for item in candidates]

        controller_output = self.llm.complete_structured(
            GUIDANCE_SYSTEM_PROMPT,
            guidance_user_prompt(
                source=source,
                sentences=sentence_payload,
                candidates=candidate_payload,
                max_guidance_items=self.max_guidance_items,
            ),
            GuidanceControllerOutput,
        )
        guidance = self._build_guidance(
            source=source,
            sentences=sentences,
            candidates=candidates,
            controller_output=controller_output,
        )

        return {
            "sentences": sentence_payload,
            "guidance_candidates": candidate_payload,
            "guidance": [item.model_dump(mode="json") for item in guidance],
        }

    def _cap_candidates(
        self, candidates: list[ConceptCandidate]
    ) -> list[ConceptCandidate]:
        if len(candidates) <= self.max_match_candidates:
            return candidates
        selected = sorted(
            candidates,
            key=lambda item: (-item.match_score, item.char_start, item.char_end),
        )[: self.max_match_candidates]
        return sorted(selected, key=lambda item: (item.char_start, item.char_end))

    def _build_guidance(
        self,
        source: str,
        sentences: list[SentenceSpan],
        candidates: list[ConceptCandidate],
        controller_output: GuidanceControllerOutput,
    ) -> list[GuidanceItem]:
        sentence_map = {item.sentence_id: item for item in sentences}
        candidate_map = {item.candidate_id: item for item in candidates}

        grouped_terms: dict[int, list[GuidanceTerm]] = defaultdict(list)
        group_priority: dict[int, int] = {}
        seen_terms: set[tuple[int, int, str]] = set()
        priority_counter = 0

        selected_ids: list[str] = []
        seen_ids: set[str] = set()
        for candidate_id in controller_output.selected_candidate_ids:
            if candidate_id in candidate_map and candidate_id not in seen_ids:
                selected_ids.append(candidate_id)
                seen_ids.add(candidate_id)

        for candidate_id in selected_ids:
            candidate = candidate_map[candidate_id]
            sentence = sentence_map.get(candidate.sentence_id)
            if sentence is None:
                continue
            if source[candidate.char_start : candidate.char_end] != candidate.span_text:
                continue
            term_key = (
                candidate.char_start,
                candidate.char_end,
                candidate.preferred_term.casefold(),
            )
            if term_key in seen_terms:
                continue
            seen_terms.add(term_key)
            group_priority.setdefault(candidate.sentence_id, priority_counter)
            priority_counter += 1
            grouped_terms[candidate.sentence_id].append(
                GuidanceTerm(
                    source_span=candidate.span_text,
                    normalized_term=candidate.preferred_term,
                    char_start=candidate.char_start,
                    char_end=candidate.char_end,
                    cui=candidate.cui,
                    semantic_types=candidate.semantic_types,
                    match_score=candidate.match_score,
                    origin=candidate.origin,
                )
            )

        for mention in controller_output.additional_mentions:
            sentence = sentence_map.get(mention.sentence_id)
            if sentence is None:
                continue
            relative_start = sentence.text.find(mention.exact_span)
            if relative_start < 0:
                continue
            char_start = sentence.start + relative_start
            char_end = char_start + len(mention.exact_span)
            if source[char_start:char_end] != mention.exact_span:
                continue
            term_key = (char_start, char_end, mention.normalized_term.casefold())
            if term_key in seen_terms:
                continue
            seen_terms.add(term_key)
            group_priority.setdefault(mention.sentence_id, priority_counter)
            priority_counter += 1
            grouped_terms[mention.sentence_id].append(
                GuidanceTerm(
                    source_span=mention.exact_span,
                    normalized_term=mention.normalized_term,
                    char_start=char_start,
                    char_end=char_end,
                    origin=CandidateOrigin.LLM,
                )
            )

        ranked_sentence_ids = sorted(
            grouped_terms,
            key=lambda sentence_id: (
                group_priority.get(sentence_id, 10**9),
                sentence_id,
            ),
        )[: self.max_guidance_items]

        guidance: list[GuidanceItem] = []
        for sentence_id in ranked_sentence_ids:
            sentence = sentence_map[sentence_id]
            terms = sorted(
                grouped_terms[sentence_id],
                key=lambda item: (item.char_start, item.char_end),
            )
            guidance.append(
                GuidanceItem(
                    guidance_id=f"g-{len(guidance):02d}",
                    sentence_id=sentence_id,
                    source_sentence=source[sentence.start : sentence.end],
                    sentence_start=sentence.start,
                    sentence_end=sentence.end,
                    terms=terms,
                )
            )
        return guidance
