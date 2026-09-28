from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import ConceptCandidate
from ..text_utils import SentenceSpan


class TermMatcher(ABC):
    @abstractmethod
    def match(
        self,
        source: str,
        sentences: list[SentenceSpan],
    ) -> list[ConceptCandidate]:
        raise NotImplementedError
