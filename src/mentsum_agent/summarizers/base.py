from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import GuidanceItem


class Summarizer(ABC):
    @abstractmethod
    def summarize(self, source: str, guidance: list[GuidanceItem]) -> str:
        raise NotImplementedError
