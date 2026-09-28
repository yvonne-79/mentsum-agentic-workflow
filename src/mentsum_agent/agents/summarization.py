from __future__ import annotations

from ..schemas import GuidanceItem
from ..state import WorkflowState
from ..summarizers import Summarizer


class SummarizationAgent:
    def __init__(self, summarizer: Summarizer):
        self.summarizer = summarizer

    def __call__(self, state: WorkflowState) -> dict:
        guidance = [GuidanceItem.model_validate(item) for item in state.get("guidance", [])]
        summary = self.summarizer.summarize(state["source"], guidance).strip()
        if not summary:
            raise RuntimeError("The summarizer returned an empty summary.")
        return {
            "initial_summary": summary,
            "best_summary": summary,
            "revision_round": 0,
            "revision_history": [],
            "revision_acceptance_history": [],
            "verification_history": [],
        }
