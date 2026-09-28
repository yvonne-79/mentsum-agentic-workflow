from __future__ import annotations

from dataclasses import dataclass

from .agents import (
    FaithfulnessChecker,
    GuidanceAgent,
    RevisionAgent,
    SafetyChecker,
    SummarizationAgent,
    VerificationAgent,
)
from .config import Settings
from .graph import WorkflowGraphFactory
from .llm import LLMClient, build_llm
from .matchers import LexiconMatcher, QuickUMLSMatcher, TermMatcher
from .prompts import PROMPT_VERSIONS
from .revision_acceptance import RevisionAcceptanceController
from .safety_cues import RiskCueScanner
from .state import WorkflowState
from .summarizers import ExternalGSumSummarizer, LLMSummarizer, Summarizer


@dataclass
class WorkflowRuntime:
    settings: Settings
    llm: LLMClient
    graph: object

    def initial_state(
        self,
        example_id: str,
        source: str,
    ) -> WorkflowState:
        return {
            "example_id": example_id,
            "source": source,
            "model_id": self.llm.model_id,
            "prompt_versions": dict(PROMPT_VERSIONS),
            "workflow_config": self.settings.public_workflow_config(),
            "revision_round": 0,
            "revision_history": [],
            "revision_acceptance_history": [],
            "verification_history": [],
        }


def _build_matcher(settings: Settings) -> TermMatcher:
    if settings.term_matcher == "lexicon":
        return LexiconMatcher(settings.lexicon_path)
    if settings.quickumls_path is None:
        raise ValueError(
            "QUICKUMLS_PATH is required when TERM_MATCHER=quickumls."
        )
    return QuickUMLSMatcher(
        quickumls_path=settings.quickumls_path,
        threshold=settings.quickumls_threshold,
        accepted_semtypes=settings.accepted_semtypes() or None,
        allowed_cuis_path=settings.quickumls_allowed_cuis_path,
        cui_labels_path=settings.quickumls_cui_labels_path,
    )


def build_summarizer(
    settings: Settings, llm: LLMClient | None = None
) -> Summarizer:
    if settings.summarizer_backend == "llm":
        llm = llm or build_llm(settings)
        return LLMSummarizer(llm, max_summary_words=settings.max_summary_words)
    return ExternalGSumSummarizer(
        command=settings.gsum_command,
        timeout_seconds=settings.gsum_timeout_seconds,
    )


def build_guidance_agent(settings: Settings, llm: LLMClient) -> GuidanceAgent:
    return GuidanceAgent(
        matcher=_build_matcher(settings),
        llm=llm,
        max_guidance_items=settings.max_guidance_items,
        max_match_candidates=settings.max_match_candidates,
    )


def build_runtime(settings: Settings | None = None) -> WorkflowRuntime:
    settings = settings or Settings()
    llm = build_llm(settings)
    cue_scanner = RiskCueScanner(settings.risk_cue_path)
    summarizer = build_summarizer(settings, llm)

    guidance_agent = build_guidance_agent(settings, llm)
    summarization_agent = SummarizationAgent(summarizer)
    verification_agent = VerificationAgent(
        faithfulness_checker=FaithfulnessChecker(llm),
        safety_checker=SafetyChecker(llm, cue_scanner),
    )
    revision_agent = RevisionAgent(llm=llm)
    revision_acceptance_controller = RevisionAcceptanceController()
    graph = WorkflowGraphFactory(
        guidance_agent=guidance_agent,
        summarization_agent=summarization_agent,
        verification_agent=verification_agent,
        revision_agent=revision_agent,
        revision_acceptance_controller=revision_acceptance_controller,
        max_revision_rounds=settings.max_revision_rounds,
    ).compile()
    return WorkflowRuntime(settings=settings, llm=llm, graph=graph)
