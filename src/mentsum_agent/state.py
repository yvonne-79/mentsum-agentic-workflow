from __future__ import annotations

from typing import Any, TypedDict


class WorkflowState(TypedDict, total=False):
    example_id: str
    source: str

    sentences: list[dict[str, Any]]
    guidance_candidates: list[dict[str, Any]]
    guidance: list[dict[str, Any]]

    initial_summary: str
    best_summary: str
    best_verification: dict[str, Any]
    candidate_summary: str
    candidate_verification: dict[str, Any]

    final_summary: str

    verification_history: list[dict[str, Any]]

    revision_round: int
    revision_history: list[dict[str, Any]]
    target_issue_ids: list[str]
    revision_acceptance: dict[str, Any]
    revision_acceptance_history: list[dict[str, Any]]

    stop_reason: str
    model_id: str
    prompt_versions: dict[str, str]
    workflow_config: dict[str, Any]
