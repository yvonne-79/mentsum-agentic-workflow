from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CandidateOrigin(str, Enum):
    QUICKUMLS = "quickumls"
    LEXICON = "lexicon"
    LLM = "llm"


class ConceptCandidate(BaseModel):
    candidate_id: str
    sentence_id: int
    sentence_text: str
    span_text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    preferred_term: str
    cui: str | None = None
    semantic_types: list[str] = Field(default_factory=list)
    match_score: float = Field(ge=0.0, le=1.0)
    origin: CandidateOrigin


class AdditionalMention(BaseModel):
    sentence_id: int = Field(ge=0)
    exact_span: str = Field(min_length=1)
    normalized_term: str = Field(min_length=1)


class GuidanceControllerOutput(BaseModel):
    selected_candidate_ids: list[str] = Field(default_factory=list)
    additional_mentions: list[AdditionalMention] = Field(default_factory=list)


class GuidanceTerm(BaseModel):
    source_span: str
    normalized_term: str
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    cui: str | None = None
    semantic_types: list[str] = Field(default_factory=list)
    match_score: float | None = Field(default=None, ge=0.0, le=1.0)
    origin: CandidateOrigin


class GuidanceItem(BaseModel):
    guidance_id: str
    sentence_id: int = Field(ge=0)
    source_sentence: str
    sentence_start: int = Field(ge=0)
    sentence_end: int = Field(gt=0)
    terms: list[GuidanceTerm] = Field(min_length=1)


class ClaimSupportLabel(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


class ClaimAssessmentDraft(BaseModel):
    summary_claim: str = Field(min_length=1)
    summary_evidence: str = Field(min_length=1)
    label: ClaimSupportLabel
    source_evidence: list[str] = Field(default_factory=list)
    correction: str = ""


class SafetyIssueType(str, Enum):
    RISK_OMISSION = "risk_omission"
    RISK_ESCALATION = "risk_escalation"
    RISK_DEESCALATION = "risk_deescalation"
    CLINICAL_OVERREACH = "clinical_overreach"


class SafetyIssueDraft(BaseModel):
    issue_type: SafetyIssueType
    summary_evidence: str = ""
    source_evidence: list[str] = Field(default_factory=list)
    description: str = Field(min_length=1)
    recommended_edit: str = ""


class FaithfulnessDraft(BaseModel):
    claim_assessments: list[ClaimAssessmentDraft] = Field(default_factory=list)


class SafetyDraft(BaseModel):
    safety_issues: list[SafetyIssueDraft] = Field(default_factory=list)


class EvidenceStatus(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    EMPTY = "empty"


class ClaimAssessment(ClaimAssessmentDraft):
    issue_id: str
    summary_evidence_status: EvidenceStatus
    source_evidence_statuses: list[EvidenceStatus] = Field(default_factory=list)


class SafetyIssue(SafetyIssueDraft):
    issue_id: str
    summary_evidence_status: EvidenceStatus
    source_evidence_statuses: list[EvidenceStatus] = Field(default_factory=list)


class RiskCueHit(BaseModel):
    category: str
    span_text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)


class RiskCandidateSentence(BaseModel):
    sentence_id: int = Field(ge=0)
    sentence_text: str = Field(min_length=1)
    sentence_start: int = Field(ge=0)
    sentence_end: int = Field(gt=0)
    risk_cues: list[RiskCueHit] = Field(min_length=1)


class VerificationReport(BaseModel):
    faithfulness_passed: bool
    safety_passed: bool
    passed: bool
    claim_assessments: list[ClaimAssessment] = Field(default_factory=list)
    safety_issues: list[SafetyIssue] = Field(default_factory=list)
    source_risk_candidates: list[RiskCandidateSentence] = Field(
        default_factory=list
    )
    summary_risk_candidates: list[RiskCandidateSentence] = Field(
        default_factory=list
    )
    actionable_issue_ids: list[str] = Field(default_factory=list)


class RevisionEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_id: str = Field(
        min_length=1,
        description="Actionable verification issue addressed by this edit.",
    )
    operation: Literal["replace", "delete", "insert_after"]
    target_span: str = Field(
        min_length=1,
        description=(
            "Exact contiguous span from the current summary; for insert_after, "
            "this is the exact anchor."
        ),
    )
    replacement: str = Field(
        description=(
            "Verbatim replacement or insertion text; use an empty string only "
            "for delete."
        )
    )

    @model_validator(mode="after")
    def validate_replacement(self) -> RevisionEdit:
        if self.operation == "delete" and self.replacement != "":
            raise ValueError("A delete edit must have an empty replacement.")
        if self.operation in {"replace", "insert_after"} and not self.replacement.strip():
            raise ValueError(
                f"A {self.operation} edit must have a non-empty replacement."
            )
        return self


class RevisionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edits: list[RevisionEdit]
    no_change: bool = Field(
        description="True only when no safe local edit can be proposed."
    )

    @model_validator(mode="after")
    def validate_edit_choice(self) -> RevisionOutput:
        if self.no_change and self.edits:
            raise ValueError("no_change cannot be returned with edit operations.")
        if not self.no_change and not self.edits:
            raise ValueError("Return at least one edit or set no_change to true.")
        return self


class RevisionRecord(BaseModel):
    round_index: int = Field(ge=1)
    previous_summary: str
    candidate_summary: str
    target_issue_ids: list[str] = Field(default_factory=list)
    applied_issue_ids: list[str] = Field(default_factory=list)
    proposed_edits: list[RevisionEdit] = Field(default_factory=list)
    applied_edits: list[RevisionEdit] = Field(default_factory=list)
    no_change: bool = False
    edit_rejection_reason: str | None = None
    changed: bool


class VerificationMetrics(BaseModel):
    num_safety_issues: int = Field(ge=0)
    num_contradicted: int = Field(ge=0)
    num_unsupported: int = Field(ge=0)
    num_partially_supported: int = Field(ge=0)


class RevisionAcceptanceDecision(BaseModel):
    accepted: bool
    reason: str = Field(min_length=1)
    target_issue_ids: list[str] = Field(default_factory=list)
    summary_changed: bool
    previous_metrics: VerificationMetrics
    candidate_metrics: VerificationMetrics


class StopReason(str, Enum):
    PASSED = "passed"
    MAX_REVISION_ROUNDS = "max_revision_rounds"
    NO_ACTIONABLE_ISSUES = "no_actionable_issues"


class RunOutput(BaseModel):
    status: Literal["ok"] = "ok"
    example_id: str
    source_sha256: str
    guidance_candidates: list[ConceptCandidate] = Field(default_factory=list)
    guidance: list[GuidanceItem] = Field(default_factory=list)
    initial_summary: str
    final_summary: str
    verification_history: list[VerificationReport] = Field(default_factory=list)
    revision_history: list[RevisionRecord] = Field(default_factory=list)
    revision_acceptance_history: list[RevisionAcceptanceDecision] = Field(
        default_factory=list
    )
    stop_reason: StopReason
    model_id: str
    prompt_versions: dict[str, str]
    workflow_config: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None


class LexiconEntry(BaseModel):
    surface_forms: list[str] = Field(min_length=1)
    preferred_term: str
    cui: str | None = None
    semantic_types: list[str] = Field(default_factory=list)


class RiskCueEntry(BaseModel):
    category: str
    surface_forms: list[str] = Field(min_length=1)


SummaryBackend = Literal["llm", "external_gsum"]
TermMatcherBackend = Literal["lexicon", "quickumls"]
LLMBackend = Literal["openai", "mock"]
OpenRouterProvider = Literal["openai"]
