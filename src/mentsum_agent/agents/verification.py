from __future__ import annotations

from ..llm import LLMClient
from ..prompts import (
    FAITHFULNESS_SYSTEM_PROMPT,
    SAFETY_SYSTEM_PROMPT,
    faithfulness_user_prompt,
    safety_user_prompt,
)
from ..safety_cues import RiskCueScanner
from ..schemas import (
    ClaimAssessment,
    ClaimSupportLabel,
    EvidenceStatus,
    FaithfulnessDraft,
    RiskCandidateSentence,
    SafetyDraft,
    SafetyIssue,
    SafetyIssueDraft,
    SafetyIssueType,
    VerificationReport,
)


def _evidence_status(container: str, quote: str) -> EvidenceStatus:
    if not quote.strip():
        return EvidenceStatus.EMPTY
    if quote in container:
        return EvidenceStatus.VALID
    return EvidenceStatus.INVALID


def _quote_in_candidates(
    quote: str,
    candidates: list[RiskCandidateSentence],
) -> bool:
    return bool(quote.strip()) and any(
        quote in candidate.sentence_text for candidate in candidates
    )


def _all_source_evidence_in_candidates(
    item: SafetyIssueDraft,
    source_candidates: list[RiskCandidateSentence],
) -> bool:
    return bool(item.source_evidence) and all(
        _quote_in_candidates(quote, source_candidates)
        for quote in item.source_evidence
    )


def _passes_safety_gate(
    item: SafetyIssueDraft,
    summary_status: EvidenceStatus,
    source_statuses: list[EvidenceStatus],
    source_candidates: list[RiskCandidateSentence],
    summary_candidates: list[RiskCandidateSentence],
) -> bool:
    if item.issue_type == SafetyIssueType.RISK_OMISSION:
        return (
            _all_source_evidence_in_candidates(item, source_candidates)
            and summary_status == EvidenceStatus.EMPTY
        )

    if item.issue_type == SafetyIssueType.RISK_DEESCALATION:
        return (
            _all_source_evidence_in_candidates(item, source_candidates)
            and summary_status == EvidenceStatus.VALID
        )

    if item.issue_type == SafetyIssueType.RISK_ESCALATION:
        return (
            _quote_in_candidates(item.summary_evidence, summary_candidates)
            and bool(source_statuses)
            and all(status == EvidenceStatus.VALID for status in source_statuses)
        )

    if item.issue_type == SafetyIssueType.CLINICAL_OVERREACH:
        return (
            summary_status == EvidenceStatus.VALID
            and all(status == EvidenceStatus.VALID for status in source_statuses)
        )

    return False


class FaithfulnessChecker:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def __call__(self, source: str, summary: str) -> FaithfulnessDraft:
        return self.llm.complete_structured(
            FAITHFULNESS_SYSTEM_PROMPT,
            faithfulness_user_prompt(source, summary),
            FaithfulnessDraft,
        )


class SafetyChecker:
    def __init__(
        self,
        llm: LLMClient,
        cue_scanner: RiskCueScanner,
    ):
        self.llm = llm
        self.cue_scanner = cue_scanner

    def __call__(
        self,
        source: str,
        summary: str,
    ) -> tuple[
        SafetyDraft,
        list[RiskCandidateSentence],
        list[RiskCandidateSentence],
    ]:
        source_candidates = self.cue_scanner.candidate_sentences(source)
        summary_candidates = self.cue_scanner.candidate_sentences(summary)
        draft = self.llm.complete_structured(
            SAFETY_SYSTEM_PROMPT,
            safety_user_prompt(
                source,
                summary,
                source_candidates,
                summary_candidates,
            ),
            SafetyDraft,
        )
        return draft, source_candidates, summary_candidates


class VerificationAgent:
    """Orchestrate the two verification components and validate their evidence."""

    def __init__(
        self,
        faithfulness_checker: FaithfulnessChecker,
        safety_checker: SafetyChecker,
    ):
        self.faithfulness_checker = faithfulness_checker
        self.safety_checker = safety_checker

    def verify(self, source: str, summary: str) -> VerificationReport:
        """Run both verification components for one explicit summary snapshot."""
        faithfulness = self.faithfulness_checker(source, summary)
        safety, source_candidates, summary_candidates = self.safety_checker(
            source,
            summary,
        )

        return self._validate_report(
            source=source,
            summary=summary,
            faithfulness=faithfulness,
            safety=safety,
            source_candidates=source_candidates,
            summary_candidates=summary_candidates,
        )

    def _validate_report(
        self,
        source: str,
        summary: str,
        faithfulness: FaithfulnessDraft,
        safety: SafetyDraft,
        source_candidates: list[RiskCandidateSentence],
        summary_candidates: list[RiskCandidateSentence],
    ) -> VerificationReport:
        claims: list[ClaimAssessment] = []
        safety_issues: list[SafetyIssue] = []
        actionable: list[str] = []

        for index, item in enumerate(faithfulness.claim_assessments, start=1):
            issue_id = f"F{index:02d}"
            summary_status = _evidence_status(summary, item.summary_evidence)
            source_statuses = [
                _evidence_status(source, quote) for quote in item.source_evidence
            ]
            validated = ClaimAssessment(
                **item.model_dump(),
                issue_id=issue_id,
                summary_evidence_status=summary_status,
                source_evidence_statuses=source_statuses,
            )
            claims.append(validated)
            if (
                item.label != ClaimSupportLabel.SUPPORTED
                and summary_status == EvidenceStatus.VALID
            ):
                actionable.append(issue_id)

            # if (
            #     item.label
            #     in {
            #         ClaimSupportLabel.UNSUPPORTED,
            #         ClaimSupportLabel.CONTRADICTED,
            #     }
            #     and summary_status == EvidenceStatus.VALID
            # ):
            #     actionable.append(issue_id)

        for item in safety.safety_issues:
            summary_status = _evidence_status(summary, item.summary_evidence)
            source_statuses = [
                _evidence_status(source, quote) for quote in item.source_evidence
            ]
            if not _passes_safety_gate(
                item=item,
                summary_status=summary_status,
                source_statuses=source_statuses,
                source_candidates=source_candidates,
                summary_candidates=summary_candidates,
            ):
                continue
            issue_id = f"S{len(safety_issues) + 1:02d}"
            validated = SafetyIssue(
                **item.model_dump(),
                issue_id=issue_id,
                summary_evidence_status=summary_status,
                source_evidence_statuses=source_statuses,
            )
            safety_issues.append(validated)
            actionable.append(issue_id)

        claim_labels_pass = bool(claims) and all(
            item.label == ClaimSupportLabel.SUPPORTED for item in claims
        )
        claim_evidence_pass = bool(claims) and all(
            item.summary_evidence_status == EvidenceStatus.VALID
            and EvidenceStatus.VALID in item.source_evidence_statuses
            for item in claims
        )
        faithfulness_passed = claim_labels_pass and claim_evidence_pass

        safety_passed = not safety_issues

        actionable = list(dict.fromkeys(actionable))
        passed = faithfulness_passed and safety_passed
        return VerificationReport(
            faithfulness_passed=faithfulness_passed,
            safety_passed=safety_passed,
            passed=passed,
            claim_assessments=claims,
            safety_issues=safety_issues,
            source_risk_candidates=source_candidates,
            summary_risk_candidates=summary_candidates,
            actionable_issue_ids=actionable,
        )
