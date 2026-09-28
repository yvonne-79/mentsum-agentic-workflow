from __future__ import annotations

from .schemas import (
    ClaimSupportLabel,
    RevisionAcceptanceDecision,
    VerificationMetrics,
    VerificationReport,
)
from .text_utils import normalize_space


def _verification_metrics(report: VerificationReport) -> VerificationMetrics:
    labels = [item.label for item in report.claim_assessments]
    return VerificationMetrics(
        num_safety_issues=len(report.safety_issues),
        num_contradicted=labels.count(ClaimSupportLabel.CONTRADICTED),
        num_unsupported=labels.count(ClaimSupportLabel.UNSUPPORTED),
        num_partially_supported=labels.count(
            ClaimSupportLabel.PARTIALLY_SUPPORTED
        ),
    )


class RevisionAcceptanceController:
    """Deterministically decide whether a revision candidate becomes the new best."""

    def __call__(
        self,
        previous_summary: str,
        previous_verification: VerificationReport | dict,
        candidate_summary: str,
        candidate_verification: VerificationReport | dict,
        target_issue_ids: list[str],
    ) -> RevisionAcceptanceDecision:
        previous_report = VerificationReport.model_validate(previous_verification)
        candidate_report = VerificationReport.model_validate(candidate_verification)
        previous = _verification_metrics(previous_report)
        candidate = _verification_metrics(candidate_report)
        targets = list(dict.fromkeys(target_issue_ids))
        summary_changed = normalize_space(candidate_summary) != normalize_space(
            previous_summary
        )

        if candidate.num_safety_issues > previous.num_safety_issues:
            return self._decision(
                False,
                "safety_regression",
                targets,
                summary_changed,
                previous,
                candidate,
            )

        if candidate.num_contradicted > previous.num_contradicted:
            return self._decision(
                False,
                "faithfulness_regression",
                targets,
                summary_changed,
                previous,
                candidate,
            )

        if candidate.num_unsupported > previous.num_unsupported:
            return self._decision(
                False,
                "faithfulness_regression",
                targets,
                summary_changed,
                previous,
                candidate,
            )

        improved = (
            candidate.num_safety_issues < previous.num_safety_issues
            or candidate.num_contradicted < previous.num_contradicted
            or candidate.num_unsupported < previous.num_unsupported
            or candidate.num_partially_supported
            < previous.num_partially_supported
        )
        if not summary_changed or not improved:
            return self._decision(
                False,
                "no_improvement",
                targets,
                summary_changed,
                previous,
                candidate,
            )

        return self._decision(
            True,
            "improved",
            targets,
            summary_changed,
            previous,
            candidate,
        )

    @staticmethod
    def _decision(
        accepted: bool,
        reason: str,
        target_issue_ids: list[str],
        summary_changed: bool,
        previous_metrics: VerificationMetrics,
        candidate_metrics: VerificationMetrics,
    ) -> RevisionAcceptanceDecision:
        return RevisionAcceptanceDecision(
            accepted=accepted,
            reason=reason,
            target_issue_ids=target_issue_ids,
            summary_changed=summary_changed,
            previous_metrics=previous_metrics,
            candidate_metrics=candidate_metrics,
        )
