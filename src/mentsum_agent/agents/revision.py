from __future__ import annotations

from ..llm import LLMClient
from ..prompts import REVISION_SYSTEM_PROMPT, revision_user_prompt
from ..revision_edits import RevisionEditApplicationError, apply_revision_edits
from ..schemas import RevisionOutput, RevisionRecord, VerificationReport
from ..state import WorkflowState


class RevisionAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def __call__(self, state: WorkflowState) -> dict:
        source = state["source"]
        best_summary = state["best_summary"]
        report = VerificationReport.model_validate(state["best_verification"])
        if not report.actionable_issue_ids:
            raise RuntimeError(
                "RevisionAgent requires at least one actionable issue."
            )

        output = self.llm.complete_structured(
            REVISION_SYSTEM_PROMPT,
            revision_user_prompt(
                source=source,
                best_summary=best_summary,
                report=report,
            ),
            RevisionOutput,
        )
        target_issue_ids = list(dict.fromkeys(report.actionable_issue_ids))
        applied_edits = []
        edit_rejection_reason = None
        if output.no_change:
            candidate_summary = best_summary
        else:
            try:
                candidate_summary, applied_edits = apply_revision_edits(
                    current_summary=best_summary,
                    edits=output.edits,
                    actionable_issue_ids=target_issue_ids,
                )
            except RevisionEditApplicationError as exc:
                candidate_summary = best_summary
                edit_rejection_reason = exc.reason

        applied_ids = list(
            dict.fromkeys(edit.issue_id for edit in applied_edits)
        )
        changed = candidate_summary != best_summary
        round_index = int(state.get("revision_round", 0)) + 1
        record = RevisionRecord(
            round_index=round_index,
            previous_summary=best_summary,
            candidate_summary=candidate_summary,
            target_issue_ids=target_issue_ids,
            applied_issue_ids=applied_ids,
            proposed_edits=output.edits,
            applied_edits=applied_edits,
            no_change=output.no_change,
            edit_rejection_reason=edit_rejection_reason,
            changed=changed,
        )
        history = list(state.get("revision_history", []))
        history.append(record.model_dump(mode="json"))
        return {
            "candidate_summary": candidate_summary,
            "target_issue_ids": target_issue_ids,
            "revision_round": round_index,
            "revision_history": history,
        }
