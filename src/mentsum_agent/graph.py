from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agents import GuidanceAgent, RevisionAgent, SummarizationAgent, VerificationAgent
from .revision_acceptance import RevisionAcceptanceController
from .schemas import StopReason, VerificationReport
from .state import WorkflowState


class WorkflowGraphFactory:
    def __init__(
        self,
        guidance_agent: GuidanceAgent,
        summarization_agent: SummarizationAgent,
        verification_agent: VerificationAgent,
        revision_agent: RevisionAgent,
        revision_acceptance_controller: RevisionAcceptanceController,
        max_revision_rounds: int,
    ):
        self.guidance_agent = guidance_agent
        self.summarization_agent = summarization_agent
        self.verification_agent = verification_agent
        self.revision_agent = revision_agent
        self.revision_acceptance_controller = revision_acceptance_controller
        self.max_revision_rounds = max_revision_rounds

    def compile(self):
        builder = StateGraph(WorkflowState)
        builder.add_node("extract_guidance", self.guidance_agent)
        builder.add_node("generate_summary", self.summarization_agent)
        builder.add_node("verify_best_summary", self._verify_best_summary)
        builder.add_node("revise_summary", self.revision_agent)
        builder.add_node("verify_candidate_summary", self._verify_candidate_summary)
        builder.add_node("evaluate_revision", self._evaluate_revision)
        builder.add_node("finalize", self._finalize)

        builder.add_edge(START, "extract_guidance")
        builder.add_edge("extract_guidance", "generate_summary")
        builder.add_edge("generate_summary", "verify_best_summary")
        builder.add_conditional_edges(
            "verify_best_summary",
            self._route_from_best,
            {"revise": "revise_summary", "finalize": "finalize"},
        )
        builder.add_edge("revise_summary", "verify_candidate_summary")
        builder.add_edge("verify_candidate_summary", "evaluate_revision")
        builder.add_conditional_edges(
            "evaluate_revision",
            self._route_after_revision,
            {"revise": "revise_summary", "finalize": "finalize"},
        )
        builder.add_edge("finalize", END)
        return builder.compile()

    def _verify_best_summary(self, state: WorkflowState) -> dict:
        report = self.verification_agent.verify(
            state["source"], state["best_summary"]
        )
        report_payload = report.model_dump(mode="json")
        history = list(state.get("verification_history", []))
        history.append(report_payload)
        return {
            "best_verification": report_payload,
            "verification_history": history,
        }

    def _verify_candidate_summary(self, state: WorkflowState) -> dict:
        report = self.verification_agent.verify(
            state["source"], state["candidate_summary"]
        )
        report_payload = report.model_dump(mode="json")
        history = list(state.get("verification_history", []))
        history.append(report_payload)
        return {
            "candidate_verification": report_payload,
            "verification_history": history,
        }

    def _evaluate_revision(self, state: WorkflowState) -> dict:
        decision = self.revision_acceptance_controller(
            previous_summary=state["best_summary"],
            previous_verification=state["best_verification"],
            candidate_summary=state["candidate_summary"],
            candidate_verification=state["candidate_verification"],
            target_issue_ids=state["target_issue_ids"],
        )
        decision_payload = decision.model_dump(mode="json")
        history = list(state.get("revision_acceptance_history", []))
        history.append(decision_payload)
        update: dict = {
            "revision_acceptance": decision_payload,
            "revision_acceptance_history": history,
        }
        if decision.accepted:
            candidate_report = VerificationReport.model_validate(
                state["candidate_verification"]
            )
            report_payload = candidate_report.model_dump(mode="json")
            update.update(
                {
                    "best_summary": state["candidate_summary"],
                    "best_verification": report_payload,
                }
            )
        return update

    def _route_from_best(self, state: WorkflowState) -> str:
        return self._route_best_snapshot(state)

    def _route_after_revision(self, state: WorkflowState) -> str:
        return self._route_best_snapshot(state)

    def _route_best_snapshot(self, state: WorkflowState) -> str:
        report = VerificationReport.model_validate(state["best_verification"])
        if report.passed:
            return "finalize"
        if not report.actionable_issue_ids:
            return "finalize"
        if int(state.get("revision_round", 0)) >= self.max_revision_rounds:
            return "finalize"
        return "revise"

    def _finalize(self, state: WorkflowState) -> dict:
        report = VerificationReport.model_validate(state["best_verification"])
        if report.passed:
            reason = StopReason.PASSED
        elif not report.actionable_issue_ids:
            reason = StopReason.NO_ACTIONABLE_ISSUES
        else:
            reason = StopReason.MAX_REVISION_ROUNDS
        return {
            "final_summary": state["best_summary"],
            "stop_reason": reason.value,
        }
