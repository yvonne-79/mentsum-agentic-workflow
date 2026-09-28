from __future__ import annotations

from dataclasses import dataclass

from .schemas import RevisionEdit


class RevisionEditApplicationError(ValueError):
    """Raised when an edit set cannot be applied safely and atomically."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class _ResolvedEdit:
    edit: RevisionEdit
    start: int
    end: int

    @property
    def patch_start(self) -> int:
        if self.edit.operation == "insert_after":
            return self.end
        return self.start

    @property
    def patch_end(self) -> int:
        if self.edit.operation == "insert_after":
            return self.end
        return self.end


def _resolve_unique_span(current_summary: str, target_span: str) -> tuple[int, int]:
    start = current_summary.find(target_span)
    if start < 0:
        raise RevisionEditApplicationError("target_span_not_found")
    if current_summary.find(target_span, start + 1) >= 0:
        raise RevisionEditApplicationError("ambiguous_target_span")
    return start, start + len(target_span)


def apply_revision_edits(
    current_summary: str,
    edits: list[RevisionEdit],
    actionable_issue_ids: list[str],
) -> tuple[str, list[RevisionEdit]]:
    """Validate and atomically apply exact-span edits from right to left."""

    if not edits:
        raise RevisionEditApplicationError("no_edits")

    allowed_issue_ids = set(actionable_issue_ids)
    resolved: list[_ResolvedEdit] = []
    for edit in edits:
        if edit.issue_id not in allowed_issue_ids:
            raise RevisionEditApplicationError("non_actionable_issue_id")
        start, end = _resolve_unique_span(current_summary, edit.target_span)
        resolved.append(_ResolvedEdit(edit=edit, start=start, end=end))

    by_target_position = sorted(resolved, key=lambda item: (item.start, item.end))
    for previous, current in zip(by_target_position, by_target_position[1:]):
        if current.start < previous.end:
            raise RevisionEditApplicationError("overlapping_edits")

    candidate_summary = current_summary
    application_order = sorted(
        resolved,
        key=lambda item: (item.patch_start, item.patch_end),
        reverse=True,
    )
    for resolved_edit in application_order:
        edit = resolved_edit.edit
        start = resolved_edit.patch_start
        end = resolved_edit.patch_end
        if edit.operation == "delete":
            replacement = ""
        else:
            replacement = edit.replacement
        candidate_summary = (
            candidate_summary[:start] + replacement + candidate_summary[end:]
        )

    if not candidate_summary.strip():
        raise RevisionEditApplicationError("empty_candidate_summary")

    return candidate_summary, list(edits)
