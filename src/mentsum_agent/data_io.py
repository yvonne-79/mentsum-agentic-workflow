from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterator

from .schemas import RunOutput
from .state import WorkflowState


def iter_source_examples(source_path: Path) -> Iterator[tuple[int, str]]:
    with source_path.open("r", encoding="utf-8") as source_file:
        for index, source_line in enumerate(source_file):
            yield index, source_line.rstrip("\r\n")


def source_sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def build_run_output(
    state: WorkflowState,
    include_source: bool = False,
) -> RunOutput:
    return RunOutput(
        example_id=state["example_id"],
        source_sha256=source_sha256(state["source"]),
        guidance_candidates=state.get("guidance_candidates", []),
        guidance=state.get("guidance", []),
        initial_summary=state["initial_summary"],
        final_summary=state["final_summary"],
        verification_history=state.get("verification_history", []),
        revision_history=state.get("revision_history", []),
        revision_acceptance_history=state.get(
            "revision_acceptance_history", []
        ),
        stop_reason=state["stop_reason"],
        model_id=state["model_id"],
        prompt_versions=state["prompt_versions"],
        workflow_config=state["workflow_config"],
        source=state["source"] if include_source else None,
    )


def read_completed_ids(output_path: Path, retry_errors: bool = False) -> set[str]:
    if not output_path.exists():
        return set()
    completed: set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {output_path} at line {line_number}."
                ) from exc
            example_id = str(record.get("example_id", ""))
            if not example_id:
                continue
            if retry_errors and record.get("status") == "error":
                continue
            completed.add(example_id)
    return completed


def append_jsonl_record(output_path: Path, record: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
