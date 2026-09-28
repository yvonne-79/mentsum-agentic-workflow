from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from .config import Settings
from .data_io import (
    append_jsonl_record,
    build_run_output,
    iter_source_examples,
    read_completed_ids,
    source_sha256,
)
from .factory import build_runtime
from .llm import usage_delta


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the agentic summarization workflow on source data."
    )
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--id-prefix", type=str, default="test")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include-source", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.start < 0:
        raise ValueError("--start must be non-negative.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")

    settings = Settings(_env_file=args.env_file)
    runtime = build_runtime(settings)
    examples = list(iter_source_examples(args.source_file))
    examples = examples[args.start :]
    if args.limit is not None:
        examples = examples[: args.limit]

    completed = read_completed_ids(args.output, retry_errors=args.retry_errors)
    pending = [
        item
        for item in examples
        if f"{args.id_prefix}-{item[0]:04d}" not in completed
    ]

    print(f"Selected examples: {len(examples)}")
    print(f"Already completed: {len(examples) - len(pending)}")
    print(f"Remaining: {len(pending)}")
    print(f"Output file: {args.output}")

    for index, source in tqdm(pending, desc="Running workflow"):
        example_id = f"{args.id_prefix}-{index:04d}"
        usage_before = runtime.llm.usage_snapshot()
        try:
            final_state = runtime.graph.invoke(
                runtime.initial_state(example_id, source),
                config={"recursion_limit": 30},
            )
            output = build_run_output(
                final_state, include_source=args.include_source
            )
            record = output.model_dump(mode="json")
            record["llm_usage"] = usage_delta(
                usage_before,
                runtime.llm.usage_snapshot(),
            )
        except Exception as exc:
            if args.fail_fast:
                raise
            record = {
                "status": "error",
                "example_id": example_id,
                "source_sha256": source_sha256(source),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "llm_usage": usage_delta(
                    usage_before,
                    runtime.llm.usage_snapshot(),
                ),
                "model_id": runtime.llm.model_id,
                "workflow_config": settings.public_workflow_config(),
            }
        append_jsonl_record(args.output, record)


if __name__ == "__main__":
    main()
