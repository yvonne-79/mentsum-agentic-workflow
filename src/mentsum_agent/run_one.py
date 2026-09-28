from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Settings
from .data_io import build_run_output, iter_source_examples
from .factory import build_runtime


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the agentic summarization workflow on one example."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source-text", type=str)
    source_group.add_argument("--source-file", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--example-id", type=str)
    parser.add_argument("--output", type=Path, default=Path("outputs/run_one.json"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--include-source", action="store_true")
    return parser.parse_args()


def _read_selected_source(args: argparse.Namespace) -> str:
    if args.source_text is not None:
        return args.source_text
    assert args.source_file is not None
    if args.index < 0:
        raise ValueError("--index must be non-negative.")
    for index, source in iter_source_examples(args.source_file):
        if index == args.index:
            return source
    raise IndexError(f"Index {args.index} is outside the source file.")


def main() -> None:
    args = _parse_args()
    settings = Settings(_env_file=args.env_file)
    runtime = build_runtime(settings)
    source = _read_selected_source(args)
    example_id = args.example_id or f"example-{args.index:04d}"

    final_state = runtime.graph.invoke(
        runtime.initial_state(example_id, source),
        config={"recursion_limit": 30},
    )
    output = build_run_output(final_state, include_source=args.include_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved: {args.output}")
    print(f"Stop reason: {output.stop_reason.value}")
    print(f"Final summary: {output.final_summary}")


if __name__ == "__main__":
    main()
