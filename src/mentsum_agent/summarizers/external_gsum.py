from __future__ import annotations

import json
import shlex
import subprocess

from ..schemas import GuidanceItem
from .base import Summarizer


class ExternalGSumSummarizer(Summarizer):
    """Calls a GSum environment through a JSON stdin/stdout command contract."""

    def __init__(
        self,
        command: str,
        timeout_seconds: float = 300.0,
    ):
        if not command.strip():
            raise ValueError(
                "GSUM_COMMAND is required when SUMMARIZER_BACKEND=external_gsum."
            )
        self.command = shlex.split(command)
        self.timeout_seconds = timeout_seconds

    def summarize(self, source: str, guidance: list[GuidanceItem]) -> str:
        ordered_guidance = sorted(
            guidance,
            key=lambda item: (item.sentence_start, item.sentence_end),
        )
        payload = {
            "source": source,
            "guidance_sentences": [
                item.source_sentence for item in ordered_guidance
            ],
        }
        completed = subprocess.run(
            self.command,
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "External GSum command failed with exit code "
                f"{completed.returncode}: {completed.stderr.strip()}"
            )
        stdout = completed.stdout.strip()
        if not stdout:
            raise RuntimeError("External GSum command returned empty stdout.")
        try:
            result = json.loads(stdout)
        except json.JSONDecodeError:
            return stdout
        if not isinstance(result, dict) or not isinstance(result.get("summary"), str):
            raise RuntimeError(
                "External GSum stdout must be plain summary text or JSON with a "
                "string field named 'summary'."
            )
        return result["summary"].strip()
