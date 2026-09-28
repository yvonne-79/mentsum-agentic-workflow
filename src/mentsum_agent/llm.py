from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from typing import TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel, ValidationError

from .config import Settings
from .schemas import (
    FaithfulnessDraft,
    GuidanceControllerOutput,
    RevisionOutput,
    SafetyDraft,
)

T = TypeVar("T", bound=BaseModel)


_USAGE_KEYS = (
    "api_calls_attempted",
    "api_calls_succeeded",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)


def empty_usage_snapshot() -> dict[str, int]:
    return {key: 0 for key in _USAGE_KEYS}


def usage_delta(
    before: dict[str, int],
    after: dict[str, int],
) -> dict[str, int]:
    return {
        key: max(0, int(after.get(key, 0)) - int(before.get(key, 0)))
        for key in _USAGE_KEYS
    }


def _openrouter_provider_extra_body(settings: Settings) -> dict | None:
    if not settings.llm_base_url:
        return None
    hostname = (urlparse(settings.llm_base_url).hostname or "").casefold()
    if hostname not in {"openrouter.ai", "www.openrouter.ai"}:
        return None
    provider: dict = {
        "allow_fallbacks": settings.openrouter_allow_fallbacks,
    }
    if settings.openrouter_provider:
        provider["only"] = [settings.openrouter_provider]
    return {"provider": provider}


class LLMClient(ABC):
    model_id: str

    def usage_snapshot(self) -> dict[str, int]:
        return empty_usage_snapshot()

    @abstractmethod
    def complete_text(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
    ) -> T:
        raise NotImplementedError


def _extract_json_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
        if not isinstance(value, dict):
            raise ValueError("The model output is not a JSON object.")
        return value
    except json.JSONDecodeError:
        first = stripped.find("{")
        last = stripped.rfind("}")
        if first < 0 or last <= first:
            raise ValueError("No JSON object was found in the model output.")
        value = json.loads(stripped[first : last + 1])
        if not isinstance(value, dict):
            raise ValueError("The model output is not a JSON object.")
        return value


class OpenAICompatibleLLM(LLMClient):
    def __init__(self, settings: Settings):
        if not settings.llm_api_key:
            raise ValueError("LLM_API_KEY is required when LLM_BACKEND=openai.")
        if settings.llm_model == "replace-with-model-id":
            raise ValueError("Set LLM_MODEL to a valid model identifier.")

        from openai import OpenAI

        client_kwargs: dict = {
            "api_key": settings.llm_api_key,
            "timeout": settings.llm_timeout_seconds,
        }
        if settings.llm_base_url:
            client_kwargs["base_url"] = settings.llm_base_url

        self.client = OpenAI(**client_kwargs)
        self.model_id = settings.llm_model
        self.temperature = settings.llm_temperature
        self.use_json_mode = settings.llm_use_json_mode
        self.max_retries = settings.max_llm_retries
        self.provider_extra_body = _openrouter_provider_extra_body(settings)
        self._usage = empty_usage_snapshot()

    def usage_snapshot(self) -> dict[str, int]:
        return dict(self._usage)

    def _chat(self, messages: list[dict[str, str]], json_mode: bool = False) -> str:
        kwargs: dict = {
            "model": self.model_id,
            "messages": messages,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.provider_extra_body is not None:
            kwargs["extra_body"] = self.provider_extra_body

        if not hasattr(self, "_usage"):
            self._usage = empty_usage_snapshot()
        self._usage["api_calls_attempted"] += 1
        completion = self.client.chat.completions.create(**kwargs)
        self._usage["api_calls_succeeded"] += 1
        usage = getattr(completion, "usage", None)
        if usage is not None:
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(
                getattr(usage, "completion_tokens", 0) or 0
            )
            total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
            self._usage["prompt_tokens"] += prompt_tokens
            self._usage["completion_tokens"] += completion_tokens
            self._usage["total_tokens"] += (
                total_tokens or prompt_tokens + completion_tokens
            )
        content = completion.choices[0].message.content
        if content is None:
            raise RuntimeError("The model returned an empty message.")
        return content

    def complete_text(self, system_prompt: str, user_prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._chat(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                ).strip()
            except Exception as exc:  # Provider SDKs expose different error classes.
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError("Text generation failed after retries.") from last_error

    def complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
    ) -> T:
        schema_text = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        repair_note = ""
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            effective_system = (
                f"{system_prompt}\n\n"
                "Return one JSON object only. It must match this JSON Schema:\n"
                f"{schema_text}{repair_note}"
            )
            messages = [
                {"role": "system", "content": effective_system},
                {"role": "user", "content": user_prompt},
            ]
            try:
                try:
                    raw = self._chat(messages, json_mode=self.use_json_mode)
                except Exception:
                    if not self.use_json_mode:
                        raise
                    raw = self._chat(messages, json_mode=False)
                payload = _extract_json_object(raw)
                return schema.model_validate(payload)
            except (ValueError, ValidationError, json.JSONDecodeError) as exc:
                last_error = exc
                repair_note = (
                    "\nThe previous output failed validation. Correct the JSON without "
                    f"adding commentary. Validation error: {exc}"
                )
            except Exception as exc:
                last_error = exc

            if attempt < self.max_retries:
                time.sleep(min(2 ** (attempt - 1), 8))

        raise RuntimeError("Structured generation failed after retries.") from last_error


class MockLLM(LLMClient):
    """A deterministic backend for graph smoke tests only."""

    model_id = "mock-llm"

    def complete_text(self, system_prompt: str, user_prompt: str) -> str:
        source_match = re.search(
            r"SOURCE:\n(.*?)(?:\n\nGUIDANCE:|\Z)", user_prompt, flags=re.DOTALL
        )
        source = source_match.group(1).strip() if source_match else user_prompt.strip()
        first_sentence = re.split(r"(?<=[.!?])\s+|\n+", source, maxsplit=1)[0]
        return first_sentence.strip() or "No summary generated."

    def complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
    ) -> T:
        if schema is GuidanceControllerOutput:
            candidate_ids = re.findall(r'"candidate_id"\s*:\s*"([^"]+)"', user_prompt)
            payload = {
                "selected_candidate_ids": candidate_ids[:10],
                "additional_mentions": [],
            }
            return schema.model_validate(payload)

        if schema is FaithfulnessDraft:
            source_match = re.search(
                r"SOURCE:\n(.*?)(?:\n\nSUMMARY:|\Z)",
                user_prompt,
                flags=re.DOTALL,
            )
            summary_match = re.search(
                r"SUMMARY:\n(.*)\Z",
                user_prompt,
                flags=re.DOTALL,
            )
            source = source_match.group(1).strip() if source_match else ""
            summary = summary_match.group(1).strip() if summary_match else ""
            supported = bool(summary and summary in source)
            claim = {
                "summary_claim": summary or "No summary generated.",
                "summary_evidence": summary or "No summary generated.",
                "label": "supported" if supported else "unsupported",
                "source_evidence": [summary] if supported else [],
                "correction": "",
            }
            payload = {
                "claim_assessments": [claim],
            }
            return schema.model_validate(payload)

        if schema is SafetyDraft:
            return schema.model_validate({"safety_issues": []})

        if schema is RevisionOutput:
            return schema.model_validate({"edits": [], "no_change": True})

        raise TypeError(f"MockLLM does not support schema {schema.__name__}.")


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm_backend == "mock":
        return MockLLM()
    return OpenAICompatibleLLM(settings)
