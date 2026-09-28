from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .schemas import (
    LLMBackend,
    OpenRouterProvider,
    SummaryBackend,
    TermMatcherBackend,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    llm_backend: LLMBackend = "mock"
    llm_api_key: str = ""
    llm_base_url: str | None = None
    llm_model: str = "replace-with-model-id"
    llm_temperature: float | None = 0.0
    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    llm_use_json_mode: bool = True
    max_llm_retries: int = Field(default=3, ge=1, le=10)
    openrouter_provider: OpenRouterProvider | None = "openai"
    openrouter_allow_fallbacks: bool = False

    term_matcher: TermMatcherBackend = "lexicon"
    lexicon_path: Path = Path("resources/demo_mental_health_lexicon.json")
    quickumls_path: Path | None = None
    quickumls_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    quickumls_accepted_semtypes: str = ""
    quickumls_allowed_cuis_path: Path | None = None
    quickumls_cui_labels_path: Path | None = None
    max_match_candidates: int = Field(default=80, ge=1, le=500)

    risk_cue_path: Path | None = Path("resources/demo_critical_risk_cues.json")

    summarizer_backend: SummaryBackend = "llm"
    gsum_command: str = ""
    gsum_timeout_seconds: float = Field(default=300.0, gt=0)

    max_guidance_items: int = Field(default=10, ge=1, le=30)
    max_summary_words: int = Field(default=80, ge=10, le=300)
    max_revision_rounds: int = Field(default=2, ge=0, le=5)

    @field_validator("llm_base_url", mode="before")
    @classmethod
    def empty_base_url_to_none(cls, value: object) -> object:
        if value == "":
            return None
        return value

    @field_validator("openrouter_provider", mode="before")
    @classmethod
    def empty_provider_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("openrouter_allow_fallbacks")
    @classmethod
    def openrouter_fallbacks_must_remain_disabled(cls, value: bool) -> bool:
        if value:
            raise ValueError("OpenRouter provider fallbacks must remain disabled.")
        return value

    @field_validator(
        "quickumls_path",
        "quickumls_allowed_cuis_path",
        "quickumls_cui_labels_path",
        "risk_cue_path",
        mode="before",
    )
    @classmethod
    def empty_path_to_none(cls, value: object) -> object:
        if value == "":
            return None
        return value

    def accepted_semtypes(self) -> set[str]:
        return {
            item.strip()
            for item in self.quickumls_accepted_semtypes.split(",")
            if item.strip()
        }

    def public_workflow_config(self) -> dict[str, object]:
        config: dict[str, object] = {
            "llm_backend": self.llm_backend,
            "llm_base_url": self.llm_base_url,
            "openrouter_provider": self.openrouter_provider,
            "openrouter_allow_fallbacks": self.openrouter_allow_fallbacks,
            "term_matcher": self.term_matcher,
            "quickumls_threshold": self.quickumls_threshold,
            "max_match_candidates": self.max_match_candidates,
            "risk_cues_enabled": self.risk_cue_path is not None,
            "summarizer_backend": self.summarizer_backend,
            "max_guidance_items": self.max_guidance_items,
            "max_revision_rounds": self.max_revision_rounds,
        }
        if self.summarizer_backend == "llm":
            config["max_summary_words"] = self.max_summary_words
        return config
