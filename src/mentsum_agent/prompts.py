from __future__ import annotations

import json

from .schemas import GuidanceItem, RiskCandidateSentence, VerificationReport


PROMPT_VERSIONS = {
    "guidance_controller": "guidance-controller-v1",
    "summarizer": "guided-summarizer-v1",
    "faithfulness_checker": "faithfulness-checker-v3",
    "safety_checker": "safety-checker-v4",
    "revision_editor": "revision-editor-v6",
}


GUIDANCE_SYSTEM_PROMPT = """You are a guidance controller in a mental-health post summarization workflow.
Your task is to select source-grounded mental-health candidates.

Remove incidental, weak, or contextually irrelevant matches.
You may add an important mental-health term expressed through informal user language and missed by the matcher.
Normalize it conservatively, but do not invent a diagnosis, CUI, or source sentence.

Return selected candidate IDs in descending importance order and any additional informal expressions.
"""


SUMMARIZER_SYSTEM_PROMPT = """You are a guided summarizer in a mental-health post summarization workflow.
Your task is to generate a concise first-person summary using the guidance as salience signals for content selection.

Return only the summary, with no heading or explanation.
"""


FAITHFULNESS_SYSTEM_PROMPT = """You are a faithfulness verifier in a mental-health post summarization workflow.
Your task is to decompose the summary into atomic claims and evaluate each claim against the source.

Label each claim as `supported`, `partially_supported`, `unsupported`, or `contradicted`.
Use `partially_supported` only when a claim mixes supported and unsupported information, and `contradicted` only when the source explicitly conflicts with the claim.

Return an exact contiguous summary span for each claim and exact contiguous source evidence when available.
Copy all evidence verbatim without adding quotation marks or changing capitalization, apostrophes, spelling, or grammar.
"""


SAFETY_SYSTEM_PROMPT = """You are a safety-consistency verifier in a mental-health post summarization workflow.
Your task is to compare the full source with the summary and report concrete safety-critical issues.

- `risk_omission`: a material in-scope source risk has no corresponding representation in the summary.
- `risk_escalation`: the summary represents the same source risk but materially strengthens its certainty, status, severity, or immediacy.
- `risk_deescalation`: the summary represents the same source risk but materially weakens its certainty, status, severity, or immediacy.
- `clinical_overreach`: the summary overstates a diagnosis, treatment, or crisis advice beyond source support.

Scope risk checks to the provided risk candidates: source candidates for `risk_omission` and `risk_deescalation`, and summary candidates for `risk_escalation`.
Candidates are eligibility hard gates, not safety labels. Assess `clinical_overreach` over the full source and full summary.

Return exact verbatim evidence. Leave summary evidence empty only for `risk_omission`; otherwise return both summary and source evidence.
"""


REVISION_SYSTEM_PROMPT = """You are a conservative revision editor in a mental-health post summarization workflow.
Your task is to revise the current summary only to resolve the listed verified issues.

Preserve every non-target span verbatim.
Do not paraphrase, reorder, compress, expand, or stylistically improve the summary.
For an unsupported or contradicted span, replace or delete only that span.
For a risk omission, insert the shortest source-grounded clause at an exact anchor.

Return edit operations rather than a rewritten summary.
Return no_change when no safe local edit is possible.
"""


def guidance_user_prompt(
    source: str,
    sentences: list[dict],
    candidates: list[dict],
    max_guidance_items: int,
) -> str:
    return (
        f"Select guidance that can be grouped into at most {max_guidance_items} source sentences.\n\n"
        f"SOURCE:\n{source}\n\n"
        "SENTENCES:\n"
        f"{json.dumps(sentences, ensure_ascii=False, indent=2)}\n\n"
        "MATCHER CANDIDATES:\n"
        f"{json.dumps(candidates, ensure_ascii=False, indent=2)}"
    )


def summarizer_user_prompt(
    source: str,
    guidance: list[GuidanceItem],
    max_summary_words: int,
) -> str:
    guidance_payload = [
        {
            "source_sentence": item.source_sentence,
            "mental_health_terms": [term.normalized_term for term in item.terms],
        }
        for item in guidance
    ]
    return (
        f"Write one summary of at most {max_summary_words} words.\n\n"
        f"SOURCE:\n{source}\n\n"
        "GUIDANCE:\n"
        f"{json.dumps(guidance_payload, ensure_ascii=False, indent=2)}"
    )


def faithfulness_user_prompt(source: str, summary: str) -> str:
    return f"SOURCE:\n{source}\n\nSUMMARY:\n{summary}"


def safety_user_prompt(
    source: str,
    summary: str,
    source_risk_candidates: list[RiskCandidateSentence],
    summary_risk_candidates: list[RiskCandidateSentence],
) -> str:
    return (
        f"SOURCE:\n{source}\n\n"
        f"SUMMARY:\n{summary}\n\n"
        "SOURCE RISK-CANDIDATE SENTENCES:\n"
        f"{json.dumps([item.model_dump() for item in source_risk_candidates], ensure_ascii=False, indent=2)}\n\n"
        "SUMMARY RISK-CANDIDATE SENTENCES:\n"
        f"{json.dumps([item.model_dump() for item in summary_risk_candidates], ensure_ascii=False, indent=2)}"
    )


def revision_user_prompt(
    source: str,
    best_summary: str,
    report: VerificationReport,
) -> str:
    actionable = {
        "actionable_issue_ids": report.actionable_issue_ids,
        "claim_assessments": [
            item.model_dump()
            for item in report.claim_assessments
            if item.issue_id in report.actionable_issue_ids
        ],
        "safety_issues": [
            item.model_dump()
            for item in report.safety_issues
            if item.issue_id in report.actionable_issue_ids
        ],
    }
    return (
        f"SOURCE:\n{source}\n\n"
        f"CURRENT SUMMARY:\n{best_summary}\n\n"
        "VERIFICATION ISSUES:\n"
        f"{json.dumps(actionable, ensure_ascii=False, indent=2)}"
    )
