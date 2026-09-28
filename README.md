# MentSum Agentic Workflow

The framework uses four functional agents with distinct responsibilities, aiming to generate a concise summary that captures domain-relevant information while remaining faithful to the source and preserving the meaning of safety-sensitive content.

1. The Guidance Extraction Agent selects source-grounded, domain-aware sentences;
2. The Guided Summarization Agent uses these sentences as guidance to generate a task-adapted initial summary;
3. The Verification Agent then assesses the summary for faithfulness and safety consistency and identify actionable issues;
4. the Revision Agent proposes targeted edits, and decides whether to accept the revision after re-verification.

The workflow is intended for research and prototyping. It is not a clinical system, diagnostic tool, or substitute for professional judgment.

## Requirements

- Python 3.10, 3.11, or 3.12
- An API key for an OpenAI-compatible endpoint when using a real model

## Quick start

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/YOUR_ACCOUNT/mentsum-agentic-workflow.git
cd mentsum-agentic-workflow
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
cp .env.example .env
```

The default configuration uses a deterministic mock model, so the complete graph can be checked without network access or an API key:

```bash
python -m mentsum_agent.run_one \
  --source-file examples/example.source \
  --index 0 \
  --include-source \
  --env-file .env.example \
  --output outputs/example.json
```

Run all synthetic examples as a batch:

```bash
python -m mentsum_agent.run_batch \
  --source-file examples/example.source \
  --include-source \
  --env-file .env.example \
  --id-prefix example \
  --output outputs/example.jsonl
```

Batch output is appended one record at a time. If the command is restarted, completed example IDs are skipped.

## Use a real model

Edit `.env` and set at least:

```dotenv
LLM_BACKEND=openai
LLM_API_KEY=your-api-key
LLM_MODEL=your-model-id
```

Set `LLM_BASE_URL` when using another OpenAI-compatible endpoint. Some models do not support JSON response mode; use `LLM_USE_JSON_MODE=false` for those models. Never commit `.env` or API keys.

Start with a small input file before running a large collection:

```bash
python -m mentsum_agent.run_batch \
  --source-file data/your_data.source \
  --limit 5 \
  --output outputs/pilot.jsonl
```

## Input and output

Input is a UTF-8 text file with one complete source post per line. Blank lines are treated as examples, so remove them unless they are intentional. See [`examples/example.source`](examples/example.source) for synthetic input.

`run_one` writes formatted JSON. `run_batch` writes JSON Lines, with one result per source line. Each successful result includes the source hash, extracted guidance, initial and final summaries, verification and revision histories, stopping reason, model ID, prompt versions, and public workflow configuration. See [`examples/example_output.jsonl`](examples/example_output.jsonl) for mock output.

Real datasets and experiment outputs are intentionally excluded from this repository. Local files under `data/`, `outputs/`, and `outputs_test/` are ignored by Git.

## Optional backends

The default matcher uses the small transparent lexicon in `resources/`. QuickUMLS can be installed with:

```bash
pip install -e ".[quickumls]"
```

Then configure `TERM_MATCHER=quickumls` and the `QUICKUMLS_*` values in `.env`. A local UMLS installation and appropriate license are required.

An external GSum-compatible process can be selected with `SUMMARIZER_BACKEND=external_gsum` and `GSUM_COMMAND`. The command receives one JSON object on standard input and must return either summary text or `{"summary": "..."}`.

## Repository layout

```text
examples/             Synthetic input and mock output
resources/            Small demonstration lexicons
src/mentsum_agent/    Workflow implementation
.env.example          Safe configuration template
pyproject.toml        Package metadata and dependencies
```

## Privacy and limitations

- Treat source posts and generated outputs as potentially sensitive data.
- Review files before sharing them, even when they are de-identified.
- The included lexicons are demonstrations, not validated clinical resources.
- Model output can be incomplete, incorrect, or unsafe and requires independent evaluation.
- The synthetic examples are fictional and do not represent real people.

## License

This project is released under the [MIT License](LICENSE).
