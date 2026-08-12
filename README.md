# Grounded Research Assistant

An AI research assistant for science that answers **only from real, peer-reviewed sources**,
**cites every claim**, **verifies its own answer**, and **refuses to guess** when the evidence is
not there. It is a small, working version of the pattern used in applied-AI science teams: bring
trusted content into a model through a gateway, and keep the model honest.

Built on real, open sources (OpenAlex + arXiv), Claude via Anthropic, an MCP gateway, a LangGraph
pipeline, and optional LangSmith tracing.

## See it work for free (no key needed)

The trusted-content layer needs no API key, so you can verify it in one command:

```bash
pip install -r requirements.txt
python -m src.cli --search-only "what limits lithium-ion cathode cycle life?"
```

That returns real papers with DOIs. To get a **grounded, cited answer**, add an Anthropic key
(see below) and drop `--search-only`.

## How it works

```
        question
           |
   [ retrieve ]   trusted-content layer: OpenAlex (peer-reviewed) + arXiv (preprints)
           |      (also exposed as an MCP gateway, see src/gateway_server.py)
   [ synthesize ] Claude answers using ONLY those sources, citing each claim [n]
           |
   [ verify ]     Claude checks its own answer against the sources (responsible-AI step)
           |
   grounded, cited, self-checked answer      (+ optional LangSmith trace)
```

If retrieval finds nothing, the assistant declines rather than inventing an answer. That refusal
is a feature.

## Run it

```bash
cp .env.example .env          # then add your ANTHROPIC_API_KEY
export ANTHROPIC_API_KEY=sk-ant-...   # or rely on .env

# Full grounded answer from the command line:
python -m src.cli "what are the main mechanisms limiting lithium-ion cathode cycle life?"

# A web UI (bring your own key, never stored):
python app.py

# The AI Gateway as a standalone MCP server (register it in any MCP client):
python -m src.gateway_server
```

## Project layout

```
grounded-research-assistant/
├── src/
│   ├── content.py          # trusted-content layer (OpenAlex + arXiv), fail-soft + cached
│   ├── gateway_server.py   # the AI Gateway as an MCP server (FastMCP)
│   ├── agent.py            # LangGraph pipeline: retrieve -> synthesize -> verify
│   ├── config.py           # model + retrieval settings, keys, logging
│   └── cli.py              # command line (incl. keyless --search-only)
├── app.py                  # Gradio UI, bring-your-own-key
├── grounded_research_assistant.ipynb   # the annotated, teach-along build
├── requirements.txt
└── .env.example
```

## Responsible-AI features

- **Grounded**: the answer uses only the retrieved sources, never the model's own memory.
- **Cited**: every claim carries a `[n]` pointing at a real paper with a DOI.
- **Refuses**: no sources means no answer, instead of a confident guess.
- **Self-verifying**: a second pass lists any statement not supported by the sources.
- **Auditable**: with a LangSmith key, every step, source, and model call is traced.

## Tech stack

| Piece | What it does |
|---|---|
| OpenAlex, arXiv | Real peer-reviewed works and preprints (the trusted content) |
| requests | Fetches that data, with timeouts and fail-soft handling |
| MCP + FastMCP | The gateway: one standard endpoint any AI client can reach |
| Claude (Anthropic) | Reads the sources and writes the cited answer |
| LangChain, LangGraph | The retrieve to synthesize to verify workflow |
| LangSmith | Optional audit trail of every step |

## Honest limits

- OpenAlex and arXiv stand in for a licensed, curated corpus; the pattern is identical, the
  content source would change.
- Preprints are labelled as not-yet-peer-reviewed.
- The self-verifier is a strong first line, not a proof. A production deployment would add human
  review for high-stakes answers.
- Answer quality depends on what the sources actually contain; when they are thin, the assistant
  says so rather than filling the gap.

## About

Built by Ibtisam Ahmed Khan, a materials engineer working in data and AI.
[materialsdecoded.com](https://materialsdecoded.com) ·
[GitHub](https://github.com/ibtisamkhan96) ·
[LinkedIn](https://www.linkedin.com/in/ibtisam-ahmed-khan)
