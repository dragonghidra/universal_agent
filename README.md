# Erosolar

Erosolar packages the Universal Agent CLI so it can be installed from PyPI and launched with the `erosolar` command.

## Features

- LangGraph-based planner/executor loop with LangChain + DeepSeek for reasoning.
- Built-in research tools (Tavily search/extract, HTTP fetch, weather, shell, python).
- Optional MCP bridge so the agent can talk with Model Context Protocol servers.
- Interactive terminal UI plus lightweight REST hook (`/chat`) for programmatic prompts.

## Installation

```bash
pip install erosolar
```

## Usage

1. Export the required API keys:
   - `DEEPSEEK_API_KEY` – DeepSeek Reasoner access.
   - `TAVILY_API_KEY` – Tavily search + extract.
   - Optional: `OPENAI_API_KEY`, `TAVILY_API_BASE`, etc. (see `universal_agent.py` for the full list).
2. (Optional) Provide MCP configuration by setting `AGENT_MCP_SERVERS` to JSON or by creating `mcp_servers.json` (copy `mcp_servers.sample.json`).
3. Run the CLI:

```bash
erosolar --verbose
```

Use `exit` or `quit` to leave the session. The agent also exposes `http://127.0.0.1:9000/chat` so you can `POST {"message": "..."}` while the CLI is running.

## Development

```bash
python -m pip install -U pip build twine
python -m build
python -m twine upload dist/*
```

Set `TWINE_USERNAME=__token__` and `TWINE_PASSWORD` to a valid PyPI API token (recommended via environment variables or CI secrets). GitHub Actions users can re-use the included workflow (`.github/workflows/workflow.yml`) and store the token in `PYPI_API_TOKEN`.
