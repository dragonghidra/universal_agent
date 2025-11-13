# Erosolar

Erosolar packages the Universal Agent CLI so it can be installed from PyPI and launched with the `erosolar` command. This is a **comprehensive, Claude Code-like CLI agent** capable of accomplishing any task through iterative self-improvement, persistent file system changes, advanced code operations, Git integration, and multi-run learning.

## Features

### Core Agent Capabilities
- **Self-Improvement & Iterative Execution**: Agent can analyze task results, identify failures, store learnings, and automatically iterate until success.
- **LangGraph-based planner/executor loop** with LangChain + DeepSeek for reasoning.
- **Dynamic tool retrieval**: Automatically selects the best tools for each task from a comprehensive toolkit.
- **Interactive terminal UI** plus lightweight REST API (`/chat`) for programmatic control.

### Comprehensive Tool Suite

#### 🌐 Web & Search
- **Tavily search/extract**: Comprehensive web research with citations
- **DuckDuckGo search**: Quick web lookups
- **Weather API**: Current conditions for any location
- **Web scraping**: Headless browser automation with Playwright

#### 💻 Code Execution
- **Python execution**: Run arbitrary Python code with timeout control
- **Shell commands**: Full system access via bash with working directory support
- **Script automation**: Save and execute reusable shell scripts

#### 📁 File Operations (Claude Code-like)
- **Basic ops**: list_directory, read_text, write_text
- **Advanced editing**: `edit_file` - Precise find/replace without rewriting entire files
- **Pattern matching**: `glob_files` - Find files by pattern (e.g., `**/*.py`, `src/**/*.ts`)
- **Content search**: `grep_files` - Regex search across files with context lines and filtering
- **Jupyter notebooks**: `edit_notebook_cell` - Modify .ipynb cells programmatically

#### 🔧 Git Operations
- **git_status**: Show working tree status and changed files
- **git_diff**: View actual code changes (staged or unstaged)
- **git_commit**: Create commits with automatic staging
- **git_log**: View commit history

#### 🧪 Code Analysis & Quality
- **analyze_code_quality**: Detect long lines, count code vs comments, find potential issues
- **Static analysis**: Language-specific issue detection (Python, JavaScript, etc.)
- **Metrics**: LOC, comment ratio, complexity indicators

#### 📝 Task Management
- **manage_todos**: Track multi-step workflows with add/update/list/clear operations
- **Status tracking**: pending, in_progress, completed states
- **Workflow visibility**: See what the agent is working on

#### 🗄️ Persistent Storage
- **tool_library**: Create, update, run, delete custom shell/Python tools (SQLite-backed)
- **research_vault**: Store and recall notes across sessions with namespace organization
- **self_improve**: Analyze results, iterate on solutions, store learnings for future tasks

#### 🌍 Browser Automation
- **headless_browse**: Full Playwright integration for form filling, account creation, screenshots, JavaScript evaluation
- **Multi-browser**: Chromium, Firefox, WebKit support
- **Device emulation**: Mobile and tablet testing

#### 🔗 External Integration
- **MCP bridge**: Connect to Model Context Protocol servers
- **Custom tools**: Create domain-specific tools on the fly
- **API integration**: Wrap any external API as a reusable tool

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

### Persistent tools & research vault

- Invoke the `tool_library` tool from the agent to create new capabilities (shell or Python), update them, run them with structured arguments, or remove them entirely. Entries are stored in an on-disk SQLite database so they survive across runs.
- Use the `research_vault` tool to write, append, list, or delete research notes scoped by namespace/project. This is useful for multi-day investigations that need durable memory.
- Storage defaults to `<repo>/.agent_state/agent_state.sqlite3`. Override the location with `AGENT_STATE_DIR` (directory) and/or `AGENT_STATE_DB` (full file path) if you prefer a shared or cloud-synced location.

## Universal Task Capabilities

### 🎯 Advanced File Operations (Claude Code-like)

The agent can perform precise, surgical file edits without rewriting entire files:

```bash
erosolar
> Find all Python files with 'DEBUG = True', change them to 'DEBUG = False', and show me what was changed
```

The agent will:
1. Use `glob_files(pattern='**/*.py')` to find all Python files
2. Use `grep_files(pattern='DEBUG = True', file_pattern='*.py')` to identify affected files
3. Use `edit_file()` for each file to make precise replacements
4. Use `git_diff()` to show changes

```bash
> Search the codebase for all uses of the deprecated 'old_api' function and replace with 'new_api'
```

### 🔍 Code Analysis & Refactoring

Analyze code quality and make improvements:

```bash
> Analyze the code quality of src/main.py and fix any issues you find
```

The agent will:
1. Use `analyze_code_quality(file_path='src/main.py')` to find issues
2. Identify long lines, debug statements, bare excepts, etc.
3. Use `edit_file()` to fix each issue
4. Verify fixes with another quality check

### 🌳 Git Workflow Integration

Work with Git repositories seamlessly:

```bash
> Show me what files have changed, review the diff, and if it looks good, commit with message 'Refactor authentication module'
```

The agent will:
1. Use `git_status()` to show changed files
2. Use `git_diff()` to display actual code changes
3. Use `git_commit(message='...', add_all=True)` to create commit
4. Use `git_log()` to confirm the commit

```bash
> Create a new feature branch, make these code changes, commit them, and show me the commit history
```

### 📝 Task Management & Workflow Tracking

Track complex multi-step workflows:

```bash
> I need to refactor the authentication system. Create a todo list for this, then work through it step by step
```

The agent will:
1. Use `manage_todos(action='add')` to create subtasks
2. Mark each as `in_progress` when working on it
3. Mark as `completed` when done
4. Use `manage_todos(action='list')` to show progress

### 🧪 Jupyter Notebook Editing

Programmatically modify Jupyter notebooks:

```bash
> In analysis.ipynb, update cell 0 to import pandas and numpy, and cell 1 to load data.csv
```

The agent will:
1. Use `edit_notebook_cell(notebook_path='analysis.ipynb', cell_index=0, new_source='import pandas...')`
2. Update multiple cells as needed
3. Preserve outputs and metadata

### 🔄 Self-Improvement & Iterative Execution

The agent can automatically improve itself through multiple runs:

```bash
> Create a web scraper for product prices from example.com, and if it fails, analyze the issue and retry with improvements until it works
```

The agent will:
1. Attempt the task
2. If it fails, use `self_improve(action='analyze')` to identify issues
3. Use `self_improve(action='iterate')` to generate improvements
4. Retry with the improved approach
5. Store successful patterns with `self_improve(action='store_learning')` for future use

### 🌐 Browser Automation & Account Management

Use Playwright for complex web interactions:

```bash
> Go to example.com, create a new account with email test@example.com, fill in the signup form, and take a screenshot
```

The agent will:
1. Use `headless_browse` with Playwright
2. Navigate to the site
3. Fill forms, click buttons, handle JavaScript
4. Create accounts or perform any web actions
5. Capture screenshots or HTML as needed

### 💻 Code Generation & Execution

Generate and run code in Python:

```bash
> Write Python code to analyze the CSV file sales.csv, calculate total revenue by product, and create a bar chart
```

The agent will:
1. Use `run_python` to execute multi-step code
2. Install required libraries if needed (via shell)
3. Process data and generate outputs
4. Show results or save files

## Advanced Examples

### Example 1: Complete Refactoring Workflow
```bash
> Find all occurrences of the old API pattern, refactor them to use the new pattern, analyze code quality, run tests, and commit if they pass
```

### Example 2: Code Review & Quality Check
```bash
> Analyze all Python files in src/ for quality issues, create a todo list of fixes, then work through fixing each issue
```

### Example 3: Git-Integrated Development
```bash
> Create a feature branch 'add-logging', add logging statements to all functions in utils.py, commit the changes, and show me the diff
```

### Example 4: Multi-Step Data Pipeline
```bash
> Download data from api.example.com, clean it with Python, analyze trends, create visualizations, commit the notebook, and store findings in research vault
```

### Example 5: Automated Testing & Documentation
```bash
> Test the login flow at myapp.com/login with various inputs, document bugs in research vault, create a todo list of fixes, and analyze the codebase to understand the auth system
```

### Example 6: Codebase Exploration & Modification
```bash
> Search the codebase for all database query functions, analyze their performance patterns, suggest optimizations, and apply the changes
```

## Development

```bash
python -m pip install -U pip build twine
python -m build
python -m twine upload dist/*
```

Set `TWINE_USERNAME=__token__` and `TWINE_PASSWORD` to a valid PyPI API token (recommended via environment variables or CI secrets). GitHub Actions users can re-use the included workflow (`.github/workflows/workflow.yml`) and store the token in `PYPI_API_TOKEN`.
