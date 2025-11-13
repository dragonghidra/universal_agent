# Complete Toolset Reference

This document lists all available tools in the Erosolar agent with their signatures, parameters, and usage examples.

## 🌐 Web & Search Tools

### `tavily_search`
Comprehensive web search using Tavily API with citations and source documents.

**Signature:**
```python
tavily_search(
    query: str,
    max_results: int = 5,
    search_depth: Literal["basic", "advanced"] = "basic"
) -> str
```

**Example:**
```python
tavily_search(query="latest developments in quantum computing", max_results=10, search_depth="advanced")
```

### `tavily_extract`
Extract readable content from a specific URL.

**Signature:**
```python
tavily_extract(url: str) -> str
```

**Example:**
```python
tavily_extract(url="https://arxiv.org/abs/2301.00000")
```

### `web_search_simple`
Quick web search using DuckDuckGo's instant answer API.

**Signature:**
```python
web_search_simple(query: str, max_results: int = 5) -> str
```

**Example:**
```python
web_search_simple(query="python asyncio tutorial", max_results=5)
```

### `get_weather`
Get current weather conditions for any location using Open-Meteo API.

**Signature:**
```python
get_weather(location: str, units: Literal["us", "metric"] = "us") -> str
```

**Example:**
```python
get_weather(location="Tokyo, Japan", units="metric")
```

## 💻 Code Execution Tools

### `run_python`
Execute arbitrary Python code in a fresh interpreter and capture output.

**Signature:**
```python
run_python(code: str, timeout: int = 60) -> str
```

**Example:**
```python
run_python(code="""
import pandas as pd
data = pd.read_csv('sales.csv')
print(data.groupby('product')['revenue'].sum())
""", timeout=30)
```

### `run_shell`
Execute shell commands with bash and capture stdout/stderr.

**Signature:**
```python
run_shell(command: str, timeout: int = 60, cwd: Optional[str] = None) -> str
```

**Example:**
```python
run_shell(command="find . -name '*.py' | head -10", timeout=10)
run_shell(command="npm install", cwd="/path/to/project", timeout=120)
```

## 📁 File Operations - Basic

### `list_directory`
List files and folders at a path (non-recursive).

**Signature:**
```python
list_directory(path: str = ".") -> str
```

**Example:**
```python
list_directory(path="~/projects/myapp/src")
```

### `read_text`
Read up to max_chars from a UTF-8 text file.

**Signature:**
```python
read_text(path: str, max_chars: int = 8000) -> str
```

**Example:**
```python
read_text(path="config.json", max_chars=4000)
```

### `write_text`
Write content to a file, creating parent directories as needed.

**Signature:**
```python
write_text(
    path: str,
    content: str,
    mode: Literal["overwrite", "append"] = "overwrite"
) -> str
```

**Example:**
```python
write_text(path="logs/output.txt", content="Log entry\n", mode="append")
```

## 📝 File Operations - Advanced (Claude Code-like)

### `edit_file`
Perform exact string replacement in a file. **Safer than overwriting entire files.**

**Signature:**
```python
edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False
) -> str
```

**Example:**
```python
# Replace a single occurrence
edit_file(
    file_path="config.py",
    old_string="DEBUG = True",
    new_string="DEBUG = False"
)

# Replace all occurrences
edit_file(
    file_path="main.py",
    old_string="old_api.call()",
    new_string="new_api.call()",
    replace_all=True
)
```

### `glob_files`
Find files matching a glob pattern (recursive pattern matching).

**Signature:**
```python
glob_files(pattern: str, path: str = ".") -> str
```

**Example:**
```python
# Find all Python files
glob_files(pattern="**/*.py", path="src")

# Find all TypeScript test files
glob_files(pattern="**/*.test.ts", path=".")

# Find all Jupyter notebooks
glob_files(pattern="**/*.ipynb", path="notebooks")
```

### `grep_files`
Search for regex pattern in files (ripgrep-like functionality).

**Signature:**
```python
grep_files(
    pattern: str,
    path: str = ".",
    file_pattern: Optional[str] = None,
    context_lines: int = 0,
    case_insensitive: bool = False,
    output_mode: Literal["files", "matches", "count"] = "files",
    max_results: int = 100
) -> str
```

**Example:**
```python
# Find files containing import statements
grep_files(
    pattern="import.*pandas",
    file_pattern="*.py",
    output_mode="files"
)

# Show actual matching lines with context
grep_files(
    pattern="def process_.*\(",
    file_pattern="*.py",
    output_mode="matches",
    context_lines=2
)

# Count matches per file
grep_files(
    pattern="TODO|FIXME",
    output_mode="count",
    case_insensitive=True
)
```

## 🔧 Git Operations

### `git_status`
Get git status showing working tree changes.

**Signature:**
```python
git_status(repo_path: str = ".") -> str
```

**Example:**
```python
git_status(repo_path="/path/to/repo")
```

### `git_diff`
Get git diff showing actual code changes.

**Signature:**
```python
git_diff(repo_path: str = ".", staged: bool = False) -> str
```

**Example:**
```python
# Show unstaged changes
git_diff(repo_path=".")

# Show staged changes
git_diff(repo_path=".", staged=True)
```

### `git_commit`
Create a git commit with optional automatic staging.

**Signature:**
```python
git_commit(
    message: str,
    repo_path: str = ".",
    add_all: bool = False
) -> str
```

**Example:**
```python
# Commit staged changes
git_commit(message="Fix authentication bug")

# Stage all changes and commit
git_commit(message="Refactor user module", add_all=True)
```

### `git_log`
View recent commit history.

**Signature:**
```python
git_log(repo_path: str = ".", max_commits: int = 10) -> str
```

**Example:**
```python
git_log(repo_path=".", max_commits=20)
```

## 🤖 Automation & Scripts

### `save_shell_automation`
Persist a reusable shell script and optionally execute it.

**Signature:**
```python
save_shell_automation(
    name: str,
    content: str,
    run: bool = False,
    timeout: int = 120
) -> str
```

**Example:**
```python
save_shell_automation(
    name="backup_project",
    content="""#!/bin/bash
tar -czf backup-$(date +%Y%m%d).tar.gz src/
echo "Backup complete"
""",
    run=True,
    timeout=60
)
```

## 🌍 Browser Automation

### `headless_browse`
Use Playwright to drive a headless browser with full automation capabilities.

**Signature:**
```python
headless_browse(
    url: str,
    wait_selector: Optional[str] = None,
    wait_seconds: Optional[int] = None,
    screenshot_path: Optional[str] = None,
    save_html_path: Optional[str] = None,
    javascript: Optional[str] = None,
    browser: Literal["chromium", "firefox", "webkit"] = "chromium",
    emulate_device: Optional[str] = None,
    timeout: int = 45
) -> str
```

**Example:**
```python
# Basic page load and screenshot
headless_browse(
    url="https://example.com",
    screenshot_path="screenshot.png"
)

# Wait for content and run JavaScript
headless_browse(
    url="https://example.com/app",
    wait_selector="div.dashboard",
    javascript="document.querySelector('h1').textContent",
    save_html_path="page.html"
)

# Mobile device emulation
headless_browse(
    url="https://example.com",
    emulate_device="iPhone 12",
    screenshot_path="mobile.png"
)
```

## 🧪 Jupyter Notebooks

### `edit_notebook_cell`
Edit a specific cell in a Jupyter notebook (.ipynb file).

**Signature:**
```python
edit_notebook_cell(
    notebook_path: str,
    cell_index: int,
    new_source: str,
    cell_type: Literal["code", "markdown"] = "code"
) -> str
```

**Example:**
```python
# Update a code cell
edit_notebook_cell(
    notebook_path="analysis.ipynb",
    cell_index=0,
    new_source="import pandas as pd\nimport numpy as np",
    cell_type="code"
)

# Update a markdown cell
edit_notebook_cell(
    notebook_path="report.ipynb",
    cell_index=1,
    new_source="# Data Analysis Results\nThis section shows...",
    cell_type="markdown"
)
```

## 📊 Code Analysis

### `analyze_code_quality`
Analyze code quality metrics and detect potential issues.

**Signature:**
```python
analyze_code_quality(
    file_path: str,
    language: Optional[str] = None
) -> str
```

**Example:**
```python
# Auto-detect language from extension
analyze_code_quality(file_path="src/main.py")

# Explicit language specification
analyze_code_quality(file_path="script.js", language="javascript")
```

**Detects:**
- Total lines, code lines, blank lines, comment lines
- Long lines (>120 chars)
- Debug print statements
- Bare except clauses
- Other language-specific issues

## 📝 Task Management

### `manage_todos`
Manage todos/tasks for tracking agent workflow.

**Signature:**
```python
manage_todos(
    action: Literal["add", "update", "list", "clear_completed"],
    content: Optional[str] = None,
    todo_id: Optional[int] = None,
    status: Optional[str] = None,
    status_filter: Optional[str] = None
) -> str
```

**Example:**
```python
# Add a new todo
manage_todos(action="add", content="Refactor authentication module")

# Update todo status
manage_todos(action="update", todo_id=1, status="in_progress")

# List all todos
manage_todos(action="list")

# List only pending todos
manage_todos(action="list", status_filter="pending")

# Clear completed todos
manage_todos(action="clear_completed")
```

**Status values:** `pending`, `in_progress`, `completed`

## 🗄️ Persistent Storage

### `tool_library`
Create, update, run, or delete custom persistent tools backed by SQLite.

**Signature:**
```python
tool_library(request: ToolLibraryRequest) -> str

# ToolLibraryRequest fields:
# - action: "list" | "create" | "update" | "delete" | "show" | "run"
# - name: str (tool name)
# - description: str (what the tool does)
# - kind: "shell" | "python"
# - body: str (script content)
# - args_schema: Dict (JSON schema for arguments)
# - metadata: Dict (custom metadata)
# - timeout: int (execution timeout)
# - arguments: Dict (args to pass when running)
```

**Example:**
```python
# Create a custom Python tool
tool_library(
    action="create",
    name="analyze_csv",
    description="Analyze CSV file and return summary statistics",
    kind="python",
    body="""
import pandas as pd
df = pd.read_csv(params['file_path'])
print(df.describe())
""",
    args_schema={"type": "object", "properties": {"file_path": {"type": "string"}}}
)

# Run the tool
tool_library(action="run", name="analyze_csv", arguments={"file_path": "data.csv"})

# List all tools
tool_library(action="list")

# Delete a tool
tool_library(action="delete", name="analyze_csv")
```

### `research_vault`
Persist, retrieve, or append research notes across agent runs.

**Signature:**
```python
research_vault(request: ResearchVaultRequest) -> str

# ResearchVaultRequest fields:
# - action: "list" | "get" | "set" | "append" | "delete"
# - namespace: str (logical bucket, default "global")
# - key: str (unique key within namespace)
# - content: str (note body)
# - metadata: Dict (custom metadata)
```

**Example:**
```python
# Store a research note
research_vault(
    action="set",
    namespace="project-x",
    key="api-findings",
    content="Discovered that API v2 requires OAuth2...",
    metadata={"date": "2024-01-15", "priority": "high"}
)

# Append to existing note
research_vault(
    action="append",
    namespace="project-x",
    key="api-findings",
    content="\n\nUpdate: Found better authentication method..."
)

# Retrieve a note
research_vault(action="get", namespace="project-x", key="api-findings")

# List all notes in namespace
research_vault(action="list", namespace="project-x")

# Delete a note
research_vault(action="delete", namespace="project-x", key="api-findings")
```

### `self_improve`
Analyze task results, iterate on solutions, store learnings, and enable self-improvement.

**Signature:**
```python
self_improve(request: SelfImproveRequest) -> str

# SelfImproveRequest fields:
# - action: "analyze" | "iterate" | "store_learning" | "get_learnings"
# - task_description: str
# - result: str
# - criteria: str (success criteria)
# - learning: str
# - category: str (e.g., "web_scraping", "code_generation")
```

**Example:**
```python
# Analyze a task result
self_improve(
    action="analyze",
    task_description="Scrape product prices from e-commerce site",
    result="Got 404 error when accessing /products",
    criteria="Extract prices for all products in catalog"
)

# Get iteration guidance
self_improve(
    action="iterate",
    task_description="Retry web scraping with improvements",
    result="404 error",
    criteria="Extract all product prices",
    category="web_scraping"
)

# Store a learning
self_improve(
    action="store_learning",
    task_description="Web scraping best practices",
    learning="Always check robots.txt before scraping. Use rate limiting.",
    category="web_scraping"
)

# Retrieve past learnings
self_improve(action="get_learnings", category="web_scraping")
```

## 🎯 Complete Usage Workflows

### Workflow 1: Code Refactoring
```python
# 1. Find all files with old pattern
glob_files(pattern="**/*.py", path="src")

# 2. Search for specific pattern
grep_files(pattern="old_api\\.call", file_pattern="*.py", output_mode="matches")

# 3. Edit each file
edit_file(
    file_path="src/module.py",
    old_string="old_api.call(data)",
    new_string="new_api.call(data)",
    replace_all=True
)

# 4. Check quality
analyze_code_quality(file_path="src/module.py")

# 5. Review changes
git_diff()

# 6. Commit
git_commit(message="Migrate from old_api to new_api", add_all=True)
```

### Workflow 2: Research & Documentation
```python
# 1. Research topic
tavily_search(query="best practices for API rate limiting", max_results=10)

# 2. Extract detailed content
tavily_extract(url="https://example.com/article")

# 3. Store findings
research_vault(
    action="set",
    namespace="api-design",
    key="rate-limiting",
    content="Summary of best practices..."
)

# 4. Create implementation tool
tool_library(
    action="create",
    name="rate_limiter",
    description="Implement rate limiting middleware",
    kind="python",
    body="# Implementation code here"
)
```

### Workflow 3: Data Analysis Pipeline
```python
# 1. Create analysis notebook
edit_notebook_cell(
    notebook_path="analysis.ipynb",
    cell_index=0,
    new_source="import pandas as pd\ndf = pd.read_csv('data.csv')"
)

# 2. Run Python analysis
run_python(code="""
import pandas as pd
df = pd.read_csv('data.csv')
print(df.describe())
""")

# 3. Store results
research_vault(
    action="set",
    namespace="data-analysis",
    key="sales-summary",
    content="Analysis results..."
)

# 4. Commit notebook
git_commit(message="Add sales analysis notebook", add_all=True)
```

## 🔍 Tool Selection Tips

The agent uses **dynamic tool retrieval** based on your task description. Tools are automatically selected based on:

- **Task keywords**: Specific words trigger relevant tools (e.g., "git", "search", "edit")
- **Tool tags**: Each tool has semantic tags (e.g., git_status has tags: ["git", "vcs", "status"])
- **Risk level**: High-risk tools (shell, git_commit) require explicit task context
- **Sticky tools**: Common tools (tavily_search, read_text, glob_files) are always available

You don't need to specify tool names - just describe what you want naturally!
