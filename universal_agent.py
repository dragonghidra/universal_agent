# pip install -U langgraph langchain langchain-openai requests pydantic mcp
# export DEEPSEEK_API_KEY="sk-..."  # DeepSeek API key
# export TAVILY_API_KEY="tvly-..."  # Tavily API key
# Optional MCP config: set AGENT_MCP_SERVERS='[{"name":"...","command":"..."}]' or create mcp_servers.json

import argparse
import concurrent.futures  # parallel worker pool (kept for optional future use)
import json
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Literal, Optional, Sequence, Set
from urllib.parse import urlparse

import requests

import cli_ui
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool, tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from agent_toolkit import (
    MAX_TOOL_OUTPUT,
    get_weather as core_get_weather,
    headless_browse as core_headless_browse,
    list_directory as core_list_directory,
    read_text as core_read_text,
    run_python as core_run_python,
    run_shell as core_run_shell,
    save_shell_automation as core_save_shell_automation,
    write_text as core_write_text,
)
from enhanced_tools import (
    edit_file as core_edit_file,
    glob_files as core_glob_files,
    grep_files as core_grep_files,
    git_status as core_git_status,
    git_diff as core_git_diff,
    git_commit as core_git_commit,
    git_log as core_git_log,
    manage_todos as core_manage_todos,
    edit_notebook_cell as core_edit_notebook_cell,
    analyze_code_quality as core_analyze_code_quality,
    web_search_simple as core_web_search_simple,
)
from mcp_integration import load_mcp_tools
from persistent_tools import STORE as PERSISTENT_STORE
from tool_retrieval import Embedder, ScoredTool, ToolRecord, ToolRetriever


# --------- Real tools (they perform actual HTTP requests) ----------

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise RuntimeError("Set TAVILY_API_KEY in your environment before running this script.")

TAVILY_SEARCH_URL = os.environ.get("TAVILY_SEARCH_URL", "https://api.tavily.com/search")
TAVILY_EXTRACT_URL = os.environ.get("TAVILY_EXTRACT_URL", "https://api.tavily.com/extract")


def _tavily_post(endpoint: str, payload: Dict[str, Any], *, timeout: int = 30) -> Dict[str, Any]:
    """Send a POST request to a Tavily endpoint and return JSON."""
    body = dict(payload)
    body["api_key"] = TAVILY_API_KEY
    resp = requests.post(endpoint, json=body, timeout=timeout)
    resp.raise_for_status()
    try:
        return resp.json()
    except ValueError as exc:
        raise RuntimeError(f"Tavily returned invalid JSON for {endpoint}") from exc


def _format_search_results(results: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for idx, item in enumerate(results, 1):
        title = item.get("title") or "Untitled result"
        url = item.get("url") or "Unknown URL"
        content = (item.get("content") or item.get("snippet") or "").strip()
        lines.append(f"{idx}. {title}\n{url}")
        if content:
            lines.append(content)
    return "\n".join(lines).strip() or "No supporting documents were returned."


@tool
def tavily_search(
    query: str,
    *,
    max_results: int = 5,
    search_depth: Literal["basic", "advanced"] = "basic",
) -> str:
    """Use Tavily to search the web for fresh information about a topic."""
    bounded_results = max(1, min(int(max_results), 20))
    payload = {
        "query": query,
        "max_results": bounded_results,
        "search_depth": search_depth,
    }
    data = _tavily_post(TAVILY_SEARCH_URL, payload)
    answer = data.get("answer") or data.get("summary")
    results = data.get("results") or []
    formatted_results = _format_search_results(results)
    if answer:
        return f"{answer}\n\nSources:\n{formatted_results}"
    return f"No direct answer from Tavily.\n\nSources:\n{formatted_results}"


def _is_valid_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except Exception:  # pragma: no cover - urlparse rarely raises
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


@tool
def tavily_extract(url: str) -> str:
    """Extract readable text from a known HTTP(S) URL (e.g. from search results)."""
    cleaned = (url or "").strip()
    if not cleaned:
        return "tavily_extract needs a concrete URL to fetch."
    if not _is_valid_http_url(cleaned):
        return f"tavily_extract only accepts full http(s) URLs. Got: {cleaned}"

    payload = {"urls": [cleaned]}
    try:
        data = _tavily_post(TAVILY_EXTRACT_URL, payload, timeout=45)
    except requests.HTTPError as exc:
        return f"Tavily extract failed for {cleaned}: {exc}"
    except requests.RequestException as exc:  # safety for timeouts, etc.
        return f"Tavily extract request error for {cleaned}: {exc}"

    # Tavily may return content at the top level or inside results[0].
    result_block = None
    if isinstance(data, dict):
        result_block = data
        if not (data.get("content") or data.get("text")):
            results = data.get("results")
            if isinstance(results, list) and results:
                result_block = results[0]
    else:
        return f"Unexpected Tavily response for {cleaned}."

    content = (
        result_block.get("content")
        or result_block.get("text")
        or result_block.get("raw_content")
        or ""
    )
    if not content:
        return f"Tavily did not return any content for {cleaned}."

    metadata = result_block.get("metadata") or {}
    title = metadata.get("title") or result_block.get("title")
    header = f"{title}\n{cleaned}\n\n" if title else f"{cleaned}\n\n"
    return header + content.strip()


@tool
def get_weather(location: str, units: Literal["us", "metric"] = "us") -> str:
    """Get current weather for a place name using the Open-Meteo API."""
    return core_get_weather(location, units)


@tool
def run_python(code: str, timeout: int = 60) -> str:
    """Execute arbitrary Python code in a fresh interpreter process and capture stdout/stderr."""
    return core_run_python(code, timeout)


@tool
def run_shell(command: str, timeout: int = 60, cwd: Optional[str] = None) -> str:
    """Run a shell command with /bin/bash -lc and return stdout/stderr."""
    return core_run_shell(command, timeout=timeout, cwd=cwd)


@tool
def list_directory(path: str = ".") -> str:
    """List files/folders at a path (non-recursive)."""
    return core_list_directory(path)


@tool
def read_text(path: str, max_chars: int = 8000) -> str:
    """Read up to max_chars from a UTF-8 text file."""
    return core_read_text(path, max_chars=max_chars)


@tool
def write_text(path: str, content: str, mode: Literal["overwrite", "append"] = "overwrite") -> str:
    """Write content to a file, creating parent directories as needed."""
    return core_write_text(path, content, mode=mode)


@tool
def save_shell_automation(
    name: str,
    content: str,
    run: bool = False,
    timeout: int = 120,
) -> str:
    """Persist a reusable shell script under automation_scripts/ and optionally execute it."""
    return core_save_shell_automation(name, content, run=run, timeout=timeout)


@tool
def headless_browse(
    url: str,
    *,
    wait_selector: Optional[str] = None,
    wait_seconds: Optional[int] = None,
    screenshot_path: Optional[str] = None,
    save_html_path: Optional[str] = None,
    javascript: Optional[str] = None,
    browser: Literal["chromium", "firefox", "webkit"] = "chromium",
    emulate_device: Optional[str] = None,
    timeout: int = 45,
) -> str:
    """Use Playwright to load a page headlessly, wait for selectors, run JS, and capture output."""
    return core_headless_browse(
        url,
        wait_selector=wait_selector,
        wait_seconds=wait_seconds,
        screenshot_path=screenshot_path,
        save_html_path=save_html_path,
        javascript=javascript,
        browser=browser,
        emulate_device=emulate_device,
        timeout=timeout,
    )


# ==================== ENHANCED FILE OPERATIONS ====================

@tool
def edit_file(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Perform exact string replacement in a file. Safer than overwriting entire file."""
    return core_edit_file(file_path, old_string, new_string, replace_all)


@tool
def glob_files(pattern: str, path: str = ".") -> str:
    """Find files matching a glob pattern (e.g., '**/*.py', 'src/**/*.ts')."""
    return core_glob_files(pattern, path)


@tool
def grep_files(
    pattern: str,
    path: str = ".",
    *,
    file_pattern: Optional[str] = None,
    context_lines: int = 0,
    case_insensitive: bool = False,
    output_mode: Literal["files", "matches", "count"] = "files",
    max_results: int = 100,
) -> str:
    """Search for regex pattern in files (ripgrep-like). Supports filtering by file type."""
    return core_grep_files(
        pattern,
        path,
        file_pattern=file_pattern,
        context_lines=context_lines,
        case_insensitive=case_insensitive,
        output_mode=output_mode,
        max_results=max_results,
    )


# ==================== GIT OPERATIONS ====================

@tool
def git_status(repo_path: str = ".") -> str:
    """Get git status for a repository showing changed files."""
    return core_git_status(repo_path)


@tool
def git_diff(repo_path: str = ".", staged: bool = False) -> str:
    """Get git diff showing actual code changes. Use staged=True for staged changes."""
    return core_git_diff(repo_path, staged)


@tool
def git_commit(message: str, repo_path: str = ".", add_all: bool = False) -> str:
    """Create a git commit. Set add_all=True to stage all changes first."""
    return core_git_commit(message, repo_path, add_all)


@tool
def git_log(repo_path: str = ".", max_commits: int = 10) -> str:
    """Get recent git commit history."""
    return core_git_log(repo_path, max_commits)


# ==================== TASK MANAGEMENT ====================

@tool
def manage_todos(
    action: Literal["add", "update", "list", "clear_completed"],
    content: Optional[str] = None,
    todo_id: Optional[int] = None,
    status: Optional[str] = None,
    status_filter: Optional[str] = None,
) -> str:
    """Manage todos/tasks for tracking workflow. Actions: add, update, list, clear_completed."""
    return core_manage_todos(action, content, todo_id, status, status_filter)


# ==================== JUPYTER NOTEBOOKS ====================

@tool
def edit_notebook_cell(
    notebook_path: str,
    cell_index: int,
    new_source: str,
    cell_type: Literal["code", "markdown"] = "code",
) -> str:
    """Edit a specific cell in a Jupyter notebook (.ipynb file)."""
    return core_edit_notebook_cell(notebook_path, cell_index, new_source, cell_type)


# ==================== CODE ANALYSIS ====================

@tool
def analyze_code_quality(file_path: str, language: Optional[str] = None) -> str:
    """Analyze code quality metrics, detect long lines, and find potential issues."""
    return core_analyze_code_quality(file_path, language)


# ==================== WEB OPERATIONS ====================

@tool
def web_search_simple(query: str, max_results: int = 5) -> str:
    """Simple web search using DuckDuckGo API. For comprehensive search use tavily_search."""
    return core_web_search_simple(query, max_results)



CUSTOM_TOOL_STORE = PERSISTENT_STORE


def _coerce_json_dict(value: Any, *, field_name: str) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} must be valid JSON: {exc}") from exc
        if isinstance(parsed, dict):
            return parsed
        raise ValueError(f"{field_name} JSON payload must be an object.")
    raise ValueError(f"{field_name} must be an object or JSON string.")


class ToolLibraryRequest(BaseModel):
    action: Literal["list", "create", "update", "delete", "show", "run"]
    name: Optional[str] = Field(
        None,
        description="Name of the custom tool to operate on.",
        min_length=1,
    )
    description: Optional[str] = Field(
        None,
        description="Human readable summary for create/update operations.",
    )
    kind: Literal["shell", "python"] = Field(
        "shell",
        description="Execution backend for the tool.",
    )
    body: Optional[str] = Field(
        None,
        description="Shell command template or Python script body.",
    )
    args_schema: Optional[Any] = Field(
        None,
        description="Optional JSON schema describing expected arguments.",
    )
    metadata: Optional[Any] = Field(
        None,
        description="Optional JSON metadata stored alongside the tool.",
    )
    timeout: Optional[int] = Field(
        None,
        description="Override execution timeout (seconds).",
    )
    arguments: Optional[Any] = Field(
        None,
        description="Arguments to pass when running the stored tool.",
    )


def _format_tool_record(record) -> str:
    args_text = json.dumps(record.args_schema or {}, indent=2, ensure_ascii=False)
    meta_text = json.dumps(record.metadata or {}, indent=2, ensure_ascii=False)
    return "\n".join(
        [
            f"Name: {record.name}",
            f"Kind: {record.kind}",
            f"Timeout: {record.timeout}s",
            f"Description: {record.description}",
            f"Created: {record.created_at}",
            f"Updated: {record.updated_at}",
            f"Args schema: {args_text}",
            f"Metadata: {meta_text}",
            f"Body:\n{record.body}",
        ]
    )


@tool("tool_library", args_schema=ToolLibraryRequest)
def tool_library(request: ToolLibraryRequest) -> str:
    """Create, update, run, or list persistent custom tools backed by SQLite."""
    store = CUSTOM_TOOL_STORE
    try:
        if request.action == "list":
            records = store.list_tools()
            if not records:
                return "tool_library: No custom tools have been saved yet."
            lines = ["tool_library: Stored tools:"]
            for record in records:
                lines.append(
                    f"- {record.name} ({record.kind}, timeout {record.timeout}s): {record.description}"
                )
            return "\n".join(lines)

        if request.action == "show":
            if not request.name:
                raise ValueError("Provide 'name' to inspect a tool.")
            record = store.get_tool(request.name)
            if record is None:
                return f"tool_library: '{request.name}' was not found."
            return _format_tool_record(record)

        if request.action == "delete":
            if not request.name:
                raise ValueError("Provide 'name' to delete a tool.")
            removed = store.delete_tool(request.name)
            if not removed:
                return f"tool_library: '{request.name}' did not exist."
            return f"tool_library: Deleted '{request.name}'."

        if request.action == "create":
            if not request.name or not request.body or not request.description:
                raise ValueError("name, description, and body are required to create a tool.")
            args_schema = _coerce_json_dict(request.args_schema, field_name="args_schema") or {}
            metadata = _coerce_json_dict(request.metadata, field_name="metadata") or {}
            record = store.create_tool(
                name=request.name,
                description=request.description,
                kind=request.kind,
                body=request.body,
                args_schema=args_schema,
                metadata=metadata,
                timeout=request.timeout or 180,
            )
            return f"tool_library: Created '{record.name}'."

        if request.action == "update":
            if not request.name:
                raise ValueError("Provide 'name' to update a tool.")
            args_schema = (
                _coerce_json_dict(request.args_schema, field_name="args_schema")
                if request.args_schema is not None
                else None
            )
            metadata = (
                _coerce_json_dict(request.metadata, field_name="metadata")
                if request.metadata is not None
                else None
            )
            record = store.update_tool(
                request.name,
                description=request.description,
                kind=request.kind,
                body=request.body,
                args_schema=args_schema,
                metadata=metadata,
                timeout=request.timeout,
            )
            return f"tool_library: Updated '{record.name}'."

        if request.action == "run":
            if not request.name:
                raise ValueError("Provide 'name' to run a stored tool.")
            arguments = request.arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:  # pragma: no cover - user error
                    raise ValueError(f"arguments must be JSON: {exc}") from exc
            if arguments is not None and not isinstance(arguments, dict):
                raise ValueError("arguments must be an object mapping parameter names to values.")
            output = store.run_tool(request.name, arguments=arguments or {})
            return f"tool_library: Ran '{request.name}'.\n{output}"

        raise ValueError(f"Unknown action '{request.action}'.")
    except ValueError as exc:
        return f"tool_library error: {exc}"


class ResearchVaultRequest(BaseModel):
    action: Literal["list", "get", "set", "append", "delete"]
    namespace: str = Field(
        "global",
        description="Logical bucket (e.g., project name) for the stored note.",
    )
    key: Optional[str] = Field(
        None,
        description="Unique key within the namespace.",
    )
    content: Optional[str] = Field(
        None,
        description="Note body for set/append operations.",
    )
    metadata: Optional[Any] = Field(
        None,
        description="Optional JSON metadata stored with the note.",
    )


@tool("research_vault", args_schema=ResearchVaultRequest)
def research_vault(request: ResearchVaultRequest) -> str:
    """Persist, retrieve, or append research notes across agent runs."""
    store = CUSTOM_TOOL_STORE
    try:
        if request.action == "list":
            notes = store.list_notes(namespace=request.namespace if request.namespace else None)
            if not notes:
                scope = request.namespace or "all namespaces"
                return f"research_vault: No notes found for {scope}."
            lines = [
                f"research_vault: Notes for {request.namespace or 'all namespaces'}:",
            ]
            for note in notes:
                preview = (note.content or "").strip().splitlines()[:3]
                summary = " ".join(preview)[:200] if preview else "(empty)"
                lines.append(
                    f"- {note.namespace}::{note.key} ({note.updated_at}) – {summary}"
                )
            return "\n".join(lines)

        if request.action == "get":
            if not request.key:
                raise ValueError("Provide 'key' to retrieve a note.")
            note = store.get_note(request.namespace, request.key)
            if note is None:
                return f"research_vault: '{request.namespace}::{request.key}' not found."
            metadata = json.dumps(note.metadata or {}, indent=2, ensure_ascii=False)
            return "\n".join(
                [
                    f"Namespace: {note.namespace}",
                    f"Key: {note.key}",
                    f"Created: {note.created_at}",
                    f"Updated: {note.updated_at}",
                    f"Metadata: {metadata}",
                    "Content:",
                    note.content,
                ]
            )

        if request.action in {"set", "append"}:
            if not request.key:
                raise ValueError("Provide 'key' to store a note.")
            if request.content is None:
                raise ValueError("Provide 'content' when writing to the vault.")
            metadata = _coerce_json_dict(request.metadata, field_name="metadata")
            note = store.write_note(
                request.namespace,
                request.key,
                request.content,
                metadata=metadata,
                mode="append" if request.action == "append" else "overwrite",
            )
            return f"research_vault: Stored '{note.namespace}::{note.key}' ({note.updated_at})."

        if request.action == "delete":
            if not request.key:
                raise ValueError("Provide 'key' to delete a note.")
            removed = store.delete_note(request.namespace, request.key)
            if not removed:
                return f"research_vault: '{request.namespace}::{request.key}' did not exist."
            return f"research_vault: Deleted '{request.namespace}::{request.key}'."

        raise ValueError(f"Unknown action '{request.action}'.")
    except ValueError as exc:
        return f"research_vault error: {exc}"


class SelfImproveRequest(BaseModel):
    action: Literal["analyze", "iterate", "store_learning", "get_learnings"]
    task_description: Optional[str] = Field(
        None,
        description="Description of the task being worked on.",
    )
    result: Optional[str] = Field(
        None,
        description="Result or output to analyze.",
    )
    criteria: Optional[str] = Field(
        None,
        description="Success criteria or requirements to check against.",
    )
    learning: Optional[str] = Field(
        None,
        description="Learning or insight to store for future reference.",
    )
    category: Optional[str] = Field(
        None,
        description="Category for the learning (e.g., 'code_generation', 'api_usage', 'debugging').",
    )


@tool("self_improve", args_schema=SelfImproveRequest)
def self_improve(request: SelfImproveRequest) -> str:
    """Analyze task results, store learnings, and enable iterative self-improvement."""
    store = CUSTOM_TOOL_STORE

    try:
        if request.action == "analyze":
            if not request.task_description or not request.result:
                raise ValueError("task_description and result are required for analysis.")

            # Provide structured analysis framework
            analysis = [
                "Self-Improvement Analysis Framework:",
                "",
                f"Task: {request.task_description}",
                f"Result: {request.result[:500]}..." if len(request.result) > 500 else f"Result: {request.result}",
                "",
                "Analysis Questions:",
                "1. Did the result meet the requirements?",
                "2. What worked well?",
                "3. What could be improved?",
                "4. What errors or issues occurred?",
                "5. What should be tried differently?",
                "",
            ]

            if request.criteria:
                analysis.extend([
                    f"Success Criteria: {request.criteria}",
                    "Check each criterion and identify gaps.",
                    "",
                ])

            analysis.extend([
                "Recommendations:",
                "- Use 'iterate' action to generate improved version",
                "- Use 'store_learning' to save insights for future tasks",
                "- Review stored learnings with 'get_learnings' before similar tasks",
            ])

            return "\n".join(analysis)

        elif request.action == "iterate":
            if not request.task_description:
                raise ValueError("task_description required for iteration.")

            # Retrieve relevant learnings
            category = request.category or "general"
            past_learnings = store.list_notes(namespace=f"learnings_{category}")

            iteration_guide = [
                "Iteration Guide:",
                "",
                f"Task: {request.task_description}",
                "",
                "Previous Result Analysis:",
            ]

            if request.result:
                iteration_guide.append(f"Previous Result: {request.result[:300]}...")

            if request.criteria:
                iteration_guide.extend([
                    "",
                    f"Success Criteria: {request.criteria}",
                    "Focus improvements on meeting these criteria.",
                ])

            if past_learnings:
                iteration_guide.extend([
                    "",
                    "Relevant Past Learnings:",
                ])
                for note in past_learnings[:5]:  # Show top 5
                    preview = note.content[:150].replace("\n", " ")
                    iteration_guide.append(f"- {note.key}: {preview}")

            iteration_guide.extend([
                "",
                "Iteration Strategy:",
                "1. Identify specific gaps or failures in previous attempt",
                "2. Generate concrete improvements addressing each gap",
                "3. Apply learnings from similar past tasks",
                "4. Test improvements incrementally",
                "5. Document what works for future use",
            ])

            return "\n".join(iteration_guide)

        elif request.action == "store_learning":
            if not request.learning:
                raise ValueError("learning content required to store.")

            category = request.category or "general"
            timestamp = datetime.now(timezone.utc).isoformat()
            key = f"learning_{timestamp[:19].replace(':', '-')}"

            learning_content = [
                f"Task: {request.task_description or 'N/A'}",
                f"Category: {category}",
                f"Timestamp: {timestamp}",
                "",
                "Learning:",
                request.learning,
            ]

            if request.criteria:
                learning_content.extend([
                    "",
                    f"Context/Criteria: {request.criteria}",
                ])

            note = store.write_note(
                namespace=f"learnings_{category}",
                key=key,
                content="\n".join(learning_content),
                metadata={"category": category, "timestamp": timestamp},
            )

            return f"Stored learning in {note.namespace}::{note.key}. Use 'get_learnings' with category='{category}' to retrieve."

        elif request.action == "get_learnings":
            category = request.category or "general"
            learnings = store.list_notes(namespace=f"learnings_{category}")

            if not learnings:
                return f"No learnings found for category '{category}'. Available categories can be checked in research_vault."

            output = [f"Stored Learnings (category: {category}):", ""]
            for note in learnings[-10:]:  # Show last 10
                output.append(f"--- {note.key} ({note.updated_at}) ---")
                output.append(note.content[:400])
                output.append("")

            return "\n".join(output)

        else:
            raise ValueError(f"Unknown action '{request.action}'.")

    except ValueError as exc:
        return f"self_improve error: {exc}"


TOOLS = [
    # Web & Search
    tavily_search,
    tavily_extract,
    web_search_simple,
    get_weather,
    # Code Execution
    run_python,
    run_shell,
    # File Operations - Basic
    list_directory,
    read_text,
    write_text,
    # File Operations - Enhanced
    edit_file,
    glob_files,
    grep_files,
    # Git Operations
    git_status,
    git_diff,
    git_commit,
    git_log,
    # Automation & Scripts
    save_shell_automation,
    # Browser Automation
    headless_browse,
    # Jupyter Notebooks
    edit_notebook_cell,
    # Code Analysis
    analyze_code_quality,
    # Task Management
    manage_todos,
    # Persistent Storage
    tool_library,
    research_vault,
    self_improve,
]

MCP_BRIDGE = None
try:
    mcp_tools, MCP_BRIDGE, mcp_messages = load_mcp_tools(MAX_TOOL_OUTPUT)
except Exception as exc:  # noqa: BLE001
    mcp_tools = []
    MCP_BRIDGE = None
    mcp_messages = [f"MCP client failed to initialize: {exc}"]

for message in mcp_messages:
    cli_ui.print_status(f"[mcp] {message}", kind="warning")

if mcp_tools:
    TOOLS.extend(mcp_tools)

tool_node = ToolNode(TOOLS)  # Executes one or more tool calls emitted by the LLM

# --------- Tool retrieval + dynamic binding ----------

TOOL_REGISTRY: Dict[str, BaseTool] = {tool.name: tool for tool in TOOLS}
TOOL_TAG_OVERRIDES: Dict[str, List[str]] = {
    "tavily_search": ["web", "search", "news"],
    "tavily_extract": ["web", "scrape"],
    "web_search_simple": ["web", "search", "quick"],
    "get_weather": ["weather", "api"],
    "run_python": ["python", "code", "analysis"],
    "run_shell": ["shell", "terminal", "system"],
    "list_directory": ["filesystem", "inspect"],
    "read_text": ["filesystem", "read"],
    "write_text": ["filesystem", "write"],
    "edit_file": ["filesystem", "edit", "refactor"],
    "glob_files": ["filesystem", "search", "pattern"],
    "grep_files": ["filesystem", "search", "regex", "code"],
    "git_status": ["git", "vcs", "status"],
    "git_diff": ["git", "vcs", "diff", "changes"],
    "git_commit": ["git", "vcs", "commit"],
    "git_log": ["git", "vcs", "history"],
    "save_shell_automation": ["automation", "scripts"],
    "headless_browse": ["browser", "automation", "playwright"],
    "edit_notebook_cell": ["jupyter", "notebook", "ipynb", "edit"],
    "analyze_code_quality": ["code", "analysis", "quality", "review"],
    "manage_todos": ["task", "todo", "workflow", "planning"],
    "tool_library": ["tools", "automation", "meta"],
    "research_vault": ["memory", "notes", "persistent"],
    "self_improve": ["improvement", "iteration", "learning", "meta"],
}
TOOL_EXAMPLE_OVERRIDES: Dict[str, List[str]] = {
    "tavily_search": ["tavily_search(query='latest ai models', max_results=5)"],
    "tavily_extract": ["tavily_extract(url='https://example.com/article')"],
    "web_search_simple": ["web_search_simple(query='python tutorials', max_results=5)"],
    "get_weather": ["get_weather(location='Paris, France', units='metric')"],
    "run_python": ["run_python(code='print(1+1)', timeout=30)"],
    "run_shell": ["run_shell(command='ls -la', timeout=60)"],
    "list_directory": ["list_directory(path='~/projects')"],
    "read_text": ["read_text(path='README.md', max_chars=4000)"],
    "write_text": ["write_text(path='notes/todo.txt', content='- item', mode='append')"],
    "edit_file": ["edit_file(file_path='config.py', old_string='DEBUG = False', new_string='DEBUG = True')"],
    "glob_files": ["glob_files(pattern='**/*.py', path='src')"],
    "grep_files": ["grep_files(pattern='import.*pandas', file_pattern='*.py', output_mode='matches')"],
    "git_status": ["git_status(repo_path='.')"],
    "git_diff": ["git_diff(repo_path='.', staged=False)"],
    "git_commit": ["git_commit(message='Fix bug in parser', add_all=True)"],
    "git_log": ["git_log(repo_path='.', max_commits=10)"],
    "save_shell_automation": ["save_shell_automation(name='backup', content='tar -czf backup.tgz .', run=False)"],
    "headless_browse": ["headless_browse(url='https://example.com', wait_selector='article h1')"],
    "edit_notebook_cell": ["edit_notebook_cell(notebook_path='analysis.ipynb', cell_index=0, new_source='import pandas as pd')"],
    "analyze_code_quality": ["analyze_code_quality(file_path='main.py', language='python')"],
    "manage_todos": ["manage_todos(action='add', content='Implement feature X')"],
    "tool_library": [
        "tool_library(action='create', name='process_csv', description='Clean CSV files', kind='python', body='...')"
    ],
    "research_vault": [
        "research_vault(action='set', namespace='cancer-research', key='hypotheses', content='1. ...')"
    ],
    "self_improve": [
        "self_improve(action='analyze', task_description='scrape website', result='got 404 error', criteria='extract all product names')",
        "self_improve(action='iterate', task_description='retry scraping', result='404', criteria='extract products')",
        "self_improve(action='store_learning', learning='Always check robots.txt before scraping', category='web_scraping')",
    ],
}
TOOL_RISK_OVERRIDES: Dict[str, str] = {
    "run_shell": "high",
    "save_shell_automation": "high",
    "headless_browse": "high",
    "run_python": "medium",
    "tool_library": "high",
    "edit_file": "medium",
    "write_text": "medium",
    "git_commit": "medium",
    "edit_notebook_cell": "medium",
}

STICKY_TOOL_NAMES = [
    name for name in ("tavily_search", "run_python", "read_text", "glob_files", "grep_files") if name in TOOL_REGISTRY
]
SAFE_FALLBACK_TOOL_NAMES = [
    name for name in (
        "tavily_search", "run_python", "read_text", "list_directory",
        "glob_files", "grep_files", "git_status", "manage_todos"
    ) if name in TOOL_REGISTRY
]
SAFE_FALLBACK_TOOLS: List[BaseTool] = [TOOL_REGISTRY[name] for name in SAFE_FALLBACK_TOOL_NAMES]
if not SAFE_FALLBACK_TOOLS:
    SAFE_FALLBACK_TOOLS = list(TOOLS)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


TOOL_SELECTION_TOPK = _int_env("AGENT_TOOL_TOPK", 12)
TOOL_EXPANSION_TOPK = _int_env("AGENT_TOOL_TOPK_EXPAND", 24)
EMBED_DIM = _int_env("AGENT_TOOL_EMBED_DIM", 1024)
EMBED_MODEL = os.environ.get("AGENT_TOOL_EMBED_MODEL")


def _args_schema_for_tool(tool: BaseTool) -> Dict[str, Any]:
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is None:
        return {"type": "object", "properties": {}}
    try:
        return args_schema.model_json_schema()
    except Exception:  # noqa: BLE001 - defensive
        return {"type": "object", "properties": {}}


def _tool_tags(tool: BaseTool) -> List[str]:
    tags = list(TOOL_TAG_OVERRIDES.get(tool.name, []))
    for tag in getattr(tool, "tags", []) or []:
        if tag not in tags:
            tags.append(tag)
    return tags


def _tool_examples(tool: BaseTool) -> List[str]:
    if tool.name in TOOL_EXAMPLE_OVERRIDES:
        return list(TOOL_EXAMPLE_OVERRIDES[tool.name])
    description = (tool.description or "").strip()
    return [description[:200]] if description else []


def _tool_risk(name: str, default: str = "low") -> str:
    return TOOL_RISK_OVERRIDES.get(name, default)


def _build_tool_records(tool_list: Sequence[BaseTool], bridge) -> List[ToolRecord]:
    records: List[ToolRecord] = []
    spec_by_name: Dict[str, Any] = {}
    if bridge is not None:
        try:
            spec_by_name = {f"mcp::{spec.qualified_name}": spec for spec in bridge.tool_specs}
        except Exception:  # noqa: BLE001 - best effort
            spec_by_name = {}
    for tool in tool_list:
        spec = spec_by_name.get(tool.name)
        server_label = spec.server_label if spec else "local"
        description = (spec.description if spec else tool.description) or tool.__doc__ or tool.name
        schema = spec.input_schema if spec else _args_schema_for_tool(tool)
        tags = _tool_tags(tool)
        examples = _tool_examples(tool)
        if spec and spec.description and not examples:
            examples = [spec.description[:160]]
        default_risk = "medium" if spec is not None else "low"
        record = ToolRecord(
            name=tool.name,
            server=server_label,
            description=str(description or tool.name).strip(),
            input_schema=schema or {"type": "object", "properties": {}},
            examples=examples,
            tags=tags,
            risk=_tool_risk(tool.name, default=default_risk),
            aliases=None,
        )
        records.append(record)
    return records


TOOL_RETRIEVER: Optional[ToolRetriever]
try:
    _TOOL_RECORDS = _build_tool_records(TOOLS, MCP_BRIDGE)
    _EMBEDDER = Embedder(dim=EMBED_DIM, model_name=EMBED_MODEL)
    TOOL_RETRIEVER = ToolRetriever(
        _EMBEDDER,
        include_defaults=STICKY_TOOL_NAMES,
        risky_opt_in=True,
        popularity_prior={},
    )
    TOOL_RETRIEVER.build_catalog(_TOOL_RECORDS)
    TOOL_RETRIEVER.build_indexes()
except Exception as exc:  # noqa: BLE001
    TOOL_RETRIEVER = None
    cli_ui.print_status(f"[tools] vector retriever disabled: {exc}", kind="warning")


def _primary_user_goal(messages: List[BaseMessage]) -> str:
    fallback = ""
    for msg in reversed(messages):
        if getattr(msg, "type", None) != "human":
            continue
        content = ensure_message_content(msg.content)
        if not content:
            continue
        name = getattr(msg, "name", "") or ""
        if name.startswith("task:"):
            fallback = fallback or content
            continue
        return content
    return fallback


def _task_hint_from_messages(messages: List[BaseMessage]) -> str:
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "human" and str(getattr(msg, "name", "")).startswith("task:"):
            return ensure_message_content(msg.content)
    return ""


def _plan_hint_from_messages(messages: List[BaseMessage]) -> str:
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "ai" and getattr(msg, "name", "") == "planner":
            return ensure_message_content(msg.content)
    return ""


def _recent_context_snippet(messages: List[BaseMessage], *, limit_chars: int = 1500, max_messages: int = 8) -> str:
    snippets: List[str] = []
    for msg in reversed(messages):
        msg_type = getattr(msg, "type", None)
        if msg_type not in {"ai", "tool"}:
            continue
        text = ensure_message_content(msg.content)
        if not text:
            continue
        label = getattr(msg, "name", "") or msg_type
        snippets.append(f"{label}: {text}")
        if len(snippets) >= max_messages:
            break
    combined = "\n".join(reversed(snippets))
    if len(combined) > limit_chars:
        return combined[-limit_chars:]
    return combined


def _should_allow_risky(*texts: str) -> bool:
    combined = " ".join(filter(None, texts)).lower()
    risk_tokens = ("shell", "terminal", "browser", "headless", "playwright", "ssh", "automation")
    return any(token in combined for token in risk_tokens)


def _bindings_from_shortlist(shortlist: List[ScoredTool]) -> List[BaseTool]:
    seen: Set[str] = set()
    bindings: List[BaseTool] = []
    for scored in shortlist:
        tool = TOOL_REGISTRY.get(scored.tool.name)
        if tool and tool.name not in seen:
            bindings.append(tool)
            seen.add(tool.name)
    return bindings


def _unknown_tool_requested(message: AIMessage, allowed_names: Set[str]) -> bool:
    tool_calls = getattr(message, "tool_calls", []) or []
    for call in tool_calls:
        if isinstance(call, dict):
            name = call.get("name") or call.get("function", {}).get("name")
        else:
            name = getattr(call, "name", None)
            if not name and hasattr(call, "function"):
                name = getattr(call.function, "name", None)
        if name and name not in allowed_names:
            return True
    return False


def _invoke_with_dynamic_tools(
    base_llm: ChatOpenAI,
    messages: List[BaseMessage],
    *,
    plan_hint_override: str = "",
    user_goal_override: str = "",
) -> AIMessage:
    if TOOL_RETRIEVER is None:
        return base_llm.bind_tools(TOOLS).invoke(messages)

    user_goal = user_goal_override or _primary_user_goal(messages) or "general task"
    plan_hint = plan_hint_override or _plan_hint_from_messages(messages)
    recent_context = _recent_context_snippet(messages)
    allow_risky = _should_allow_risky(user_goal, plan_hint)
    shortlist = TOOL_RETRIEVER.select(
        user_goal,
        recent_context,
        plan_hint,
        K_dynamic=max(4, TOOL_SELECTION_TOPK),
        allow_risky=allow_risky,
    )
    if not shortlist:
        shortlist = TOOL_RETRIEVER.select(
            user_goal,
            recent_context,
            plan_hint,
            K_dynamic=max(4, TOOL_SELECTION_TOPK),
            allow_risky=True,
        )

    return _apply_shortlist_with_retry(
        base_llm,
        messages,
        shortlist,
        user_goal,
        recent_context,
        plan_hint,
    )


def _apply_shortlist_with_retry(
    base_llm: ChatOpenAI,
    messages: List[BaseMessage],
    shortlist: List[ScoredTool],
    user_goal: str,
    recent_context: str,
    plan_hint: str,
) -> AIMessage:
    attempts = 0
    current_shortlist = shortlist
    while True:
        bindings = _bindings_from_shortlist(current_shortlist) or SAFE_FALLBACK_TOOLS
        allowed_names = {tool.name for tool in bindings}
        response = base_llm.bind_tools(bindings).invoke(messages)
        if TOOL_RETRIEVER is None or attempts > 0:
            return response
        if _unknown_tool_requested(response, allowed_names):
            expanded = TOOL_RETRIEVER.expand_and_retry(
                user_goal,
                recent_context,
                plan_hint,
                current_shortlist,
                K_expand=max(TOOL_EXPANSION_TOPK, TOOL_SELECTION_TOPK),
            )
            expanded_names = {entry.tool.name for entry in expanded}
            current_names = {entry.tool.name for entry in current_shortlist}
            if expanded_names and expanded_names != current_names:
                current_shortlist = expanded
                attempts += 1
                continue
        return response

# --------- Real LLMs ----------

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise RuntimeError("Set DEEPSEEK_API_KEY in your environment before running this script.")

DEEPSEEK_API_BASE = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-reasoner")


def _make_deepseek_chat_model(*, streaming: bool = False):
    """Factory so every LLM shares the same DeepSeek configuration."""
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        temperature=0,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_API_BASE,
        streaming=streaming,
    )


# Worker LLM (tool-calling, dynamic tool binding)
worker_llm = _make_deepseek_chat_model(streaming=True)

# Planner LLM (no tools)
planner_llm = _make_deepseek_chat_model()

# --------- Worker agent (same loop as your starter) ----------

def agent(state: MessagesState):
    """Single LLM step for a worker. Returns an AIMessage that may include tool_calls."""
    plan_hint = _task_hint_from_messages(state["messages"])
    resp = _invoke_with_dynamic_tools(
        worker_llm,
        state["messages"],
        plan_hint_override=plan_hint,
    )
    return {"messages": [resp]}

worker_workflow = StateGraph(MessagesState)
worker_workflow.add_node("agent", agent)
worker_workflow.add_node("tools", tool_node)
worker_workflow.add_edge(START, "agent")
worker_workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
worker_workflow.add_edge("tools", "agent")
worker_graph = worker_workflow.compile()  # compiled worker

# --------- Planning schema & node ----------

class PlanStep(BaseModel):
    id: str = Field(..., description="Short unique id: 'step1', 'weather_sf', etc.")
    description: str = Field(..., description="What a worker should do with available tools.")

class Plan(BaseModel):
    mode: Literal["single", "sequential", "parallel"] = Field(
        ..., description="Pick 'single', 'sequential', or 'parallel'."
    )
    steps: List[PlanStep] = Field(..., description="1–8 concrete steps.")


def _extract_plan_payload(message: Optional[BaseMessage]) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of planner JSON regardless of how it was stored."""
    if not message:
        return None
    extras = getattr(message, "additional_kwargs", None) or {}
    plan_payload = extras.get("plan")
    if isinstance(plan_payload, dict):
        return plan_payload
    content = getattr(message, "content", None)
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and "steps" in block:
                return block
    if isinstance(content, str):
        try:
            decoded = json.loads(content)
        except json.JSONDecodeError:
            # Attempt to recover a JSON object embedded inside free-form text.
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    decoded = json.loads(content[start : end + 1])
                except json.JSONDecodeError:
                    return None
            else:
                return None
        if isinstance(decoded, dict):
            return decoded
    return None

# Use the base planner LLM (DeepSeek currently doesn't support JSON response_format)
planner = planner_llm

PLANNER_SYSTEM = SystemMessage(
    content=(
        "You are a planning agent. Read the user's goal and design a minimal plan the workers can execute. "
        "Workers and executors have the following tools:\n"
        "WEB & SEARCH:\n"
        "- tavily_search(query, max_results, search_depth) for comprehensive web research\n"
        "- tavily_extract(url) for scraping known pages\n"
        "- web_search_simple(query, max_results) for quick DuckDuckGo searches\n"
        "- get_weather(location, units) for weather data\n"
        "CODE EXECUTION:\n"
        "- run_python(code, timeout) to execute Python code\n"
        "- run_shell(command, timeout, cwd) for terminal commands\n"
        "FILE OPERATIONS:\n"
        "- list_directory(path), read_text(path, max_chars), write_text(path, content, mode)\n"
        "- edit_file(file_path, old_string, new_string, replace_all) for precise edits\n"
        "- glob_files(pattern, path) to find files by pattern (e.g., '**/*.py')\n"
        "- grep_files(pattern, path, file_pattern, output_mode) for regex search in files\n"
        "GIT OPERATIONS:\n"
        "- git_status(repo_path), git_diff(repo_path, staged), git_commit(message, add_all), git_log(repo_path)\n"
        "AUTOMATION:\n"
        "- save_shell_automation(name, content, run, timeout) to persist/execute scripts\n"
        "- headless_browse(...) for browser automation (forms, screenshots, JS)\n"
        "JUPYTER:\n"
        "- edit_notebook_cell(notebook_path, cell_index, new_source, cell_type)\n"
        "CODE ANALYSIS:\n"
        "- analyze_code_quality(file_path, language) for code metrics and issue detection\n"
        "TASK MANAGEMENT:\n"
        "- manage_todos(action, content, todo_id, status) to track workflow\n"
        "PERSISTENT STORAGE:\n"
        "- tool_library(...) to create/update/run custom tools\n"
        "- research_vault(...) to persist notes across runs\n"
        "- self_improve(...) to analyze/iterate/learn from attempts\n\n"
        "Choose mode: 'single' if one step suffices, 'sequential' if steps depend on each other, "
        "'parallel' if steps are independent and can run concurrently. Keep steps crisp and tool-friendly. "
        "Respond ONLY with a JSON object shaped like "
        '{"mode": "...", "steps": [{"id": "...", "description": "..."}]}.'
    )
)


def _default_plan_for(user_message: HumanMessage) -> Plan:
    """Fallback single-step plan mirroring the user request."""
    description = user_message.content
    if not isinstance(description, str):
        description = ensure_message_content(description)
    return Plan(
        mode="single",
        steps=[PlanStep(id="direct", description=str(description))],
    )


def planning_node(state: MessagesState):
    # Find last user request
    user_msgs = [m for m in state["messages"] if getattr(m, "type", None) == "human"]
    last_user = user_msgs[-1] if user_msgs else HumanMessage(content="Do something useful.")

    planner_response = planner.invoke([PLANNER_SYSTEM, last_user])
    plan_payload = _extract_plan_payload(planner_response)
    plan_model: Plan
    if plan_payload:
        try:
            plan_model = Plan.model_validate(plan_payload)
        except ValidationError:
            plan_model = _default_plan_for(last_user)
    else:
        plan_model = _default_plan_for(last_user)
    plan_dict = plan_model.model_dump()
    # Attach the plan as an AI message named 'planner' so the executor can find it
    plan_msg = AIMessage(
        name="planner",
        content=ensure_message_content(plan_dict),
        additional_kwargs={"plan": plan_dict},
    )
    return {"messages": [plan_msg]}

# --------- Helpers ----------

def _extract_final_ai(messages: List[BaseMessage]) -> Optional[AIMessage]:
    final_ai = None
    for msg in messages:
        if getattr(msg, "type", None) == "ai":
            final_ai = msg
    return final_ai

def _run_worker_for_step(step: PlanStep, user_context: HumanMessage, extra_context: Optional[List[BaseMessage]] = None):
    """Invoke the worker graph for a single step."""
    sys = SystemMessage(
        content=(
            "You are a focused worker agent. Only complete the assigned step using available tools. "
            "You can search, extract URLs, query weather, run Python, execute shell commands, manage scripts, drive headless browsers, and read/write files. "
            "Be concise, but include the essential details and any URLs you used verbatim."
        )
    )
    task_msg = HumanMessage(name=f"task:{step.id}", content=f"Step: {step.description}")
    messages = [sys, user_context, task_msg]
    if extra_context:
        messages.extend(extra_context)
    result = worker_graph.invoke({"messages": messages})
    return _extract_final_ai(result["messages"])

# --------- NEW: ReAct Executor (agent keeps deciding to use more tools until done) ----------

# Dedicated executor LLM (dynamic tools)
executor_llm = _make_deepseek_chat_model(streaming=True)

EXECUTOR_SYSTEM = SystemMessage(
    content=(
        "You are the EXECUTOR. Follow the provided plan as a guide and use ReAct:\n"
        "Think about what to do next, choose a tool if needed, observe the result, and repeat until you can answer.\n\n"
        "Available tools (COMPREHENSIVE TOOLKIT):\n"
        "WEB & SEARCH: tavily_search, tavily_extract, web_search_simple, get_weather\n"
        "CODE EXECUTION: run_python, run_shell (full system access)\n"
        "FILE OPS (Basic): list_directory, read_text, write_text\n"
        "FILE OPS (Advanced): edit_file (precise find/replace), glob_files (pattern matching), grep_files (regex search)\n"
        "GIT: git_status, git_diff, git_commit, git_log\n"
        "AUTOMATION: save_shell_automation, headless_browse (browser automation for forms/accounts/screenshots)\n"
        "JUPYTER: edit_notebook_cell (modify .ipynb cells)\n"
        "CODE ANALYSIS: analyze_code_quality (metrics, issues, long lines)\n"
        "TASK MANAGEMENT: manage_todos (add/update/list/clear todos)\n"
        "PERSISTENT: tool_library (custom tools), research_vault (notes), self_improve (analyze/iterate/learn)\n\n"
        "You may call tools multiple times. If a task fails, use self_improve to analyze why and iterate with improvements.\n"
        "Use manage_todos to track complex multi-step workflows.\n"
        "Use edit_file for surgical code changes instead of rewriting entire files.\n"
        "Use glob_files and grep_files to find files and search code efficiently.\n"
        "Use git operations to inspect and commit changes.\n"
        "Stop calling tools when you have enough information or achieved the goal.\n"
        "When finished, reply with a concise, user-ready answer. Do not include internal chain-of-thought."
    )
)

def executor_agent(state: MessagesState):
    """One step of the executor (tool-capable)."""
    plan_hint = _plan_hint_from_messages(state["messages"])
    resp = _invoke_with_dynamic_tools(
        executor_llm,
        state["messages"],
        plan_hint_override=plan_hint,
    )
    return {"messages": [resp]}

# ReAct loop for the executor: agent -> (tools?) -> agent ... until done
executor_workflow = StateGraph(MessagesState)
executor_workflow.add_node("agent", executor_agent)
executor_workflow.add_node("tools", tool_node)
executor_workflow.add_edge(START, "agent")
executor_workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
executor_workflow.add_edge("tools", "agent")
executor_graph = executor_workflow.compile()

# --------- Execution node (uses ReAct executor) ----------

def execution_node(state: MessagesState):
    """
    Read the most recent planner output and then run a ReAct loop that keeps deciding
    whether to use more tools until the model stops emitting tool calls.
    """
    # Locate the latest plan emitted by the planner
    plan_msg = None
    for msg in reversed(state["messages"]):
        if getattr(msg, "type", None) == "ai" and getattr(msg, "name", "") == "planner":
            plan_msg = msg
            break

    # Last user message for grounding
    user_msgs = [m for m in state["messages"] if getattr(m, "type", None) == "human"]
    last_user = user_msgs[-1] if user_msgs else HumanMessage(content="Do something useful.")

    # If for any reason the planner isn't present, fall back to a single worker run on the user prompt.
    plan_payload = _extract_plan_payload(plan_msg)
    if not plan_msg or not plan_payload:
        fallback_step = PlanStep(id="direct", description=last_user.content)
        final = _run_worker_for_step(fallback_step, last_user)
        summary = final.content if final else "No result."
        return {"messages": [AIMessage(name="executor", content=summary)]}

    # Seed the ReAct executor with a system prompt, the user's request, and the plan as context.
    # NOTE: We pass these into a *subgraph*; we will return only the newly created messages,
    # so we don't duplicate context in the outer transcript.
    plan_context = AIMessage(
        name="planner",
        content=plan_msg.content,
        additional_kwargs={"plan": plan_payload},
    )

    seeded_messages: List[BaseMessage] = [
        EXECUTOR_SYSTEM,
        last_user,
        plan_context,
    ]

    # Run the ReAct loop inside the subgraph
    sub_result = executor_graph.invoke({"messages": seeded_messages})

    # Only return messages created by the subgraph *after* our seeds, so outer history stays clean.
    produced = sub_result["messages"][len(seeded_messages):]

    # Safety fallback: if nothing was produced (unlikely), return a basic executor message.
    if not produced:
        return {"messages": [AIMessage(name="executor", content="No result.")]}

    return {"messages": produced}

# --------- Controller graph (planner -> ReAct executor) ----------

controller = StateGraph(MessagesState)
controller.add_node("plan", planning_node)
controller.add_node("execute", execution_node)
controller.add_edge(START, "plan")
controller.add_edge("plan", "execute")
controller.add_edge("execute", END)
controller_graph = controller.compile()

# For the rest of the program, we expose this as `graph`
graph = controller_graph

# --------- Runtime helpers & interactive entrypoints (unchanged) ---------

CLI_READY = threading.Event()
API_HOST = os.environ.get("WEATHER_AGENT_API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("WEATHER_AGENT_API_PORT", "8080"))


def env_flag(name: str, default: str = "") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def make_json_safe(value: Any):
    if isinstance(value, BaseMessage):
        payload = {
            "type": value.type,
            "content": value.content,
        }
        if getattr(value, "name", None):
            payload["name"] = value.name
        if getattr(value, "tool_calls", None):
            payload["tool_calls"] = value.tool_calls
        return make_json_safe(payload)
    if isinstance(value, dict):
        return {key: make_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


def ensure_message_content(value: Any):
    """Return a value that satisfies LangChain's str-or-list content rule."""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, (str, dict)) for item in value):
        return value
    try:
        return json.dumps(make_json_safe(value), ensure_ascii=False)
    except TypeError:
        return str(value)


SSE_FORCE = env_flag("WEATHER_AGENT_FORCE_SSE")
IDE_INTEGRATION = env_flag("ENABLE_IDE_INTEGRATION")
SSE_PORT_PRESENT = bool(os.environ.get("CLAUDE_CODE_SSE_PORT"))
INTERACTIVE_SHELL_ENABLED = SSE_FORCE or IDE_INTEGRATION or SSE_PORT_PRESENT
VERBOSE = False


class InteractiveShellStreamer:
    """Minimal SSE emitter for the Codex interactive shell."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._write_lock = threading.Lock()
        self._session_lock = threading.Lock()
        self._next_session_id = 0

    def begin(self, *, source: str, prompt: str, metadata: dict | None = None):
        if not self.enabled:
            return None
        with self._session_lock:
            self._next_session_id += 1
            session_id = self._next_session_id
        payload = {"session_id": session_id, "source": source, "prompt": prompt}
        if metadata:
            payload.update(metadata)
        self._emit("response_start", payload)
        return session_id

    def message(self, message, session_id: int | None = None):
        if not self.enabled:
            return
        payload = serialize_message(message)
        if session_id is not None:
            payload["session_id"] = session_id
        self._emit("response_message", payload)

    def end(self, session_id: int | None = None, status: str = "ok", metadata: dict | None = None):
        if not self.enabled:
            return
        payload = {"status": status}
        if session_id is not None:
            payload["session_id"] = session_id
        if metadata:
            payload.update(metadata)
        self._emit("response_end", payload)

    def error(self, session_id: int | None, message: str):
        if not self.enabled:
            return
        payload = {"error": message}
        if session_id is not None:
            payload["session_id"] = session_id
        self._emit("response_error", payload)

    def _emit(self, event: str, payload):
        safe_payload = make_json_safe(payload)
        with self._write_lock:
            print(f"event: {event}", flush=True)
            print(f"data: {json.dumps(safe_payload, ensure_ascii=False)}", flush=True)
            print(flush=True)


SHELL_STREAMER = InteractiveShellStreamer(INTERACTIVE_SHELL_ENABLED)


class ConversationManager:
    """Thread-safe wrapper that keeps conversation state in sync with the graph."""

    def __init__(self, compiled_graph):
        self.graph = compiled_graph
        self.history = []
        self._lock = threading.Lock()

    @contextmanager
    def locked_submit(self, content: str, source: str = "cli"):
        """Run the graph while holding the manager lock until responses are consumed."""

        with self._lock:
            self.history.append(HumanMessage(content=ensure_message_content(content), name=source))
            reply_start = len(self.history)
            result = self.graph.invoke({"messages": self.history})
            self.history = result["messages"]
            responses = [msg for msg in self.history[reply_start:] if msg.type != "human"]
            yield responses

    def submit(self, content: str, source: str = "cli"):
        """Compatibility helper for callers that don't need extended locking."""

        with self.locked_submit(content, source) as responses:
            return responses


def stringify_content(content):
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except TypeError:
        return str(content)


def _message_style(message_type: str) -> str:
    return {
        "ai": "assistant",
        "assistant": "assistant",
        "tool": "tool",
        "system": "muted",
        "human": "user",
    }.get(message_type, "accent")


def print_message(message):
    """Pretty-print one message plus any tool calls."""
    label = message.type.upper()
    if getattr(message, "name", None):
        label = f"{label}[{message.name}]"
    label_text = cli_ui.color_text(label.ljust(12), style=_message_style(message.type), bold=True)
    content = stringify_content(getattr(message, "content", ""))
    print(f"{label_text} {content}")
    for tc in getattr(message, "tool_calls", []) or []:
        tool_label = cli_ui.color_text("  -> tool_call", style="tool")
        print(f"{tool_label} {stringify_content(tc)}")


def _extract_final_ai_global(messages):
    final_ai = None
    for msg in messages:
        if getattr(msg, "type", None) == "ai":
            final_ai = msg
    return final_ai


def _print_pretty_ai(message):
    if not message:
        return
    content = stringify_content(getattr(message, "content", "")).strip()
    if not content:
        return
    cli_ui.print_panel("Assistant", content, style="assistant")


def display_responses(messages, *, streamer=None, session_id=None, verbose=None):
    is_verbose = VERBOSE if verbose is None else verbose

    if is_verbose:
        for msg in messages:
            print_message(msg)
            if streamer is not None:
                streamer.message(msg, session_id=session_id)
        return

    final_ai = _extract_final_ai_global(messages)
    if final_ai:
        _print_pretty_ai(final_ai)
        if streamer is not None:
            streamer.message(final_ai, session_id=session_id)
    else:
        # Fall back to verbose output when no assistant reply is present.
        for msg in messages:
            print_message(msg)
            if streamer is not None:
                streamer.message(msg, session_id=session_id)


def reprint_prompt():
    if CLI_READY.is_set():
        print(cli_ui.prompt_label("You"), end="", flush=True)


def run_cli_chat(conversation: ConversationManager, stop_event: threading.Event):
    """Interactive multi-turn chat loop in the terminal."""
    cli_ui.print_banner(
        "Erosolar",
        "Universal AGI Agent - Advanced reasoning, coding, research, and task automation.",
    )
    cli_ui.print_status("Type 'exit' or 'quit' to leave. Use --verbose for tool traces.", kind="info")
    CLI_READY.set()

    try:
        while not stop_event.is_set():
            try:
                user_text = input(cli_ui.prompt_label("You")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                cli_ui.print_status("Exiting.", kind="warning")
                stop_event.set()
                break

            if not user_text:
                continue
            if user_text.lower() in {"exit", "quit"}:
                cli_ui.print_status("Goodbye!", kind="success")
                stop_event.set()
                break

            session_id = SHELL_STREAMER.begin(
                source="cli",
                prompt=user_text,
                metadata={"origin": "cli"}
            ) if SHELL_STREAMER.enabled else None

            try:
                with conversation.locked_submit(user_text, source="cli") as responses:
                    display_responses(
                        responses,
                        streamer=SHELL_STREAMER if SHELL_STREAMER.enabled else None,
                        session_id=session_id,
                    )
            except Exception as exc:  # noqa: BLE001
                cli_ui.print_status(f"[cli] agent error: {exc}", kind="error")
                if session_id is not None:
                    SHELL_STREAMER.error(session_id, str(exc))
            else:
                if session_id is not None:
                    SHELL_STREAMER.end(session_id=session_id, metadata={"origin": "cli"})
    finally:
        CLI_READY.clear()


def serialize_message(message):
    payload = {
        "type": message.type,
        "name": getattr(message, "name", None),
    }
    content = getattr(message, "content", "")
    payload["content"] = make_json_safe(content)
    if getattr(message, "tool_calls", None):
        payload["tool_calls"] = make_json_safe(message.tool_calls)
    return payload


def start_background_api(conversation: ConversationManager, host: str, port: int):
    """Launch a simple HTTP endpoint so external processes can inject prompts."""

    conv = conversation

    class ConversationHandler(BaseHTTPRequestHandler):
        conversation = conv

        def do_POST(self):
            if self.path != "/chat":
                self.send_error(404, "POST /chat to talk to the agent.")
                return

            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw_body.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON payload.")
                return

            message = payload.get("message")
            if not message:
                self.send_error(400, "Field 'message' is required.")
                return

            source = str(payload.get("source", "api"))
            try:
                responses = self.conversation.submit(str(message), source=source)
            except Exception as exc:  # noqa: BLE001
                self.send_error(500, f"Agent error: {exc}")
                return

            response_body = json.dumps(
                {"responses": [serialize_message(msg) for msg in responses]},
                ensure_ascii=True,
            ).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        def log_message(self, fmt, *args):  # noqa: D401
            print(f"[api] {fmt % args}")

    server = ThreadingHTTPServer((host, port), ConversationHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    cli_ui.print_status(
        f"REST API listening on http://{host}:{port}/chat (POST {{'message': '...'}})",
        kind="info",
    )
    return server



def parse_args():
    parser = argparse.ArgumentParser(description="LangGraph planner+executor (ReAct) CLI.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every intermediate message and tool call.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    global VERBOSE
    VERBOSE = args.verbose

    conversation = ConversationManager(graph)
    stop_event = threading.Event()

    api_server = None
    try:
        api_server = start_background_api(conversation, API_HOST, API_PORT)
    except OSError as exc:
        cli_ui.print_status(
            f"[api] Unable to start server on {API_HOST}:{API_PORT}: {exc}",
            kind="warning",
        )

    try:
        run_cli_chat(conversation, stop_event)
    finally:
        stop_event.set()
        if api_server:
            api_server.shutdown()
            api_server.server_close()
        if MCP_BRIDGE is not None:
            MCP_BRIDGE.close()


if __name__ == "__main__":
    main()
