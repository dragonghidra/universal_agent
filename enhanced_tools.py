"""Enhanced tools for comprehensive CLI agent capabilities (Claude Code-like features)."""

from __future__ import annotations

import glob as glob_module
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from agent_toolkit import clean_cwd, truncate_output


# ==================== FILE OPERATIONS ====================

def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """
    Perform exact string replacement in a file.

    Args:
        file_path: Path to the file to edit
        old_string: The exact string to find and replace
        new_string: The replacement string
        replace_all: If True, replace all occurrences; if False, only replace if unique

    Returns:
        Success or error message
    """
    target = clean_cwd(file_path)

    if not target.exists():
        return f"edit_file: {target} does not exist."

    if target.is_dir():
        return f"edit_file: {target} is a directory."

    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"edit_file: {target} is not valid UTF-8 text."

    if old_string not in content:
        return f"edit_file: old_string not found in {target}."

    occurrences = content.count(old_string)

    if not replace_all and occurrences > 1:
        return (
            f"edit_file: old_string appears {occurrences} times in {target}. "
            f"Provide a longer, unique string or set replace_all=True."
        )

    if replace_all:
        new_content = content.replace(old_string, new_string)
    else:
        new_content = content.replace(old_string, new_string, 1)

    target.write_text(new_content, encoding="utf-8")

    action = f"Replaced {occurrences} occurrence(s)" if replace_all else "Replaced 1 occurrence"
    return f"edit_file: {action} in {target}."


def glob_files(pattern: str, path: str = ".") -> str:
    """
    Find files matching a glob pattern (e.g., '**/*.py', 'src/**/*.ts').

    Args:
        pattern: Glob pattern to match
        path: Base directory to search from

    Returns:
        Newline-separated list of matching file paths
    """
    base = clean_cwd(path)

    if not base.exists():
        return f"glob_files: {base} does not exist."

    if not base.is_dir():
        return f"glob_files: {base} is not a directory."

    # Use pathlib for recursive glob
    matches = list(base.glob(pattern))

    # Sort by modification time (most recent first)
    matches.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

    if not matches:
        return f"glob_files: No files matching '{pattern}' in {base}."

    # Return relative paths
    results = []
    for match in matches:
        try:
            rel_path = match.relative_to(base)
            results.append(str(rel_path))
        except ValueError:
            results.append(str(match))

    return "\n".join(results)


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
    """
    Search for regex pattern in files (ripgrep-like functionality).

    Args:
        pattern: Regex pattern to search for
        path: Directory to search in
        file_pattern: Optional glob pattern to filter files (e.g., '*.py')
        context_lines: Number of context lines to show around matches
        case_insensitive: Case-insensitive search
        output_mode: 'files' = list files, 'matches' = show matches, 'count' = count matches
        max_results: Maximum number of results to return

    Returns:
        Search results based on output_mode
    """
    base = clean_cwd(path)

    if not base.exists():
        return f"grep_files: {base} does not exist."

    if not base.is_dir():
        return f"grep_files: {base} is not a directory."

    try:
        regex = re.compile(pattern, re.IGNORECASE if case_insensitive else 0)
    except re.error as exc:
        return f"grep_files: Invalid regex pattern: {exc}"

    # Determine files to search
    if file_pattern:
        search_files = list(base.glob(f"**/{file_pattern}"))
    else:
        search_files = [f for f in base.rglob("*") if f.is_file()]

    matches_by_file: Dict[Path, List[tuple[int, str]]] = {}
    total_matches = 0

    for file_path in search_files:
        if not file_path.is_file():
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue

        lines = content.splitlines()
        file_matches = []

        for line_num, line in enumerate(lines, 1):
            if regex.search(line):
                file_matches.append((line_num, line))
                total_matches += 1

                if total_matches >= max_results:
                    break

        if file_matches:
            matches_by_file[file_path] = file_matches

        if total_matches >= max_results:
            break

    if not matches_by_file:
        return f"grep_files: No matches found for pattern '{pattern}' in {base}."

    # Format output based on mode
    if output_mode == "count":
        lines = []
        for file_path, file_matches in matches_by_file.items():
            rel_path = file_path.relative_to(base) if file_path.is_relative_to(base) else file_path
            lines.append(f"{rel_path}: {len(file_matches)} matches")
        return "\n".join(lines)

    elif output_mode == "files":
        files = []
        for file_path in matches_by_file.keys():
            rel_path = file_path.relative_to(base) if file_path.is_relative_to(base) else file_path
            files.append(str(rel_path))
        return "\n".join(files)

    else:  # matches
        lines = []
        for file_path, file_matches in matches_by_file.items():
            rel_path = file_path.relative_to(base) if file_path.is_relative_to(base) else file_path
            lines.append(f"\n{rel_path}:")

            for line_num, line in file_matches:
                lines.append(f"  {line_num}: {line}")

                # Add context if requested
                if context_lines > 0:
                    try:
                        all_lines = file_path.read_text(encoding="utf-8").splitlines()
                        start = max(0, line_num - context_lines - 1)
                        end = min(len(all_lines), line_num + context_lines)

                        for ctx_num in range(start, line_num - 1):
                            lines.append(f"  {ctx_num + 1}- {all_lines[ctx_num]}")
                        for ctx_num in range(line_num, end):
                            lines.append(f"  {ctx_num + 1}+ {all_lines[ctx_num]}")
                    except Exception:
                        pass

        return truncate_output("\n".join(lines))


# ==================== GIT OPERATIONS ====================

def git_status(repo_path: str = ".") -> str:
    """Get git status for a repository."""
    cwd = clean_cwd(repo_path)

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=str(cwd),
            timeout=10,
        )

        if result.returncode != 0:
            return f"git_status: Error - {result.stderr or 'Not a git repository'}"

        if not result.stdout.strip():
            return "git_status: Working tree clean."

        return f"git_status:\n{result.stdout}"

    except subprocess.TimeoutExpired:
        return "git_status: Command timed out."
    except FileNotFoundError:
        return "git_status: git command not found."


def git_diff(repo_path: str = ".", staged: bool = False) -> str:
    """Get git diff for a repository."""
    cwd = clean_cwd(repo_path)

    cmd = ["git", "diff"]
    if staged:
        cmd.append("--staged")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(cwd),
            timeout=30,
        )

        if result.returncode != 0:
            return f"git_diff: Error - {result.stderr or 'Failed'}"

        if not result.stdout.strip():
            return "git_diff: No changes."

        return truncate_output(f"git_diff:\n{result.stdout}", limit=10000)

    except subprocess.TimeoutExpired:
        return "git_diff: Command timed out."
    except FileNotFoundError:
        return "git_diff: git command not found."


def git_commit(message: str, repo_path: str = ".", add_all: bool = False) -> str:
    """
    Create a git commit.

    Args:
        message: Commit message
        repo_path: Repository path
        add_all: If True, run 'git add -A' before committing

    Returns:
        Result message
    """
    cwd = clean_cwd(repo_path)

    try:
        if add_all:
            add_result = subprocess.run(
                ["git", "add", "-A"],
                capture_output=True,
                text=True,
                cwd=str(cwd),
                timeout=10,
            )

            if add_result.returncode != 0:
                return f"git_commit: git add failed - {add_result.stderr}"

        result = subprocess.run(
            ["git", "commit", "-m", message],
            capture_output=True,
            text=True,
            cwd=str(cwd),
            timeout=30,
        )

        if result.returncode != 0:
            return f"git_commit: Commit failed - {result.stderr or result.stdout}"

        return f"git_commit: {result.stdout.strip()}"

    except subprocess.TimeoutExpired:
        return "git_commit: Command timed out."
    except FileNotFoundError:
        return "git_commit: git command not found."


def git_log(repo_path: str = ".", max_commits: int = 10) -> str:
    """Get recent git commit history."""
    cwd = clean_cwd(repo_path)

    try:
        result = subprocess.run(
            ["git", "log", f"-{max_commits}", "--oneline", "--decorate"],
            capture_output=True,
            text=True,
            cwd=str(cwd),
            timeout=10,
        )

        if result.returncode != 0:
            return f"git_log: Error - {result.stderr or 'Failed'}"

        if not result.stdout.strip():
            return "git_log: No commits yet."

        return f"git_log:\n{result.stdout}"

    except subprocess.TimeoutExpired:
        return "git_log: Command timed out."
    except FileNotFoundError:
        return "git_log: git command not found."


# ==================== TASK MANAGEMENT ====================

class TodoManager:
    """In-memory todo/task manager for agent workflow tracking."""

    def __init__(self):
        self.todos: List[Dict[str, Any]] = []

    def add_todo(self, content: str, status: str = "pending", active_form: Optional[str] = None) -> str:
        """Add a new todo item."""
        todo = {
            "id": len(self.todos) + 1,
            "content": content,
            "status": status,
            "activeForm": active_form or f"{content}...",
        }
        self.todos.append(todo)
        return f"Added todo #{todo['id']}: {content}"

    def update_todo(self, todo_id: int, status: Optional[str] = None, content: Optional[str] = None) -> str:
        """Update an existing todo."""
        for todo in self.todos:
            if todo["id"] == todo_id:
                if status:
                    todo["status"] = status
                if content:
                    todo["content"] = content
                return f"Updated todo #{todo_id}"
        return f"Todo #{todo_id} not found"

    def list_todos(self, status_filter: Optional[str] = None) -> str:
        """List todos, optionally filtered by status."""
        if not self.todos:
            return "No todos yet."

        filtered = self.todos
        if status_filter:
            filtered = [t for t in self.todos if t["status"] == status_filter]

        if not filtered:
            return f"No todos with status '{status_filter}'."

        lines = ["Current todos:"]
        for todo in filtered:
            status_symbol = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}.get(
                todo["status"], "❓"
            )
            lines.append(f"{status_symbol} #{todo['id']} [{todo['status']}] {todo['content']}")

        return "\n".join(lines)

    def clear_completed(self) -> str:
        """Remove completed todos."""
        before = len(self.todos)
        self.todos = [t for t in self.todos if t["status"] != "completed"]
        after = len(self.todos)
        removed = before - after
        return f"Removed {removed} completed todo(s)."


TODO_MANAGER = TodoManager()


def manage_todos(
    action: Literal["add", "update", "list", "clear_completed"],
    content: Optional[str] = None,
    todo_id: Optional[int] = None,
    status: Optional[str] = None,
    status_filter: Optional[str] = None,
) -> str:
    """
    Manage todos/tasks for tracking agent workflow.

    Args:
        action: Action to perform (add, update, list, clear_completed)
        content: Todo content (for add/update)
        todo_id: Todo ID (for update)
        status: New status (for update)
        status_filter: Filter by status (for list)

    Returns:
        Result message
    """
    if action == "add":
        if not content:
            return "manage_todos: content required for add action."
        return TODO_MANAGER.add_todo(content)

    elif action == "update":
        if todo_id is None:
            return "manage_todos: todo_id required for update action."
        return TODO_MANAGER.update_todo(todo_id, status=status, content=content)

    elif action == "list":
        return TODO_MANAGER.list_todos(status_filter=status_filter)

    elif action == "clear_completed":
        return TODO_MANAGER.clear_completed()

    return f"manage_todos: Unknown action '{action}'."


# ==================== JUPYTER NOTEBOOK OPERATIONS ====================

def edit_notebook_cell(
    notebook_path: str,
    cell_index: int,
    new_source: str,
    cell_type: Literal["code", "markdown"] = "code",
) -> str:
    """
    Edit a cell in a Jupyter notebook.

    Args:
        notebook_path: Path to .ipynb file
        cell_index: 0-based cell index
        new_source: New cell content
        cell_type: Cell type (code or markdown)

    Returns:
        Success or error message
    """
    target = clean_cwd(notebook_path)

    if not target.exists():
        return f"edit_notebook_cell: {target} does not exist."

    if not target.suffix == ".ipynb":
        return f"edit_notebook_cell: {target} is not a .ipynb file."

    try:
        with open(target, "r", encoding="utf-8") as f:
            notebook = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return f"edit_notebook_cell: Failed to parse notebook - {exc}"

    cells = notebook.get("cells", [])

    if not (0 <= cell_index < len(cells)):
        return f"edit_notebook_cell: Cell index {cell_index} out of range (0-{len(cells)-1})."

    # Update cell
    cells[cell_index]["cell_type"] = cell_type
    cells[cell_index]["source"] = new_source.splitlines(keepends=True)

    # Clear outputs if code cell
    if cell_type == "code":
        cells[cell_index]["outputs"] = []
        cells[cell_index]["execution_count"] = None

    # Write back
    with open(target, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    return f"edit_notebook_cell: Updated cell {cell_index} in {target}."


# ==================== CODE ANALYSIS & DIAGNOSTICS ====================

def analyze_code_quality(file_path: str, language: Optional[str] = None) -> str:
    """
    Analyze code quality using basic static analysis.

    Args:
        file_path: Path to source file
        language: Programming language (auto-detected if not provided)

    Returns:
        Analysis results
    """
    target = clean_cwd(file_path)

    if not target.exists():
        return f"analyze_code_quality: {target} does not exist."

    if target.is_dir():
        return f"analyze_code_quality: {target} is a directory."

    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"analyze_code_quality: {target} is not valid UTF-8 text."

    # Auto-detect language
    if not language:
        ext = target.suffix.lower()
        lang_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".cpp": "cpp",
            ".c": "c",
        }
        language = lang_map.get(ext, "unknown")

    lines = content.splitlines()

    # Basic metrics
    total_lines = len(lines)
    blank_lines = sum(1 for line in lines if not line.strip())
    comment_lines = 0

    # Language-specific comment detection
    if language == "python":
        comment_lines = sum(1 for line in lines if line.strip().startswith("#"))
    elif language in {"javascript", "typescript", "java", "cpp", "c", "go", "rust"}:
        comment_lines = sum(1 for line in lines if line.strip().startswith("//"))

    code_lines = total_lines - blank_lines - comment_lines

    # Find long lines (>120 chars)
    long_lines = [(i+1, len(line)) for i, line in enumerate(lines) if len(line) > 120]

    # Find potential issues
    issues = []

    if language == "python":
        # Check for common Python issues
        for i, line in enumerate(lines, 1):
            if "except:" in line and "pass" in lines[i] if i < len(lines) else False:
                issues.append(f"Line {i}: Bare except with pass")
            if line.strip().startswith("print(") and "debug" in line.lower():
                issues.append(f"Line {i}: Debug print statement")

    # Build report
    report = [
        f"Code Quality Analysis: {target.name}",
        f"Language: {language}",
        f"",
        f"Metrics:",
        f"  Total lines: {total_lines}",
        f"  Code lines: {code_lines}",
        f"  Blank lines: {blank_lines}",
        f"  Comment lines: {comment_lines}",
        f"",
    ]

    if long_lines:
        report.append(f"Long lines (>120 chars): {len(long_lines)}")
        for line_num, length in long_lines[:5]:
            report.append(f"  Line {line_num}: {length} chars")
        if len(long_lines) > 5:
            report.append(f"  ... and {len(long_lines) - 5} more")
        report.append("")

    if issues:
        report.append("Potential Issues:")
        for issue in issues[:10]:
            report.append(f"  {issue}")
        if len(issues) > 10:
            report.append(f"  ... and {len(issues) - 10} more")
    else:
        report.append("No obvious issues detected.")

    return "\n".join(report)


# ==================== WEB OPERATIONS ====================

def web_search_simple(query: str, max_results: int = 5) -> str:
    """
    Simple web search using DuckDuckGo's instant answer API.

    Args:
        query: Search query
        max_results: Maximum results

    Returns:
        Search results
    """
    try:
        import requests
    except ImportError:
        return "web_search_simple: requests library not available."

    try:
        # Use DuckDuckGo instant answer API
        response = requests.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        results = []

        # Abstract
        if data.get("Abstract"):
            results.append(f"Summary: {data['Abstract']}")
            if data.get("AbstractURL"):
                results.append(f"Source: {data['AbstractURL']}")

        # Related topics
        related = data.get("RelatedTopics", [])
        if related:
            results.append("\nRelated:")
            for i, topic in enumerate(related[:max_results], 1):
                if isinstance(topic, dict) and "Text" in topic:
                    results.append(f"{i}. {topic['Text']}")
                    if "FirstURL" in topic:
                        results.append(f"   {topic['FirstURL']}")

        if not results:
            return f"web_search_simple: No results found for '{query}'."

        return "\n".join(results)

    except Exception as exc:
        return f"web_search_simple: Search failed - {exc}"


__all__ = [
    "edit_file",
    "glob_files",
    "grep_files",
    "git_status",
    "git_diff",
    "git_commit",
    "git_log",
    "manage_todos",
    "TODO_MANAGER",
    "edit_notebook_cell",
    "analyze_code_quality",
    "web_search_simple",
]
