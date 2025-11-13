"""Shared tool implementations used by the universal agent and MCP server."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import requests


MAX_TOOL_OUTPUT = int(os.environ.get("AGENT_MAX_TOOL_OUTPUT", "8000"))
AUTOMATION_DIR = Path(os.environ.get("AGENT_AUTOMATION_DIR", "automation_scripts")).expanduser()


def truncate_output(text: str, *, limit: int = MAX_TOOL_OUTPUT) -> str:
    """Trim long stdout/stderr blobs so they fit within model context budgets."""
    if text is None:
        return ""
    normalized = text.strip("\x00")
    if len(normalized) <= limit:
        return normalized
    omitted = len(normalized) - limit
    return normalized[:limit] + f"\n... (truncated {omitted} characters)"


def clean_cwd(cwd: Optional[str]) -> Path:
    """Resolve relative working directories safely."""
    base = Path.cwd()
    if not cwd:
        return base
    target = Path(cwd).expanduser()
    if not target.is_absolute():
        target = base / target
    return target


def safe_slug(value: str, fallback: str = "automation") -> str:
    """Convert arbitrary labels into filesystem-friendly slugs."""
    cleaned = "".join(ch for ch in value if ch.isalnum() or ch in {"-", "_"}).strip("_-")
    return cleaned or fallback


def automation_path(name: str, suffix: str = ".sh") -> Path:
    """Return the on-disk path for a saved automation script."""
    slug = safe_slug(name)
    AUTOMATION_DIR.mkdir(parents=True, exist_ok=True)
    return AUTOMATION_DIR / f"{slug}{suffix}"


def format_process_result(result: subprocess.CompletedProcess) -> str:
    """Render stdout/stderr plus exit code in a deterministic format."""
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    combined = ""
    if stdout:
        combined += f"[stdout]\\n{stdout}"
    if stderr:
        if combined:
            combined += "\n"
        combined += f"[stderr]\n{stderr}"
    combined = combined or "(no output)"
    combined += f"\n(exit code {result.returncode})"
    return truncate_output(combined)


def run_python(code: str, timeout: int = 60) -> str:
    """Execute arbitrary Python code in a fresh interpreter process and capture stdout/stderr."""
    safe_timeout = max(1, min(int(timeout), 300))
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=safe_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        partial_out = (exc.stdout or "") + (exc.stderr or "")
        return truncate_output(f"Python timed out after {safe_timeout}s.\n{partial_out}")
    except FileNotFoundError:
        return "python3 is not available in this environment."
    return format_process_result(result)


def run_shell(command: str, timeout: int = 60, cwd: Optional[str] = None) -> str:
    """Run a shell command with /bin/bash -lc and return stdout/stderr."""
    safe_timeout = max(1, min(int(timeout), 300))
    working_dir = clean_cwd(cwd)
    try:
        result = subprocess.run(
            ["bash", "-lc", command],
            capture_output=True,
            text=True,
            timeout=safe_timeout,
            cwd=str(working_dir),
        )
    except subprocess.TimeoutExpired as exc:
        partial_out = (exc.stdout or "") + (exc.stderr or "")
        return truncate_output(f"Shell command timed out after {safe_timeout}s.\n{partial_out}")
    return format_process_result(result)


def list_directory(path: str = ".") -> str:
    """List files/folders at a path (non-recursive)."""
    target = clean_cwd(path)
    if not target.exists():
        return f"{target} does not exist."
    if not target.is_dir():
        return f"{target} is not a directory."
    entries = []
    for entry in sorted(target.iterdir()):
        kind = "dir" if entry.is_dir() else "file"
        size = entry.stat().st_size
        entries.append(f"{entry.name}\t{kind}\t{size} bytes")
    return "\n".join(entries) or "(empty directory)"


def read_text(path: str, max_chars: int = 8000) -> str:
    """Read up to max_chars from a UTF-8 text file."""
    target = clean_cwd(path)
    if not target.exists():
        return f"{target} does not exist."
    if target.is_dir():
        return f"{target} is a directory. Provide a file path."
    try:
        data = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"{target} is not valid UTF-8 text."
    limit = max(1, min(int(max_chars), 20000))
    return truncate_output(data, limit=limit)


def write_text(path: str, content: str, mode: Literal["overwrite", "append"] = "overwrite") -> str:
    """Write content to a file, creating parent directories as needed."""
    target = clean_cwd(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_mode = "w" if mode == "overwrite" else "a"
    with target.open(write_mode, encoding="utf-8") as handle:
        handle.write(content)
    action = "overwritten" if mode == "overwrite" else "appended"
    return f"{target} {action} ({len(content)} characters)."


def save_shell_automation(
    name: str,
    content: str,
    run: bool = False,
    timeout: int = 120,
) -> str:
    """Persist a reusable shell script under automation_scripts/ and optionally execute it."""
    target = automation_path(name, ".sh")
    target.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")
    mode = target.stat().st_mode
    target.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    message = f"Saved automation script to {target}"
    if not run:
        return message + f". Run later with: bash {target}"

    safe_timeout = max(1, min(int(timeout), 600))
    try:
        result = subprocess.run(
            ["bash", str(target)],
            capture_output=True,
            text=True,
            timeout=safe_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        partial = (exc.stdout or "") + (exc.stderr or "")
        return truncate_output(
            f"{message}. Execution timed out after {safe_timeout}s.\n{partial}"
        )
    output = format_process_result(result)
    return f"{message} and executed it.\n{output}"


def get_weather(location: str, units: Literal["us", "metric"] = "us") -> str:
    """Get current weather for a place name using the Open-Meteo API."""
    geo_resp = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location, "count": 1, "language": "en", "format": "json"},
        timeout=10,
    )
    geo_resp.raise_for_status()
    geo = geo_resp.json()
    if not geo.get("results"):
        return f"Couldn't find '{location}'. Try a more specific name."

    place = geo["results"][0]
    lat, lon = place["latitude"], place["longitude"]
    resolved = ", ".join(
        p for p in [place.get("name"), place.get("admin1"), place.get("country")] if p
    )

    temp_unit = "fahrenheit" if units == "us" else "celsius"
    wind_unit = "mph" if units == "us" else "kmh"

    current_vars = ",".join(
        [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "weather_code",
            "precipitation",
            "cloud_cover",
        ]
    )

    wx_resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": current_vars,
            "temperature_unit": temp_unit,
            "wind_speed_unit": wind_unit,
            "timezone": "auto",
        },
        timeout=10,
    )
    wx_resp.raise_for_status()
    wx = wx_resp.json().get("current", {})

    def get(key, default="—"):
        return wx.get(key, default)

    WMO = {
        0: "Clear",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Drizzle",
        55: "Heavy drizzle",
        61: "Light rain",
        63: "Rain",
        65: "Heavy rain",
        71: "Light snow",
        73: "Snow",
        75: "Heavy snow",
        80: "Rain showers",
        81: "Heavy rain showers",
        82: "Violent rain showers",
        95: "Thunderstorm",
        96: "Thunderstorm w/ hail",
        99: "Severe thunderstorm w/ hail",
    }

    code = int(get("weather_code", -1)) if str(get("weather_code", "")).isdigit() else -1
    desc = WMO.get(code, "Unknown")

    return (
        f"{resolved} — {get('temperature_2m')}°{'F' if units=='us' else 'C'}, "
        f"RH {get('relative_humidity_2m')}%, "
        f"wind {get('wind_speed_10m')} {wind_unit} "
        f"({get('wind_direction_10m')}°), "
        f"precip {get('precipitation')} mm, cloud {get('cloud_cover')}%, "
        f"{desc}. "
        f"[{get('time','now')}]"
    )


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
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return (
            "Playwright is not installed. Run `pip install playwright` "
            "and `playwright install` to enable headless browsing."
        )

    safe_timeout = max(5, min(int(timeout), 300)) * 1000
    screenshot_saved = ""
    html_saved = ""
    js_result = None
    html_preview = ""

    with sync_playwright() as pw:
        browser_type = getattr(pw, browser, pw.chromium)
        launch = browser_type.launch(headless=True)
        context = None
        try:
            context_kwargs: Dict[str, Any] = {}
            if emulate_device:
                device = pw.devices.get(emulate_device)
                if not device:
                    return f"Unknown Playwright device profile '{emulate_device}'."
                context_kwargs.update(device)
            context = launch.new_context(**context_kwargs)
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=safe_timeout)
            if wait_selector:
                page.wait_for_selector(wait_selector, timeout=safe_timeout)
            if wait_seconds:
                page.wait_for_timeout(max(0, int(wait_seconds)) * 1000)
            if javascript:
                try:
                    js_result = page.evaluate(javascript)
                except Exception as exc:  # noqa: BLE001
                    js_result = f"JavaScript error: {exc}"
            if screenshot_path:
                shot_target = clean_cwd(screenshot_path)
                shot_target.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(shot_target), full_page=True)
                screenshot_saved = f"Screenshot saved to {shot_target}."
            if save_html_path:
                html_target = clean_cwd(save_html_path)
                html_target.parent.mkdir(parents=True, exist_ok=True)
                html_target.write_text(page.content(), encoding="utf-8")
                html_saved = f"HTML saved to {html_target}."
            html_preview = page.content()
        finally:
            if context:
                context.close()
            launch.close()

    parts = [
        f"Loaded {url} with {browser} headless browser.",
    ]
    if wait_selector:
        parts.append(f"Waited for CSS selector '{wait_selector}'.")
    if wait_seconds:
        parts.append(f"Waited an additional {wait_seconds}s.")
    if screenshot_saved:
        parts.append(screenshot_saved)
    if html_saved:
        parts.append(html_saved)
    if js_result is not None:
        parts.append(f"JavaScript result: {js_result}")
    parts.append("Page preview:\n" + truncate_output(html_preview, limit=4000))
    return "\n".join(parts)


__all__ = [
    "AUTOMATION_DIR",
    "MAX_TOOL_OUTPUT",
    "automation_path",
    "clean_cwd",
    "format_process_result",
    "get_weather",
    "headless_browse",
    "list_directory",
    "read_text",
    "run_python",
    "run_shell",
    "save_shell_automation",
    "safe_slug",
    "truncate_output",
    "write_text",
]
