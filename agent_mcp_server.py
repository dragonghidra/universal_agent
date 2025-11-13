"""Expose the universal agent's local capabilities as an MCP server."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from mcp.server.fastmcp import FastMCP

from agent_toolkit import (
    AUTOMATION_DIR,
    get_weather as core_get_weather,
    headless_browse as core_headless_browse,
    run_shell as core_run_shell,
)


app = FastMCP("universal-agent-tools")


@app.tool(name="weather.report", description="Fetch current weather for a city or place name.")
def get_weather(location: str, units: Literal["us", "metric"] = "us") -> str:
    return core_get_weather(location, units)


@app.tool(
    name="shell.run",
    description="Execute a bash command with optional timeout and working directory.",
)
def run_shell(command: str, timeout: int = 60, cwd: Optional[str] = None) -> str:
    return core_run_shell(command, timeout=timeout, cwd=cwd)


@app.tool(
    name="browser.headless",
    description="Drive a headless Playwright browser to capture screenshots and HTML.",
)
def headless_browse(
    url: str,
    wait_selector: Optional[str] = None,
    wait_seconds: Optional[int] = None,
    screenshot_path: Optional[str] = None,
    save_html_path: Optional[str] = None,
    javascript: Optional[str] = None,
    browser: Literal["chromium", "firefox", "webkit"] = "chromium",
    emulate_device: Optional[str] = None,
    timeout: int = 45,
) -> str:
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


def _list_automation_scripts() -> str:
    AUTOMATION_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for path in sorted(AUTOMATION_DIR.glob("*")):
        if path.is_file():
            entries.append(f"{path.name}\t{path.stat().st_size} bytes")
    return "\n".join(entries) or "No automation scripts have been saved yet."


@app.resource("automation://scripts", name="automation.scripts", description="List saved automation shell scripts.")
def automation_index() -> str:
    return _list_automation_scripts()


@app.resource(
    "automation://script/{name}",
    name="automation.script",
    description="Fetch the contents of a saved automation script.",
)
def automation_script(name: str) -> str:
    safe_name = Path(name).name
    target = AUTOMATION_DIR / safe_name
    if not target.exists():
        return f"{target} does not exist."
    if target.is_dir():
        return f"{target} is a directory."
    return target.read_text(encoding="utf-8")


@app.prompt(
    name="automation_runbook",
    title="Automation Runbook",
    description="Kick off a shell automation plan for a given task.",
)
def automation_prompt(task: str) -> list[Dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are an experienced automation engineer. Provide a numbered checklist "
                        "that can be executed via shell scripts or CLI commands."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"Task to automate: {task}"}],
        },
    ]


@app.prompt(
    name="web_capture_brief",
    title="Headless browse template",
    description="Template prompt for capturing screenshots and HTML snapshots.",
)
def browser_prompt(url: str) -> list[Dict[str, Any]]:
    instructions = (
        "Use the 'browser.headless' tool to capture the page, wait for key selectors if necessary, "
        "and summarize any critical findings."
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": instructions}]},
        {"role": "user", "content": [{"type": "text", "text": f"Target URL: {url}"}]},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the universal-agent MCP server.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable_http"],
        default="stdio",
        help="Transport to expose. stdio is ideal for local hosts; SSE/Streamable HTTP work for network hosts.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for SSE/Streamable HTTP transports.")
    parser.add_argument("--port", type=int, default=3928, help="Port for SSE/Streamable HTTP transports.")
    parser.add_argument(
        "--mount-path",
        default="/mcp",
        help="Mount path for SSE transport (ignored for stdio).",
    )
    return parser.parse_args()


def configure_server(host: str, port: int) -> None:
    app.settings.host = host
    app.settings.port = port


def main() -> None:
    args = parse_args()
    configure_server(args.host, args.port)
    if args.transport == "stdio":
        asyncio.run(app.run_stdio_async())
    elif args.transport == "sse":
        asyncio.run(app.run_sse_async(mount_path=args.mount_path))
    else:
        asyncio.run(app.run_streamable_http_async())


if __name__ == "__main__":
    main()
