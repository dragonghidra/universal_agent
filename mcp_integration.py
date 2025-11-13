"""Utilities for bridging LangChain tools with MCP servers."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

from anyio.from_thread import start_blocking_portal
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, ValidationError, ConfigDict, create_model

from agent_toolkit import truncate_output

try:  # pragma: no cover - optional MCP dependency
    import mcp
    from mcp import types
    from mcp.client.session_group import (
        ClientSessionGroup,
        SseServerParameters,
        StdioServerParameters,
        StreamableHttpParameters,
    )
except ImportError:  # pragma: no cover - handled gracefully at runtime
    mcp = None
    types = None
    ClientSessionGroup = None
    StdioServerParameters = None
    SseServerParameters = None
    StreamableHttpParameters = None


LOG = logging.getLogger(__name__)
DEFAULT_CONFIG_LOCATIONS = [
    Path(os.environ.get("AGENT_MCP_CONFIG", "")).expanduser()
    if os.environ.get("AGENT_MCP_CONFIG")
    else None,
    Path("mcp_servers.json"),
]


class MCPClientError(RuntimeError):
    """Raised when MPC client setup fails."""


@dataclass
class MCPServerConfig:
    """Describes how to connect to a single MCP server."""

    name: str
    transport: str = "stdio"
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    cwd: Optional[str] = None
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    timeout: Optional[float] = None
    sse_read_timeout: Optional[float] = None
    namespace: Optional[str] = None

    def label(self, server_info: Optional[types.Implementation]) -> str:
        """Create a human-friendly server label for tool descriptions."""
        if self.namespace:
            return self.namespace
        if server_info and server_info.name:
            return server_info.name
        if self.name:
            return self.name
        return self.transport

    def to_parameters(self):
        if self.transport == "stdio":
            if not StdioServerParameters:
                raise MCPClientError("MCP stdio transport is unavailable.")
            if not self.command:
                raise MCPClientError(f"Server '{self.name}' is missing a 'command' for stdio transport.")
            return StdioServerParameters(
                command=self.command,
                args=self.args or [],
                env=self.env,
                cwd=self.cwd,
            )
        if self.transport == "sse":
            if not SseServerParameters:
                raise MCPClientError("MCP SSE transport is unavailable.")
            if not self.url:
                raise MCPClientError(f"Server '{self.name}' is missing a 'url' for SSE transport.")
            return SseServerParameters(
                url=self.url,
                headers=self.headers,
                timeout=self.timeout or 5,
                sse_read_timeout=self.sse_read_timeout or 300,
            )
        if self.transport == "streamable_http":
            if not StreamableHttpParameters:
                raise MCPClientError("MCP Streamable HTTP transport is unavailable.")
            if not self.url:
                raise MCPClientError(f"Server '{self.name}' is missing a 'url' for Streamable HTTP transport.")
            return StreamableHttpParameters(
                url=self.url,
                headers=self.headers,
                timeout=timedelta(seconds=self.timeout or 30),
                sse_read_timeout=timedelta(seconds=self.sse_read_timeout or 300),
                terminate_on_close=True,
            )
        raise MCPClientError(f"Unsupported transport '{self.transport}' for server '{self.name}'.")


@dataclass
class MCPToolSpec:
    """Metadata for a single remote MCP tool."""

    qualified_name: str
    display_name: str
    description: str
    input_schema: Dict[str, Any] | None
    server_label: str

    def summarize_schema(self) -> str:
        schema = self.input_schema or {}
        props = schema.get("properties") or {}
        if not props:
            return "Arguments: none."
        required = set(schema.get("required") or [])
        lines = []
        for field, meta in props.items():
            schema_type = meta.get("type") or meta.get("anyOf") or meta.get("oneOf") or "any"
            if isinstance(schema_type, list):
                schema_type = ", ".join(schema_type)
            desc = meta.get("description") or ""
            flag = "required" if field in required else "optional"
            snippet = f"- {field} ({schema_type}, {flag})"
            if desc:
                snippet += f": {desc}"
            lines.append(snippet)
        return "Arguments:\n" + "\n".join(lines)

    def safe_name(self) -> str:
        return "".join(ch if ch.isalnum() else "_" for ch in self.qualified_name)

    def to_langchain_tool(self, bridge: "MCPClientBridge") -> BaseTool:
        args_model = build_args_model(f"{self.safe_name()}Args", self.input_schema)

        def _call(**kwargs):
            return bridge.call_tool(self.qualified_name, kwargs)

        description = self.description or "Remote MCP tool."
        description += f"\nProvided by MCP server: {self.server_label}."
        description += f"\n{self.summarize_schema()}"

        structured_tool = StructuredTool.from_function(
            func=_call,
            name=f"mcp::{self.qualified_name}",
            description=description.strip(),
            args_schema=args_model,
        )
        return structured_tool


def build_args_model(class_name: str, schema: Dict[str, Any] | None) -> Type[BaseModel]:
    """Convert a JSON schema to a pydantic model for LangChain's StructuredTool."""
    schema = schema or {}
    properties: Dict[str, Any] = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    model_fields: Dict[str, Tuple[Any, Field]] = {}

    def pick_type(meta: Dict[str, Any]) -> Any:
        raw_type = meta.get("type")
        if isinstance(raw_type, list):
            raw_type = next((t for t in raw_type if t != "null"), raw_type[0] if raw_type else None)
        if raw_type == "string":
            return str
        if raw_type == "integer":
            return int
        if raw_type == "number":
            return float
        if raw_type == "boolean":
            return bool
        if raw_type == "array":
            inner = pick_type(meta.get("items") or {})
            return List[inner]  # type: ignore[misc]
        if raw_type == "object":
            return Dict[str, Any]
        return Any

    for key, meta in properties.items():
        annotation = pick_type(meta)
        field_description = meta.get("description")
        default = meta.get("default", None if key not in required else ...)
        if default is ...:
            field_info = Field(..., description=field_description)
            model_fields[key] = (annotation, field_info)
        else:
            optional_type = annotation if annotation is Any else Optional[annotation]  # type: ignore[misc]
            field_info = Field(default, description=field_description)
            model_fields[key] = (optional_type, field_info)

    if not model_fields:
        return create_model(class_name, __config__=ConfigDict(extra="allow"))

    return create_model(class_name, __config__=ConfigDict(extra="allow"), **model_fields)


def load_server_configs() -> tuple[List[MCPServerConfig], List[str]]:
    """Load MCP server configuration from env vars or JSON files."""
    warnings: List[str] = []
    configs: List[MCPServerConfig] = []
    inline = os.environ.get("AGENT_MCP_SERVERS")
    sources: List[Dict[str, Any]] = []

    if inline:
        try:
            payload = json.loads(inline)
            if isinstance(payload, dict):
                payload = [payload]
            if isinstance(payload, list):
                sources.extend(payload)
            else:
                warnings.append("AGENT_MCP_SERVERS must be a JSON object or list.")
        except json.JSONDecodeError as exc:
            warnings.append(f"Invalid JSON in AGENT_MCP_SERVERS: {exc}")
    else:
        for candidate in DEFAULT_CONFIG_LOCATIONS:
            if not candidate:
                continue
            if candidate.exists():
                try:
                    payload = json.loads(candidate.read_text())
                except json.JSONDecodeError as exc:
                    warnings.append(f"Invalid JSON in {candidate}: {exc}")
                    continue
                if isinstance(payload, dict):
                    payload = [payload]
                if isinstance(payload, list):
                    sources.extend(payload)
                else:
                    warnings.append(f"{candidate} must contain a JSON object or array of server entries.")
                break

    for idx, raw in enumerate(sources):
        if not isinstance(raw, dict):
            warnings.append(f"Ignoring MCP server entry #{idx+1}: expected object.")
            continue
        entry = dict(raw)
        name = str(entry.get("name") or entry.get("id") or f"server{idx+1}")
        config = MCPServerConfig(
            name=name,
            transport=str(entry.get("transport", "stdio")),
            command=entry.get("command"),
            args=entry.get("args"),
            env=entry.get("env"),
            cwd=entry.get("cwd"),
            url=entry.get("url"),
            headers=entry.get("headers"),
            timeout=entry.get("timeout"),
            sse_read_timeout=entry.get("sse_read_timeout"),
            namespace=entry.get("namespace"),
        )
        configs.append(config)

    return configs, warnings


class MCPClientBridge:
    """Manages persistent MCP connections and exposes LangChain-friendly hooks."""

    def __init__(self, configs: Sequence[MCPServerConfig], *, max_output: int = 4000):
        if not configs:
            raise MCPClientError("No MCP servers were configured.")
        if ClientSessionGroup is None or mcp is None:
            raise MCPClientError("The 'mcp' Python package is required but not installed.")
        self._configs = list(configs)
        self._max_output = max_output
        self._portal_cm = start_blocking_portal()
        self._portal = self._portal_cm.__enter__()
        self._group: ClientSessionGroup | None = None
        self._session_labels: Dict[int, str] = {}
        self._tool_specs: List[MCPToolSpec] = []
        self.connection_warnings: List[str] = []
        try:
            self._group = self._portal.call(self._async_setup_group)
            self._tool_specs = self._portal.call(self._collect_tool_specs)
        except Exception:
            self.close()
            raise
        if not self._tool_specs:
            self.close()
            raise MCPClientError("Connected to MCP servers but no tools were exposed.")

    @property
    def tool_specs(self) -> List[MCPToolSpec]:
        return list(self._tool_specs)

    def close(self) -> None:
        """Shut down the background MCP session(s)."""
        if self._group is not None:
            try:
                self._portal.call(self._async_teardown_group)
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                LOG.warning("Error while closing MCP session group: %s", exc)
            finally:
                self._group = None
        if self._portal_cm is not None:
            self._portal_cm.__exit__(None, None, None)
            self._portal_cm = None

    async def _async_setup_group(self) -> ClientSessionGroup:
        assert ClientSessionGroup is not None
        group = ClientSessionGroup(component_name_hook=self._component_name_hook)
        await group.__aenter__()
        warnings: List[str] = []
        for config in self._configs:
            try:
                params = config.to_parameters()
                server_info, session = await group._establish_session(params)  # noqa: SLF001
                await group.connect_with_session(server_info, session)
                self._session_labels[id(session)] = config.label(server_info)
            except Exception as exc:  # noqa: BLE001
                warning = f"Failed to connect to MCP server '{config.name}': {exc}"
                LOG.warning(warning)
                warnings.append(warning)
        self.connection_warnings = warnings
        return group

    async def _collect_tool_specs(self) -> List[MCPToolSpec]:
        assert self._group is not None
        specs: List[MCPToolSpec] = []
        for qualified_name, tool in (self._group.tools or {}).items():
            session = self._group._tool_to_session.get(qualified_name)  # noqa: SLF001
            label = self._session_labels.get(id(session), "mcp-server")
            specs.append(
                MCPToolSpec(
                    qualified_name=qualified_name,
                    display_name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema,
                    server_label=label,
                )
            )
        return specs

    async def _async_teardown_group(self) -> None:
        assert self._group is not None
        await self._group.__aexit__(None, None, None)

    @staticmethod
    def _component_name_hook(component_name: str, server_info: types.Implementation) -> str:
        prefix = server_info.name or "mcp"
        version = server_info.version
        if version:
            prefix = f"{prefix}@{version}"
        safe_prefix = "".join(ch if ch.isalnum() else "_" for ch in prefix)
        return f"{safe_prefix}.{component_name}"

    def call_tool(self, qualified_name: str, arguments: Dict[str, Any]) -> str:
        """Synchronously invoke an MCP tool and return formatted text."""
        try:
            result = self._portal.call(self._async_call_tool, qualified_name, arguments or {})
        except Exception as exc:  # noqa: BLE001
            return f"MCP tool '{qualified_name}' failed: {exc}"
        return render_call_result(result, limit=self._max_output)

    async def _async_call_tool(self, qualified_name: str, arguments: Dict[str, Any]):
        assert self._group is not None
        return await self._group.call_tool(qualified_name, arguments)


def render_call_result(result: types.CallToolResult, *, limit: int) -> str:
    """Convert a CallToolResult into human-readable text."""
    if not isinstance(result, types.CallToolResult):
        return truncate_output(str(result), limit=limit)

    lines: List[str] = []
    if result.isError:
        lines.append("[MCP ERROR]")

    for block in result.content or []:
        if hasattr(block, "model_dump"):
            data = block.model_dump()
        elif isinstance(block, dict):
            data = block
        else:
            data = {"type": "text", "text": str(block)}
        block_type = data.get("type")
        if block_type == "text":
            lines.append(data.get("text", ""))
        elif block_type == "image":
            payload = data.get("data") or ""
            lines.append(f"[image:{data.get('mimeType','application/octet-stream')} - {len(payload)} base64 chars]")
        elif block_type == "audio":
            payload = data.get("data") or ""
            lines.append(f"[audio:{data.get('mimeType','application/octet-stream')} - {len(payload)} base64 chars]")
        elif block_type == "resource":
            resource = data.get("resource") or {}
            uri = resource.get("uri") or resource.get("name") or "resource"
            lines.append(f"[resource] {uri}")
            contents = resource.get("text") or resource.get("content")
            if contents:
                lines.append(str(contents))
        elif block_type == "resource_link":
            uri = data.get("uri") or "resource"
            lines.append(f"[resource link] {uri}")
        else:
            lines.append(json.dumps(data, ensure_ascii=False))

    if result.structuredContent:
        structured = json.dumps(result.structuredContent, indent=2, ensure_ascii=False)
        lines.append("Structured output:\n" + structured)

    text = "\n".join(line for line in lines if line).strip() or "(no content)"
    return truncate_output(text, limit=limit)


def load_mcp_tools(max_output: int) -> tuple[List[BaseTool], Optional[MCPClientBridge], List[str]]:
    """Discover MCP tools and wrap them for LangGraph."""
    configs, warnings = load_server_configs()
    if not configs:
        return [], None, warnings
    if mcp is None:
        warnings.append("Install the 'mcp' package to enable MCP tool calls.")
        return [], None, warnings
    try:
        bridge = MCPClientBridge(configs, max_output=max_output)
    except MCPClientError as exc:
        warnings.append(f"MCP disabled: {exc}")
        return [], None, warnings
    warnings.extend(bridge.connection_warnings)
    tools = [spec.to_langchain_tool(bridge) for spec in bridge.tool_specs]
    return tools, bridge, warnings


__all__ = ["load_mcp_tools", "MCPClientBridge", "MCPClientError", "MCPServerConfig"]
