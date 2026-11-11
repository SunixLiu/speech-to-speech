"""MCP server manager for speech-to-speech service.

Manages connections to multiple MCP servers and provides a unified interface
for tool discovery and execution.
"""

import asyncio
import json
import logging
from pathlib import Path
from threading import Thread
from typing import Any, Callable

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import Tool

logger = logging.getLogger(__name__)


class MCPServer:
    """Represents a single MCP server connection."""

    def __init__(self, config: dict[str, Any]):
        self.name = config["name"]
        self.transport_type = config.get("transport", "stdio")
        self.command = config.get("command")
        self.args = config.get("args", [])
        self.env = config.get("env", None)
        self.url = config.get("url")
        self.headers = config.get("headers", None)
        self.enabled = config.get("enabled", True)
        self.session: ClientSession | None = None
        self._transport_ctx = None
        self.tools: list[Tool] = []

    async def connect(self) -> None:
        """Connect to the MCP server and discover available tools."""
        if self.transport_type == "stdio":
            if not self.command:
                raise ValueError(f"stdio transport requires 'command' for server {self.name}")
            server_params = StdioServerParameters(command=self.command, args=self.args, env=self.env)
            self._transport_ctx = stdio_client(server_params)
            streams = await self._transport_ctx.__aenter__()
            read_stream, write_stream = streams
        elif self.transport_type == "http":
            if not self.url:
                raise ValueError(f"http transport requires 'url' for server {self.name}")
            from mcp.client.streamable_http import streamablehttp_client

            self._transport_ctx = streamablehttp_client(url=self.url, headers=self.headers)
            streams = await self._transport_ctx.__aenter__()
            read_stream, write_stream = streams
        elif self.transport_type == "websocket":
            if not self.url:
                raise ValueError(f"websocket transport requires 'url' for server {self.name}")
            from mcp.client.websocket import websocket_client

            self._transport_ctx = websocket_client(url=self.url)
            streams = await self._transport_ctx.__aenter__()
            read_stream, write_stream = streams
        else:
            raise ValueError(f"Unknown transport type: {self.transport_type}")

        self.session = ClientSession(read_stream, write_stream)
        await self.session.__aenter__()
        # Initialize the session before listing tools
        await self.session.initialize()
        result = await self.session.list_tools()
        self.tools = result.tools
        logger.info("Connected to MCP server '%s', discovered %d tools", self.name, len(self.tools))

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on this MCP server."""
        if not self.session:
            raise RuntimeError(f"MCP server '{self.name}' is not connected")
        result = await self.session.call_tool(name, arguments)
        return result

    async def close(self) -> None:
        """Close the connection to this MCP server."""
        if self.session:
            try:
                await self.session.__aexit__(None, None, None)
            except Exception as e:
                logger.debug("Error closing session for '%s': %s", self.name, e)
            self.session = None
            
        if hasattr(self, '_transport_ctx') and self._transport_ctx is not None:
            try:
                await self._transport_ctx.__aexit__(None, None, None)
            except Exception as e:
                logger.debug("Error closing transport for '%s': %s", self.name, e)
            self._transport_ctx = None
            
        logger.info("Closed connection to MCP server '%s'", self.name)


class MCPManager:
    """Manages multiple MCP server connections and provides unified tool access."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.servers: list[MCPServer] = []
        self.tools: list[dict[str, Any]] = []
        self.tool_to_server: dict[str, tuple[MCPServer, str]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: Thread | None = None

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Return the manager's dedicated event loop, starting it if needed.

        MCP ``ClientSession`` objects are bound to the loop in which they were
        created, so every coroutine (initialize, execute_tool, shutdown) must
        run on this same long-lived loop — never on a temporary one that gets
        closed after ``run_until_complete``.
        """
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            self._loop_thread = Thread(
                target=self._loop.run_forever,
                daemon=True,
                name="mcp-event-loop",
            )
            self._loop_thread.start()
        return self._loop

    def initialize_sync(self) -> None:
        """Connect to all servers and discover tools (thread-safe)."""
        loop = self._ensure_loop()
        asyncio.run_coroutine_threadsafe(self.initialize(), loop).result()

    def execute_tool_sync(
        self,
        name: str,
        arguments: dict[str, Any],
        timeout: float = 30.0,
    ) -> str:
        """Execute a tool from a synchronous context (thread-safe)."""
        loop = self._ensure_loop()
        return asyncio.run_coroutine_threadsafe(
            self.execute_tool(name, arguments), loop
        ).result(timeout)

    def shutdown_sync(self) -> None:
        """Shut down all servers and stop the dedicated event loop."""
        if self._loop is None or self._loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(self.shutdown(), self._loop).result(timeout=10.0)
        except Exception as e:
            logger.warning("Error during MCP shutdown: %s", e)
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5.0)
        self._loop.close()
        self._loop = None
        self._loop_thread = None

    def load_config(self) -> None:
        """Load MCP server configuration from JSON file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"MCP config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Support both array format ("servers") and object format ("mcpServers")
        if "servers" in config:
            # Array format: [{"name": "...", "command": "...", ...}]
            for server_config in config["servers"]:
                if server_config.get("enabled", True):
                    self.servers.append(MCPServer(server_config))
        elif "mcpServers" in config:
            # Object format (Claude Desktop style): {"server-name": {"command": "...", ...}}
            for name, server_config in config["mcpServers"].items():
                if isinstance(server_config, dict) and server_config.get("enabled", True):
                    server_config_with_name = {"name": name, **server_config}
                    self.servers.append(MCPServer(server_config_with_name))

        logger.info("Loaded MCP configuration with %d enabled servers", len(self.servers))

    async def initialize(self) -> None:
        """Connect to all configured MCP servers and discover tools."""
        for server in self.servers:
            try:
                await server.connect()
                for tool in server.tools:
                    prefixed_name = f"{server.name}_{tool.name}"
                    
                    # 兼容 MCP 2.x (input_schema) 和 MCP 1.x (inputSchema)
                    schema = getattr(tool, "input_schema", getattr(tool, "inputSchema", {}))
                    
                    self.tools.append(
                        {
                            "type": "function",
                            "name": prefixed_name,
                            "description": tool.description,
                            "parameters": schema,
                        }
                    )
                    self.tool_to_server[prefixed_name] = (server, tool.name)
            except Exception as e:
                logger.warning("Failed to connect to MCP server '%s': %s", server.name, e)

        logger.info("MCP manager initialized with %d tools", len(self.tools))

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool by its prefixed name and return its text result."""
        if name not in self.tool_to_server:
            raise ValueError(f"Unknown MCP tool: {name}")

        server, original_name = self.tool_to_server[name]
        result = await server.call_tool(original_name, arguments)
        return self._result_to_text(result)

    @staticmethod
    def _result_to_text(result: Any) -> str:
        """Flatten a ``CallToolResult`` into the text the LLM should see.

        The pipeline's tool handler only knows how to feed strings back to the
        model; a raw pydantic result would fail ``json.dumps`` and surface to
        the LLM as an opaque serialization error.
        """
        if isinstance(result, str):
            return result
        content = getattr(result, "content", None)
        if content is not None:
            parts = [
                block.text
                for block in content
                if getattr(block, "type", None) == "text" and getattr(block, "text", None)
            ]
            text = "\n".join(parts)
            is_error = getattr(result, "is_error", None)
            if is_error is None:
                is_error = getattr(result, "isError", False)
            if is_error:
                return f"Tool returned an error: {text}" if text else "Tool returned an error"
            if text:
                return text
        return json.dumps(result, ensure_ascii=False, default=str)

    async def shutdown(self) -> None:
        """Close all MCP server connections."""
        for server in self.servers:
            try:
                await server.close()
            except Exception as e:
                logger.warning("Failed to close MCP server '%s': %s", server.name, e)

    def get_tools(self) -> list[dict[str, Any]]:
        """Get all discovered tools in OpenAI-compatible format."""
        return self.tools

    def get_tool_executor(self) -> Callable:
        """Get a synchronous tool executor wrapper bound to the manager's loop."""
        return self.execute_tool_sync