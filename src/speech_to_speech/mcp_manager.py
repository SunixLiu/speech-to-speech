"""MCP server manager for speech-to-speech service.

Manages connections to multiple MCP servers and provides a unified interface
for tool discovery and execution.
"""

import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from mcp import ClientSession
from mcp.client.stdio import StdioTransport
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
        self.tools: list[Tool] = []

    async def connect(self) -> None:
        """Connect to the MCP server and discover available tools."""
        if self.transport_type == "stdio":
            if not self.command:
                raise ValueError(f"stdio transport requires 'command' for server {self.name}")
            transport = StdioTransport(self.command, self.args, env=self.env)
        elif self.transport_type == "http":
            if not self.url:
                raise ValueError(f"http transport requires 'url' for server {self.name}")
            from mcp.client.streamable_http import StreamableHttpTransport

            transport = StreamableHttpTransport(url=self.url, headers=self.headers)
        elif self.transport_type == "websocket":
            if not self.url:
                raise ValueError(f"websocket transport requires 'url' for server {self.name}")
            from mcp.client.websocket import WebSocketTransport

            transport = WebSocketTransport(url=self.url)
        else:
            raise ValueError(f"Unknown transport type: {self.transport_type}")

        self.session = ClientSession(transport)
        await self.session.__aenter__()
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
            await self.session.__aexit__(None, None, None)
            self.session = None
            logger.info("Closed connection to MCP server '%s'", self.name)


class MCPManager:
    """Manages multiple MCP server connections and provides unified tool access."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.servers: list[MCPServer] = []
        self.tools: list[dict[str, Any]] = []
        self.tool_to_server: dict[str, tuple[MCPServer, str]] = {}

    def load_config(self) -> None:
        """Load MCP server configuration from JSON file."""
        with open(self.config_path, "r") as f:
            config = json.load(f)

        for server_config in config.get("servers", []):
            if server_config.get("enabled", True):
                self.servers.append(MCPServer(server_config))

        logger.info("Loaded MCP configuration with %d enabled servers", len(self.servers))

    async def initialize(self) -> None:
        """Connect to all configured MCP servers and discover tools."""
        for server in self.servers:
            try:
                await server.connect()
                for tool in server.tools:
                    prefixed_name = f"{server.name}_{tool.name}"
                    self.tools.append(
                        {
                            "type": "function",
                            "name": prefixed_name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        }
                    )
                    self.tool_to_server[prefixed_name] = (server, tool.name)
            except Exception as e:
                logger.warning("Failed to connect to MCP server '%s': %s", server.name, e)

        logger.info("MCP manager initialized with %d tools", len(self.tools))

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool by its prefixed name."""
        if name not in self.tool_to_server:
            raise ValueError(f"Unknown MCP tool: {name}")

        server, original_name = self.tool_to_server[name]
        result = await server.call_tool(original_name, arguments)
        return result

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
        """Get a synchronous tool executor wrapper."""
        import asyncio

        async def _async_executor(name: str, arguments: dict[str, Any]) -> Any:
            return await self.execute_tool(name, arguments)

        def _sync_executor(name: str, arguments: dict[str, Any]) -> Any:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(_async_executor(name, arguments))
            finally:
                loop.close()

        return _sync_executor
