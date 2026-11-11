"""
Voice Tools - Server-side tool definitions and executor for dynamic voice switching.

Provides two tools:
  - list_voices:  Return all registered voices with metadata
  - switch_voice: Switch the TTS voice to a registered profile

These tools are exposed to the LLM via the tool-calling system prompt so the
model can invoke them when the user requests a voice change.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from openai.types.realtime import RealtimeFunctionTool

from speech_to_speech.LLM.tool_call.function_tool import FunctionTool
from speech_to_speech.TTS.voice_registry import VoiceRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions (JSON Schema for LLM)
# ---------------------------------------------------------------------------

LIST_VOICES_TOOL = FunctionTool(
    name="list_voices",
    description="List all available voices that can be switched to. Returns voice names, display names, and descriptions. ABSOLUTELY MUST call this tool when asked about available voices or how many voices exist. You CANNOT answer this question without calling this tool first. There is no other way to know voice information.",
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
)

SWITCH_VOICE_TOOL = FunctionTool(
    name="switch_voice",
    description="Switch the TTS voice to a different registered voice. Use this when the user asks to change voice, tone, or speaker.",
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The voice identifier to switch to (e.g. 'news_male', 'warm_female'). Must be one of the registered voice names.",
            },
        },
        "required": ["name"],
    },
)


def _switch_voice_description(registry: Optional[VoiceRegistry] = None) -> str:
    """Build the switch_voice description, listing registered voices when known.

    Listing the voices (display name -> exact name) lets the LLM map a user
    request like "switch to Indian English" to the exact registered name and
    call the tool directly, instead of guessing whether the voice exists and
    refusing to switch.
    """
    base = (
        "Switch the TTS voice to a different registered voice. Use this when the "
        "user asks to change voice, tone, or speaker."
    )
    if registry is None:
        return base
    voices = registry.list_voices()
    if not voices:
        return base
    lines = [f"- {v.display_name} -> {v.name}" for v in voices]
    return (
        base
        + " Available voices (pass the exact name after '->' as the 'name' argument):\n"
        + "\n".join(lines)
    )


def build_voice_tools(registry: Optional[VoiceRegistry] = None) -> list[dict[str, object]]:
    """Build the list of voice-related tools as TypedDict-compatible dicts.

    Must return plain dicts (not Pydantic models) because the OpenAI SDK's
    responses.create() serializes TypedDict inputs correctly but may drop
    fields from BaseModel instances that inherit from RealtimeFunctionTool,
    resulting in empty 'type' fields that local LLM servers reject.

    Pass *registry* to embed the list of registered voices in the switch_voice
    description so the LLM knows the exact voice names it may request.
    """
    return [
        {
            "type": "function",
            "name": LIST_VOICES_TOOL.name,
            "description": LIST_VOICES_TOOL.description,
            "parameters": LIST_VOICES_TOOL.parameters,
        },
        {
            "type": "function",
            "name": SWITCH_VOICE_TOOL.name,
            "description": _switch_voice_description(registry),
            "parameters": SWITCH_VOICE_TOOL.parameters,
        },
    ]


# ---------------------------------------------------------------------------
# Server-side executor
# ---------------------------------------------------------------------------

VoiceToolCallback = Callable[[str, dict[str, Any]], str]


class VoiceToolExecutor:
    """
    Executes voice tool calls on the server side.

    This class holds a reference to the VoiceRegistry and a callback to the
    TTS handler's switch_voice() method. When the LLM generates a voice tool
    call, the LLM handler delegates execution here instead of sending it to
    the client.
    """

    def __init__(
        self,
        registry: Optional[VoiceRegistry] = None,
        switch_callback: Optional[VoiceToolCallback] = None,
    ) -> None:
        self.registry = registry
        self._switch_callback = switch_callback

    def set_switch_callback(self, callback: VoiceToolCallback) -> None:
        """Set or update the callback used for switch_voice actions."""
        self._switch_callback = callback

    def execute(self, tool_name: str, args: dict[str, Any]) -> str:
        """Execute a voice tool call and return the result string."""
        if tool_name == "list_voices":
            return self._handle_list_voices()
        elif tool_name == "switch_voice":
            return self._handle_switch_voice(args)
        return f"Unknown voice tool: {tool_name}"

    def _handle_list_voices(self) -> str:
        """Handle list_voices tool call."""
        if self.registry is None:
            return "No voice registry loaded. Cannot list voices."

        voices = self.registry.list_voices()
        if not voices:
            return "No voices registered."

        lines = []
        for v in voices:
            tag_str = f" [{', '.join(v.tags)}]" if v.tags else ""
            desc = f" — {v.description}" if v.description else ""
            lines.append(f"- {v.display_name} ({v.name}){tag_str}{desc}")
        return "\n".join(lines)

    def _handle_switch_voice(self, args: dict[str, Any]) -> str:
        """Handle switch_voice tool call."""
        voice_name = args.get("name", "")
        if not voice_name:
            return "Error: 'name' argument is required for switch_voice"

        if self._switch_callback is not None:
            try:
                result = self._switch_callback("switch", {"name": voice_name})
                if isinstance(result, str):
                    return result
                return f"Switched to {voice_name}"
            except Exception as e:
                logger.warning("Voice switch callback failed: %s", e, exc_info=True)
                return f"Error switching voice: {e}"

        # Fallback: apply directly from registry
        if self.registry is None:
            return "Error: no voice registry loaded"

        profile = self.registry.get(voice_name)
        if profile is None:
            available = self.registry.list_names()
            return f"Error: voice '{voice_name}' not found. Available: {', '.join(available)}"

        return f"Switched to {profile.display_name} ({profile.name})"
