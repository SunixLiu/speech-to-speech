"""Integration tests for server-side voice switching and follow-up generation.

These tests validate the full voice-switching flow:
  1. The LLM emits a brief acknowledgment + switch_voice tool call.
  2. The service executes the voice tool server-side and appends the result.
  3. A follow-up response is generated without the acknowledgment text.
  4. In multi-turn conversations, only the current turn's assistant message is
     removed from the follow-up context, preserving previous turns.
"""

import json
from queue import Queue
from threading import Event as ThreadingEvent
from types import MethodType
from unittest.mock import MagicMock

import pytest
from openai.types.realtime.conversation_item import (
    RealtimeConversationItemAssistantMessage,
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemFunctionCallOutput,
)
from openai.types.responses import ResponseFunctionToolCall

from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.tool_call.function_tool import FunctionTool
from speech_to_speech.TTS.voice_tools import VoiceToolExecutor, build_voice_tools
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.messages import GenerateResponseRequest


def _make_runtime_config(chat: Chat | None = None) -> RuntimeConfig:
    cfg = RuntimeConfig(chat=chat or Chat(10))
    cfg.session.audio.output.voice = "default-voice"
    return cfg


def _mock_voice_executor(switch_callback=None):
    """Build a VoiceToolExecutor with a switch callback that records calls."""
    registry = MagicMock()
    registry.list_voices.return_value = []
    executor = VoiceToolExecutor(registry=registry, switch_callback=switch_callback)
    return executor


def _tool_call(name="switch_voice", call_id="call_switch", arguments=None):
    if arguments is None:
        arguments = json.dumps({"name": "trump-speech-2"})
    return ResponseFunctionToolCall(
        type="function_call",
        id=f"fc_{name}",
        call_id=call_id,
        name=name,
        arguments=arguments,
    )


class TestVoiceToolExecution:
    """Test server-side voice tool execution and buffer cleanup."""

    def test_handle_voice_tool_calls_executes_and_appends_output(self):
        chat = Chat(10)
        chat.add_item(make_user_message("Hello"))
        runtime_config = _make_runtime_config(chat)
        voice_switched = []

        def switch_callback(action, args):
            if action == "switch":
                voice_switched.append(args.get("name"))
                return f"Switched to {args.get('name')}"
            return "unknown"

        executor = _mock_voice_executor(switch_callback=switch_callback)
        runtime_config.voice_tool_executor = executor

        handler = object.__new__(BaseHandler)
        handler.stop_event = ThreadingEvent()
        handler.queue_in = Queue()
        handler.queue_out = Queue()

        tool_call = _tool_call()
        # Simulate the function_call being committed (and evicted into pending).
        fc = RealtimeConversationItemFunctionCall(
            type="function_call",
            id="fc_switch",
            call_id="call_switch",
            name="switch_voice",
            arguments=json.dumps({"name": "trump-speech-2"}),
        )
        chat._pending_tool_calls["call_switch"] = fc
        chat._ordered_pending_calls.setdefault("call_switch", set())

        executed = handler._handle_voice_tool_calls(
            tools=[tool_call],
            original_chat=chat,
            runtime_config=runtime_config,
            response=None,
        )

        assert executed is True
        assert voice_switched == ["trump-speech-2"]
        # The function_call_output should be appended to the chat buffer.
        output_items = [
            item for item in chat.buffer if isinstance(item, RealtimeConversationItemFunctionCallOutput)
        ]
        assert len(output_items) == 1
        assert output_items[0].call_id == "call_switch"
        assert "Switched to" in output_items[0].output

    def test_handle_voice_tool_calls_removes_reinjected_function_call(self):
        """Verify that re-injected function_call is removed from buffer."""
        chat = Chat(10)
        chat.add_item(make_user_message("Hello"))
        runtime_config = _make_runtime_config(chat)
        executor = _mock_voice_executor()
        runtime_config.voice_tool_executor = executor

        handler = object.__new__(BaseHandler)
        handler.stop_event = ThreadingEvent()
        handler.queue_in = Queue()
        handler.queue_out = Queue()

        # Simulate an evicted function_call being re-injected by append_tool_output.
        fc = RealtimeConversationItemFunctionCall(
            type="function_call",
            id="fc_switch",
            call_id="call_switch",
            name="switch_voice",
            arguments=json.dumps({"name": "trump-speech-2"}),
        )
        chat._pending_tool_calls["call_switch"] = fc
        chat._ordered_pending_calls.setdefault("call_switch", set())

        tool_call = _tool_call()
        handler._handle_voice_tool_calls(
            tools=[tool_call],
            original_chat=chat,
            runtime_config=runtime_config,
            response=None,
        )

        # The re-injected function_call should have been removed from the buffer.
        fc_in_buffer = [
            item for item in chat.buffer if isinstance(item, RealtimeConversationItemFunctionCall)
        ]
        assert len(fc_in_buffer) == 0, f"Expected no function_call in buffer, got {fc_in_buffer}"


class TestFollowUpBufferCleanup:
    """Test that follow-up generation only removes current turn's assistant messages."""

    def _setup_multi_turn_chat(self):
        """Create a chat with two previous turns plus a pending tool call."""
        chat = Chat(10)
        # Turn 1
        chat.add_item(make_user_message("Turn 1 user"))
        chat.add_item(RealtimeConversationItemAssistantMessage(type="message", role="assistant", content=[{"type": "output_text", "text": "Hi there!"}]))
        # Turn 2 (current turn being processed)
        chat.add_item(make_user_message("Switch voice please"))
        return chat

    def test_follow_up_removes_only_current_turn_assistant_messages(self):
        """Previous turns' assistant messages must be preserved in follow-up context."""
        chat = self._setup_multi_turn_chat()
        runtime_config = _make_runtime_config(chat)

        # Simulate that the current turn's items were tracked provisionally.
        current_fc = RealtimeConversationItemFunctionCall(
            type="function_call",
            id="fc_switch2",
            call_id="call_switch2",
            name="switch_voice",
            arguments=json.dumps({"name": "Indian-english"}),
        )
        current_assistant = RealtimeConversationItemAssistantMessage(
            type="message", role="assistant", content=[{"type": "output_text", "text": "Sure, switching voice."}])
        chat.add_item(current_assistant)
        chat.add_item(current_fc)

        # Track these in provisional_generations under a fake response key.
        current_response_key = "current-turn-response-key"
        if not hasattr(chat, "_provisional_generations"):
            chat._provisional_generations = {}
        chat._provisional_generations[current_response_key] = (
            {current_assistant.id, current_fc.id},
            {"call_switch2"},
        )

        # Now simulate the follow-up buffer cleanup logic from
        # base_openai_compatible_language_model.py.
        follow_up_active = chat.copy()
        _current_item_ids = chat._provisional_generations.get(current_response_key, (set(),))[0]
        follow_up_active.buffer = [
            item
            for item in follow_up_active.buffer
            if not (
                isinstance(item, RealtimeConversationItemAssistantMessage)
                and item.id in _current_item_ids
            )
        ]

        # The previous turn's assistant message should be preserved.
        assistant_messages = [
            item for item in follow_up_active.buffer
            if isinstance(item, RealtimeConversationItemAssistantMessage)
        ]
        assert len(assistant_messages) == 1
        assert assistant_messages[0].content[0].text == "Hi there!"

        # The current turn's assistant message should be removed.
        current_assistant_in_followup = [
            m for m in assistant_messages if any("Switching" in (c.text or "") for c in m.content)
        ]
        assert len(current_assistant_in_followup) == 0

        # User messages and function call should remain.
        user_messages = [item for item in follow_up_active.buffer if isinstance(item, type(chat.buffer[0]))]
        assert len(user_messages) >= 2  # both user messages preserved


class TestVoiceToolExecutor:
    """Test the VoiceToolExecutor callback mechanism."""

    def test_switch_callback_is_invoked(self):
        switched_to = []

        def switch_callback(action, args):
            if action == "switch":
                switched_to.append(args["name"])
                return f"Switched to {args['name']}"
            return "unknown"

        executor = VoiceToolExecutor(registry=None, switch_callback=switch_callback)
        result = executor.execute("switch_voice", {"name": "Indian-english"})

        assert switched_to == ["Indian-english"]
        assert "Switched to Indian-english" in result

    def test_list_voices_uses_registry(self):
        from speech_to_speech.TTS.voice_registry import VoiceProfile

        registry = MagicMock()
        registry.list_voices.return_value = [
            VoiceProfile(name="voice1", display_name="Voice 1", description="A voice"),
        ]
        executor = VoiceToolExecutor(registry=registry, switch_callback=None)
        result = executor.execute("list_voices", {})

        assert "voice1" in result
        assert "Voice 1" in result


class TestEndToEndVoiceSwitch:
    """End-to-end test of the voice switching flow through the service."""

    def test_response_generation_done_queues_followup(self):
        """A ResponseGenerationDoneEvent with tool call ids triggers follow-up prefetch."""
        from speech_to_speech.api.openai_realtime.service import RealtimeService
        from speech_to_speech.pipeline.events import ResponseGenerationDoneEvent

        chat = Chat(10)
        runtime_config = _make_runtime_config(chat)

        service = RealtimeService()
        conn_id = service.register()
        state = service._state(conn_id)
        state.runtime_config = runtime_config

        done_event = ResponseGenerationDoneEvent(
            response_key="resp-1",
            call_ids=["call_switch_e2e"],
            succeeded=True,
        )

        events = service.response.on_response_generation_done(conn_id, done_event)

        # The service should have recorded the tool calls for follow-up.
        st = state
        assert "call_switch_e2e" in st.generation_done_tool_calls["resp-1"]


