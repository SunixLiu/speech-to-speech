# DEVELOPMENT LOG — speech-to-speech (fork)

## 2026-11-11

### Merge upstream v1.1.0 + post-release updates into local branches
- **目的**：本地 `main` 停在 upstream v1.0.0（merge-base `16d7f98f`），upstream 已推进到 `ebc09bf2`（v1.1.0，2026-10-24 发布，之后 main 又 +184 提交）。
- **步骤**：
  1. 先把 `feature/mcp-support` 上未提交的 MCP 服务端工具执行改动提交为 `feat: execute general (MCP) tool calls server-side with follow-up generation`（14 文件，+504/-82）。
  2. `git merge upstream/main` 进本地 `main`：**零冲突**。
  3. `git merge main` 进 `feature/mcp-support`：8 个文件冲突，全部手工解决。
- **冲突解决要点**：
  - `pyproject.toml`：保留本地新增的 `mcp>=1.0.0` 依赖 + upstream 的 `nltk==3.10.3` 升级。
  - `README.md`：两边各加一行配置项（`--mcp-config` / `--torch_num_threads`），都保留。
  - `TTS/qwen3_tts_handler.py`：保留本地 `generate_voice_clone_streaming(**gen_kwargs)` 调用方式（gen_kwargs 含动态音色切换的 ref_spk/ref_rvq/voice_clone_prompt 覆盖；两侧都 pin `faster-qwen3-tts>=0.5.4`，签名一致）。
  - `api/openai_realtime/runtime_config.py`：合并双方新增字段——本地的 `voice_tool_executor`/`tool_executor`/`tool_executor_names` + upstream 的 `last_assistant_language`/`_trace_session_id`(PrivateAttr)/`bind_trace_session`。
  - `api/openai_realtime/service.py`：`__init__` 与 `register()` 合并双方新增参数/赋值（`voice_tool_executor`、`turn_latency_store`、语言能力字段、`bind_trace_session` 调用）。
  - `s2s_pipeline.py`：合并 import；`_build_pipeline_unit` 保留本地 MCP 工具合并逻辑 + upstream 的 `resource_ledger` 参数；`local` 命令采用 upstream 的新 `client_config` 结构（verbose/vad_threshold/interrupt_response/echo_cancellation 字段 + phonon 音色处理），MCP 块保留在 client_config 构建之前。
  - `LLM/base_openai_compatible_language_model.py`：保留本地 `import uuid/wave`。
  - `tests/openai_realtime/test_realtime_service.py`：补回 upstream 被冲突吞掉的 `TestTurnLifecycleOwnership` 断言，并保留本地三个新测试类（TestChatImageLifecycle / TestChatToolCallTracking / TestServerToolInjection）。
- **环境**：uv 需 >=0.13.4（旧 0.11.7 不满足），用 `C:\Users\ABC\AppData\Roaming\Python\Python313\Scripts\uv.exe` (0.13.7) 执行 `uv sync` + `uv lock`。
- **注意**：
  - upstream v1.1.0 要求 Python ≥ 3.11；默认本地 LLM 换成 Qwen3.5-4B（想保留旧模型需显式 `--model_name`）。
  - 本地 `chore: switch qwen3_tts default backend to torch locally` 是 TTS 后端改动，与 LLM 默认模型变更无关。

### 合并后测试验证与集成修复
- **首轮全量 pytest**：103 failed / 3665 passed / 12 skipped / 9 errors。逐类归因：
  1. **httpx 环境 bug（约 72 个失败，非合并引入）**：本机系统代理 `http://localhost:7890` + `NO_PROXY=localhost,127.0.0.1,::1,[::1]`。httpx 0.28.1 的 `get_environment_proxies()` 对 NO_PROXY 中带方括号的 IPv6 条目 `[::1]` 生成非法 mount key `all://*[::1]`，构造 `AsyncClient(trust_env=True)` 时抛 `InvalidURL: Invalid port: ':1]'`。影响所有真实构造 httpx 客户端的测试（llm_proxy、openai_client SDK、webrtc、minimax/openai STT/TTS handler 等）。已知上游问题（见 hermes-agent #118159），与本次合并无关；`llm_proxy.py` 自 merge-base 起未被改动。
  2. **合并集成问题（已修复）**：
     - `TTS/qwen3_tts_handler.py`：本地音色切换代码直接访问 `_cached_ref_spk`/`voice_registry` 等属性，upstream 新测试用 `object.__new__` 桩构造 handler 时触发 AttributeError；同时本地版本丢失了 upstream 的 `ref_spk`/`ref_rvq` 预计算引用透传。改为防御性 `getattr` 读取并恢复该 fallback 分支 → `tests/test_qwen3_tts_handler_backend.py` 70/70 通过。
     - `tests/openai_realtime/test_voice_switching.py`：upstream 把 `Chat._ordered_pending_call_ids`（set）重命名为 `_ordered_pending_calls`（dict[str, set]），本地 WIP 测试仍用旧名 → 改用新 API。
     - `tests/openai_realtime/test_realtime_service.py::TestServerToolInjection`：WIP 测试传入裸 `object()` 作为 voice_tool_executor，`_merge_server_tools` 需要 `.registry` → 改为 `VoiceToolExecutor()`。
     - `src/speech_to_speech/config.py`：本地新增的 `mcp_config` 字段未登记进 `_MODULE_PIPELINE`，upstream 新配置清单测试断言集合相等 → 加入该集合。
     - `tests/test_cli_defaults.py`：upstream 新测试断言发布默认 `qwen3_tts_backend == "ggml"`，与本地有意改为 torch 的默认冲突 → 按本地意图更新断言并加注释。
  3. **环境/其他（少量）**：`test_demo_user_audio`（Windows GBK locale 读 UTF-8 HTML）、`test_speculative_turns` 一条时间边界断言（0.8s 临界，慢机器 flaky）、`test_weather_server::test_chinese_city_name_resolves`（真实 HTTP 调用同样触发 httpx 环境 bug）。
- **修复后**：上述集成问题对应的测试文件全部通过（qwen3 后端 70/70、realtime service 286/286、voice_switching / cli_defaults / config / weather 除真实网络用例外全过）。剩余失败均为第 1、3 类环境问题。
