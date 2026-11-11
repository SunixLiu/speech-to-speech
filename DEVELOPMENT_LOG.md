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

### venv 依赖修复（serve 的 TTS 阶段）
- **背景**：`serve` 依次卡在 STT→TTS。STT 缺 funasr（已装 funasr 1.4.24 + modelscope 1.40.3）。TTS 阶段连续暴露多个 `.venv` 依赖问题，逐一修复（均为环境改动，非源码）。
- **改动**：
  1. torch 由 CPU 版换成 CUDA 版 `torch 2.6.0+cu126`（本地缓存 `D:\download\whl` 有 cp312 wheel）。
  2. torchaudio 对齐到 `2.6.0+cu126`。本地缓存只有 cp310/cp39，cp312 需从官方索引下载：`pip install "torchaudio==2.6.0" --index-url https://download.pytorch.org/whl/cu126`。**torch/torchaudio 版本必须一致**，否则 `_torchaudio` 扩展加载报 `OSError: [WinError 127]`。
  3. `qwen-tts-hf 0.1.1.post1` 安装不完整（`qwen_tts/` 只剩 `_transformers_compat.py`，缺 `__init__.py`/modeling 等共 37 个文件）→ `pip install --force-reinstall --no-deps "qwen-tts-hf==0.1.1.post1"` 补回。
  4. 修复一次被中断的重装进程顺带升级的包：numpy 2.5.4→**2.4.3**（numba 需 ≤2.4）、llvmlite 0.50.0→**0.48.0**（numba 0.66.0 需 <0.49）、pydantic-core 2.50.4→**2.47.0**（pydantic 2.14.0a1 需 ==2.47.0）、sympy 1.14.0→**1.13.1**（torch 2.6.0 需 ==1.13.1）。
- **验证**：`pip check` 无冲突；`import speech_to_speech.s2s_pipeline`、`from qwen_tts import Qwen3TTSModel` 均通过；`torch.cuda.is_available()==True`。
- **注意**：
  - 重装本项目依赖（如 `pip install -e ".[paraformer]"`）会重新解析 torch，可能把 CUDA 版换回 CPU 版或打乱版本；重装本项目本身建议加 `--no-deps`，装完确认 torch 仍是 `+cu126`。
  - flash-attn 未安装，Qwen3-TTS 走手动 PyTorch 路径（仅警告，不影响功能）。
- **修复后**：上述集成问题对应的测试文件全部通过（qwen3 后端 70/70、realtime service 286/286、voice_switching / cli_defaults / config / weather 除真实网络用例外全过）。剩余失败均为第 1、3 类环境问题。

### paraformer STT 语言校验修复（realtime 前端报 "Language 'zh' is not supported"）
- **现象**：前端 `reachy_mini_conversation_app` 连 speech-to-speech realtime，session.update 带 `language='zh'`，服务端报 `Language 'zh' is not supported by the active STT backend.`。
- **根因**：`s2s_pipeline._stt_session_languages()` 把 `paraformer` 归入「返回空集合 `set()`」的一组；`api/openai_realtime/handlers/session.py` 用 `stt_supported_languages` 校验时，`set()` 非 None 且 `'zh' not in set()` → 拒绝一切语言。但 paraformer-zh 本身支持中文。
- **改动**：`_stt_session_languages` 为 `paraformer` 单独返回其配置语言 `{handler.language}`（默认 `zh`），其余后端保持原逻辑。
- **验证**：单元测试 `_stt_session_languages(paraformer, handler.language='zh') -> {'zh'}`，`'zh' in {'zh'}` 通过；TTS 侧 `QWEN3_LANGUAGE_ALIASES` 含 `zh`，故 TTS 校验也通过。
