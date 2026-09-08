"""
Voice Extraction Tool - Batch extract voice embeddings from audio+text pairs.

Supports both .wav and .mp3 audio files. Matches audio files with corresponding *.txt files.

Usage:
    python scripts/tools/voice_extractor.py --input-dir ./samples --output-dir ./voices --backend ggml
    python scripts/tools/voice_extractor.py --input-dir ./samples --output-dir ./voices --backend torch

This tool scans a directory for matching *.wav or *.mp3 + *.txt pairs, extracts voice
embeddings using Qwen3-TTS, and saves them as cache files (.spk/.rvq for GGML
or .npz for Torch) along with a metadata JSON file.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import librosa
try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int = 24000) -> np.ndarray:
    """Simple linear resampling to target sample rate."""
    if src_sr == dst_sr:
        return np.ascontiguousarray(audio, dtype=np.float32)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    duration = audio.shape[0] / float(src_sr)
    dst_n = max(1, int(round(duration * dst_sr)))
    src_x = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    dst_x = np.linspace(0.0, duration, num=dst_n, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


def load_audio_24k(audio_path: Path) -> np.ndarray:
    """Load audio file (wav/mp3) and resample to 24kHz mono float32."""
    # librosa handles both wav and mp3 natively, resampling to sr=24000
    audio, sr = librosa.load(str(audio_path), sr=24000, mono=True, dtype=np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


def extract_ggml(
    audio_24k: np.ndarray,
    model_name: str,
    output_dir: Path,
    voice_name: str,
    ref_text: str,
    xvec_only: bool = False,
    gguf_talker_path: str | Path | None = None,
    gguf_codec_path: str | Path | None = None,
) -> dict[str, Any]:
    """Extract voice reference using GGML backend (qwentts_cpp)."""
    from qwentts_cpp import QwenTTS, resolve_gguf_paths

    # Check if libqwen.dll exists
    import sys
    if sys.platform == "win32":
        import os
        lib_dir = pathlib.Path(__file__).parent.parent.parent / ".venv" / "Lib" / "site-packages" / "qwentts_cpp" / "lib"
        if not lib_dir.exists():
            logger.warning("libqwen.dll not found on Windows. Use --backend torch instead.")
            # Fall through to use default behavior (will fail if DLL missing)
            pass
    if gguf_talker_path is not None and gguf_codec_path is not None:

        talker_path = gguf_talker_path

        codec_path = gguf_codec_path

        logger.info("Using custom GGUF: talker=%s, codec=%s", talker_path, codec_path)

    else:

        talker_path, codec_path = resolve_gguf_paths(model_name)

    runtime = QwenTTS(talker_path=talker_path, codec_path=codec_path)

    try:
        audio_flat = np.ascontiguousarray(audio_24k, dtype=np.float32).reshape(-1)
        voice_ref = runtime.extract_voice_ref(audio_flat)

        spk_path = output_dir / f"{voice_name}.spk"
        rvq_path = output_dir / f"{voice_name}.rvq"
        voice_ref.save(spk_path, rvq_path)

        result = {
            "backend": "ggml",
            "spk_file": str(spk_path),
            "rvq_file": str(rvq_path),
            "ref_text": ref_text,
            "xvec_only": xvec_only,
            "spk_emb_shape": list(voice_ref.ref_spk_emb.shape),
            "ref_codes_shape": list(voice_ref[0].ref_codes.shape) if not xvec_only else None,
        }
        logger.info("GGML extraction done: %s.spk, %s.rvq", spk_path, rvq_path)
        return result
    finally:
        runtime.close()


def extract_torch(
    audio_24k: np.ndarray,
    model_name: str,
    output_dir: Path,
    voice_name: str,
    ref_text: str,
    xvec_only: bool = False,
) -> dict[str, Any]:
    """Extract voice embedding using Torch backend."""
    import soundfile as sf
    import torch
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    # Save audio to temp file for create_voice_clone_prompt
    import tempfile

    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(temp_wav.name, audio_24k, 24000, format="WAV", subtype="PCM_16")
    temp_wav.close()

    try:
        model = Qwen3TTSModel.from_pretrained(model_name)
        voice_clone_prompt = model.create_voice_clone_prompt(
            ref_audio=temp_wav.name,
            ref_text=ref_text,
            x_vector_only_mode=xvec_only,
        )

        npz_path = output_dir / f"{voice_name}.npz"
        np.savez(
            npz_path,
            ref_spk_embedding=voice_clone_prompt[0].ref_spk_embedding.numpy(),
            ref_codes=voice_clone_prompt[0].ref_code.numpy() if not xvec_only else None,
            ref_text=np.array(ref_text),
            xvec_only=np.array(xvec_only),
        )

        result = {
            "backend": "torch",
            "npz_file": str(npz_path),
            "ref_text": ref_text,
            "xvec_only": xvec_only,
            "spk_emb_shape": list(voice_clone_prompt[0].ref_spk_embedding.shape),
            "ref_codes_shape": list(voice_clone_prompt[0].ref_code.shape) if not xvec_only else None,
        }
        logger.info("Torch extraction done: %s", npz_path)
        return result
    finally:
        Path(temp_wav.name).unlink(missing_ok=True)


def scan_pairs(input_dir: Path) -> list[tuple[Path, Path]]:
    """Scan directory for matching audio (wav/mp3) + txt pairs."""
    audio_exts = ("*.wav", "*.mp3")
    audio_files = []
    for ext in audio_exts:
        audio_files.extend(input_dir.glob(ext))
    audio_files = sorted(audio_files, key=lambda p: p.stem)
    pairs = []
    for audio_path in audio_files:
        txt = audio_path.with_suffix(".txt")
        if txt.exists():
            pairs.append((audio_path, txt))
    return pairs




def _yaml_scalar_repr(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='' if data and any(c in data for c in ':{}[]&,*') else None)


def write_registry_yaml(registry_path, entries, output_dir, default_voice=None):
    if yaml is None:
        logger.warning('PyYAML not installed; skipping YAML generation.')
        return
    registry_path = Path(registry_path)
    # If registry_path is a directory, use voices.yaml inside it
    if registry_path.is_dir():
        registry_path = registry_path / 'voices.yaml'
    if not registry_path.suffix:
        registry_path = registry_path.with_suffix('.yaml')
    existing = {}
    existing_default = ''
    if registry_path.exists():
        with open(registry_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        if data:
            existing_default = data.get('default_voice', '')
            existing_voices = data.get('voices', [])
            existing = {v['name']: v for v in existing_voices}
    for entry in entries:
        name = entry['name']
        ve = {'name': name, 'display_name': name.replace('_', ' ').title(), 'description': '', 'tags': [], 'backend': entry['backend'], 'ref_text': entry.get('ref_text', '')}
        reg_dir = registry_path.parent
        if entry['backend'] == 'ggml':
            spk, rvq = Path(entry['spk_file']), Path(entry['rvq_file'])
            ve['spk_file'] = str(spk.relative_to(reg_dir)) if spk.is_relative_to(reg_dir) else str(spk)
            ve['rvq_file'] = str(rvq.relative_to(reg_dir)) if rvq.is_relative_to(reg_dir) else str(rvq)
        else:
            npz = Path(entry['npz_file'])
            ve['npz_file'] = str(npz.relative_to(reg_dir)) if npz.is_relative_to(reg_dir) else str(npz)
        existing[name] = ve
        if not default_voice and not existing_default:
            default_voice = name
    final_default = default_voice or existing_default
    if not final_default and existing:
        final_default = next(iter(existing))
    doc = {'default_voice': final_default, 'voices': list(existing.values())}
    with open(registry_path, 'w', encoding='utf-8') as f:
        yaml.dump(doc, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    logger.info('YAML registry written to %s (%d voice(s))', registry_path, len(existing))
def main():
    parser = argparse.ArgumentParser(
        description="Batch extract voice embeddings from audio+text pairs."
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Directory containing *.wav or *.mp3 audio files with matching *.txt files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to save extracted voice files.",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Qwen3-TTS model name or path. Default: Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    )
    parser.add_argument(
        "--gguf-talker", type=str, default=None,
        help="Path to local GGUF talker file (skips auto-download).",
    )
    parser.add_argument(
        "--gguf-codec", type=str, default=None,
        help="Path to local GGUF codec file (skips auto-download).",
    )
    parser.add_argument(
        "--backend", type=str, choices=["ggml", "torch"], default="torch",
        help="Backend to use. Default: torch (Windows compatible)",
    )
    parser.add_argument(
        "--xvec-only", action="store_true",
        help="Use x-vector only mode (cleaner starts, better language switching).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging.",
    )
    parser.add_argument(
        '--registry', type=Path, default=None,
        help='Path to voices.yaml to generate or append.',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = scan_pairs(input_dir)
    if not pairs:
        logger.error("No matching audio + txt pairs found in %s", input_dir)
        sys.exit(1)

    logger.info("Found %d audio+text pair(s) in %s", len(pairs), input_dir)
    gguf_note = ""

    if args.gguf_talker and args.gguf_codec:

        gguf_note = " | GGUF: custom paths"

    logger.info("Backend: %s | Model: %s | xvec_only: %s%s", args.backend, args.model, args.xvec_only, gguf_note)


    registry = []
    success_count = 0
    fail_count = 0

    for wav_path, txt_path in pairs:
        voice_name = wav_path.stem
        ref_text = txt_path.read_text(encoding="utf-8").strip()

        logger.info("Processing: %s (%s)", voice_name, ref_text[:60])

        start_time = time.perf_counter()
        try:
            audio_24k = load_audio_24k(wav_path)
            logger.debug("Loaded audio: %d samples (%.2fs)", audio_24k.shape[0], audio_24k.shape[0] / 24000)

            if args.backend == "ggml":
                result = extract_ggml(

                    audio_24k, args.model, output_dir, voice_name, ref_text,

                    args.xvec_only,

                    gguf_talker_path=args.gguf_talker,

                    gguf_codec_path=args.gguf_codec,

                )

            else:
                result = extract_torch(audio_24k, args.model, output_dir, voice_name, ref_text, args.xvec_only)

            elapsed = time.perf_counter() - start_time
            result["source_wav"] = str(wav_path)
            result["source_txt"] = str(txt_path)
            result["extract_seconds"] = round(elapsed, 2)

            # Save metadata JSON
            meta_path = output_dir / f"{voice_name}.json"
            meta_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

            registry.append({
                "name": voice_name,
                "backend": result["backend"],
                "ref_text": ref_text,
                "extract_seconds": result["extract_seconds"],
            })
            if result["backend"] == "ggml":
                registry[-1]["spk_file"] = result["spk_file"]
                registry[-1]["rvq_file"] = result["rvq_file"]
            else:
                registry[-1]["npz_file"] = result["npz_file"]

            success_count += 1
            logger.info("  OK: %s (%.2fs)", voice_name, elapsed)

        except Exception as e:
            fail_count += 1
            logger.error("  FAILED: %s - %s", voice_name, e, exc_info=True)

    # Save registry
    registry_path = output_dir / "registry.json"
    registry_path.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Registry saved to %s", registry_path)

    logger.info(
        "Done: %d succeeded, %d failed out of %d pairs",
        success_count, fail_count, len(pairs),
    )

    if args.registry and success_count > 0:
        write_registry_yaml(args.registry, registry, output_dir)

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
