"""
Voice Registry - Load and manage pre-extracted voice profiles.

Provides a registry of named voices with metadata, loaded from a YAML config.
Each voice entry points to pre-extracted voice cache files (.spk/.rvq for GGML
or .npz for Torch).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class VoiceProfile:
    """A single registered voice with its cache files and metadata."""

    name: str
    display_name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    backend: str = "ggml"
    spk_file: Optional[str] = None
    rvq_file: Optional[str] = None
    npz_file: Optional[str] = None
    ref_text: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_ggml(self) -> bool:
        return self.backend == "ggml"

    @property
    def is_torch(self) -> bool:
        return self.backend == "torch"

    def validate(self) -> list[str]:
        """Check that required files exist. Returns list of errors."""
        errors = []
        if self.is_ggml:
            if not self.spk_file:
                errors.append(f"Voice '{self.name}': missing spk_file for GGML backend")
            elif not Path(self.spk_file).exists():
                errors.append(f"Voice '{self.name}': spk_file not found: {self.spk_file}")
            if not self.rvq_file:
                errors.append(f"Voice '{self.name}': missing rvq_file for GGML backend")
            elif not Path(self.rvq_file).exists():
                errors.append(f"Voice '{self.name}': rvq_file not found: {self.rvq_file}")
        elif self.is_torch:
            if not self.npz_file:
                errors.append(f"Voice '{self.name}': missing npz_file for Torch backend")
            elif not Path(self.npz_file).exists():
                errors.append(f"Voice '{self.name}': npz_file not found: {self.npz_file}")
        return errors


@dataclass
class VoiceRegistry:
    """Registry of available voices, loaded from config."""

    voices: dict[str, VoiceProfile] = field(default_factory=dict)
    default_voice: str = ""
    config_path: Optional[str] = None

    def add(self, profile: VoiceProfile) -> None:
        self.voices[profile.name] = profile

    def get(self, name: str) -> Optional[VoiceProfile]:
        return self.voices.get(name)

    def get_default(self) -> Optional[VoiceProfile]:
        if self.default_voice and self.default_voice in self.voices:
            return self.voices[self.default_voice]
        if self.voices:
            return next(iter(self.voices.values()))
        return None

    def list_voices(self) -> list[VoiceProfile]:
        return list(self.voices.values())

    def list_names(self) -> list[str]:
        return list(self.voices.keys())

    def search(self, query: str) -> list[VoiceProfile]:
        """Search voices by name, display_name, description, or tags."""
        query_lower = query.lower()
        results = []
        for v in self.voices.values():
            if (
                query_lower in v.name.lower()
                or query_lower in v.display_name.lower()
                or query_lower in v.description.lower()
                or any(query_lower in tag.lower() for tag in v.tags)
            ):
                results.append(v)
        return results

    def validate_all(self) -> list[str]:
        """Validate all voices. Returns list of errors."""
        errors = []
        for v in self.voices.values():
            errors.extend(v.validate())
        return errors


def load_registry(config_path: str | Path) -> VoiceRegistry:
    """Load a VoiceRegistry from a YAML config file."""
    if not HAS_YAML:
        raise ImportError("PyYAML is required to load voice registry configs")

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Voice registry config not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    registry = VoiceRegistry(config_path=str(path))

    default = data.get("default_voice")
    if default:
        registry.default_voice = default

    for entry in data.get("voices", []):
        profile = VoiceProfile(
            name=entry["name"],
            display_name=entry.get("display_name", entry["name"]),
            description=entry.get("description", ""),
            tags=entry.get("tags", []),
            backend=entry.get("backend", "ggml"),
            spk_file=entry.get("spk_file"),
            rvq_file=entry.get("rvq_file"),
            npz_file=entry.get("npz_file"),
            ref_text=entry.get("ref_text", ""),
            meta=entry.get("meta", {}),
        )
        registry.add(profile)

    errors = registry.validate_all()
    if errors:
        logger.warning("Voice registry validation warnings: %s", "; ".join(errors))

    logger.info(
        "Loaded voice registry from %s: %d voice(s), default=%r",
        path,
        len(registry.voices),
        registry.default_voice,
    )
    return registry
