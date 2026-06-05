from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ENDPOINT = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-v4-flash"


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key: str
    endpoint: str
    model: str


def load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def load_deepseek_config(config_path: str | Path) -> DeepSeekConfig:
    file_values = load_env_file(Path(config_path))
    api_key = os.getenv("DEEPSEEK_API_KEY") or file_values.get("DEEPSEEK_API_KEY", "")
    endpoint = os.getenv("DEEPSEEK_ENDPOINT") or file_values.get("DEEPSEEK_ENDPOINT", DEFAULT_ENDPOINT)
    model = os.getenv("DEEPSEEK_MODEL") or file_values.get("DEEPSEEK_MODEL", DEFAULT_MODEL)
    return DeepSeekConfig(api_key=api_key, endpoint=endpoint, model=model)
