"""Generation over a vLLM OpenAI-compatible endpoint (the ONE backend).

Same backend is used to produce gate-training labels and to serve, so the
feature distribution the gate is calibrated on never shifts. Generations are
disk-cached on sha1(model|messages) so full eval runs resume after a Colab drop.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _cache_key(model: str, messages: list[dict], max_tokens: int, temperature: float) -> str:
    payload = json.dumps(
        {"m": model, "msgs": messages, "mt": max_tokens, "t": temperature},
        sort_keys=True,
    ).encode()
    return hashlib.sha1(payload).hexdigest()


class Generator:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        cache_dir: str | Path = "artifacts/gen_cache",
        max_tokens: int = 256,
        temperature: float = 0.0,
    ):
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
        self.model = model or os.environ.get("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "EMPTY")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            from openai import OpenAI  # lazy
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def generate(self, messages: list[dict]) -> str:
        key = _cache_key(self.model, messages, self.max_tokens, self.temperature)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())["text"]

        resp = self._client_lazy().chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        text = resp.choices[0].message.content.strip()
        cache_file.write_text(json.dumps({"text": text}))
        return text
