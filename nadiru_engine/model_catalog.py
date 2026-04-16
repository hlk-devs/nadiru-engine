"""Model metadata catalog and merge utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .providers.base import ModelInfo


_DEFAULT_REGISTRY: dict[str, Any] = {'models': {'claude-sonnet-4': {'provider': 'anthropic', 'cost_per_1m_input': 3.0, 'cost_per_1m_output': 15.0, 'max_output_tokens': 8192, 'max_context_window': 200000, 'speed_tier': 'medium', 'quality_tier': 4}, 'claude-haiku-4-5': {'provider': 'anthropic', 'cost_per_1m_input': 0.8, 'cost_per_1m_output': 4.0, 'max_output_tokens': 8192, 'max_context_window': 200000, 'speed_tier': 'fast', 'quality_tier': 3}, 'claude-opus-4': {'provider': 'anthropic', 'cost_per_1m_input': 15.0, 'cost_per_1m_output': 75.0, 'max_output_tokens': 8192, 'max_context_window': 200000, 'speed_tier': 'slow', 'quality_tier': 5}, 'gpt-4o': {'provider': 'openai', 'cost_per_1m_input': 2.5, 'cost_per_1m_output': 10.0, 'max_output_tokens': 16384, 'max_context_window': 128000, 'speed_tier': 'medium', 'quality_tier': 4}, 'gpt-4o-mini': {'provider': 'openai', 'cost_per_1m_input': 0.15, 'cost_per_1m_output': 0.6, 'max_output_tokens': 16384, 'max_context_window': 128000, 'speed_tier': 'fast', 'quality_tier': 3}, 'gpt-4.1': {'provider': 'openai', 'cost_per_1m_input': 2.0, 'cost_per_1m_output': 8.0, 'max_output_tokens': 32768, 'max_context_window': 1047576, 'speed_tier': 'medium', 'quality_tier': 4}, 'gpt-4.1-mini': {'provider': 'openai', 'cost_per_1m_input': 0.4, 'cost_per_1m_output': 1.6, 'max_output_tokens': 32768, 'max_context_window': 1047576, 'speed_tier': 'fast', 'quality_tier': 3}, 'gemini-2.5-flash': {'provider': 'google', 'cost_per_1m_input': 0.15, 'cost_per_1m_output': 0.6, 'max_output_tokens': 8192, 'max_context_window': 1048576, 'speed_tier': 'fast', 'quality_tier': 3}, 'gemini-2.0-flash-lite': {'provider': 'google', 'cost_per_1m_input': 0.0, 'cost_per_1m_output': 0.0, 'max_output_tokens': 8192, 'max_context_window': 1048576, 'speed_tier': 'fast', 'quality_tier': 2}, 'llama-3.3-70b-versatile': {'provider': 'groq', 'cost_per_1m_input': 0.59, 'cost_per_1m_output': 0.79, 'max_output_tokens': 8192, 'max_context_window': 128000, 'speed_tier': 'fast', 'quality_tier': 3}, 'llama-3.1-8b-instant': {'provider': 'groq', 'cost_per_1m_input': 0.05, 'cost_per_1m_output': 0.08, 'max_output_tokens': 8192, 'max_context_window': 128000, 'speed_tier': 'fast', 'quality_tier': 2}, 'mixtral-8x7b-32768': {'provider': 'groq', 'cost_per_1m_input': 0.24, 'cost_per_1m_output': 0.24, 'max_output_tokens': 32768, 'max_context_window': 32768, 'speed_tier': 'fast', 'quality_tier': 3}, 'deepseek-chat': {'provider': 'deepseek', 'cost_per_1m_input': 0.14, 'cost_per_1m_output': 0.28, 'max_output_tokens': 8192, 'max_context_window': 64000, 'speed_tier': 'medium', 'quality_tier': 3}, 'meta-llama/Llama-4-Scout-17B-16E-Instruct': {'provider': 'together', 'cost_per_1m_input': 0.18, 'cost_per_1m_output': 0.18, 'max_output_tokens': 8192, 'max_context_window': 512000, 'speed_tier': 'fast', 'quality_tier': 3}, 'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8': {'provider': 'together', 'cost_per_1m_input': 0.27, 'cost_per_1m_output': 0.27, 'max_output_tokens': 8192, 'max_context_window': 512000, 'speed_tier': 'medium', 'quality_tier': 4}, 'Qwen/Qwen2.5-72B-Instruct-Turbo': {'provider': 'together', 'cost_per_1m_input': 0.6, 'cost_per_1m_output': 0.6, 'max_output_tokens': 8192, 'max_context_window': 131072, 'speed_tier': 'medium', 'quality_tier': 4}, 'sonar-pro': {'provider': 'perplexity', 'cost_per_1m_input': 3.0, 'cost_per_1m_output': 15.0, 'max_output_tokens': 8192, 'max_context_window': 200000, 'speed_tier': 'medium', 'quality_tier': 4}, 'sonar': {'provider': 'perplexity', 'cost_per_1m_input': 1.0, 'cost_per_1m_output': 1.0, 'max_output_tokens': 8192, 'max_context_window': 128000, 'speed_tier': 'fast', 'quality_tier': 3}, 'llama-4-scout-17b-16e-instruct': {'provider': 'cerebras', 'cost_per_1m_input': 0.1, 'cost_per_1m_output': 0.1, 'max_output_tokens': 8192, 'max_context_window': 128000, 'speed_tier': 'fast', 'quality_tier': 3}}, 'default_metadata': {'cost_per_1m_input': 1.0, 'cost_per_1m_output': 2.0, 'max_output_tokens': 4096, 'max_context_window': 128000, 'speed_tier': 'medium', 'quality_tier': 3}}


class ModelCatalog:
    """Loads model metadata registry and merges API-discovered model IDs."""

    def __init__(self, registry_path: str | Path | None = None):
        self.registry_path = Path(registry_path) if registry_path else Path(__file__).with_name("model_registry.json")
        self._data = self._load_registry()

    def _load_registry(self) -> dict[str, Any]:
        if self.registry_path.exists():
            try:
                return json.loads(self.registry_path.read_text())
            except Exception:
                return _DEFAULT_REGISTRY
        return _DEFAULT_REGISTRY

    @property
    def default_metadata(self) -> dict[str, Any]:
        return self._data.get("default_metadata", _DEFAULT_REGISTRY["default_metadata"])

    def get_metadata(self, model_id: str, provider_name: str | None = None) -> ModelInfo:
        model_meta = self._data.get("models", {}).get(model_id, {})
        defaults = self.default_metadata
        provider = model_meta.get("provider") or provider_name or "unknown"
        return ModelInfo(
            name=model_id,
            provider=provider,
            cost_per_1m_input=float(model_meta.get("cost_per_1m_input", defaults["cost_per_1m_input"])),
            cost_per_1m_output=float(model_meta.get("cost_per_1m_output", defaults["cost_per_1m_output"])),
            max_output_tokens=int(model_meta.get("max_output_tokens", defaults["max_output_tokens"])),
            max_context_window=int(model_meta.get("max_context_window", defaults["max_context_window"])),
            speed_tier=str(model_meta.get("speed_tier", defaults["speed_tier"])),
            quality_tier=int(model_meta.get("quality_tier", defaults["quality_tier"])),
        )

    def merge_discovered(self, provider_name: str, discovered_model_ids: list[str]) -> list[ModelInfo]:
        seen: set[str] = set()
        merged: list[ModelInfo] = []
        for model_id in discovered_model_ids:
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)
            merged.append(self.get_metadata(model_id, provider_name=provider_name))
        return merged

    def models_for_provider(self, provider_name: str) -> list[ModelInfo]:
        entries = self._data.get("models", {})
        out: list[ModelInfo] = []
        for model_id, meta in entries.items():
            if meta.get("provider") == provider_name:
                out.append(self.get_metadata(model_id, provider_name=provider_name))
        return out
