"""Ollama provider adapter."""

import json
import httpx
from typing import Optional
from ..model_catalog import ModelCatalog
from .base import BaseProvider, ModelInfo, GenerateResult


class OllamaProvider(BaseProvider):
    """Local Ollama models. No API key needed."""

    def __init__(self, base_url: str = "http://localhost:11434",
                 conductor_model: str = "llama3.1:8b"):
        super().__init__(api_key="local")
        self.base_url = base_url.rstrip("/")
        self.conductor_model = conductor_model
        self._catalog = ModelCatalog()
        defaults = self._catalog.models_for_provider(self.name)
        if defaults:
            self.set_active_models(defaults)
        else:
            self.set_active_models([
                ModelInfo(
                    name=self.conductor_model,
                    provider="ollama",
                    cost_per_1m_input=0.0,
                    cost_per_1m_output=0.0,
                    max_output_tokens=4096,
                    max_context_window=128000,
                    speed_tier="fast",
                    quality_tier=2,
                )
            ])

    @property
    def name(self) -> str:
        return "ollama"

    async def discover_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
            return [
                model.get("name")
                for model in data.get("models", [])
                if isinstance(model, dict) and model.get("name")
            ]
        except Exception:
            return []

    async def generate(self, model_name: str, prompt: str,
                       messages: list[dict] = None,
                       system_prompt: str = None,
                       max_tokens: int = None,
                       temperature: float = 0.7) -> GenerateResult:
        """Generate via Ollama /api/chat endpoint."""
        chat_messages = []

        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})

        if messages:
            chat_messages.extend(messages)

        chat_messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model_name,
            "messages": chat_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()

            content = data.get("message", {}).get("content", "")
            tokens_in = data.get("prompt_eval_count", 0)
            tokens_out = data.get("eval_count", 0)

            return GenerateResult(
                content=content,
                model=model_name,
                provider="ollama",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_estimate=0.0,
            )

        except httpx.ConnectError:
            return GenerateResult(
                content="ERROR: Cannot connect to Ollama. Is it running?",
                model=model_name,
                provider="ollama",
                error="connection_failed",
            )
        except httpx.HTTPStatusError as e:
            return GenerateResult(
                content=f"ERROR: Ollama returned {e.response.status_code}",
                model=model_name,
                provider="ollama",
                error=f"http_{e.response.status_code}",
            )
        except Exception as e:
            return GenerateResult(
                content=f"ERROR: {str(e)}",
                model=model_name,
                provider="ollama",
                error="unknown",
            )
    async def stream_generate(
        self, model_name: str, prompt: str,
        messages: list[dict] = None,
        system_prompt: str = None,
        max_tokens: int = None,
        temperature: float = 0.7,
    ):
        chat_messages = []
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        if messages:
            chat_messages.extend(messages)
        chat_messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model_name,
            "messages": chat_messages,
            "stream": True,
            "options": {"temperature": temperature},
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        prev = ""
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/api/chat", json=payload
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        full = (obj.get("message") or {}).get("content") or ""
                        if len(full) > len(prev):
                            yield full[len(prev):]
                            prev = full
        except Exception:
            result = await self.generate(
                model_name, prompt, messages, system_prompt, max_tokens, temperature
            )
            if result.content:
                yield result.content


