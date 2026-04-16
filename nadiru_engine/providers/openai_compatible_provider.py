"""Generic OpenAI-compatible HTTP adapter for third-party APIs."""

import os
from typing import Optional

import httpx

from ..model_catalog import ModelCatalog
from .base import BaseProvider, GenerateResult


class OpenAICompatibleProvider(BaseProvider):
    """Generic adapter for any OpenAI-compatible API.

    Works with: OpenRouter, Mistral, Fireworks, AI21,
    Azure OpenAI, and any other OpenAI-format API.
    """

    def __init__(
        self,
        name: str,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        auth_header: str = "Authorization",
        auth_prefix: str = "Bearer",
        azure_openai: bool = False,
        api_version: Optional[str] = None,
    ):
        super().__init__(api_key=api_key)
        self._name = name
        self.base_url = base_url.rstrip("/")
        self.auth_header = auth_header
        self.auth_prefix = auth_prefix
        self._azure_openai = azure_openai
        self._api_version = api_version or "2024-08-01-preview"
        self._resource_endpoint = (
            os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
            if azure_openai
            else None
        )
        self._catalog = ModelCatalog()
        self.set_active_models(self._catalog.models_for_provider(self.name))

    @property
    def name(self) -> str:
        return self._name

    def _auth_headers(self) -> dict[str, str]:
        if not self.api_key:
            return {"Content-Type": "application/json"}
        if self.auth_prefix:
            token = f"{self.auth_prefix} {self.api_key}".strip()
        else:
            token = self.api_key
        return {
            self.auth_header: token,
            "Content-Type": "application/json",
        }

    def _models_url(self) -> str:
        if self._azure_openai and self._resource_endpoint:
            return (
                f"{self._resource_endpoint}/openai/models"
                f"?api-version={self._api_version}"
            )
        return f"{self.base_url}/models"

    def _chat_completions_url(self, model_name: str) -> str:
        if self._azure_openai:
            base = self.base_url.rstrip("/")
            return (
                f"{base}/{model_name}/chat/completions"
                f"?api-version={self._api_version}"
            )
        return f"{self.base_url}/chat/completions"

    async def discover_models(self) -> list[str]:
        if not self.api_key:
            return []
        if self._azure_openai and not self._resource_endpoint:
            return []
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(self._models_url(), headers=self._auth_headers())
                resp.raise_for_status()
                data = resp.json()
            models = data.get("data", [])
            return [
                m["id"]
                for m in models
                if isinstance(m, dict) and "id" in m
            ]
        except Exception:
            return []

    async def generate(
        self,
        model_name: str,
        prompt: str,
        messages: list[dict] = None,
        system_prompt: str = None,
        max_tokens: int = None,
        temperature: float = 0.7,
    ) -> GenerateResult:
        if not self.api_key:
            return GenerateResult(
                content=f"ERROR: No API key configured for {self._name}",
                model=model_name,
                provider=self._name,
                error="no_api_key",
            )

        model_info = self.get_model(model_name)
        if max_tokens is None:
            max_tokens = model_info.max_output_tokens if model_info else 4096

        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        if messages:
            api_messages.extend(messages)
        api_messages.append({"role": "user", "content": prompt})

        payload = {
            "max_tokens": max_tokens,
            "messages": api_messages,
            "temperature": temperature,
        }
        if not self._azure_openai:
            payload["model"] = model_name

        url = self._chat_completions_url(model_name)

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(url, json=payload, headers=self._auth_headers())
                resp.raise_for_status()
                data = resp.json()

            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            tokens_in = usage.get("prompt_tokens", 0)
            tokens_out = usage.get("completion_tokens", 0)

            return GenerateResult(
                content=content,
                model=model_name,
                provider=self._name,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_estimate=self.estimate_cost(model_name, tokens_in, tokens_out),
            )

        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            error_type = {
                401: "auth_failed",
                404: "model_not_found",
                429: "rate_limited",
            }.get(code, f"http_{code}")
            return GenerateResult(
                content=f"ERROR: {self._name} returned {code}",
                model=model_name,
                provider=self._name,
                error=error_type,
            )
        except Exception as e:
            return GenerateResult(
                content=f"ERROR: {str(e)}",
                model=model_name,
                provider=self._name,
                error="unknown",
            )


