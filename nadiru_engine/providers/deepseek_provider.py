"""DeepSeek provider adapter."""

import httpx
from typing import Optional
from ..model_catalog import ModelCatalog
from .base import BaseProvider, GenerateResult


class DeepSeekProvider(BaseProvider):

    API_URL = "https://api.deepseek.com/v1/chat/completions"

    MODELS_URL = "https://api.deepseek.com/v1/models"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key=api_key)
        self._catalog = ModelCatalog()
        self.set_active_models(self._catalog.models_for_provider(self.name))

    @property
    def name(self) -> str:
        return "deepseek"

    async def discover_models(self) -> list[str]:
        if not self.api_key:
            return []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(self.MODELS_URL, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            return [m.get("id") for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
        except Exception:
            return []

    async def generate(self, model_name: str, prompt: str,
                       messages: list[dict] = None,
                       system_prompt: str = None,
                       max_tokens: int = None,
                       temperature: float = 0.7) -> GenerateResult:
        if not self.api_key:
            return GenerateResult(
                content="ERROR: No DeepSeek API key configured",
                model=model_name, provider="deepseek", error="no_api_key",
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
            "model": model_name,
            "max_tokens": max_tokens,
            "messages": api_messages,
            "temperature": temperature,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(self.API_URL, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            tokens_in = usage.get("prompt_tokens", 0)
            tokens_out = usage.get("completion_tokens", 0)

            return GenerateResult(
                content=content, model=model_name, provider="deepseek",
                tokens_in=tokens_in, tokens_out=tokens_out,
                cost_estimate=self.estimate_cost(model_name, tokens_in, tokens_out),
            )
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            error_type = {401: "auth_failed", 404: "model_not_found",
                         429: "rate_limited"}.get(code, f"http_{code}")
            return GenerateResult(
                content=f"ERROR: DeepSeek returned {code}",
                model=model_name, provider="deepseek", error=error_type,
            )
        except Exception as e:
            return GenerateResult(
                content=f"ERROR: {str(e)}",
                model=model_name, provider="deepseek", error="unknown",
            )
