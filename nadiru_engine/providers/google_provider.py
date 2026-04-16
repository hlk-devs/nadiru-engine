"""Google provider adapter."""

import json
import httpx
from typing import Optional
from ..model_catalog import ModelCatalog
from .base import BaseProvider, GenerateResult


class GoogleProvider(BaseProvider):

    API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key=api_key)
        self._catalog = ModelCatalog()
        self.set_active_models(self._catalog.models_for_provider(self.name))

    @property
    def name(self) -> str:
        return "google"


    async def generate(self, model_name: str, prompt: str,
                       messages: list[dict] = None,
                       system_prompt: str = None,
                       max_tokens: int = None,
                       temperature: float = 0.7) -> GenerateResult:
        if not self.api_key:
            return GenerateResult(
                content="ERROR: No Google API key configured",
                model=model_name, provider="google",
                error="no_api_key",
            )

        model_info = self.get_model(model_name)
        if max_tokens is None:
            max_tokens = model_info.max_output_tokens if model_info else 4096

        # Build contents array
        contents = []
        if messages:
            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg["content"]}]
                })
        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }

        url = f"{self.API_BASE}/{model_name}:generateContent?key={self.api_key}"

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()


            # Extract content
            candidates = data.get("candidates", [])
            if not candidates:
                return GenerateResult(
                    content="ERROR: No response from Google",
                    model=model_name, provider="google", error="empty_response",
                )

            finish_reasons = [c.get("finishReason") for c in candidates if isinstance(c, dict)]

            parts_per_candidate = [
                len(c.get("content", {}).get("parts", [])) if isinstance(c, dict) else 0
                for c in candidates
            ]

            # Current behavior: use first candidate and join all its parts.
            parts = candidates[0].get("content", {}).get("parts", [])
            content = "".join(part.get("text", "") for part in parts)

            usage = data.get("usageMetadata", {})
            tokens_in = usage.get("promptTokenCount", 0)
            tokens_out = usage.get("candidatesTokenCount", 0)

            return GenerateResult(
                content=content,
                model=model_name,
                provider="google",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_estimate=self.estimate_cost(model_name, tokens_in, tokens_out),
            )

        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            error_type = {401: "auth_failed", 404: "model_not_found",
                         429: "rate_limited"}.get(code, f"http_{code}")
            return GenerateResult(
                content=f"ERROR: Google returned {code}",
                model=model_name, provider="google", error=error_type,
            )
        except Exception as e:
            return GenerateResult(
                content=f"ERROR: {str(e)}",
                model=model_name, provider="google", error="unknown",
            )
