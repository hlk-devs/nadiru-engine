"""Provider registry and environment-based loader."""

import importlib
import logging
import os
import re
from typing import Any

from .base import BaseProvider
from .ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)

_ENV_SUB = re.compile(r"\$\{([^}]+)\}")


def _substitute_env(value: str) -> str:
    """Replace ${VAR_NAME} with os.environ[VAR_NAME] (empty if unset)."""

    def _repl(match: re.Match[str]) -> str:
        return os.getenv(match.group(1), "")

    return _ENV_SUB.sub(_repl, value)


def _resolve_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in kwargs.items():
        if isinstance(val, str):
            out[key] = _substitute_env(val)
        else:
            out[key] = val
    return out


# Maps env var name -> module, class, and optional constructor kwargs.
# The engine loads a provider ONLY if its env var is set and non-empty.
# Users can extend this by adding entries before calling load_providers().
PROVIDER_MAP: dict[str, dict[str, Any]] = {
    "ANTHROPIC_API_KEY": {
        "module": "nadiru_engine.providers.anthropic_provider",
        "class": "AnthropicProvider",
    },
    "OPENAI_API_KEY": {
        "module": "nadiru_engine.providers.openai_provider",
        "class": "OpenAIProvider",
    },
    "GOOGLE_API_KEY": {
        "module": "nadiru_engine.providers.google_provider",
        "class": "GoogleProvider",
    },
    "GROQ_API_KEY": {
        "module": "nadiru_engine.providers.groq_provider",
        "class": "GroqProvider",
    },
    "DEEPSEEK_API_KEY": {
        "module": "nadiru_engine.providers.deepseek_provider",
        "class": "DeepSeekProvider",
    },
    "TOGETHER_API_KEY": {
        "module": "nadiru_engine.providers.together_provider",
        "class": "TogetherProvider",
    },
    "PERPLEXITY_API_KEY": {
        "module": "nadiru_engine.providers.perplexity_provider",
        "class": "PerplexityProvider",
    },
    "CEREBRAS_API_KEY": {
        "module": "nadiru_engine.providers.cerebras_provider",
        "class": "CerebrasProvider",
    },
    "OPENROUTER_API_KEY": {
        "module": "nadiru_engine.providers.openai_compatible_provider",
        "class": "OpenAICompatibleProvider",
        "kwargs": {
            "name": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
        },
    },
    "MISTRAL_API_KEY": {
        "module": "nadiru_engine.providers.openai_compatible_provider",
        "class": "OpenAICompatibleProvider",
        "kwargs": {
            "name": "mistral",
            "base_url": "https://api.mistral.ai/v1",
        },
    },
    "FIREWORKS_API_KEY": {
        "module": "nadiru_engine.providers.openai_compatible_provider",
        "class": "OpenAICompatibleProvider",
        "kwargs": {
            "name": "fireworks",
            "base_url": "https://api.fireworks.ai/inference/v1",
        },
    },
    "AI21_API_KEY": {
        "module": "nadiru_engine.providers.openai_compatible_provider",
        "class": "OpenAICompatibleProvider",
        "kwargs": {
            "name": "ai21",
            "base_url": "https://api.ai21.com/studio/v1",
        },
    },
    "COHERE_API_KEY": {
        "module": "nadiru_engine.providers.openai_compatible_provider",
        "class": "OpenAICompatibleProvider",
        "kwargs": {
            "name": "cohere",
            "base_url": "https://api.cohere.com/v2",
        },
    },
    "AZURE_OPENAI_API_KEY": {
        "module": "nadiru_engine.providers.openai_compatible_provider",
        "class": "OpenAICompatibleProvider",
        "kwargs": {
            "name": "azure_openai",
            "base_url": "${AZURE_OPENAI_ENDPOINT}/openai/deployments",
            "auth_header": "api-key",
            "auth_prefix": "",
            "azure_openai": True,
        },
    },
}


def load_providers(ollama: OllamaProvider) -> dict[str, BaseProvider]:
    """
    Auto-discover and load providers based on environment variables.

    Returns a dict of provider_name -> provider_instance.
    Ollama is always included. Paid providers are loaded only if their
    API key env var is set.
    """
    providers: dict[str, BaseProvider] = {"ollama": ollama}

    for env_var, config in PROVIDER_MAP.items():
        api_key = os.getenv(env_var, "").strip()
        if not api_key:
            continue
        if env_var == "AZURE_OPENAI_API_KEY":
            if not os.getenv("AZURE_OPENAI_ENDPOINT", "").strip():
                logger.warning(
                    "Skipping Azure OpenAI: AZURE_OPENAI_ENDPOINT is not set"
                )
                continue

        try:
            module = importlib.import_module(config["module"])
            provider_class = getattr(module, config["class"])
            kwargs = _resolve_kwargs(dict(config.get("kwargs", {})))
            instance = provider_class(api_key=api_key, **kwargs)
            providers[instance.name] = instance
        except (ImportError, AttributeError, TypeError) as e:
            logger.warning("Could not load provider for %s: %s", env_var, e)
            continue

    return providers
