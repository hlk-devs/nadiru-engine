"""Provider registry and environment-based loader."""

import importlib
import logging
import os
from .base import BaseProvider
from .ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)

# Maps env var name -> (module path, class name)
# The engine loads a provider ONLY if its env var is set and non-empty.
# Users can extend this by adding entries before calling load_providers().
PROVIDER_MAP: dict[str, tuple[str, str]] = {
    "ANTHROPIC_API_KEY": ("nadiru_engine.providers.anthropic_provider", "AnthropicProvider"),
    "OPENAI_API_KEY": ("nadiru_engine.providers.openai_provider", "OpenAIProvider"),
    "GOOGLE_API_KEY": ("nadiru_engine.providers.google_provider", "GoogleProvider"),
    "GROQ_API_KEY": ("nadiru_engine.providers.groq_provider", "GroqProvider"),
    "DEEPSEEK_API_KEY": ("nadiru_engine.providers.deepseek_provider", "DeepSeekProvider"),
    "TOGETHER_API_KEY": ("nadiru_engine.providers.together_provider", "TogetherProvider"),
    "PERPLEXITY_API_KEY": ("nadiru_engine.providers.perplexity_provider", "PerplexityProvider"),
    "CEREBRAS_API_KEY": ("nadiru_engine.providers.cerebras_provider", "CerebrasProvider"),
}


def load_providers(ollama: OllamaProvider) -> dict[str, BaseProvider]:
    """
    Auto-discover and load providers based on environment variables.
    
    Returns a dict of provider_name -> provider_instance.
    Ollama is always included. Paid providers are loaded only if their
    API key env var is set.
    """
    providers: dict[str, BaseProvider] = {"ollama": ollama}

    for env_var, (module_path, class_name) in PROVIDER_MAP.items():
        api_key = os.getenv(env_var, "").strip()
        if not api_key:
            continue

        try:
            module = importlib.import_module(module_path)
            provider_class = getattr(module, class_name)
            instance = provider_class(api_key=api_key)
            providers[instance.name] = instance
        except (ImportError, AttributeError) as e:
            logger.warning("Could not load provider for %s: %s", env_var, e)
            continue

    return providers
