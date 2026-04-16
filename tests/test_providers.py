"""Tests for provider adapters."""

import pytest
from nadiru_engine.providers.base import ModelInfo, GenerateResult, BaseProvider
from nadiru_engine.providers.ollama_provider import OllamaProvider
from nadiru_engine.providers.anthropic_provider import AnthropicProvider
from nadiru_engine.providers.openai_provider import OpenAIProvider
from nadiru_engine.providers.google_provider import GoogleProvider
from nadiru_engine.conductor import Conductor


def test_model_info():
    m = ModelInfo("test-model", "test", 1.0, 2.0, 4096, 128000, "fast", 3)
    assert m.name == "test-model"
    assert m.cost_per_1m_input == 1.0


def test_generate_result():
    r = GenerateResult(content="Hello", model="test", provider="test")
    assert r.content == "Hello"
    assert r.error is None


def test_cost_estimation():
    provider = AnthropicProvider(api_key=None)
    # claude-sonnet-4: $3/M input, $15/M output
    cost = provider.estimate_cost("claude-sonnet-4", 1000, 500)
    expected = (1000 / 1_000_000 * 3.0) + (500 / 1_000_000 * 15.0)
    assert abs(cost - expected) < 0.0001


def test_get_model():
    provider = AnthropicProvider(api_key=None)
    model = provider.get_model("claude-sonnet-4")
    assert model is not None
    assert model.quality_tier == 4

    missing = provider.get_model("nonexistent")
    assert missing is None


def test_provider_not_available_without_key():
    provider = AnthropicProvider(api_key=None)
    assert not provider.is_available

    provider = AnthropicProvider(api_key="sk-test")
    assert provider.is_available


def test_ollama_always_available():
    provider = OllamaProvider()
    assert provider.is_available
    assert provider.name == "ollama"
    assert len(provider.models) >= 1
    assert provider.models[0].cost_per_1m_input == 0.0


def test_anthropic_models():
    provider = AnthropicProvider(api_key="test")
    assert provider.name == "anthropic"
    names = [m.name for m in provider.models]
    assert "claude-sonnet-4" in names
    assert "claude-opus-4" in names


def test_openai_models():
    provider = OpenAIProvider(api_key="test")
    assert provider.name == "openai"
    names = [m.name for m in provider.models]
    assert "gpt-4o" in names
    assert "gpt-4o-mini" in names


def test_google_models():
    provider = GoogleProvider(api_key="test")
    assert provider.name == "google"
    names = [m.name for m in provider.models]
    assert "gemini-2.5-flash" in names


@pytest.mark.asyncio
async def test_anthropic_no_key():
    provider = AnthropicProvider(api_key=None)
    result = await provider.generate("claude-sonnet-4", "Hello")
    assert result.error == "no_api_key"


@pytest.mark.asyncio
async def test_openai_no_key():
    provider = OpenAIProvider(api_key=None)
    result = await provider.generate("gpt-4o", "Hello")
    assert result.error == "no_api_key"


@pytest.mark.asyncio
async def test_google_no_key():
    provider = GoogleProvider(api_key=None)
    result = await provider.generate("gemini-2.5-flash", "Hello")
    assert result.error == "no_api_key"


def test_cold_start_rates():
    rates = {
        "conversation": Conductor._cold_start_rate("conversation"),
        "code": Conductor._cold_start_rate("code"),
        "complex": Conductor._cold_start_rate("complex"),
    }
    assert rates["conversation"] > rates["code"]
    assert rates["code"] > rates["complex"]


