"""Tests for Conductor routing logic."""

import pytest
from unittest.mock import AsyncMock
from nadiru_engine.conductor import Conductor
from nadiru_engine.memory import MemoryStore
from nadiru_engine.providers.ollama_provider import OllamaProvider
from nadiru_engine.providers.anthropic_provider import AnthropicProvider


@pytest.fixture
def memory(tmp_path):
    db = MemoryStore(str(tmp_path / "test.db"))
    yield db
    db.close()


@pytest.fixture
def conductor(memory):
    ollama = OllamaProvider(conductor_model="test-model")
    providers = {
        "ollama": ollama,
        "anthropic": AnthropicProvider(api_key="test-key"),
    }
    return Conductor(
        memory=memory,
        conductor_provider=ollama,
        conductor_model="test-model",
        providers=providers,
        is_local_conductor=True,
    )


def test_conductor_init(conductor):
    assert conductor._conductor_provider is not None
    assert "anthropic" in conductor.providers
    assert "ollama" in conductor.providers


def test_get_available_providers(conductor):
    available = conductor._get_available_providers()
    assert len(available) >= 1
    names = [p["name"] for p in available]
    assert "anthropic" in names
    assert "ollama" not in names


def test_pick_fallback(conductor):
    fallback = conductor._pick_fallback("anthropic", "claude-sonnet-4")
    assert fallback is not None
    provider, _ = fallback
    assert provider == "ollama"


def test_pick_fallback_multiple_providers(memory):
    from nadiru_engine.providers.openai_provider import OpenAIProvider
    ollama = OllamaProvider(conductor_model="test-model")
    providers = {
        "ollama": ollama,
        "anthropic": AnthropicProvider(api_key="test"),
        "openai": OpenAIProvider(api_key="test"),
    }
    cond = Conductor(
        memory=memory,
        conductor_provider=ollama,
        conductor_model="test-model",
        providers=providers,
        is_local_conductor=True,
    )

    fallback = cond._pick_fallback("anthropic", "claude-sonnet-4")
    assert fallback is not None
    provider, _ = fallback
    assert provider == "openai"


def test_temp_for_type():
    assert Conductor._temp_for_type("code") == 0.3
    assert Conductor._temp_for_type("creative") == 0.8
    assert Conductor._temp_for_type("conversation") == 0.7
    assert Conductor._temp_for_type("nonsense") == 0.5


def test_get_signals_cold_start(conductor):
    signals = conductor._get_signals()
    assert signals["total_interactions"] == 0
    assert signals["local_success_rate"] == {}


@pytest.mark.asyncio
async def test_handle_request_routes_locally(conductor, memory):
    nadi = memory.register_nadi("test-nadi")

    conductor._classify = AsyncMock(return_value={
        "type": "conversation", "confidence": 0.95, "complexity": 1
    })
    conductor._route = AsyncMock(return_value={
        "route": "local", "provider": None, "model": None,
        "reason": "Simple greeting", "estimated_cost": 0.0
    })

    from nadiru_engine.providers.base import GenerateResult
    conductor._conductor_provider.generate = AsyncMock(return_value=GenerateResult(
        content="Hello! How can I help?",
        model="test-model", provider="ollama",
        tokens_in=5, tokens_out=8,
    ))

    result = await conductor.handle_request(
        nadi_id=nadi["nadi_id"],
        prompt="Hi there",
    )

    assert result["content"] == "Hello! How can I help?"
    assert result["provider"] == "ollama"
    assert result["cost_estimate"] == 0.0
    assert "request_id" in result

    query = memory.query_interactions(nadi_id=nadi["nadi_id"])
    assert query["total"] == 1


@pytest.mark.asyncio
async def test_handle_request_delegates(conductor, memory):
    nadi = memory.register_nadi("test-nadi")

    conductor._classify = AsyncMock(return_value={
        "type": "code", "confidence": 0.9, "complexity": 4
    })
    conductor._route = AsyncMock(return_value={
        "route": "delegate", "provider": "anthropic",
        "model": "claude-sonnet-4",
        "reason": "Complex code task", "estimated_cost": 0.003
    })

    from nadiru_engine.providers.base import GenerateResult
    conductor.providers["anthropic"].generate = AsyncMock(return_value=GenerateResult(
        content="Here's the code...",
        model="claude-sonnet-4", provider="anthropic",
        tokens_in=100, tokens_out=500, cost_estimate=0.003,
    ))

    result = await conductor.handle_request(
        nadi_id=nadi["nadi_id"],
        prompt="Write a FastAPI endpoint",
        priority="quality",
    )

    assert result["content"] == "Here's the code..."
    assert result["provider"] == "anthropic"
    assert result["model"] == "claude-sonnet-4"


@pytest.mark.asyncio
async def test_handle_request_fallback_on_error(conductor, memory):
    nadi = memory.register_nadi("test-nadi")

    conductor._classify = AsyncMock(return_value={
        "type": "code", "confidence": 0.9, "complexity": 4
    })
    conductor._route = AsyncMock(return_value={
        "route": "delegate", "provider": "anthropic",
        "model": "claude-sonnet-4",
        "reason": "Code task", "estimated_cost": 0.003
    })

    from nadiru_engine.providers.base import GenerateResult

    conductor.providers["anthropic"].generate = AsyncMock(return_value=GenerateResult(
        content="ERROR: rate limited",
        model="claude-sonnet-4", provider="anthropic",
        error="rate_limited",
    ))

    conductor._conductor_provider.generate = AsyncMock(return_value=GenerateResult(
        content="Here's my best attempt at the code...",
        model="test-model", provider="ollama",
        tokens_in=50, tokens_out=200,
    ))

    result = await conductor.handle_request(
        nadi_id=nadi["nadi_id"],
        prompt="Write a function",
    )

    assert result["provider"] == "ollama"
    assert "Fallback" in result["routing_reason"]
