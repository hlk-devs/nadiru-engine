"""Tests for MemoryStore."""

import os
import pytest
from nadiru_engine.memory import MemoryStore


@pytest.fixture
def memory(tmp_path):
    db = MemoryStore(str(tmp_path / "test.db"))
    yield db
    db.close()


def test_register_nadi(memory):
    result = memory.register_nadi("test-nadi", "A test agent", "balanced")
    assert "nadi_id" in result
    assert "connected_at" in result
    assert len(result["nadi_id"]) == 36  # UUID format


def test_get_nadi(memory):
    result = memory.register_nadi("test-nadi")
    nadi = memory.get_nadi(result["nadi_id"])
    assert nadi is not None
    assert nadi["name"] == "test-nadi"


def test_get_nadi_not_found(memory):
    nadi = memory.get_nadi("nonexistent-id")
    assert nadi is None


def test_log_interaction(memory):
    nadi = memory.register_nadi("test-nadi")
    request_id = memory.log_interaction(
        nadi_id=nadi["nadi_id"],
        prompt="Hello world",
        content="Hi there!",
        model="llama3.1:8b",
        provider="ollama",
        tokens_in=10,
        tokens_out=5,
        cost_estimate=0.0,
        latency_ms=200,
        task_type="conversation",
        complexity=1,
        routing_reason="Simple greeting, handled locally",
    )
    assert len(request_id) == 36


def test_query_interactions(memory):
    nadi = memory.register_nadi("test-nadi")
    nid = nadi["nadi_id"]

    for i in range(5):
        memory.log_interaction(
            nadi_id=nid, prompt=f"Question {i}", content=f"Answer {i}",
            model="llama3.1:8b", provider="ollama",
        )

    result = memory.query_interactions(nadi_id=nid)
    assert result["total"] == 5
    assert len(result["interactions"]) == 5


def test_query_with_filters(memory):
    nadi = memory.register_nadi("test-nadi")
    nid = nadi["nadi_id"]

    memory.log_interaction(
        nadi_id=nid, prompt="Cheap", content="Result",
        model="llama3.1:8b", provider="ollama", cost_estimate=0.0,
    )
    memory.log_interaction(
        nadi_id=nid, prompt="Expensive", content="Result",
        model="claude-sonnet-4", provider="anthropic", cost_estimate=0.01,
    )

    result = memory.query_interactions(provider="anthropic")
    assert result["total"] == 1
    assert result["interactions"][0]["provider"] == "anthropic"


def test_query_pagination(memory):
    nadi = memory.register_nadi("test-nadi")
    nid = nadi["nadi_id"]

    for i in range(10):
        memory.log_interaction(
            nadi_id=nid, prompt=f"Q{i}", content=f"A{i}",
            model="test", provider="ollama",
        )

    page1 = memory.query_interactions(limit=3, offset=0)
    page2 = memory.query_interactions(limit=3, offset=3)
    assert len(page1["interactions"]) == 3
    assert len(page2["interactions"]) == 3
    assert page1["total"] == 10


def test_implicit_feedback_rejected(memory):
    """Similar prompt within 60s = rejected."""
    nadi = memory.register_nadi("test-nadi")
    nid = nadi["nadi_id"]

    # First request
    memory.log_interaction(
        nadi_id=nid, prompt="How do I fix this Python error with imports",
        content="Try checking your sys.path",
        model="llama3.1:8b", provider="ollama",
    )

    # Very similar prompt right after (simulates re-prompt)
    memory.log_interaction(
        nadi_id=nid, prompt="How do I fix this Python error with imports please",
        content="You need to install the package",
        model="claude-sonnet-4", provider="anthropic",
    )

    # Check the first interaction's outcome
    result = memory.query_interactions(nadi_id=nid, limit=10)
    interactions = sorted(result["interactions"], key=lambda x: x["timestamp"])
    # First should be marked rejected (similar prompt came quickly)
    assert interactions[0]["outcome"] == "rejected"


def test_implicit_feedback_accepted(memory):
    """Different prompt = accepted (moved on)."""
    nadi = memory.register_nadi("test-nadi")
    nid = nadi["nadi_id"]

    memory.log_interaction(
        nadi_id=nid, prompt="What is the capital of France",
        content="Paris",
        model="llama3.1:8b", provider="ollama",
    )

    memory.log_interaction(
        nadi_id=nid, prompt="How do I bake chocolate chip cookies",
        content="Here's a recipe...",
        model="llama3.1:8b", provider="ollama",
    )

    result = memory.query_interactions(nadi_id=nid, limit=10)
    interactions = sorted(result["interactions"], key=lambda x: x["timestamp"])
    assert interactions[0]["outcome"] == "accepted"


def test_compute_user_signals_empty(memory):
    signals = memory.compute_user_signals()
    assert signals["total_interactions"] == 0
    assert signals["local_success_rate"] == {}


def test_compute_user_signals_with_data(memory):
    nadi = memory.register_nadi("test-nadi")
    nid = nadi["nadi_id"]

    # Log some interactions with outcomes
    for i in range(5):
        rid = memory.log_interaction(
            nadi_id=nid, prompt=f"Code question {i}",
            content=f"Answer {i}",
            model="llama3.1:8b", provider="ollama",
            task_type="code",
        )

    # Manually set some outcomes for testing
    memory._conn.execute(
        "UPDATE interactions SET outcome = 'accepted' "
        "WHERE task_type = 'code' AND rowid <= 3"
    )
    memory._conn.execute(
        "UPDATE interactions SET outcome = 'rejected' "
        "WHERE task_type = 'code' AND rowid > 3"
    )
    memory._conn.commit()

    signals = memory.compute_user_signals()
    assert signals["total_interactions"] == 5
    assert "code" in signals["local_success_rate"]


def test_jaccard_similarity():
    sim = MemoryStore._jaccard_similarity(
        "how do I fix python import errors",
        "how do I fix python import errors please help"
    )
    assert sim > 0.7  # Very similar

    sim = MemoryStore._jaccard_similarity(
        "how do I fix python import errors",
        "what is the best pizza in New York"
    )
    assert sim < 0.2  # Very different
