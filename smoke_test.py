"""
Smoke test — run this after starting the engine to verify the full loop.

Usage:
  1. Start the engine: python -m nadiru_engine
  2. In another terminal: python smoke_test.py
"""

import httpx
import sys
import time

ENGINE = "http://localhost:8765"


def main():
    print("Nadiru Engine Smoke Test")
    print("=" * 40)

    # 1. Health check
    print("\n1. Health check...")
    try:
        resp = httpx.get(f"{ENGINE}/health", timeout=5.0)
        resp.raise_for_status()
        health = resp.json()
        print(f"   Status: {health['status']}")
        conductor_provider = health.get('conductor_provider', 'ollama')
        print(f"   Conductor: {conductor_provider}/{health['conductor_model']}")
        print(f"   Interactions logged: {health['interactions']}")
    except httpx.ConnectError:
        print("   FAILED — is the engine running? Start it with: python -m nadiru_engine")
        sys.exit(1)

    # 2. Connect a test Nadi
    print("\n2. Connecting test Nadi...")
    resp = httpx.post(f"{ENGINE}/connect", json={
        "name": "smoke-test",
        "description": "Verifying the engine works",
        "default_priority": "balanced",
    })
    resp.raise_for_status()
    nadi = resp.json()
    nadi_id = nadi["nadi_id"]
    print(f"   Nadi ID: {nadi_id}")
    print(f"   Connected at: {nadi['connected_at']}")

    # 3. Simple generate — should route locally
    print("\n3. Generate (simple question — should route locally)...")
    start = time.time()
    resp = httpx.post(f"{ENGINE}/generate", json={
        "nadi_id": nadi_id,
        "prompt": "What is 2 + 2?",
        "priority": "cost",
    }, timeout=120.0)
    elapsed = time.time() - start
    resp.raise_for_status()
    result = resp.json()
    print(f"   Response: {result['content'][:100]}...")
    print(f"   Model: {result['model']}")
    print(f"   Provider: {result['provider']}")
    print(f"   Cost: ${result['cost_estimate']:.6f}")
    print(f"   Latency: {result['latency_ms']}ms (wall: {elapsed:.1f}s)")
    print(f"   Routing: {result['routing_reason']}")

    # 4. Complex generate — may delegate if providers configured
    print("\n4. Generate (complex question — may delegate)...")
    start = time.time()
    resp = httpx.post(f"{ENGINE}/generate", json={
        "nadi_id": nadi_id,
        "prompt": "Write a Python function that implements binary search on a sorted list, with proper error handling and type hints.",
        "priority": "quality",
    }, timeout=120.0)
    elapsed = time.time() - start
    resp.raise_for_status()
    result = resp.json()
    print(f"   Response: {result['content'][:100]}...")
    print(f"   Model: {result['model']}")
    print(f"   Provider: {result['provider']}")
    print(f"   Cost: ${result['cost_estimate']:.6f}")
    print(f"   Latency: {result['latency_ms']}ms (wall: {elapsed:.1f}s)")
    print(f"   Routing: {result['routing_reason']}")

    # 5. Query history
    print("\n5. Querying interaction history...")
    resp = httpx.get(f"{ENGINE}/query", params={
        "nadi_id": nadi_id,
        "limit": 10,
    })
    resp.raise_for_status()
    history = resp.json()
    print(f"   Total interactions: {history['total']}")
    for ix in history["interactions"]:
        print(f"   - [{ix['provider']}/{ix['model']}] {ix['prompt'][:50]}... "
              f"(${ix['cost_estimate']:.6f})")

    # 6. Final health
    print("\n6. Final health check...")
    resp = httpx.get(f"{ENGINE}/health")
    health = resp.json()
    print(f"   Interactions logged: {health['interactions']}")
    print(f"   Nadis registered: {health['nadis']}")

    print("\n" + "=" * 40)
    print("SMOKE TEST PASSED")
    print("The engine is working. Start building Nadis.")


if __name__ == "__main__":
    main()
