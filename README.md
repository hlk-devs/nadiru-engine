# Nadiru Engine

Sovereign AI orchestration engine with a Conductor that learns your patterns and routes requests intelligently.

## Why Nadiru?

Most orchestration layers are static routers around API keys. Nadiru is different: every request is classified, routed, logged, and used to improve the next decision. The Conductor adapts to your acceptance patterns, provider reliability, and cost constraints.

![Nadiru Dashboard](docs/images/dashboard.png)
*Intelligent routing in action: simple questions route to free models, complex tasks delegate to quality providers. Five responses, three different models, two providers, total cost $0.007.*

## Conductor Tiers

- **Tier 1: Cloud Conductor (recommended)**
  - Examples: `google/gemini-2.5-flash`, `anthropic/claude-haiku-4-5-20251001`
  - Best quality/latency for JSON decisions, no cold-start training needed.
- **Tier 2: Beefy Local Conductor**
  - Example: `ollama/llama3.1:70b`
  - Private and capable, with strong local routing quality.
- **Tier 3: Small Local Conductor**
  - Example: `ollama/qwen2.5:14b`
  - Works well with delegate-first cold-start protection.

## Architecture Highlights

- **Provider-agnostic Conductor**: any configured provider can run classify/route calls.
- **Dynamic model discovery**: providers discover available model IDs at startup.
- **Registry-backed metadata**: `model_catalog.py` + `model_registry.json` provide model cost/quality metadata.
- **Cold start protection**: local conductors default to delegate-first until local success is proven.
- **Three-endpoint API**: small surface area, easy to embed in any app ("Nadi").

## API Contract

### POST /connect
Register a Nadi (agent/application) with the engine.

**Request:**
```json
{
  "name": "my-app",
  "description": "Optional description",
  "default_priority": "balanced"
}
```

**Response:**
```json
{
  "nadi_id": "uuid",
  "connected_at": "2026-04-14T12:00:00Z"
}
```

### POST /generate
Send a prompt, get a response. The Conductor decides routing.

**Request:**
```json
{
  "nadi_id": "uuid",
  "prompt": "Your request here",
  "messages": [
    {"role": "user", "content": "previous message"},
    {"role": "assistant", "content": "previous response"}
  ],
  "priority": "balanced",
  "max_cost": 0.05,
  "prefer_provider": "anthropic"
}
```

**Response:**
```json
{
  "content": "The response text",
  "model": "claude-sonnet-4-6",
  "provider": "anthropic",
  "tokens_in": 150,
  "tokens_out": 420,
  "cost_estimate": 0.0023,
  "latency_ms": 1840,
  "request_id": "uuid",
  "routing_reason": "Complex code task delegated to high-quality provider"
}
```

### GET /query
Read-only access to interaction history.

**Parameters:** `nadi_id`, `since`, `until`, `model`, `provider`, `min_cost`, `max_cost`, `limit`, `offset`

## Building a Nadi

A Nadi is any application that connects to the engine: CLI, web app, bot, dashboard, automation worker.

```python
import httpx

ENGINE = "http://localhost:8765"

resp = httpx.post(f"{ENGINE}/connect", json={"name": "my-nadi"})
nadi_id = resp.json()["nadi_id"]

resp = httpx.post(f"{ENGINE}/generate", json={
    "nadi_id": nadi_id,
    "prompt": "Explain quantum entanglement simply"
})
print(resp.json()["content"])

resp = httpx.get(f"{ENGINE}/query", params={"nadi_id": nadi_id, "limit": 10})
print(resp.json()["total"])
```

## Quick Start

```bash
# 1) Clone + install
git clone https://github.com/hlk-devs/nadiru-engine.git
cd nadiru-engine
pip install -r requirements.txt

# 2) Configure .env
cp .env.example .env
# Option A: Install Ollama and set CONDUCTOR_PROVIDER=ollama
# Option B: Set CONDUCTOR_PROVIDER to anthropic/openai/google and add API key

# 3) Run
python -m nadiru_engine
```

## Project Structure

```text
nadiru-engine/
+-- README.md
+-- LICENSE
+-- requirements.txt
+-- .env.example
+-- pyproject.toml
+-- docs/
   +-- CONDUCTOR_DESIGN.md
+-- community-nadis/
   +-- README.md
+-- nadiru_engine/
   +-- __init__.py
   +-- __main__.py
   +-- service.py
   +-- conductor.py
   +-- memory.py
   +-- model_catalog.py
   +-- model_registry.json
   +-- providers/
       +-- __init__.py
       +-- base.py
       +-- registry.py
       +-- ollama_provider.py
       +-- anthropic_provider.py
       +-- openai_provider.py
       +-- google_provider.py
       +-- groq_provider.py
       +-- deepseek_provider.py
       +-- together_provider.py
       +-- perplexity_provider.py
       +-- cerebras_provider.py
+-- tests/
    +-- test_conductor.py
    +-- test_memory.py
    +-- test_providers.py
```

## Philosophy

The engine stays intentionally small: routing, memory, providers, and API. Domain logic belongs in Nadis.

*From the deepest point flows the purest intelligence.*

## License

MIT.

## Roadmap

- [ ] Response streaming (v0.2.0)
- [ ] Web dashboard Nadi (v0.2.0)
- [ ] Periodic model re-discovery (v0.2.0)
- [ ] Nadi SDK Python package (v0.3.0)

