# Nadiru

**An AI orchestration engine that picks the right model for the job.**

You send it a prompt. A Conductor LLM classifies what kind of task it is, picks the right provider and model based on cost, quality, and what's worked before, and routes the request. It learns from outcomes. It handles refusals. It works with local models, cloud APIs, or any mix you want to wire up.

![Nadiru Dashboard](docs/images/dashboard.png)

*Routing in action. Simple questions go to fast cheap models. Complex tasks delegate to better ones. The dashboard shows what got routed where and why.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Tests](https://img.shields.io/badge/tests-34_passing-green)
![Python](https://img.shields.io/badge/python-3.11+-blue)

## Why this exists

I was paying for Anthropic, OpenAI, and Google subscriptions, plus running Ollama locally, and manually deciding which one to use for what. Quick math problem? Free Gemini Flash. Complex code refactor? Claude Sonnet. Creative writing? Different Claude. It was annoying to keep track of, and I kept making the wrong call and either burning money on overkill models or getting bad output from undersized ones.

OpenRouter and LiteLLM both solve part of this. They give you one API to talk to many providers. But they don't actually think about the request. They route based on static rules you configure, not based on what the prompt is actually asking for.

So I built Nadiru. The core idea is that a small fast model (the Conductor) reads every incoming prompt, classifies it, and decides where it should go. Over time it learns which models are good at which tasks based on whether you re-prompted, how long things took, what cost what. The longer you run it, the better it gets at sending things to the right place.

I wrote the longer story on [dev.to](https://dev.to/homelesslighthousekeeper/i-built-an-open-source-ai-orchestration-engine-that-learns-how-you-work-550l) if you want context.

## What makes it different

**The Conductor is itself a model, and you pick which one.** Run a free local model on Ollama for routing. Or use Gemini Flash or Claude Haiku in the cloud for better routing JSON without needing a GPU. Or burn the latency and use a beefy model for the smartest decisions. It's your call.

**It learns from outcomes.** Every interaction logs task type, latency, cost, refusals, retries, and whether you came back and rephrased. The Conductor uses these signals to tighten future routing. If a 14B local model has a 70% success rate on summaries, future summaries route locally. If GPT-4o keeps refusing a certain task type, that task starts going elsewhere.

**It handles refusals.** When a model returns a policy refusal, Nadiru can retry once on a less restrictive provider. You get an answer instead of a dead end.

**It's actually provider-agnostic.** 15+ adapters built in. Any OpenAI-compatible API drops in by adding an env var to the registry. The Conductor itself can be any of them.

**It logs everything.** Cost per request, tokens used, routing reason, fallback attempts, refusals. Every decision is auditable.

## Currently used in production

Nadiru is the orchestration layer behind [Paaseki](https://paaseki.com), an AI-powered SEO audit service. Each audit costs about half a cent in AI fees because the Conductor routes most work to cheap models and only escalates when needed. That kind of cost control would be miserable to manage by hand.

If you build something on top of Nadiru, open an issue or PR and I'll add it here.

## Quick start

About 90 seconds if you have an API key handy.

```bash
git clone https://github.com/hlk-devs/nadiru-engine.git
cd nadiru-engine
pip install -r requirements.txt
cp .env.example .env
# edit .env, add at least one provider key (Anthropic, OpenAI, Google, etc)
python -m nadiru_engine
```

Engine starts on `http://localhost:8765`. In another terminal:

```python
import httpx

ENGINE = "http://localhost:8765"

# Register your client (called a "Nadi")
r = httpx.post(f"{ENGINE}/connect", json={
    "name": "demo",
    "default_priority": "balanced"
})
nadi_id = r.json()["nadi_id"]

# Generate something
r = httpx.post(f"{ENGINE}/generate", json={
    "nadi_id": nadi_id,
    "prompt": "Explain quantum entanglement to a 10 year old",
})
result = r.json()

print(result["content"])
print(f"\nRouted to: {result['provider']}/{result['model']}")
print(f"Reason: {result['routing_reason']}")
print(f"Cost: ${result['cost_estimate']:.6f}")
print(f"Latency: {result['latency_ms']}ms")
```

That's it. The Conductor read your prompt, decided where to send it, and routed accordingly.

## The Conductor decision

The Conductor's job is small (read prompt, classify, return JSON), so you don't need an expensive model for it. Three reasonable defaults:

| Tier | Conductor | When to use |
|------|-----------|-------------|
| Cloud routing | Gemini 2.5 Flash, Claude Haiku 4.5 | Best routing JSON, no GPU needed, roughly $0.0001 per decision |
| Beefy local | 70B+ on Ollama | Full privacy, no per-request cost, needs decent GPU |
| Small local | 8 to 14B on Ollama | Free, slower, delegates more during cold start until it earns trust |

Set your choice with `CONDUCTOR_PROVIDER` and `CONDUCTOR_MODEL` in your `.env`. Gemini Flash is what I use day to day. It's cheap, fast, and the JSON is reliable.

There's a longer write-up of how the Conductor works in [docs/CONDUCTOR_DESIGN.md](docs/CONDUCTOR_DESIGN.md) if you want the details.

## API surface

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/connect` | Register a client, returns a `nadi_id` |
| `POST` | `/generate` | One-shot generation, full metadata in response |
| `POST` | `/generate/stream` | SSE streaming. Routing event first, then token chunks, then `done` |
| `GET`  | `/query` | Paginated interaction history per Nadi |
| `GET`  | `/providers` | Configured providers and discovered models |
| `POST` | `/test-provider` | Direct provider health check |
| `GET`  | `/health` | Engine and Conductor status |

Every `/generate` response includes which provider and model handled it, why, what it cost, and how long it took. No black boxes.

## What's a Nadi?

A Nadi is anything that talks to the engine. CLI, web app, automation script, IDE plugin, whatever. You register it once, get an ID, and from then on the engine logs interactions under that ID and learns from them. Each Nadi has its own history and the Conductor uses that history when routing.

There's a companion repo at [nadiru-nadis](https://github.com/hlk-devs/nadiru-nadis) with thirteen example Nadis covering chat, code review, cost tracking, web scraping, code generation, and more. Worth cloning if you want to see what building a Nadi looks like.

## Supported providers

| Provider | Adapter | Auth env var |
|----------|---------|--------------|
| Ollama | `OllamaProvider` | local, optional `OLLAMA_BASE_URL` |
| Anthropic | `AnthropicProvider` | `ANTHROPIC_API_KEY` |
| OpenAI | `OpenAIProvider` | `OPENAI_API_KEY` |
| Google | `GoogleProvider` | `GOOGLE_API_KEY` |
| Groq | `GroqProvider` | `GROQ_API_KEY` |
| DeepSeek | `DeepSeekProvider` | `DEEPSEEK_API_KEY` |
| Together | `TogetherProvider` | `TOGETHER_API_KEY` |
| Perplexity | `PerplexityProvider` | `PERPLEXITY_API_KEY` |
| Cerebras | `CerebrasProvider` | `CEREBRAS_API_KEY` |
| OpenRouter | `OpenAICompatibleProvider` | `OPENROUTER_API_KEY` |
| Mistral | `OpenAICompatibleProvider` | `MISTRAL_API_KEY` |
| Fireworks | `OpenAICompatibleProvider` | `FIREWORKS_API_KEY` |
| AI21 | `OpenAICompatibleProvider` | `AI21_API_KEY` |
| Cohere | `OpenAICompatibleProvider` | `COHERE_API_KEY` |
| Azure OpenAI | `OpenAICompatibleProvider` | `AZURE_OPENAI_API_KEY` plus `AZURE_OPENAI_ENDPOINT` |

Any other OpenAI-compatible API works. Add an env var and a `PROVIDER_MAP` entry in `nadiru_engine/providers/registry.py` and you're set. Models are discovered dynamically, no hardcoded lists.

## Project layout

```
nadiru-engine/
â”œâ”€â”€ README.md
â”œâ”€â”€ CHANGELOG.md
â”œâ”€â”€ CONTRIBUTING.md
â”œâ”€â”€ CODE_OF_CONDUCT.md
â”œâ”€â”€ LICENSE
â”œâ”€â”€ pyproject.toml
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ .env.example
â”œâ”€â”€ docs/
â”‚   â”œâ”€â”€ CONDUCTOR_DESIGN.md
â”‚   â””â”€â”€ images/
â”‚       â””â”€â”€ dashboard.png
â”œâ”€â”€ nadiru_engine/
â”‚   â”œâ”€â”€ __main__.py
â”‚   â”œâ”€â”€ service.py
â”‚   â”œâ”€â”€ conductor.py
â”‚   â”œâ”€â”€ memory.py
â”‚   â”œâ”€â”€ model_catalog.py
â”‚   â”œâ”€â”€ model_registry.json
â”‚   â””â”€â”€ providers/
â”‚       â”œâ”€â”€ base.py
â”‚       â”œâ”€â”€ registry.py
â”‚       â”œâ”€â”€ ollama_provider.py
â”‚       â”œâ”€â”€ openai_compatible_provider.py
â”‚       â””â”€â”€ ...
â””â”€â”€ tests/
```

The engine stays small on purpose. Routing, memory, providers, API. Everything else lives in Nadis.

## Roadmap

- [x] Response streaming (v0.1.1)
- [x] Refusal detection and retry (v0.1.1)
- [x] Generic OpenAI-compatible provider (v0.1.1)
- [x] Direct provider health testing (v0.1.1)
- [ ] Configurable routing prompts (v0.2.0)
- [ ] Periodic model re-discovery (v0.2.0)
- [ ] Python SDK package (v0.3.0)
- [ ] Pipeline / multi-step orchestration (v0.3.0)

## Contributing

Issues and PRs welcome. The bar is "does it keep the engine small and the routing debuggable." See [CONTRIBUTING.md](CONTRIBUTING.md) for development flow and the right way to add a provider.

## Philosophy

Most AI tools today are wrappers around a single provider. They become useless the day that provider raises prices, changes terms, or gets bought. Nadiru is the opposite shape. It gets more useful as the provider landscape gets messier, because the orchestration layer is where the value sits when there are a hundred models and you need to pick the right one.

Your data, your providers, your routing logic. The Conductor is yours to choose.

*From the deepest point flows the purest intelligence.*

## License

MIT. Copyright HLK Devs. See [LICENSE](LICENSE).
