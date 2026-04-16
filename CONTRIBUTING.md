# Contributing to Nadiru Engine

Thanks for contributing.

## Principles

1. Keep the engine small and composable.
2. Keep app/domain behavior in Nadis, not in core engine modules.
3. Keep routing deterministic and debuggable.
4. Add tests for all behavior changes.

## Development Flow

1. Fork and branch.
2. Make focused changes.
3. Run `pytest tests/ -v`.
4. Open a PR with a clear description and test evidence.

## Adding a Provider

### OpenAI-compatible APIs

For providers that expose an OpenAI-style HTTP API, use **`OpenAICompatibleProvider`**: add an entry to **`PROVIDER_MAP`** in `nadiru_engine/providers/registry.py` (env var name, module/class, optional `kwargs` such as `base_url` and `name`). The engine loads the provider when that env var is set.

### Non-standard APIs

For bespoke APIs, subclass **`BaseProvider`**, implement **`generate()`** and **`stream_generate()`**, and **`discover_models()`**. Register the provider in **`PROVIDER_MAP`** with the env var that gates loading.

### Metadata

Add model metadata to **`nadiru_engine/model_registry.json`** so routing and cost estimates stay accurate.

### Tests

Add tests under `tests/test_providers.py` for new behavior.

Provider files should not contain hardcoded model lists. Models are discovered dynamically and merged with metadata from `model_registry.json` via `model_catalog.py`.

## Code Style

- Prefer explicit types and readable method boundaries.
- Keep provider adapters focused and free of dead/commented code.
- Keep public behavior documented in README and design docs.

## Where to Contribute

- Engine routing, memory, providers, and service API: this repository.
- End-user applications (Nadis): community repos.
