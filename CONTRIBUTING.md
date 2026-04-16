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

When adding a provider adapter:

1. Subclass `BaseProvider`.
2. Implement `generate()` and `discover_models()`.
3. Add its env-var mapping to `PROVIDER_MAP` in `nadiru_engine/providers/registry.py`.
4. Add provider model metadata to `nadiru_engine/model_registry.json`.
5. Add tests under `tests/test_providers.py`.

Provider files should not contain hardcoded model lists. Models are discovered dynamically and merged with metadata from `model_registry.json` via `model_catalog.py`.

## Code Style

- Prefer explicit types and readable method boundaries.
- Keep provider adapters focused and free of dead/commented code.
- Keep public behavior documented in README and design docs.

## Where to Contribute

- Engine routing, memory, providers, and service API: this repository.
- End-user applications (Nadis): community repos.
