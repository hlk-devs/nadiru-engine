# Changelog

## [0.1.1] - 2026-04-16

### Added
- Response streaming via POST /generate/stream (SSE)
- Refusal detection with automatic retry to less restrictive providers
- Generic OpenAI-compatible provider adapter
- Support for OpenRouter, Mistral, Fireworks, AI21, Cohere, Azure OpenAI
- GET /providers endpoint listing all configured providers
- POST /test-provider for direct provider health testing
- Web dashboard (nadi-dashboard) with streaming support
- CORS middleware for browser-based Nadis
- Model validation preventing routing to non-existent models
- Conductor model exclusion from generation pool

### Fixed
- Gemini thinking tokens causing truncated routing responses
- Anthropic model ID mismatches from dynamic discovery
- Cold start override bypassing Conductor self-assessment
- Dashboard UTF-8 encoding issues

### Changed
- Cold start defaults to delegate-first philosophy
- Provider registry now uses dict format with optional kwargs
- Model lists no longer hardcoded in provider files

## [0.1.0] - 2026-04-15

### Added
- Sovereign AI orchestration engine with three-endpoint API
- Provider-agnostic Conductor supporting cloud and local models
- Dynamic model discovery across all providers
- Cold start protection: delegate-first, earn local trust
- Community-maintainable model metadata registry
- 9 provider adapters: Ollama, Anthropic, OpenAI, Google, Groq, DeepSeek, Together, Perplexity, Cerebras
- Implicit feedback learning from user behavior
- SQLite-backed interaction memory store
- CLI chat interface (nadi-chat)
- 34 unit tests
- MIT license
