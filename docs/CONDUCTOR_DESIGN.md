# Conductor Design

## Overview

The Conductor is the routing brain for Nadiru. It is provider-agnostic and can run on local or cloud models. Every request goes through:

1. classification (`_classify`)
2. routing decision (`_route`)
3. execution (`_delegate`, local execution, or `_delegate_cheapest`)

Both classify and route prompts live in `nadiru_engine/conductor.py`.

## Provider-Agnostic Execution

The Conductor no longer depends on Ollama-specific helper methods. It accepts:

- `conductor_provider` (any `BaseProvider`)
- `conductor_model`
- `is_local_conductor`

This enables:

- cloud Conductor (Google/Anthropic/OpenAI)
- beefy local Conductor
- small local Conductor with cold-start protection

## Classification and Routing

`_classify` and `_route` both:

- call `self._conductor_provider.generate(...)`
- enforce temperature `0.0`
- request strict JSON-only outputs
- parse responses with `_parse_json_response`

## Robust JSON Parsing

`_parse_json_response` handles common model quirks:

- raw JSON object
- fenced JSON (` ```json ... ``` `)
- fenced text without language tag
- preamble text before JSON
- trailing commentary after JSON
- extraction of first valid balanced JSON object

If parsing fails, `_route` safely defaults to delegation.

## Cold Start Behavior

Cold-start guard is only active for local conductors.

- Cloud conductor: no cold-start override (`_check_cold_start_override` returns `None`)
- Local conductor: delegate-first strategy for low-confidence task types until enough successful local outcomes accumulate

Default local cold-start rates (`_cold_start_rate`) intentionally favor delegation for factual/code/analysis and keep only conversation mostly local.

## Execution Paths

After routing:

- `route == "local"` and local conductor: execute on conductor model
- `route == "local"` and cloud conductor: reinterpret as cheapest delegation via `_delegate_cheapest`
- `route == "delegate"`: execute chosen provider/model via `_delegate`
- if delegated call fails: one fallback attempt with `_pick_fallback`

## Signals and Learning

`_get_signals` uses cached/aggregated memory signals from `MemoryStore`.

Signals influence routing through:

- task-type local success rates
- preferred models by task type
- recent provider errors

All outcomes and routing reasons are logged for continuous implicit learning.

## Conductor Tiers

1. **Cloud Conductor (recommended)**
   - best JSON reliability and low latency for routing calls
2. **Beefy Local Conductor**
   - private, strong local quality
3. **Small Local Conductor**
   - efficient with delegate-first training period

## Key Methods in `conductor.py`

- `handle_request`
- `_classify`
- `_route`
- `_parse_json_response`
- `_check_cold_start_override`
- `_delegate`
- `_delegate_cheapest`
- `_pick_fallback`
- `_get_signals`
- `_get_available_providers`
- `_cold_start_rate`
- `_temp_for_type`
