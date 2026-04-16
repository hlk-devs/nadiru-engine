"""Conductor: classify, route, execute, and learn from outcomes."""

import json
import logging
import time
from typing import Optional

from .memory import MemoryStore
from .providers.base import BaseProvider, GenerateResult, ModelInfo

logger = logging.getLogger(__name__)

RECOMPUTE_INTERVAL = 50
CLASSIFY_FALLBACK = {"type": "unknown", "confidence": 0.0, "complexity": 3}

CLASSIFY_SYSTEM_PROMPT = """You are a task classifier. Return ONLY valid JSON.
No markdown or explanation.
Categories: factual, creative, code, analysis, summarize, translate, conversation, complex, unknown.
JSON: {"type":"<category>","confidence":<0.0-1.0>,"complexity":<1-5>}"""

ROUTE_RULES = """1. quality -> delegate to best model
2. use user preferred model when available
3. honor preferred_provider unless recent errors
4. cost + local_success>70% -> local
5. complexity<=2 + local_success>60% -> local
6. speed -> lowest latency option
7. if max_cost blocks providers -> local
8. default: complexity>3 or local_success<50% -> delegate"""
_REFUSAL_MARKERS_STRONG = (
    "i can't assist with that",
    "i cannot assist with that",
    "i can't help with that",
    "i cannot help with that",
    "against my guidelines",
    "i'm not able to provide",
    "i am not able to provide",
    "i must decline",
    "i'm programmed to be a harmless",
    "i am programmed to be a harmless",
    "falls into a grey area",
    "could potentially be used for unethical",
    "sorry, but i can't",
    "sorry, but i cannot",
    "i'm sorry, but as an ai",
    "i am sorry, but as an ai",
)

_REFUSAL_MARKERS_WEAK_ABLE = (
    "i'm unable to",
    "i am unable to",
)

_CAPABILITY_NOT_REFUSAL = (
    "i don't have real-time",
    "i don't have access to",
    "i can't browse the internet",
)

_RETRY_AFTER_REFUSAL_ORDER = (
    "deepseek",
    "groq",
    "together",
    "cerebras",
    "openai",
    "anthropic",
    "google",
    "perplexity",
)


class Conductor:
    """Provider-agnostic routing brain used by the service layer."""

    def __init__(
        self,
        memory: MemoryStore,
        conductor_provider: BaseProvider,
        conductor_model: str,
        providers: dict[str, BaseProvider],
        is_local_conductor: bool,
    ):
        self.memory = memory
        self._conductor_provider = conductor_provider
        self._conductor_model = conductor_model
        self.providers = providers
        self._is_local_conductor = is_local_conductor
        self._cached_signals: Optional[dict] = None
        self._interactions_since_recompute = 0

    def _active_models_for_routing(self, provider_name: str, provider: BaseProvider) -> list[ModelInfo]:
        """Models allowed as generation targets; excludes the Conductor model on its own provider."""
        active = list(provider._active_models)
        if provider_name != self._conductor_provider.name:
            return active
        return [m for m in active if m.name != self._conductor_model]

    def _detect_refusal(self, content: str) -> bool:
        """True if content looks like a content-policy refusal (substring match, case-insensitive)."""
        if not content:
            return False
        lower = content.lower()
        if any(s in lower for s in _REFUSAL_MARKERS_STRONG):
            return True
        if any(s in lower for s in _REFUSAL_MARKERS_WEAK_ABLE):
            if any(c in lower for c in _CAPABILITY_NOT_REFUSAL):
                return False
            return True
        return False

    def _pick_retry_after_refusal(
        self,
        original_provider: str,
        original_model: str,
        available_providers: list[dict],
    ) -> tuple[Optional[str], Optional[str]]:
        """Pick a different provider/model for retry after refusal.
        Prefer providers known to be less restrictive."""
        for provider_name in _RETRY_AFTER_REFUSAL_ORDER:
            if provider_name == original_provider:
                continue
            for p in available_providers:
                if p["name"] == provider_name and p.get("models"):
                    model = p["models"][0]
                    return (provider_name, model)
        return (None, None)


    async def handle_request(
        self,
        nadi_id: str,
        prompt: str,
        messages: list[dict] = None,
        priority: str = "balanced",
        max_cost: float = None,
        prefer_provider: str = None,
    ) -> dict:
        """Classify, route, execute, persist interaction, and return response payload."""
        started = time.time()
        signals = self._get_signals()
        classification = await self._classify(prompt)
        task_type = classification.get("type", "unknown")
        complexity = classification.get("complexity", 3)

        available = self._get_available_providers()
        routing = self._check_cold_start_override(task_type, signals, available)
        if routing is None:
            routing = await self._route(
                prompt, classification, priority, max_cost, prefer_provider, signals, available
            )

        routing = self._repair_invalid_delegate_routing(routing, available)
        routing = self._coerce_delegate_to_active_models(routing)
        route = routing.get("route", "delegate")
        provider_name = routing.get("provider")
        model_name = routing.get("model")
        routing_reason = routing.get("reason", "")

        if route == "local" or (not provider_name and not model_name):
            if self._is_local_conductor:
                result = await self._conductor_provider.generate(
                    model_name=self._conductor_model,
                    prompt=prompt,
                    messages=messages,
                    temperature=self._temp_for_type(task_type),
                )
            else:
                result = await self._delegate_cheapest(prompt, messages, task_type)
                if not routing_reason:
                    routing_reason = "Cloud conductor local route mapped to cheapest delegated model"
        else:
            result = await self._delegate(provider_name, model_name, prompt, messages, task_type)
            if result.error and result.error != "no_api_key":
                fallback = self._pick_fallback(provider_name, model_name)
                if fallback:
                    fb_provider, fb_model = fallback
                    routing_reason += f" | Fallback from {provider_name}/{model_name}"
                    result = await self._delegate(fb_provider, fb_model, prompt, messages, task_type)

        if result.content and self._detect_refusal(result.content):
            retry_provider, retry_model = self._pick_retry_after_refusal(
                original_provider=result.provider,
                original_model=result.model,
                available_providers=available,
            )
            if retry_provider and retry_model:
                refused_provider = result.provider
                refused_model = result.model
                retry_result = await self._delegate(
                    provider_name=retry_provider,
                    model_name=retry_model,
                    prompt=prompt,
                    messages=messages,
                    task_type=task_type,
                )
                if not self._detect_refusal(retry_result.content):
                    result = retry_result
                    rr = routing_reason or ""
                    routing_reason = (
                        rr
                        + f" | Refusal from {refused_provider}/{refused_model}, "
                        + f"retried with {retry_provider}/{retry_model}"
                    )

        latency_ms = int((time.time() - started) * 1000)
        request_id = self.memory.log_interaction(
            nadi_id=nadi_id,
            prompt=prompt,
            content=result.content,
            model=result.model,
            provider=result.provider,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            cost_estimate=result.cost_estimate,
            latency_ms=latency_ms,
            task_type=task_type,
            complexity=complexity,
            routing_reason=routing_reason,
        )

        self._interactions_since_recompute += 1
        if self._interactions_since_recompute >= RECOMPUTE_INTERVAL:
            self._cached_signals = self.memory.compute_user_signals()
            self._interactions_since_recompute = 0

        return {
            "content": result.content,
            "model": result.model,
            "provider": result.provider,
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
            "cost_estimate": result.cost_estimate,
            "latency_ms": latency_ms,
            "request_id": request_id,
            "routing_reason": routing_reason,
        }

    async def _classify(self, prompt: str) -> dict:
        """Run classification prompt on configured conductor model."""
        result = await self._conductor_provider.generate(
            model_name=self._conductor_model,
            prompt=f'Request: "{prompt}"',
            system_prompt=CLASSIFY_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=1024,
        )
        if result.error:
            return CLASSIFY_FALLBACK.copy()
        parsed = self._parse_json_response(result.content)
        return parsed if parsed is not None else CLASSIFY_FALLBACK.copy()

    async def _route(
        self,
        prompt: str,
        classification: dict,
        priority: str,
        max_cost: float | None,
        prefer_provider: str | None,
        user_signals: dict,
        available_providers: list[dict],
    ) -> dict:
        """Run routing prompt and return parsed route payload."""
        task_type = classification.get("type", "unknown")
        complexity = classification.get("complexity", 3)
        confidence = classification.get("confidence", 0.5)
        local_rate = user_signals.get("local_success_rate", {}).get(task_type, self._cold_start_rate(task_type))
        preferred = user_signals.get("preferred_models", {}).get(task_type, "none")
        errors = ", ".join(user_signals.get("recent_error_providers", [])) or "none"

        provider_lines = [
            f"- {p['name']}: {', '.join(p['models'])} (avg cost: ${p.get('avg_cost', 0):.4f})"
            for p in available_providers
        ]
        providers_text = "\n".join(provider_lines) if provider_lines else "- none configured"
        local_label = "local" if self._is_local_conductor else "cloud conductor"

        system = f"""You are a routing engine. Return ONLY valid JSON.
No markdown or explanation.
Task: {task_type} (complexity={complexity}/5, confidence={confidence})
Prompt length: {len(prompt.split())} words
Priority: {priority}
Max cost: {f'${max_cost}' if max_cost else 'none'}
Preferred provider: {prefer_provider or 'none'}
Conductor model: {self._conductor_provider.name}/{self._conductor_model} ({local_label})
Local success rate for {task_type}: {int(local_rate * 100)}%
User preferred model for {task_type}: {preferred}
You MUST pick a model name EXACTLY as listed below. Do not modify, abbreviate, or invent model names.
Available providers (exact model IDs from discovery):
{providers_text}
Recent errors: {errors}
Rules:
{ROUTE_RULES}
JSON: {{"route":"local|delegate","provider":"<name|null>","model":"<name|null>","reason":"<one sentence>","estimated_cost":<float>}}"""

        result = await self._conductor_provider.generate(
            model_name=self._conductor_model,
            prompt=f'Route this request: "{prompt[:500]}"',
            system_prompt=system,
            temperature=0.0,
            max_tokens=2048,
        )
        if result.error:
            return self._delegate_fallback(available_providers, "Conductor unavailable, defaulting to delegation")

        parsed = self._parse_json_response(result.content)
        if parsed is None:
            return self._delegate_fallback(
                available_providers,
                "Could not parse routing decision, defaulting to delegation",
            )
        return parsed

    def _validate_routing(self, routing: dict, available_providers: list[dict]) -> bool:
        """Verify the routed provider/model actually exists."""
        provider = routing.get("provider")
        model = routing.get("model")
        if not provider or not model:
            return False
        for p in available_providers:
            if p["name"] == provider and model in p["models"]:
                return True
        return False

    def _best_delegate_pair(
        self, available_providers: list[dict], provider_name: str
    ) -> Optional[tuple[str, str]]:
        """Pick best tier model for a provider using only names listed in available_providers."""
        meta = next((x for x in available_providers if x["name"] == provider_name), None)
        if not meta or not meta.get("models"):
            return None
        pn = meta["name"]
        prov = self.providers.get(pn)
        if not prov:
            return None
        allowed = set(meta["models"])
        pool = [m for m in self._active_models_for_routing(pn, prov) if m.name in allowed]
        if not pool:
            return None
        best = self._pick_best_model_from_models(pool)
        chosen = best or pool[0]
        return pn, chosen.name

    def _repair_invalid_delegate_routing(
        self, routing: dict, available_providers: list[dict]
    ) -> dict:
        """If delegate target is not in the discovered list, substitute a valid pair."""
        if routing.get("route") != "delegate":
            return routing
        if not available_providers:
            return routing
        if self._validate_routing(routing, available_providers):
            return routing
        bad_p = routing.get("provider")
        bad_m = routing.get("model")
        picked = None
        if bad_p:
            picked = self._best_delegate_pair(available_providers, bad_p)
        if not picked:
            picked = self._best_delegate_pair(
                available_providers, available_providers[0]["name"]
            )
        if not picked:
            return routing
        ap, am = picked
        logger.warning(
            "Conductor suggested %s/%s which doesn't exist, using %s/%s instead",
            bad_p,
            bad_m,
            ap,
            am,
        )
        out = dict(routing)
        out["provider"] = ap
        out["model"] = am
        base = out.get("reason", "") or "Routing"
        out["reason"] = (
            f"{base} | Conductor suggested {bad_p}/{bad_m} which doesn't exist, "
            f"using {ap}/{am} instead"
        )
        return out

    def _delegate_fallback(self, available_providers: list[dict], reason: str) -> dict:
        """Pick first paid provider with non-empty startup active models."""
        for meta in available_providers:
            name = meta.get("name")
            provider = self.providers.get(name) if name else None
            if not provider or not provider._active_models:
                continue
            eligible = self._active_models_for_routing(name, provider)
            if not eligible:
                continue
            best = self._pick_best_model_from_models(eligible)
            chosen = best or eligible[0]
            return {
                "route": "delegate",
                "provider": name,
                "model": chosen.name,
                "reason": reason,
                "estimated_cost": 0.0,
            }
        return {
            "route": "delegate",
            "provider": None,
            "model": None,
            "reason": reason,
            "estimated_cost": 0.0,
        }

    async def _delegate_cheapest(
        self,
        prompt: str,
        messages: list[dict] = None,
        task_type: str = "unknown",
    ) -> GenerateResult:
        """Use the lowest-cost paid model when no local executor exists."""
        cheapest: Optional[tuple[float, str, str]] = None
        for name, provider in self.providers.items():
            if name == "ollama" or not provider.is_available:
                continue
            for model in self._active_models_for_routing(name, provider):
                option = (model.cost_per_1m_output, name, model.name)
                if cheapest is None or option[0] < cheapest[0]:
                    cheapest = option

        if cheapest:
            _, provider_name, model_name = cheapest
            return await self._delegate(provider_name, model_name, prompt, messages, task_type)

        return await self._conductor_provider.generate(
            model_name=self._conductor_model,
            prompt=prompt,
            messages=messages,
            temperature=self._temp_for_type(task_type),
        )

    async def _delegate(
        self,
        provider_name: str,
        model_name: str,
        prompt: str,
        messages: list[dict] = None,
        task_type: str = "unknown",
    ) -> GenerateResult:
        """Send generation request to a specific provider/model."""
        provider = self.providers.get(provider_name)
        if not provider:
            return GenerateResult(
                content=f"ERROR: Provider '{provider_name}' not found",
                model=model_name or "unknown",
                provider=provider_name or "unknown",
                error="provider_not_found",
            )
        if not provider.is_available:
            return GenerateResult(
                content=f"ERROR: Provider '{provider_name}' has no API key",
                model=model_name or "unknown",
                provider=provider_name,
                error="no_api_key",
            )
        return await provider.generate(
            model_name=model_name,
            prompt=prompt,
            messages=messages,
            temperature=self._temp_for_type(task_type),
        )

    def _check_cold_start_override(
        self,
        task_type: str,
        user_signals: dict,
        available_providers: list[dict],
    ) -> Optional[dict]:
        """Delegate-first override during local cold start."""
        if not self._is_local_conductor:
            return None
        interactions = self.memory.get_interaction_count()
        if interactions >= 100:
            return None
        if task_type in user_signals.get("local_success_rate", {}):
            return None
        if self._cold_start_rate(task_type) >= 0.5:
            return None

        best_paid = self._pick_best_paid_provider(available_providers)
        if not best_paid:
            return None
        provider_name, model_name = best_paid
        return {
            "route": "delegate",
            "provider": provider_name,
            "model": model_name,
            "reason": (
                f"Cold start: delegating {task_type} until local model proves capable "
                f"({interactions}/100 interactions)"
            ),
        }

    def _pick_best_paid_provider(self, available_providers: list[dict]) -> Optional[tuple[str, str]]:
        """Prefer quality tier 3-4 paid models, then cheapest tier 5."""
        candidates: list[tuple[str, ModelInfo]] = []
        for provider_meta in available_providers:
            name = provider_meta.get("name")
            provider = self.providers.get(name)
            if not provider or not provider.is_available:
                continue
            for model in self._active_models_for_routing(name, provider):
                candidates.append((name, model))

        if not candidates:
            return None

        mid = [(n, m) for n, m in candidates if 3 <= m.quality_tier <= 4]
        tier5 = [(n, m) for n, m in candidates if m.quality_tier == 5]

        if mid:
            n, m = sorted(mid, key=lambda x: (-x[1].quality_tier, x[1].cost_per_1m_output))[0]
            return n, m.name
        if tier5:
            n, m = sorted(tier5, key=lambda x: x[1].cost_per_1m_output)[0]
            return n, m.name
        n, m = sorted(candidates, key=lambda x: x[1].cost_per_1m_output)[0]
        return n, m.name

    def _pick_fallback(self, failed_provider: str, failed_model: str) -> Optional[tuple[str, str]]:
        """One retry: next provider, avoiding first model when possible."""
        for name, provider in self.providers.items():
            if name in (failed_provider, "ollama") or not provider.is_available:
                continue
            eligible = self._active_models_for_routing(name, provider)
            if not eligible:
                continue
            names = [m.name for m in eligible]
            return name, (names[1] if len(names) > 1 else names[0])
        return "ollama", self._conductor_model

    def _get_signals(self) -> dict:
        """Return cached user signals or compute from memory."""
        if self._cached_signals:
            return self._cached_signals
        cached = self.memory.get_cached_signals()
        self._cached_signals = cached or self.memory.compute_user_signals()
        return self._cached_signals

    def _get_available_providers(self) -> list[dict]:
        """Return paid providers currently eligible for delegation."""
        available = []
        for name, provider in self.providers.items():
            if name == "ollama" or not provider.is_available:
                continue
            eligible = self._active_models_for_routing(name, provider)
            if not eligible:
                continue
            models = [m.name for m in eligible]
            avg_cost = sum(m.cost_per_1m_output for m in eligible) / len(eligible)
            available.append({"name": name, "models": models, "avg_cost": avg_cost})
        return available

    @staticmethod
    def _pick_best_model_from_models(models: list[ModelInfo]) -> Optional[ModelInfo]:
        """Prefer mid-tier (3-4), then cheapest tier 5, else cheapest overall."""
        if not models:
            return None
        mid = [m for m in models if 3 <= m.quality_tier <= 4]
        if mid:
            return sorted(mid, key=lambda m: (-m.quality_tier, m.cost_per_1m_output))[0]
        tier5 = [m for m in models if m.quality_tier == 5]
        if tier5:
            return sorted(tier5, key=lambda m: m.cost_per_1m_output)[0]
        return min(models, key=lambda m: m.cost_per_1m_output)

    def _coerce_delegate_to_active_models(self, routing: dict) -> dict:
        """Remap delegate targets to IDs present in provider._active_models (post-discovery)."""
        if routing.get("route") != "delegate":
            return routing
        provider_name = routing.get("provider")
        model_name = routing.get("model")
        if not provider_name:
            return routing
        provider = self.providers.get(provider_name)
        if not provider or not provider._active_models:
            return routing
        eligible = self._active_models_for_routing(provider_name, provider)
        if not eligible:
            return routing
        if model_name and any(m.name == model_name for m in eligible):
            return routing
        best = self._pick_best_model_from_models(eligible)
        chosen = best or eligible[0]
        out = dict(routing)
        out["model"] = chosen.name
        base = routing.get("reason", "") or "Routing"
        out["reason"] = f"{base} (model coerced to active ID: {chosen.name})"
        return out

    def _parse_json_response(self, text: str) -> Optional[dict]:
        """Parse first valid JSON object from raw/fenced/noisy model output."""
        if not text:
            return None

        raw = text.strip()
        candidates: list[str] = [raw]
        if raw.startswith("```"):
            lines = raw.splitlines()
            if len(lines) >= 2:
                body = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
                fenced = "\n".join(body).strip()
                if fenced:
                    candidates.append(fenced)

        for c in list(candidates):
            a, b = c.find("{"), c.rfind("}")
            if a != -1 and b != -1 and b > a:
                candidates.append(c[a : b + 1].strip())

        for c in candidates:
            if not c:
                continue
            try:
                parsed = json.loads(c)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

        for c in candidates:
            start = c.find("{")
            if start == -1:
                continue
            depth = 0
            for i in range(start, len(c)):
                if c[i] == "{":
                    depth += 1
                elif c[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed = json.loads(c[start : i + 1])
                            if isinstance(parsed, dict):
                                return parsed
                        except json.JSONDecodeError:
                            break
        return None

    @staticmethod
    def _cold_start_rate(task_type: str) -> float:
        """Delegate-first defaults before local capability is proven."""
        return {
            "factual": 0.20,
            "creative": 0.15,
            "code": 0.10,
            "analysis": 0.10,
            "summarize": 0.30,
            "translate": 0.20,
            "conversation": 0.85,
            "complex": 0.05,
            "unknown": 0.10,
        }.get(task_type, 0.10)

    @staticmethod
    def _temp_for_type(task_type: str) -> float:
        """Temperature defaults for final generation calls."""
        return {
            "factual": 0.2,
            "code": 0.3,
            "analysis": 0.4,
            "summarize": 0.3,
            "translate": 0.2,
            "creative": 0.8,
            "conversation": 0.7,
            "complex": 0.5,
            "unknown": 0.5,
        }.get(task_type, 0.5)
