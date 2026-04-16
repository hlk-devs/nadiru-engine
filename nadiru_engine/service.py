"""
Service layer - FastAPI with three endpoints.
POST /connect, POST /generate, GET /query.
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .conductor import Conductor
from .memory import MemoryStore
from .model_catalog import ModelCatalog
from .providers.ollama_provider import OllamaProvider
from .providers.registry import load_providers


class ConnectRequest(BaseModel):
    name: str
    description: str = ""
    default_priority: str = "balanced"


class ConnectResponse(BaseModel):
    nadi_id: str
    connected_at: str


class GenerateRequest(BaseModel):
    nadi_id: str
    prompt: str
    messages: Optional[list[dict]] = None
    priority: Optional[str] = None
    max_cost: Optional[float] = None
    prefer_provider: Optional[str] = None

class GenerateResponse(BaseModel):
    content: str
    model: str
    provider: str
    tokens_in: int
    tokens_out: int
    cost_estimate: float
    latency_ms: int
    request_id: str
    routing_reason: str


class QueryResponse(BaseModel):
    interactions: list[dict]
    total: int
    limit: int
    offset: int


conductor: Optional[Conductor] = None
memory: Optional[MemoryStore] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global conductor, memory

    db_path = os.getenv("NADIRU_DB_PATH", "nadiru.db")
    memory = MemoryStore(db_path)

    conductor_provider_name = os.getenv("CONDUCTOR_PROVIDER", "ollama").strip().lower()
    conductor_model = os.getenv("CONDUCTOR_MODEL", "qwen2.5:14b")
    local_url = os.getenv("LOCAL_MODEL_URL") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    ollama = OllamaProvider(base_url=local_url, conductor_model=conductor_model)
    providers = load_providers(ollama)

    catalog = ModelCatalog()
    print("  Discovering provider models...")
    for provider_name, provider in providers.items():
        discovered_ids = await provider.discover_models()

        if discovered_ids:
            merged_models = catalog.merge_discovered(provider_name, discovered_ids)
            provider.set_active_models(merged_models)
            active_names = [m.name for m in provider.models]
            print(f"  {provider_name}: {len(active_names)} models available ({', '.join(active_names)})")
        else:
            fallback_models = catalog.models_for_provider(provider_name)
            if fallback_models:
                provider.set_active_models(fallback_models)
                fallback_names = [m.name for m in fallback_models]
                print(
                    f"  WARNING: {provider_name} discovery returned no models; "
                    f"using registry fallback ({', '.join(fallback_names)})"
                )
            else:
                print(
                    f"  WARNING: {provider_name} discovery returned no models and no registry "
                    "fallback is available"
                )

    if conductor_provider_name not in providers:
        raise RuntimeError(
            f"CONDUCTOR_PROVIDER='{conductor_provider_name}' is not configured/available"
        )

    conductor_provider = providers[conductor_provider_name]
    is_local_conductor = conductor_provider_name == "ollama"

    if is_local_conductor:
        available_local_models = [m.name for m in ollama.models]
        model_found = any(
            conductor_model == m or m.startswith(conductor_model.split(":")[0])
            for m in available_local_models
        )
        if not model_found:
            print(f"\n  WARNING: Conductor model '{conductor_model}' not found.")
            print(f"  Available local models: {', '.join(available_local_models) if available_local_models else 'none'}")
            print(f"  If using Ollama, run: ollama pull {conductor_model}")
            print("  Continuing with fallback model list.\n")

    conductor = Conductor(
        memory=memory,
        conductor_provider=conductor_provider,
        conductor_model=conductor_model,
        providers=providers,
        is_local_conductor=is_local_conductor,
    )

    provider_names = [n for n in providers.keys() if n != "ollama"]
    conductor_mode = "local" if is_local_conductor else "cloud"
    print("Nadiru engine started")
    print(f"  Conductor: {conductor_provider_name}/{conductor_model} ({conductor_mode})")
    print(f"  Paid providers: {', '.join(provider_names) if provider_names else 'none (local only)'}")
    print(f"  Database: {db_path}")

    yield

    memory.close()
    print("Nadiru engine stopped")


app = FastAPI(
    title="Nadiru Engine",
    description="Sovereign AI orchestration. Conductor routes between local and paid models.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/connect", response_model=ConnectResponse)
async def connect(req: ConnectRequest):
    result = memory.register_nadi(
        name=req.name,
        description=req.description,
        default_priority=req.default_priority,
    )
    return ConnectResponse(**result)


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    nadi = memory.get_nadi(req.nadi_id)
    if not nadi:
        raise HTTPException(status_code=404, detail=f"Nadi '{req.nadi_id}' not found")

    priority = req.priority or nadi["default_priority"]

    result = await conductor.handle_request(
        nadi_id=req.nadi_id,
        prompt=req.prompt,
        messages=req.messages,
        priority=priority,
        max_cost=req.max_cost,
        prefer_provider=req.prefer_provider,
    )

    return GenerateResponse(**result)


@app.get("/query", response_model=QueryResponse)
async def query(
    nadi_id: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    min_cost: Optional[float] = None,
    max_cost: Optional[float] = None,
    limit: int = 50,
    offset: int = 0,
):
    result = memory.query_interactions(
        nadi_id=nadi_id,
        since=since,
        until=until,
        model=model,
        provider=provider,
        min_cost=min_cost,
        max_cost=max_cost,
        limit=min(limit, 100),
        offset=offset,
    )
    return QueryResponse(**result)


@app.get("/health")
async def health():
    interaction_count = memory.get_interaction_count()
    nadi_count = memory._conn.execute("SELECT COUNT(*) FROM nadis").fetchone()[0]
    provider_names = [n for n in conductor.providers.keys() if n != "ollama"]
    return {
        "status": "ok",
        "interactions": interaction_count,
        "nadis": nadi_count,
        "conductor_model": os.getenv("CONDUCTOR_MODEL", "qwen2.5:14b"),
        "conductor_provider": os.getenv("CONDUCTOR_PROVIDER", "ollama"),
        "providers": provider_names,
    }
