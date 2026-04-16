"""Provider base types and interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelInfo:
    """Metadata about a single model."""
    name: str
    provider: str
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0
    max_output_tokens: int = 4096
    max_context_window: int = 128000
    speed_tier: str = "medium"
    quality_tier: int = 3


@dataclass
class GenerateResult:
    """Standard response from any provider."""
    content: str
    model: str
    provider: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_estimate: float = 0.0
    error: Optional[str] = None


class BaseProvider(ABC):
    """Interface for provider adapters."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._available = api_key is not None
        self._active_models: list[ModelInfo] = []

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def models(self) -> list[ModelInfo]:
        return self._active_models

    def set_active_models(self, models: list[ModelInfo]) -> None:
        self._active_models = models

    async def discover_models(self) -> list[str]:
        """Discover provider model IDs via API. Default: none discovered."""
        return []

    @abstractmethod
    async def generate(self, model_name: str, prompt: str,
                       messages: list[dict] = None,
                       system_prompt: str = None,
                       max_tokens: int = None,
                       temperature: float = 0.7) -> GenerateResult:
        ...

    def get_model(self, model_name: str) -> Optional[ModelInfo]:
        for m in self._active_models:
            if m.name == model_name:
                return m
        return None

    def estimate_cost(self, model_name: str, tokens_in: int,
                      tokens_out: int) -> float:
        model = self.get_model(model_name)
        if not model:
            return 0.0
        input_cost = (tokens_in / 1_000_000) * model.cost_per_1m_input
        output_cost = (tokens_out / 1_000_000) * model.cost_per_1m_output
        return round(input_cost + output_cost, 8)

    @property
    def is_available(self) -> bool:
        return self._available
