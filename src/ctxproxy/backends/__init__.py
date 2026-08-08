from ..config import BackendConfig
from ..store.capture import DebugCapture
from .anthropic_native import AnthropicBackend
from .base import Backend
from .openai_compat import OpenAICompatBackend


def build_backend(config: BackendConfig, capture: DebugCapture | None = None) -> Backend:
    match config.kind:
        case "anthropic":
            return AnthropicBackend(config, capture)
        case "openai":
            return OpenAICompatBackend(config, capture)
    raise ValueError(f"unknown backend kind: {config.kind!r}")


class BackendRegistry:
    """Owns one long-lived backend (and its connection pool) per config entry."""

    def __init__(
        self, configs: list[BackendConfig], capture: DebugCapture | None = None
    ) -> None:
        self._backends = {c.name: build_backend(c, capture) for c in configs}

    def get(self, name: str) -> Backend:
        try:
            return self._backends[name]
        except KeyError:
            raise KeyError(
                f"backend {name!r} is not configured; known: {sorted(self._backends)}"
            ) from None

    async def aclose(self) -> None:
        for backend in self._backends.values():
            await backend.aclose()


__all__ = [
    "Backend",
    "AnthropicBackend",
    "OpenAICompatBackend",
    "BackendRegistry",
    "build_backend",
]
