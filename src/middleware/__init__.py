# Custom middleware for LangChain agents
from src.middleware.retry_middleware import ModelRetryMiddleware

__all__ = ["ModelRetryMiddleware", "GuardrailsMiddleware"]


def __getattr__(name: str):
    """Lazy import to avoid circular dependency with src.agent.config."""
    if name == "GuardrailsMiddleware":
        from src.middleware.guardrails_middleware import GuardrailsMiddleware

        return GuardrailsMiddleware
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
