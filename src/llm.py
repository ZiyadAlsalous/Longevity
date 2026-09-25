from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel

from src.config import settings


class LLMConfigError(RuntimeError):
    pass


class StructuredLLM(Protocol):
    def generate[T: BaseModel](self, *, system: str, user: str, schema: type[T]) -> T: ...


class OllamaLLM:
    """Local model served by Ollama."""

    def __init__(self, model: str) -> None:
        import ollama

        self._model = model
        self._client = ollama.Client(
            host=settings.ollama_host, timeout=settings.llm_timeout_seconds
        )

    def generate[T: BaseModel](self, *, system: str, user: str, schema: type[T]) -> T:
        """Ask the model for output that matches the schema."""
        import ollama

        try:
            response = self._client.chat(
                model=self._model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                format=_require_every_field(schema.model_json_schema()),
                think=False,
                options={
                    "temperature": settings.llm_temperature,
                    # Stops a looping model instead of letting it run to the timeout.
                    "num_predict": settings.llm_max_tokens,
                },
            )
        except ConnectionError as exc:
            raise LLMConfigError(
                f"Ollama is not running at {settings.ollama_host}. Start the Ollama app."
            ) from exc
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                raise LLMConfigError(
                    f"Model {self._model} is not installed. Run: ollama pull {self._model}"
                ) from exc
            raise
        if response.done_reason == "length":
            raise LLMConfigError(
                f"The model stopped at the {settings.llm_max_tokens}-token limit without finishing. "
                "Run it again, or raise LLM_MAX_TOKENS in .env."
            )
        return schema.model_validate_json(response.message.content or "")


def _require_every_field(node: Any) -> Any:
    """Mark every schema field required so the model cannot skip one."""
    if isinstance(node, dict):
        if "properties" in node:
            node["required"] = list(node["properties"])
        for value in node.values():
            _require_every_field(value)
    elif isinstance(node, list):
        for value in node:
            _require_every_field(value)
    return node


def build_llm() -> StructuredLLM:
    return OllamaLLM(settings.llm_model)
