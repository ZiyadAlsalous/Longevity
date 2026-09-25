from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel

from src.config import settings


class LLMConfigError(RuntimeError):
    pass


class StructuredLLM(Protocol):
    def generate[T: BaseModel](self, *, system: str, user: str, schema: type[T]) -> T: ...


class OllamaLLM:
    """A local model served by Ollama. Nothing leaves the machine.

    Ollama constrains decoding to the schema, so the reply is always well-formed JSON;
    Pydantic still validates it, because a well-formed reply can break a field constraint.
    """

    def __init__(self, model: str) -> None:
        import ollama

        self._model = model
        self._client = ollama.Client(
            host=settings.ollama_host, timeout=settings.llm_timeout_seconds
        )

    def generate[T: BaseModel](self, *, system: str, user: str, schema: type[T]) -> T:
        import ollama

        try:
            response = self._client.chat(
                model=self._model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                format=_require_every_field(schema.model_json_schema()),
                think=False,
                options={"temperature": settings.llm_temperature},
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
        return schema.model_validate_json(response.message.content or "")


def _require_every_field(node: Any) -> Any:
    """Mark every property required, so constrained decoding cannot skip a field.

    A field with a default is optional in Pydantic's schema, and small local models omit
    optional fields: they returned a lab panel with no analytes at all. Nullable fields
    are still allowed to be null.
    """
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
