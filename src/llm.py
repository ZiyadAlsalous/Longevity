from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from src.config import settings


class LLMConfigError(RuntimeError):
    pass


class StructuredLLM(Protocol):
    def generate[T: BaseModel](self, *, system: str, user: str, schema: type[T]) -> T: ...


class LangChainLLM:
    def __init__(self, provider: str, model: str) -> None:
        if provider == "anthropic":
            if not settings.anthropic_api_key:
                raise LLMConfigError("Set ANTHROPIC_API_KEY in .env to run the pipeline.")
            from langchain_anthropic import ChatAnthropic

            self._chat = ChatAnthropic(
                model=model,
                timeout=settings.llm_timeout_seconds,
                stop=None,
                api_key=settings.anthropic_api_key,
            )
        elif provider == "openai":
            if not settings.openai_api_key:
                raise LLMConfigError("Set OPENAI_API_KEY in .env to run the pipeline.")
            from langchain_openai import ChatOpenAI

            self._chat = ChatOpenAI(
                model=model,
                temperature=settings.llm_temperature,
                timeout=settings.llm_timeout_seconds,
                api_key=settings.openai_api_key,
            )
        else:
            raise LLMConfigError(f"Unsupported provider: {provider}")

    def generate[T: BaseModel](self, *, system: str, user: str, schema: type[T]) -> T:
        result = self._chat.with_structured_output(schema).invoke(
            [("system", system), ("human", user)]
        )
        return schema.model_validate(result)


def build_llm() -> StructuredLLM:
    return LangChainLLM(settings.llm_provider, settings.llm_model)
