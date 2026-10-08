"""Pluggable LLM provider interface.

Gemini 2.5 Flash (Google AI Studio, free tier) is the default provider.
Optional providers stay inert unless selected:
  - vllm: a self-hosted vLLM server via its OpenAI-compatible API
          (VLLM_BASE_URL, VLLM_MODEL, optional VLLM_API_KEY).
  - grok: xAI Grok (OpenAI-SDK-compatible, GROK_API_KEY).
  - none: no LLM; callers fall back to computed facts/templates.

Selected via env var LLM_PROVIDER=gemini|vllm|grok|none (default: gemini).
Shared knobs: LLM_TIMEOUT_S (default 30), LLM_MAX_RETRIES (default 1),
LLM_MAX_TOKENS (default 512), LLM_TEMPERATURE (OpenAI-compatible providers;
unset = server default, 0 for reproducible evaluation).

Every provider failure surfaces as an `LLMError` subclass, so callers can
fall back with one `except LLMError` instead of catching SDK-specific
exceptions.
"""
from __future__ import annotations
import logging
import os
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Base class for any LLM failure the caller should degrade around."""


class LLMNotConfiguredError(LLMError):
    """Raised when a provider is selected but its API key/endpoint is missing."""


class LLMUnavailableError(LLMError):
    """Raised on timeouts, connection failures, or upstream error responses."""


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class LLMProvider(ABC):
    name = "base"
    model = ""

    @abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        ...

    def stream(self, prompt: str, system: Optional[str] = None) -> Iterator[str]:
        """Yields text chunks. Default: one chunk from generate(), so every
        provider supports the streaming endpoint even without native streaming."""
        yield self.generate(prompt, system=system)


class GeminiProvider(LLMProvider):
    name = "gemini"
    model = "gemini-2.5-flash"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise LLMNotConfiguredError(
                "GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey "
                "and set it in your environment or .env file."
            )
        self.timeout_s = _env_float("LLM_TIMEOUT_S", 30.0)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            from google.genai import types
            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=int(self.timeout_s * 1000)),
            )
        return self._client

    def _config(self, system: Optional[str]):
        from google.genai import types
        return types.GenerateContentConfig(system_instruction=system) if system else None

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        try:
            response = self._get_client().models.generate_content(
                model=self.model, contents=prompt, config=self._config(system),
            )
        except Exception as e:
            raise LLMUnavailableError(f"Gemini request failed: {e}") from e
        return response.text or ""

    def stream(self, prompt: str, system: Optional[str] = None) -> Iterator[str]:
        try:
            for chunk in self._get_client().models.generate_content_stream(
                model=self.model, contents=prompt, config=self._config(system),
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise LLMUnavailableError(f"Gemini stream failed: {e}") from e


class OpenAICompatibleProvider(LLMProvider):
    """Shared client for any OpenAI-compatible chat-completions endpoint
    (vLLM, xAI Grok). Subclasses only set the endpoint, key, and model."""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout_s = _env_float("LLM_TIMEOUT_S", 30.0)
        self.max_retries = _env_int("LLM_MAX_RETRIES", 1)
        self.max_tokens = _env_int("LLM_MAX_TOKENS", 512)
        t = os.environ.get("LLM_TEMPERATURE")
        self.sampling = {"temperature": float(t)} if t not in (None, "") else {}
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                                  timeout=self.timeout_s, max_retries=self.max_retries)
        return self._client

    def _messages(self, prompt: str, system: Optional[str]) -> List[dict]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        import openai
        try:
            response = self._get_client().chat.completions.create(
                model=self.model, messages=self._messages(prompt, system),
                max_tokens=self.max_tokens, **self.sampling,
            )
        except openai.OpenAIError as e:
            raise LLMUnavailableError(f"{self.name} request failed: {e}") from e
        return response.choices[0].message.content or ""

    def stream(self, prompt: str, system: Optional[str] = None) -> Iterator[str]:
        import openai
        try:
            chunks = self._get_client().chat.completions.create(
                model=self.model, messages=self._messages(prompt, system),
                max_tokens=self.max_tokens, stream=True, **self.sampling,
            )
            for chunk in chunks:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except openai.OpenAIError as e:
            raise LLMUnavailableError(f"{self.name} stream failed: {e}") from e


class VLLMProvider(OpenAICompatibleProvider):
    """Self-hosted vLLM (`vllm serve <model>`), reached over its
    OpenAI-compatible API. vLLM ignores the API key unless started with
    --api-key, so the default is a placeholder."""
    name = "vllm"

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None,
                 api_key: Optional[str] = None):
        base_url = base_url or os.environ.get("VLLM_BASE_URL")
        model = model or os.environ.get("VLLM_MODEL")
        if not base_url or not model:
            raise LLMNotConfiguredError(
                "LLM_PROVIDER=vllm needs VLLM_BASE_URL (e.g. http://localhost:8001/v1) "
                "and VLLM_MODEL (the model name passed to `vllm serve`)."
            )
        super().__init__(api_key or os.environ.get("VLLM_API_KEY", "EMPTY"), base_url, model)


class GrokProvider(OpenAICompatibleProvider):
    """xAI Grok — OpenAI-SDK-compatible. Optional; requires a separate
    key from console.x.ai (not an OpenAI key, despite the SDK shape)."""
    name = "grok"
    MODEL = "grok-2-latest"
    BASE_URL = "https://api.x.ai/v1"

    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.environ.get("GROK_API_KEY")
        if not api_key:
            raise LLMNotConfiguredError(
                "GROK_API_KEY is not set. This provider is optional — "
                "sign up at https://console.x.ai if you want to use it."
            )
        super().__init__(api_key, self.BASE_URL, self.MODEL)


class NoLLMProvider(LLMProvider):
    """Explicit opt-out: callers fall back to deterministic answers."""
    name = "none"

    def __init__(self):
        raise LLMNotConfiguredError("LLM_PROVIDER=none — LLM narration disabled.")

    def generate(self, prompt: str, system: Optional[str] = None) -> str:  # pragma: no cover
        raise LLMNotConfiguredError("LLM disabled")


_PROVIDERS = {"gemini": GeminiProvider, "vllm": VLLMProvider, "grok": GrokProvider,
              "none": NoLLMProvider}

_provider_instance: Optional[LLMProvider] = None


def get_provider() -> LLMProvider:
    global _provider_instance
    if _provider_instance is None:
        name = os.environ.get("LLM_PROVIDER", "gemini").lower()
        cls = _PROVIDERS.get(name)
        if cls is None:
            raise LLMNotConfiguredError(
                f"Unknown LLM_PROVIDER '{name}'. Choose from: {list(_PROVIDERS)}")
        _provider_instance = cls()
    return _provider_instance


def provider_info() -> dict:
    """Name/model of the configured provider, or why it is unavailable —
    for /ready and benchmark metadata. Never raises."""
    try:
        p = get_provider()
        return {"provider": p.name, "model": p.model, "configured": True}
    except LLMError as e:
        return {"provider": os.environ.get("LLM_PROVIDER", "gemini").lower(),
                "model": None, "configured": False, "detail": str(e)}


def reset_provider() -> None:
    """Test helper: clears the cached provider singleton."""
    global _provider_instance
    _provider_instance = None


def generate(prompt: str, system: Optional[str] = None) -> str:
    return get_provider().generate(prompt, system=system)


def stream(prompt: str, system: Optional[str] = None) -> Iterator[str]:
    return get_provider().stream(prompt, system=system)
