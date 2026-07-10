"""Pluggable LLM provider interface.

Gemini 2.5 Flash (Google AI Studio, free tier) is the default and only
required provider. Grok (xAI, OpenAI-SDK-compatible) is an optional
secondary provider — it is never required and stays inert unless
GROK_API_KEY is set and LLM_PROVIDER=grok.

Selected via env var LLM_PROVIDER=gemini|grok (default: gemini).
"""
from __future__ import annotations
import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

log = logging.getLogger(__name__)


class LLMNotConfiguredError(RuntimeError):
    """Raised when a provider is selected but its API key is missing."""


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        ...


class GeminiProvider(LLMProvider):
    MODEL = "gemini-2.5-flash"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise LLMNotConfiguredError(
                "GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey "
                "and set it in your environment or .env file."
            )
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        from google.genai import types
        client = self._get_client()
        config = types.GenerateContentConfig(system_instruction=system) if system else None
        response = client.models.generate_content(
            model=self.MODEL, contents=prompt, config=config,
        )
        return response.text


class GrokProvider(LLMProvider):
    """xAI Grok — OpenAI-SDK-compatible. Optional; requires a separate
    key from console.x.ai (not an OpenAI key, despite the SDK shape)."""
    MODEL = "grok-2-latest"
    BASE_URL = "https://api.x.ai/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GROK_API_KEY")
        if not self.api_key:
            raise LLMNotConfiguredError(
                "GROK_API_KEY is not set. This provider is optional — "
                "sign up at https://console.x.ai if you want to use it."
            )
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.BASE_URL)
        return self._client

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        client = self._get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(model=self.MODEL, messages=messages)
        return response.choices[0].message.content


_PROVIDERS = {"gemini": GeminiProvider, "grok": GrokProvider}

_provider_instance: Optional[LLMProvider] = None


def get_provider() -> LLMProvider:
    global _provider_instance
    if _provider_instance is None:
        name = os.environ.get("LLM_PROVIDER", "gemini").lower()
        cls = _PROVIDERS.get(name)
        if cls is None:
            raise ValueError(f"Unknown LLM_PROVIDER '{name}'. Choose from: {list(_PROVIDERS)}")
        _provider_instance = cls()
    return _provider_instance


def reset_provider() -> None:
    """Test helper: clears the cached provider singleton."""
    global _provider_instance
    _provider_instance = None


def generate(prompt: str, system: Optional[str] = None) -> str:
    return get_provider().generate(prompt, system=system)
