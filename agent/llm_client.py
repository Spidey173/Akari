"""
Thin, swappable LLM client interface.

Defines an abstract `LLMClient` to decouple the state pipeline from specific
model providers. Graph nodes depend solely on this interface, allowing
seamless switching between hosted providers (Google Gemini, Groq) and fully
local offline inference (Ollama).
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str | None = None, max_tokens: int = 2048) -> str:
        """Return the model's text response for a single-turn prompt."""
        raise NotImplementedError


class GeminiClient(LLMClient):
    """Uses Google AI Studio's free tier via the current `google-genai` SDK
    (the older `google-generativeai` package is deprecated as of 2025)."""

    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None):
        try:
            from google import genai
        except ImportError as e:
            raise ImportError("google-genai not installed. Run: pip install google-genai") from e

        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Get a free key at https://aistudio.google.com/apikey "
                "and export GEMINI_API_KEY=... (or put it in a .env file)."
            )
        self._client = genai.Client(api_key=api_key)
        self.model_name = model

    def generate(self, prompt: str, system: str | None = None, max_tokens: int = 2048) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=0.2,
        )
        # Simple retry with backoff for free-tier rate limits (429s).
        last_err = None
        for attempt in range(3):
            try:
                resp = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                return (resp.text or "").strip()
            except Exception as e:  # noqa: BLE001 - free-tier SDK raises assorted errors
                last_err = e
                wait = 2**attempt * 2
                print(f"  [llm] Gemini call failed ({e}); retrying in {wait}s...")
                time.sleep(wait)
        raise RuntimeError(f"Gemini generation failed after retries: {last_err}")


class GroqClient(LLMClient):
    """Alternative free-tier backend (fast Llama/Mixtral inference)."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        try:
            from groq import Groq
        except ImportError as e:
            raise ImportError("groq not installed. Run: pip install groq") from e

        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys"
            )
        self._client = Groq(api_key=api_key)
        self.model_name = model or os.environ.get("GROQ_MODEL") or "openai/gpt-oss-20b"

    def generate(self, prompt: str, system: str | None = None, max_tokens: int = 2048) -> str:
        messages: list[Any] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Cap max_tokens if model has a strict output token per minute ceiling (e.g. qwen on Groq on-demand)
        tokens = min(max_tokens, 950) if "qwen" in self.model_name.lower() else max_tokens

        last_err = None
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=tokens,
                    temperature=0.2,
                )
                content = resp.choices[0].message.content
                return (content or "").strip()
            except Exception as e:
                last_err = e
                err_msg = str(e).lower()
                if "reduce max_tokens" in err_msg or "limit 1000" in err_msg:
                    tokens = min(tokens, 800)
                    time.sleep(1)
                    continue
                wait = 2**attempt * 2
                print(f"  [llm] Groq call failed ({e}); retrying in {wait}s...")
                time.sleep(wait)
        raise RuntimeError(f"Groq generation failed after retries: {last_err}")


class OllamaClient(LLMClient):
    """Fully local backend — no API key, no rate limits, requires `ollama serve` running."""

    def __init__(self, model: str = "llama3.1", host: str = "http://localhost:11434"):
        self.model_name = model
        self.host = host

    def generate(self, prompt: str, system: str | None = None, max_tokens: int = 2048) -> str:
        import requests

        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        resp = requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model_name, "prompt": full_prompt, "stream": False},
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()


def build_llm_client(provider: str | None = None) -> LLMClient:
    if not provider:
        if os.environ.get("GEMINI_API_KEY"):
            provider = "gemini"
        elif os.environ.get("GROQ_API_KEY"):
            provider = "groq"
        else:
            provider = "gemini"

    provider = provider.lower()
    if provider == "gemini":
        return GeminiClient()
    if provider == "groq":
        return GroqClient()
    if provider == "ollama":
        return OllamaClient()
    raise ValueError(f"Unknown provider: {provider}")
