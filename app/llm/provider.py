"""Provider-agnostic LLM access with schema-constrained output.

Two reasons this abstraction exists rather than calling one SDK inline:

1. Graders must be able to run this without a paid account. Gemini's free tier
   is the default; Groq and Anthropic are drop-in alternatives via one env var.
2. It keeps the contract narrow — `structured()` takes a Pydantic model and
   returns instances of it. Everything downstream depends on that contract, not
   on any vendor's response shape.

Model availability moves fast (we hit a retired model id and a 503 on a busy
one while building this), so the Gemini client carries an explicit fallback
chain and retries on transient errors.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Protocol, TypeVar

from pydantic import BaseModel

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Tried in order when the configured model is unavailable.
GEMINI_FALLBACKS = ("gemini-3.6-flash", "gemini-2.5-flash", "gemini-flash-latest")

TRANSIENT_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL")


class LLMError(RuntimeError):
    pass


class LLMProvider(Protocol):
    name: str

    def structured(self, system: str, prompt: str, schema: type[T]) -> T:
        """Return one instance of `schema`, or raise LLMError."""
        ...


def _is_transient(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}"
    return any(m in text for m in TRANSIENT_MARKERS)


class GeminiProvider:
    """Google Gemini via the free AI Studio tier."""

    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        from google import genai  # imported lazily so other providers work without it

        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise LLMError("GEMINI_API_KEY is not set (copy .env.example to .env)")
        self._genai = genai
        self.client = genai.Client(api_key=key)
        configured = model or os.getenv("GEMINI_MODEL") or GEMINI_FALLBACKS[0]
        # Configured model first, then the rest of the chain, de-duplicated.
        self.models = [configured] + [m for m in GEMINI_FALLBACKS if m != configured]

    def structured(self, system: str, prompt: str, schema: type[T]) -> T:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.0,  # extraction should be reproducible, not creative
        )

        last: Exception | None = None
        for model in self.models:
            for attempt in range(3):
                try:
                    resp = self.client.models.generate_content(
                        model=model, contents=prompt, config=config
                    )
                    parsed = getattr(resp, "parsed", None)
                    if isinstance(parsed, schema):
                        return parsed
                    # Fall back to parsing the raw JSON text ourselves.
                    if resp.text:
                        return schema.model_validate(json.loads(resp.text))
                    raise LLMError("empty response")
                except Exception as exc:  # noqa: BLE001 - classified below
                    last = exc
                    if _is_transient(exc) and attempt < 2:
                        time.sleep(2 * (attempt + 1))
                        continue
                    break  # try the next model
            log.warning("gemini model %s unusable: %s", model, last)
        raise LLMError(f"all Gemini models failed; last error: {last}")


class GroqProvider:
    """Groq free tier. Uses JSON mode; schema is enforced by our own validation."""

    name = "groq"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        from groq import Groq

        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise LLMError("GROQ_API_KEY is not set")
        self.client = Groq(api_key=key)
        self.model = model or os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile"

    def structured(self, system: str, prompt: str, schema: type[T]) -> T:
        instruction = (
            f"{system}\n\nRespond with JSON matching this schema:\n"
            f"{json.dumps(schema.model_json_schema())}"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt},
            ],
        )
        return schema.model_validate(json.loads(resp.choices[0].message.content))


class AnthropicProvider:
    """Claude — highest quality, but paid, so never the default here."""

    name = "anthropic"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model or os.getenv("ANTHROPIC_MODEL") or "claude-opus-5"

    def structured(self, system: str, prompt: str, schema: type[T]) -> T:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": schema.model_json_schema(),
                }
            },
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return schema.model_validate(json.loads(text))


_PROVIDERS = {
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "anthropic": AnthropicProvider,
}

_cached: LLMProvider | None = None


def get_provider(name: str | None = None, *, refresh: bool = False) -> LLMProvider:
    """Build the configured provider. Cached, since clients are reusable."""
    global _cached
    if _cached is not None and not refresh and name is None:
        return _cached

    from dotenv import load_dotenv

    load_dotenv()
    key = (name or os.getenv("LLM_PROVIDER") or "gemini").strip().lower()
    if key not in _PROVIDERS:
        raise LLMError(f"unknown provider {key!r}; choose from {sorted(_PROVIDERS)}")
    provider = _PROVIDERS[key]()
    if name is None:
        _cached = provider
    return provider
