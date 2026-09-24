"""Swappable chat-completions client for Hugging Face routed inference.

Routed/provider inference only - no dedicated endpoints are created. Credentials come from the
environment (`HF_TOKEN`); nothing is read from source. The base URL and both model ids are
configurable so the provider can be swapped without touching call sites.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Sequence

from bow.config import ModelConfig, load_dotenv
from bow.errors import ExtractionError

DEFAULT_BASE_URL = "https://router.huggingface.co/v1"


@dataclass(frozen=True, slots=True)
class ChatResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    model: str
    provider: str
    provider_cost: Decimal | None      # the router reports its own price; prefer it over a guess


class ChatClient:
    """Minimal OpenAI-compatible client over stdlib urllib (no extra dependency)."""

    def __init__(self, config: ModelConfig | None = None, base_url: str | None = None) -> None:
        load_dotenv()
        self.config = config or ModelConfig()
        self.base_url = (base_url or os.environ.get("HF_BASE_URL")
                         or DEFAULT_BASE_URL).rstrip("/")
        self._variant = 0

    # ------------------------------------------------------------------ config

    @property
    def text_model(self) -> str:
        return os.environ.get("TEXT_MODEL") or self.config.text_model

    @property
    def vision_model(self) -> str:
        return os.environ.get("VISION_MODEL") or self.config.vision_model

    def validate(self) -> None:
        """Fail clearly when credentials are absent, before any work is attempted."""
        if not self.config.api_token:
            raise ExtractionError(
                "HF_TOKEN is not set. Put it in the environment or the repo-root .env; "
                "it is never read from source.")

    # ------------------------------------------------------------------ call

    def complete_json(self, *, model: str, system: str, user_content: Any,
                      schema: dict[str, Any], schema_name: str,
                      max_tokens: int = 700,
                      thinking: bool = True) -> tuple[dict[str, Any], ChatResult, int]:
        """Return (parsed JSON, usage, retries). Raises ExtractionError if unusable."""
        self.validate()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        retries = 0
        last_error = ""
        for attempt in range(self.config.max_retries + 1):
            if attempt:
                # Bounded corrective request: restate the schema, quote the failure.
                messages = messages[:2] + [{
                    "role": "user",
                    "content": (f"Your previous reply was not valid against the schema "
                                f"({last_error}). Reply again with JSON only, no prose and no "
                                f"code fences, matching exactly:\n{json.dumps(schema)}")}]
                retries += 1
            result = self._post(model=model, messages=messages, schema=schema,
                                schema_name=schema_name, max_tokens=max_tokens,
                                thinking=thinking)
            try:
                parsed = _parse_json(result.text)
                _validate(parsed, schema)
                return parsed, result, retries
            except (ValueError, ExtractionError) as exc:
                last_error = str(exc)[:160]
        raise ExtractionError(f"schema validation failed after {retries} retries: {last_error}")

    #: Routed inference picks a provider per call, and providers disagree about these knobs:
    #: some reject `strict: false` outright, others run thinking in "required" mode and reject
    #: `enable_thinking: false`. Rather than pin a provider (which would make the solution
    #: brittle to availability), try the variants in order and remember the first that works.
    _VARIANTS = (
        {"strict": True, "thinking_off": False},
        {"strict": True, "thinking_off": True},
        {"strict": False, "thinking_off": True},
        {"strict": False, "thinking_off": False},
    )

    def _build(self, variant: dict, *, model: str, messages, schema, schema_name: str,
               max_tokens: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": self.config.temperature,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema,
                                "strict": variant["strict"]},
            },
        }
        if variant["thinking_off"]:
            # A reasoning model left unchecked spends the whole completion budget thinking and
            # returns empty content. Extraction needs no chain of thought.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload

    def _post(self, *, model: str, messages: Sequence[dict[str, Any]],
              schema: dict[str, Any], schema_name: str, max_tokens: int,
              thinking: bool = True) -> ChatResult:
        last: Exception | None = None
        order = list(range(len(self._VARIANTS)))
        if thinking:
            order.sort(key=lambda i: 0 if i == self._variant else 1)
        else:
            # Caller needs the answer, not the reasoning: a provider that *accepts* thinking
            # mode will happily spend the whole budget on it and return empty content, so
            # "first variant that is accepted" is not enough - ask for thinking off first.
            order.sort(key=lambda i: (not self._VARIANTS[i]["thinking_off"], i))
        for index in order:
            try:
                result = self._post_once(
                    self._build(self._VARIANTS[index], model=model, messages=messages,
                                schema=schema, schema_name=schema_name,
                                max_tokens=max_tokens))
                if thinking:
                    self._variant = index
                return result
            except ExtractionError as exc:
                if "HTTP 400" not in str(exc):
                    raise
                last = exc
        raise last or ExtractionError("no request variant accepted by the provider")

    def _post_once(self, payload: dict[str, Any]) -> ChatResult:
        body = json.dumps(payload).encode("utf-8")
        model = payload["model"]
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Authorization": f"Bearer {self.config.api_token}",
                     "Content-Type": "application/json"})
        started = time.time()
        delay = 2.0
        for attempt in range(self.config.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
                    data = json.load(response)
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:300]
                if exc.code in {408, 429, 500, 502, 503, 504} and attempt < self.config.max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise ExtractionError(f"HTTP {exc.code} from provider: {detail}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < self.config.max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise ExtractionError(f"transport failure: {exc}") from exc
        else:  # pragma: no cover - loop always breaks or raises
            raise ExtractionError("request exhausted retries")

        usage = data.get("usage") or {}
        choices = data.get("choices") or []
        if not choices:
            raise ExtractionError(f"provider returned no choices: {str(data)[:200]}")
        reported = usage.get("estimated_cost")
        return ChatResult(
            text=choices[0].get("message", {}).get("content") or "",
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            latency_ms=int((time.time() - started) * 1000),
            model=data.get("model") or model,
            provider=self.config.provider,
            provider_cost=Decimal(str(reported)) if reported is not None else None,
        )


# ---------------------------------------------------------------- validation


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        cleaned = cleaned[4:] if cleaned.lower().startswith("json") else cleaned
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in reply")
    parsed = json.loads(cleaned[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("reply is not a JSON object")
    return parsed


def _validate(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    """Small structural validator - enough for closed schemas, no jsonschema dependency."""
    for field in schema.get("required", []):
        if field not in payload:
            raise ExtractionError(f"missing required field {field!r}")
    props: dict[str, Any] = schema.get("properties", {})
    for key, value in payload.items():
        spec = props.get(key)
        if spec is None:
            raise ExtractionError(f"unexpected field {key!r}")
        _check(key, value, spec)


def _check(key: str, value: Any, spec: dict[str, Any]) -> None:
    types = spec.get("type")
    types = [types] if isinstance(types, str) else list(types or [])
    if value is None:
        if "null" not in types:
            raise ExtractionError(f"{key!r} must not be null")
        return
    ok = {
        "string": isinstance(value, str),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "array": isinstance(value, list),
        "object": isinstance(value, dict),
    }
    if types and not any(ok.get(t, False) for t in types):
        raise ExtractionError(f"{key!r} has wrong type: {type(value).__name__}, want {types}")
    if "enum" in spec and value not in spec["enum"]:
        raise ExtractionError(f"{key!r}={value!r} is not one of {spec['enum']}")
    if isinstance(value, str) and "maxLength" in spec and len(value) > spec["maxLength"]:
        raise ExtractionError(f"{key!r} exceeds maxLength {spec['maxLength']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in spec and value < spec["minimum"]:
            raise ExtractionError(f"{key!r}={value} below minimum {spec['minimum']}")
        if "maximum" in spec and value > spec["maximum"]:
            raise ExtractionError(f"{key!r}={value} above maximum {spec['maximum']}")
    if isinstance(value, list):
        item_spec = spec.get("items")
        if item_spec:
            for index, item in enumerate(value):
                if item_spec.get("type") == "object":
                    _validate(item, item_spec)
                else:
                    _check(f"{key}[{index}]", item, item_spec)
