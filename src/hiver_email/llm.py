"""Thin, single-provider LLM client.

We use the OpenAI SDK and nothing else. Because the SDK takes a base_url, any
OpenAI-compatible endpoint (Gemini's compatibility layer, Together, vLLM,
Azure-style gateways) works by setting one environment variable - so we get
provider flexibility without writing a second API integration.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=False)

DEFAULT_GEN_MODEL = "gpt-4o-mini"
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"


class LLMError(RuntimeError):
    """Raised when the model cannot be reached or returns nothing usable."""


@dataclass
class LLMConfig:
    api_key: str
    base_url: str | None
    gen_model: str
    judge_model: str


def load_config() -> LLMConfig:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise LLMError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.\n"
            "Any OpenAI-compatible endpoint works via OPENAI_BASE_URL."
        )
    return LLMConfig(
        api_key=key,
        base_url=os.environ.get("OPENAI_BASE_URL", "").strip() or None,
        gen_model=os.environ.get("GEN_MODEL", DEFAULT_GEN_MODEL).strip(),
        judge_model=os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL).strip(),
    )


class LLMClient:
    """Chat-completions wrapper with bounded retries."""

    def __init__(
        self,
        config: LLMConfig | None = None,
        max_retries: int = 3,
        timeout: float | None = None,
    ) -> None:
        from openai import OpenAI  # imported lazily so tests can run without network

        self.config = config or load_config()
        self.max_retries = max_retries
        # A bounded per-request timeout matters more than it looks: the SDK
        # default is 10 minutes, so one stalled endpoint can hold a worker
        # thread long enough to stall a whole evaluation run.
        self.timeout = timeout if timeout is not None else float(
            os.environ.get("LLM_TIMEOUT_SECONDS", "90")
        )
        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.timeout,
            max_retries=0,  # retries are handled here, with our own backoff
        )

    @property
    def gen_model(self) -> str:
        return self.config.gen_model

    @property
    def judge_model(self) -> str:
        return self.config.judge_model

    def complete(
        self,
        system: str,
        user: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 900,
        json_mode: bool = False,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                text = (resp.choices[0].message.content or "").strip()
                if text:
                    return text
                # Reasoning models spend the budget on hidden thinking tokens
                # and return an empty message; widen the ceiling and retry.
                last = LLMError("model returned empty content")
                kwargs["max_tokens"] = min(int(kwargs["max_tokens"] * 2), 4096)
            except Exception as exc:  # noqa: BLE001 - surfaced after retries
                last = exc
                if json_mode and "response_format" in kwargs and _rejects_json_mode(exc):
                    kwargs.pop("response_format")  # endpoint lacks JSON mode
                    continue
            time.sleep(1.5 * (attempt + 1))
        raise LLMError(f"LLM call failed after {self.max_retries} attempts: {last}")


def _rejects_json_mode(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "response_format" in msg or "json_object" in msg


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def salvage_truncated_json(text: str) -> dict[str, Any] | None:
    """Recover a verdict whose trailing string field was cut off mid-write.

    A reasoning model that hits its token ceiling part-way through the final
    `reason` string leaves every numeric field intact and only the prose
    unfinished. Closing the open string and the open braces recovers real
    measured values; it invents nothing, because anything still missing after
    the repair fails schema validation and is rejected upstream.

    Returns None when the text is not a recoverable truncation.
    """
    start = text.find("{")
    if start == -1:
        return None
    body = text[start:].rstrip()
    # An odd number of unescaped quotes means a string is still open.
    if len(re.findall(r'(?<!\\)"', body)) % 2:
        body += '"'
    # Drop a trailing comma or a key whose value never arrived.
    body = re.sub(r',\s*"[^"]*"\s*:?\s*$', "", body).rstrip().rstrip(",")
    depth = body.count("{") - body.count("}")
    if depth <= 0:
        return None  # not a truncation - a different problem
    body += "}" * depth
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def parse_json_object(text: str) -> dict[str, Any]:
    """Decode a JSON object from a model response.

    Tolerates markdown fences and leading prose, but raises on anything that
    is not a JSON object - the caller records that as a failed judgement
    rather than silently substituting a default score.
    """
    if not text or not text.strip():
        raise ValueError("empty judge response")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(cleaned)
        if match:
            payload = json.loads(match.group(0))  # may raise - intentional
        else:
            # Last resort: a verdict cut off mid-string by the token ceiling.
            # Repair recovers only fields the model actually emitted; if the
            # rubric scores are missing, schema validation still rejects it.
            payload = salvage_truncated_json(cleaned)
            if payload is None:
                raise ValueError(
                    f"no JSON object found in judge response: {text[:200]!r}"
                )
    if not isinstance(payload, dict):
        raise ValueError(f"judge returned {type(payload).__name__}, expected object")
    return payload
