"""Qwen3-VL image-fact extraction. One call per image, cached by file content hash."""

from __future__ import annotations

import base64
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence

from bow.ai.cache import ExtractionCache, cache_key, content_hash
from bow.ai.client import ChatClient
from bow.ai.schemas import (
    EXTRACTOR_VERSION, IMAGE_PROMPT_VERSION, IMAGE_SCHEMA, IMAGE_SCHEMA_VERSION,
    IMAGE_SYSTEM,
    image_user_prompt,
)
from bow.errors import ExtractionError
from bow.models import ImageFact, ImageLabel, RawFinancialEvent, RawImage
from bow.money import Money
from bow.usage import UsageLedger

#: The extractor's vocabulary is richer than the engine's, so the mapping is explicit.
SEMANTICS_TO_LABEL: Mapping[str, ImageLabel] = {
    "net_pay": "net_pay",
    "balance_due": "balance_due",
    "grand_total": "grand_total",
    "invoice_total": "grand_total",
    "total_paid": "total_paid",
    "amount_due_by_date": "amount_due_by_date",
    "item_total": "item_total",
    "subtotal": "item_total",
    # Semantics that are never the economic value of the transaction. Accepting one of these as
    # the chosen amount is an extraction error, not a valid reading.
    "gross_pay": "net_pay",
    "total_earnings": "net_pay",
    "cash_tendered": "total_paid",
    "amount_due_after_date": "amount_due_by_date",
    "previous_balance": "balance_due",
    "other": "grand_total",
}

SUSPECT_SEMANTICS = {"gross_pay", "total_earnings", "cash_tendered",
                     "amount_due_after_date", "previous_balance", "subtotal"}

LOW_CONFIDENCE = 0.6

#: Documents print symbols, not ISO codes. Normalising them is reading the document correctly,
#: not inferring anything. An unrecognised symbol falls back to the currency the ledger already
#: records for the linked event, which is stated in the CSV rather than guessed.
SYMBOL_TO_ISO = {
    "₹": "INR", "RS": "INR", "RS.": "INR", "INR": "INR", "RUPEES": "INR", "₨": "INR",
    "$": "USD", "US$": "USD", "USD": "USD",
    "€": "EUR", "EUR": "EUR",
    "R": "ZAR", "ZAR": "ZAR",
    "RP": "IDR", "IDR": "IDR",
}


def normalise_currency(raw: object, ledger_currency: str | None) -> str | None:
    text = str(raw or "").strip().upper()
    if text in SYMBOL_TO_ISO:
        return SYMBOL_TO_ISO[text]
    if len(text) == 3 and text.isalpha():
        return text
    return ledger_currency


def encode_image(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    return base64.b64encode(raw).decode("ascii"), content_hash(raw)


def extract_image(image: RawImage, linked: RawFinancialEvent | None,
                  client: ChatClient, cache: ExtractionCache,
                  ledger: UsageLedger) -> tuple[ImageFact, bool]:
    path = Path(image.path)
    if not path.is_file():
        raise ExtractionError(f"{image.image_id}: file missing at {path}")
    encoded, digest = encode_image(path)

    model = client.vision_model
    key = cache_key(model, IMAGE_PROMPT_VERSION, IMAGE_SCHEMA_VERSION, EXTRACTOR_VERSION, digest)
    cached = cache.get(key)
    if cached is not None:
        meta = cached.get("meta", {})
        ledger.record(provider=meta.get("provider", "cache"), model=model,
                      call_type="image_extract", source_id=image.image_id,
                      input_tokens=int(meta.get("input_tokens", 0)),
                      output_tokens=int(meta.get("output_tokens", 0)),
                      latency_ms=0, cache="hit", retries=0)
        return to_fact(image, cached["fact"], _ledger_currency(linked)), True

    ledger.check_budget()
    hint = (f"{linked.description} ({linked.category}, {linked.direction}, "
            f"{linked.status}, dated {linked.event_date})") if linked else "an expense"
    currency = linked.amount.currency if linked and linked.amount else "the ledger currency"
    parsed, usage, retries = client.complete_json(
        model=model, system=IMAGE_SYSTEM,
        user_content=[
            {"type": "text", "text": image_user_prompt(hint, currency)},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{encoded}"}},
        ],
        schema=IMAGE_SCHEMA, schema_name="image_fact", max_tokens=900)
    ledger.record(provider=usage.provider, model=model, call_type="image_extract",
                  source_id=image.image_id, input_tokens=usage.input_tokens,
                  output_tokens=usage.output_tokens, latency_ms=usage.latency_ms,
                  cache="miss", retries=retries, provider_cost=usage.provider_cost)
    cache.put(key, fact=parsed, meta={
        "image_id": image.image_id, "related_event_id": image.related_event_id,
        "model": usage.model, "provider": usage.provider,
        "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
        "retries": retries, "cost_usd": str(usage.provider_cost or 0),
        "prompt_version": IMAGE_PROMPT_VERSION,
        "schema_version": IMAGE_SCHEMA_VERSION, "extractor_version": EXTRACTOR_VERSION})
    return to_fact(image, parsed, _ledger_currency(linked)), False


def to_fact(image: RawImage, payload: dict[str, Any],
            ledger_currency: str | None = None) -> ImageFact:
    semantics = payload["amount_semantics"]
    label = SEMANTICS_TO_LABEL.get(semantics)
    if label is None:
        raise ExtractionError(f"{image.image_id}: unmapped semantics {semantics!r}")
    try:
        amount = Decimal(str(payload["relevant_transaction_amount"]))
    except (InvalidOperation, KeyError) as exc:
        raise ExtractionError(f"{image.image_id}: unusable amount") from exc
    if amount < 0:
        raise ExtractionError(f"{image.image_id}: negative amount {amount}")

    currency = normalise_currency(payload.get("currency"), ledger_currency)
    if not currency or len(currency) != 3:
        raise ExtractionError(
            f"{image.image_id}: unusable currency {payload.get('currency')!r} "
            f"and no ledger currency to fall back on")

    confidence = float(payload.get("confidence") or 0.0)
    if semantics in SUSPECT_SEMANTICS:
        # The model chose a figure that is never the economic value of a transaction. Keep it,
        # but mark it for inspection rather than silently substituting a different number.
        confidence = min(confidence, 0.4)

    others = tuple(
        (str(entry.get("label")), Money(Decimal(str(entry.get("value"))), currency))
        for entry in (payload.get("other_amounts") or [])
        if entry.get("value") is not None
    )
    return ImageFact(
        image_id=image.image_id,
        related_event_id=image.related_event_id,
        chosen_amount=Money(amount, currency),
        semantic_label=label,
        rejected_candidates=others,
        confidence=confidence,
        notes=(payload.get("evidence_note") or "")[:200],
    )


def _ledger_currency(linked: RawFinancialEvent | None) -> str | None:
    """The CSV states the event's currency even when the amount is blank."""
    return linked.amount.currency if linked and linked.amount else None


def parse_document_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def low_confidence(facts: Sequence[ImageFact], threshold: float = LOW_CONFIDENCE):
    return tuple(f for f in facts if f.confidence < threshold)
