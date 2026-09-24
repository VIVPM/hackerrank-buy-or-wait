"""GLM-5.3 message-fact extraction. One call per message, cached by content hash."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

from bow.ai.cache import ExtractionCache, cache_key, content_hash
from bow.ai.client import ChatClient
from bow.ai.schemas import (
    EXTRACTOR_VERSION, MESSAGE_PROMPT_VERSION, MESSAGE_SCHEMA, MESSAGE_SCHEMA_VERSION,
    MESSAGE_SYSTEM,
    message_user_prompt,
)
from bow.errors import ExtractionError
from bow.models import FactType, MessageFact, RawFinancialEvent, RawMessage
from bow.money import Money
from bow.usage import UsageLedger

#: Targets that name a recurring *expense* rather than income.
EXPENSE_TARGETS = {"rent", "utilities", "loan", "childcare"}

#: Targets that name income. A suppression message may only stop the class it is about, which
#: `recurrence._suppressed_classes` resolves: a gig or invoice note never touches payroll.
INCOME_TARGETS = {"salary", "gig_payout", "invoice"}


def derive_fact_type(operation: str, target: str, recurrence_effect: str,
                     has_amount: bool, has_percent: bool, has_new_date: bool) -> FactType:
    """Map an extraction onto the engine's vocabulary.

    `operation` alone is not trustworthy: the extractor labels a rent increase
    `recurrence_start` and a closed prize claim `recurrence_stop`. `target`,
    `recurrence_effect` and the presence of a number are far more stable, so the decision uses
    the combination.

    The safety property this function exists to guarantee: **a message that is not about salary
    can never terminate income projection.** Fifty of the 215 messages are `income_unconfirmed`
    or `recurrence_stop` aimed at a bonus, prize, invoice, gig payout or utility bill; taking
    those at face value would zero out the salary of roughly a fifth of all users.
    """
    if target in INCOME_TARGETS:
        if has_amount and target == "salary":
            # A stated salary figure is a change, not a termination. Amount wins over a
            # "stops" flag, which the extractor applies far too readily.
            return "salary_amount_change"
        if recurrence_effect == "stops" or operation == "employment_end":
            return "income_ended"
        if target != "salary":
            # A message about gig or invoice income that states no confirmed amount leaves that
            # income unconfirmed. The specification is explicit that money which is pending,
            # awaiting approval or not yet credited is not counted, and a platform or client
            # telling the user about their payouts without naming a settled figure is exactly
            # that case. A message naming an approved amount is left alone: the recurring
            # history already carries it.
            return "no_financial_effect" if has_amount else "income_unconfirmed"
        if has_new_date or recurrence_effect == "date_changes" or operation == "amend_date":
            return "salary_date_change"
        # Salary mentioned with no figure and no schedule change: nothing to apply. The engine's
        # own income taxonomy already refuses to project unconfirmed income.
        return "no_financial_effect"

    if target in EXPENSE_TARGETS and (has_amount or has_percent) and             recurrence_effect in {"amount_changes", "starts"}:
        return "recurring_expense_change"
    if operation == "refund_notice":
        return "refund_pending"
    if operation == "duplicate_notice":
        return "duplicate_under_investigation"
    if operation == "delay":
        return "event_delayed"
    if operation == "cancel":
        return "event_cancelled"
    if operation == "confirm":
        return "event_confirmed"
    return "no_financial_effect"


LOW_CONFIDENCE = 0.5


def source_digest(message: RawMessage, linked: RawFinancialEvent | None) -> str:
    return content_hash(message.message_text, message.source_type, message.sent_at,
                        _linked_summary(linked) or "")


def _linked_summary(event: RawFinancialEvent | None) -> str | None:
    if event is None:
        return None
    amount = f"{event.amount.amount} {event.amount.currency}" if event.amount else "amount unknown"
    return (f"{event.description} | {event.category} | {event.direction} | {amount} | "
            f"{event.event_date} | status {event.status}")


def extract_message(message: RawMessage, linked: RawFinancialEvent | None,
                    client: ChatClient, cache: ExtractionCache,
                    ledger: UsageLedger) -> tuple[MessageFact, bool]:
    """Return (fact, cache_hit). Raises ExtractionError only when the model is unusable."""
    model = client.text_model
    key = cache_key(model, MESSAGE_PROMPT_VERSION, MESSAGE_SCHEMA_VERSION, EXTRACTOR_VERSION,
                    source_digest(message, linked))
    cached = cache.get(key)
    if cached is not None:
        meta = cached.get("meta", {})
        ledger.record(provider=meta.get("provider", "cache"), model=model,
                      call_type="message_extract", source_id=message.message_id,
                      input_tokens=int(meta.get("input_tokens", 0)),
                      output_tokens=int(meta.get("output_tokens", 0)),
                      latency_ms=0, cache="hit", retries=0)
        return to_fact(message, cached["fact"]), True

    ledger.check_budget()
    parsed, usage, retries = client.complete_json(
        model=model, system=MESSAGE_SYSTEM,
        user_content=message_user_prompt(message.message_text, message.source_type,
                                         message.sent_at, _linked_summary(linked)),
        schema=MESSAGE_SCHEMA, schema_name="message_fact", max_tokens=500)
    ledger.record(provider=usage.provider, model=model, call_type="message_extract",
                  source_id=message.message_id, input_tokens=usage.input_tokens,
                  output_tokens=usage.output_tokens, latency_ms=usage.latency_ms,
                  cache="miss", retries=retries, provider_cost=usage.provider_cost)
    cache.put(key, fact=parsed, meta={
        "message_id": message.message_id, "model": usage.model, "provider": usage.provider,
        "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
        "retries": retries, "cost_usd": str(usage.provider_cost or 0),
        "prompt_version": MESSAGE_PROMPT_VERSION,
        "schema_version": MESSAGE_SCHEMA_VERSION, "extractor_version": EXTRACTOR_VERSION})
    return to_fact(message, parsed), False


def to_fact(message: RawMessage, payload: dict[str, Any]) -> MessageFact:
    amount = _money(payload.get("amount"), payload.get("currency"))
    percent = payload.get("percent_change")
    multiplier = (Decimal(1) + Decimal(str(percent)) / Decimal(100)) if percent else None
    new_date = _date(payload.get("new_date"))

    fact_type = derive_fact_type(
        operation=str(payload.get("operation") or ""),
        target=str(payload.get("target") or "other"),
        recurrence_effect=str(payload.get("recurrence_effect") or "none"),
        has_amount=amount is not None,
        has_percent=multiplier is not None,
        has_new_date=new_date is not None,
    )
    # A fact is quantified when the message actually carried a number the engine can use.
    # Without one, the fact is recorded and never priced.
    quantified = amount is not None or multiplier is not None

    return MessageFact(
        message_id=message.message_id,
        user_id=message.user_id,
        fact_type=fact_type,
        subject=payload.get("target") or "other",
        target_event_id=message.related_event_id,
        effective_date=new_date or _date(payload.get("effective_date")),
        amount=amount,
        multiplier=multiplier,
        quantified=quantified,
        confidence=float(payload.get("confidence") or 0.0),
        evidence_span=(payload.get("notes_short") or "")[:200],
    )


def _money(value: Any, currency: Any) -> Money | None:
    if value is None or not currency:
        return None
    try:
        return Money(Decimal(str(value)), str(currency).strip().upper()[:3])
    except (InvalidOperation, ValueError):
        return None


def _date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def low_confidence(facts: Sequence[MessageFact], threshold: float = LOW_CONFIDENCE):
    return tuple(f for f in facts if f.confidence < threshold)
