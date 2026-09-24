"""Deterministic decision_explanation.

Templated from verified numbers. No model is called: the explanation has to agree with the plan
exactly, and a template built from the winning candidate cannot drift from it. The wording
follows the register of the solved samples.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from bow.models import CandidatePlan, EvidenceBundle, SpendingChange
from bow.money import quantize

MONTHS = ("January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December")

#: Readable names for the categories that can be changed, so the sentence reads like the samples
#: ("Stop the online backup subscription") rather than naming a raw category key.
CATEGORY_WORDS = {
    "streaming": "streaming subscription",
    "cloud_storage": "online backup subscription",
    "music_subscription": "music subscription",
    "delivery_membership": "delivery membership",
    "gym": "gym membership",
    "dining": "dining spend",
    "entertainment": "entertainment spend",
    "shopping": "shopping spend",
}


def money(currency: str, amount: Decimal) -> str:
    value = quantize(amount)
    whole = value == value.to_integral_value()
    return f"{currency} {value:,.0f}" if whole else f"{currency} {value:,.2f}"


def long_date(day: date) -> str:
    return f"{day.day} {MONTHS[day.month - 1]} {day.year}"


def _change_phrase(change: SpendingChange, currency: str) -> str:
    name = CATEGORY_WORDS.get(change.stream_key[1], f"{change.stream_key[1]} spend")
    if change.kind == "stop":
        return f"stop the {name}"
    return f"reduce the {name} to {money(currency, change.new_amount or Decimal(0))}"


def _changes_prefix(plan: CandidatePlan, currency: str) -> str:
    if not plan.changes:
        return ""
    phrases = [_change_phrase(c, currency) for c in plan.changes]
    joined = phrases[0] if len(phrases) == 1 else \
        ", ".join(phrases[:-1]) + f" and {phrases[-1]}"
    return joined[0].upper() + joined[1:] + ", then "


def build(plan: CandidatePlan, ev: EvidenceBundle, safe_today: Decimal,
          earliest: date | None) -> str:
    currency = ev.profile.home_currency
    minimum = money(currency, ev.profile.minimum_balance_to_keep)
    requested = ev.request.requested_amount
    prefix = _changes_prefix(plan, currency)

    if plan.method == "full_payment":
        head = f"{prefix}pay {money(currency, requested)} today" if prefix else \
            f"Pay {money(currency, requested)} today"
        return (f"{head}. This leaves at least {minimum} available over the next 90 days.")

    if plan.method == "wait":
        when = long_date(plan.payments[0][0])
        return (f"{prefix}pay {money(currency, requested)} in full on {when}."
                if prefix else
                f"Pay {money(currency, requested)} in full on {when}. "
                f"Paying earlier would take the balance below the {minimum} minimum.")

    if plan.method == "partial_payment":
        (_, first), (second_date, second) = plan.payments[0], plan.payments[1]
        return (f"Pay {money(currency, first)} today and the remaining "
                f"{money(currency, second)} on {long_date(second_date)}. This completes the "
                f"full request and keeps the {minimum} minimum protected.")

    if plan.method == "installments":
        count = len(plan.payments)
        each = money(currency, plan.payments[0][1])
        start = long_date(plan.payments[0][0])
        body = (f"use {count} installments of {each}, starting {start}"
                if prefix else
                f"Use {count} installments of {each}, starting {start}")
        return f"{prefix}{body}. This leaves at least {minimum} available."

    # not_recommended
    deadline = long_date(ev.request.desired_completion_date)
    if safe_today > 0 and ev.request.allows_partial_payment:
        return (f"Do not proceed with the {money(currency, requested)} request. Although "
                f"{money(currency, safe_today)} is available today, the full amount cannot be "
                f"completed safely within 90 days.")
    return (f"Do not make this payment by {deadline}. None of the available options keeps the "
            f"{minimum} minimum protected.")
