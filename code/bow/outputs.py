"""Output serialisation in exactly the required format and column order."""

from __future__ import annotations

import csv
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Sequence

from bow.models import OUTPUT_COLUMNS, CandidatePlan, PredictionResult, SpendingChange
from bow.money import format_amount


def serialise_plan(payments: Sequence[tuple[date, Decimal]]) -> str:
    """`YYYY-MM-DD:amount` entries joined by `|`, chronological, or `none`."""
    if not payments:
        return "none"
    return "|".join(f"{d.isoformat()}:{format_amount(a)}" for d, a in payments)


def serialise_changes(changes: Sequence[SpendingChange]) -> str:
    if not changes:
        return "none"
    parts = []
    for change in changes:
        if change.kind == "stop":
            parts.append(f"stop:{change.event_id}")
        else:
            parts.append(
                f"reduce_to:{change.event_id}:{format_amount(change.new_amount or Decimal(0))}")
    return "|".join(parts)


def serialise_date(value: date | None) -> str:
    return value.isoformat() if value else ""


def row_for(result: PredictionResult) -> dict[str, str]:
    return {
        "request_id": result.request_id,
        "amount_safe_to_pay": format_amount(result.amount_safe_to_pay),
        "affordability_status": result.affordability_status,
        "recommended_payment_method": result.recommended_payment_method,
        "payment_plan": result.payment_plan,
        "earliest_date_for_full_payment": result.earliest_date_for_full_payment,
        "spending_changes_needed": result.spending_changes_needed,
        "decision_explanation": result.decision_explanation,
    }


def write_output(results: Sequence[PredictionResult], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        for result in results:
            writer.writerow(row_for(result))
    return path


def plan_fields(plan: CandidatePlan) -> tuple[str, str]:
    return serialise_plan(plan.payments), serialise_changes(plan.changes)
