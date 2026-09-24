"""Permitted spending changes and their bounded enumeration.

Only a stream that is BOTH flexible in the ledger AND in a category the user said they would
change may be touched. `reduce_to` uses the stream's `minimum_allowed_amount`, which is the only
reduced value the dataset supplies.
"""

from __future__ import annotations

from decimal import Decimal
from itertools import combinations
from typing import Iterator, Sequence

from bow.models import FinancialProfile, RecurringStream, SpendingChange

MAX_CHANGES = 3
STOPPABLE = {"stoppable", "reducible_or_stoppable"}
REDUCIBLE = {"reducible", "reducible_or_stoppable"}


def permitted_changes(profile: FinancialProfile,
                      streams: Sequence[RecurringStream]) -> tuple[SpendingChange, ...]:
    """At most one change per stream, so stop and reduce can never collide."""
    out: list[SpendingChange] = []
    for stream in sorted(streams, key=lambda s: (s.kind, s.group)):
        if stream.kind != "expense":
            continue
        monthly = _monthly_equivalent(stream)
        if (stream.flexibility in STOPPABLE
                and stream.group in profile.stoppable_categories):
            out.append(SpendingChange(
                kind="stop", stream_key=stream.key, event_id=stream.latest_event_id,
                new_amount=None, monthly_saving=monthly))
        if (stream.flexibility in REDUCIBLE
                and stream.group in profile.reducible_categories
                and stream.minimum_allowed_amount is not None
                and stream.minimum_allowed_amount < stream.amount):
            saved = monthly * (stream.amount - stream.minimum_allowed_amount) / stream.amount
            out.append(SpendingChange(
                kind="reduce_to", stream_key=stream.key, event_id=stream.latest_event_id,
                new_amount=stream.minimum_allowed_amount, monthly_saving=saved))
    return tuple(out)


def change_sets(changes: Sequence[SpendingChange],
                limit: int = MAX_CHANGES) -> Iterator[tuple[SpendingChange, ...]]:
    """Exhaustive, deterministic enumeration of 1..limit changes on distinct streams.

    Flexible streams per user are few (at most a handful), so enumeration beats a heuristic and
    removes any question of the search missing a valid answer.
    """
    ordered = sorted(changes, key=lambda c: (c.stream_key, c.kind))
    for size in range(1, min(limit, len(ordered)) + 1):
        for combo in combinations(ordered, size):
            keys = [c.stream_key for c in combo]
            if len(set(keys)) != len(keys):
                continue          # never stop and reduce the same stream
            yield combo


def overrides_for(changes: Sequence[SpendingChange]) -> dict:
    return {c.stream_key: (None if c.kind == "stop" else c.new_amount) for c in changes}


def _monthly_equivalent(stream: RecurringStream) -> Decimal:
    return stream.amount * Decimal(30) / Decimal(stream.cadence_days)
