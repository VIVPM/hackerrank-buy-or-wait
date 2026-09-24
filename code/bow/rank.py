"""The specification's ranking, applied literally and deterministically.

No model is consulted. The order is fixed by the problem statement:

    1. complete the full request by desired_completion_date
    2. require no spending changes
    3. minimise the total amount paid
    4. start payment earlier
    5. use fewer payments
    6. lowest payment_option_id
"""

from __future__ import annotations

from datetime import date
from typing import Sequence

from bow.models import CandidatePlan

FAR_FUTURE = date(9999, 12, 31)

RULES = ("completes by deadline", "requires no spending changes", "lower total paid",
         "starts earlier", "fewer payments", "lower payment_option_id")


def sort_key(plan: CandidatePlan):
    starts = min((d for d, _ in plan.payments), default=FAR_FUTURE)
    return (
        0 if plan.completes_by_deadline else 1,        # 1. complete by the deadline
        1 if plan.changes else 0,                      # 2. require no spending changes
        plan.total_paid,                               # 3. minimise total paid
        starts,                                        # 4. start earlier
        len(plan.payments),                            # 5. fewer payments
        plan.source_option_id or "",                   # 6. lowest payment_option_id
    )


def rank(candidates: Sequence[CandidatePlan]) -> tuple[CandidatePlan, ...]:
    real = [c for c in candidates if c.method != "not_recommended"]
    fallback = [c for c in candidates if c.method == "not_recommended"]
    return tuple(sorted(real, key=sort_key)) + tuple(fallback)


def choose(candidates: Sequence[CandidatePlan]) -> CandidatePlan:
    ordered = rank(candidates)
    if not ordered:
        raise ValueError("no candidates, not even the fallback")
    return ordered[0]


def explain_choice(winner: CandidatePlan, runners: Sequence[CandidatePlan]) -> str:
    if winner.method == "not_recommended":
        return "no eligible candidate survived validation"
    if not runners:
        return "only eligible safe candidate"
    other = runners[0]
    win, lose = sort_key(winner), sort_key(other)
    for index, name in enumerate(RULES):
        if win[index] != lose[index]:
            return f"beat {other.method} on rule {index + 1} ({name})"
    return "tie on every rule; stable order"
