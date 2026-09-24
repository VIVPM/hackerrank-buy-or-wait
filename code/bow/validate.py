"""Independent validation of a candidate plan.

This module does not trust the planner. It re-derives the forecast itself, so a candidate that
the planner built incorrectly is still rejected here. It shares `eligibility` with the planner
on purpose - the *rules* must not drift apart - but every safety and arithmetic check is
recomputed from the evidence.

Each failure carries a reason code so the harness can group misses without parsing prose.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Sequence

from bow.eligibility import assess
from bow.forecast import ForecastContext, ForecastEngine
from bow.models import CandidatePlan, EvidenceBundle, Violation
from bow.spending import REDUCIBLE, STOPPABLE, overrides_for
from bow.plans import _flows

ZERO = Decimal(0)
MAX_CHANGES = 3


def validate(plan: CandidatePlan, ev: EvidenceBundle, engine: ForecastEngine,
             safe_amount_today: Decimal,
             earliest: "object | None" = None) -> tuple[Violation, ...]:
    request, profile = ev.request, ev.profile
    requested = request.requested_amount
    out: list[Violation] = []

    def fail(code: str, message: str) -> None:
        out.append(Violation(code=code, message=message))

    # ---- V1 amount bounds
    if not (ZERO <= safe_amount_today <= requested):
        fail("V1", f"amount_safe_to_pay {safe_amount_today} outside [0, {requested}]")

    if plan.method == "not_recommended":
        if plan.payments:
            fail("V15", "not_recommended must carry no payments")
        return tuple(out)

    # ---- V5 method eligibility
    elig = assess(profile, request, ev.options)
    if not elig.accepts(plan.method):
        fail("V5", f"user does not accept {plan.method}")

    # ---- V13 chronological order
    dates = [d for d, _ in plan.payments]
    if dates != sorted(dates):
        fail("V13", "payments are not in chronological order")
    if any(a <= ZERO for _, a in plan.payments):
        fail("V13", "a payment is not positive")

    # ---- V3 exact sums
    total = sum((a for _, a in plan.payments), ZERO)
    if plan.method in {"full_payment", "partial_payment", "wait"} and total != requested:
        fail("V3", f"payments sum to {total}, not the requested {requested}")

    # ---- V4 completion deadline
    if plan.method in {"full_payment", "partial_payment", "wait", "installments"}:
        if dates and max(dates) > request.desired_completion_date:
            fail("V4", f"last payment {max(dates)} after deadline "
                       f"{request.desired_completion_date}")

    # ---- V6 partial-payment structure
    if plan.method == "partial_payment":
        ok, why = elig.partial_allowed(safe_amount_today, earliest)   # type: ignore[arg-type]
        if not ok:
            fail("V6", f"partial payment not permitted: {why}")
        if len(plan.payments) != 2:
            fail("V6", f"partial payment must have exactly 2 payments, got {len(plan.payments)}")
        else:
            (d1, a1), (d2, a2) = plan.payments
            if d1 != request.request_date:
                fail("V6", "first partial payment is not on request_date")
            if a1 != safe_amount_today:
                fail("V6", f"first partial payment {a1} != amount_safe_to_pay "
                           f"{safe_amount_today}")
            if a2 != requested - safe_amount_today:
                fail("V6", "second partial payment is not the remainder")
            if earliest is not None and d2 != earliest:
                fail("V6", "second partial payment is not on earliest_date_for_full_payment")

    # ---- V7 installments must equal a supplied option
    if plan.method == "installments":
        match = next((v for v in elig.verdicts
                      if v.option.payment_option_id == plan.source_option_id), None)
        if match is None:
            fail("V7", f"unknown payment_option_id {plan.source_option_id!r}")
        else:
            if not match.usable:
                fail("V7", f"option not eligible: {match.reason}")
            if tuple(plan.payments) != tuple(match.option.schedule):
                fail("V7", "schedule does not match the supplied option exactly")
            if total != match.option.total_payable_amount:
                fail("V7", f"total {total} != option total "
                           f"{match.option.total_payable_amount}")
            if (profile.max_installment_months is not None
                    and match.option.number_of_payments > profile.max_installment_months):
                fail("V7", "exceeds max_installment_months")
    elif plan.source_option_id is not None:
        fail("V7", "non-installment plan cites a payment option")

    # ---- V8/V9/V10/V11/V12 spending changes
    if len(plan.changes) > MAX_CHANGES:
        fail("V10", f"{len(plan.changes)} spending changes exceeds {MAX_CHANGES}")
    seen: set = set()
    by_key = {s.key: s for s in ev.streams}
    for change in plan.changes:
        if change.stream_key in seen:
            fail("V11", f"two changes target the same stream {change.stream_key}")
        seen.add(change.stream_key)
        stream = by_key.get(change.stream_key)
        if stream is None:
            fail("V8", f"change targets unknown stream {change.stream_key}")
            continue
        if change.event_id != stream.latest_event_id:
            fail("V8", f"change cites {change.event_id}, not the latest occurrence "
                       f"{stream.latest_event_id}")
        if change.kind == "stop":
            if stream.flexibility not in STOPPABLE:
                fail("V8", f"{stream.group} is not stoppable ({stream.flexibility})")
            if stream.group not in profile.stoppable_categories:
                fail("V9", f"user will not stop {stream.group}")
        else:
            if stream.flexibility not in REDUCIBLE:
                fail("V8", f"{stream.group} is not reducible ({stream.flexibility})")
            if stream.group not in profile.reducible_categories:
                fail("V9", f"user will not reduce {stream.group}")
            if change.new_amount != stream.minimum_allowed_amount:
                fail("V12", f"reduce_to {change.new_amount} != minimum_allowed_amount "
                            f"{stream.minimum_allowed_amount}")

    # ---- V2 the minimum balance must hold, re-derived here
    ctx = ForecastContext(ev, request.request_date, _flows(plan.payments),
                          overrides_for(plan.changes))
    forecast = engine.project(ctx)
    if not forecast.is_safe:
        first = forecast.breaches[0]
        fail("V2", f"balance falls to {first[1]} on {first[0]}, below "
                   f"{profile.minimum_balance_to_keep}")

    return tuple(out)


def first_valid(candidates: Sequence[CandidatePlan], ev: EvidenceBundle,
                engine: ForecastEngine, safe_amount_today: Decimal,
                earliest) -> tuple[CandidatePlan | None,
                                   dict[str, tuple[Violation, ...]]]:
    """Return the first candidate that survives validation, plus every rejection."""
    problems: dict[str, tuple[Violation, ...]] = {}
    for plan in candidates:
        found = validate(plan, ev, engine, safe_amount_today, earliest)
        label = f"{plan.method}:{plan.source_option_id or '-'}:{len(plan.changes)}"
        if not found:
            return plan, problems
        problems[label] = found
    return None, problems
