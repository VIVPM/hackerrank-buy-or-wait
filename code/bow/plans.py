"""Candidate plan generation.

Every candidate is re-forecast from scratch with its own payments and stream overrides applied,
so a plan is never judged against a projection it does not actually cause.

Spending-change variants are generated only when no change-free candidate completes the request
by its deadline. That is not a shortcut: ranking rule 1 is "complete by the deadline" and rule 2
is "require no spending changes", so a change-free candidate that completes by the deadline can
never be beaten by one that needs changes. Skipping them there is exactly equivalent, and far
cheaper than re-forecasting every subset.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Sequence

from bow.eligibility import Eligibility, assess
from bow.forecast import Forecast, ForecastContext, ForecastEngine
from bow.models import CandidatePlan, EvidenceBundle, ProjectedCashFlow, SpendingChange
from bow.safeamount import SafeAmount, earliest_full_payment_date, safe_amount
from bow.spending import change_sets, overrides_for, permitted_changes
from bow.trace import NULL_TRACE, TraceLike

ZERO = Decimal(0)


@dataclass(frozen=True, slots=True)
class PlanningResult:
    candidates: tuple[CandidatePlan, ...]
    base_forecast: Forecast
    safe: SafeAmount
    earliest: date | None
    eligibility: Eligibility
    rejected: tuple[tuple[str, str], ...]


def _flows(payments: Sequence[tuple[date, Decimal]]) -> tuple[ProjectedCashFlow, ...]:
    return tuple(ProjectedCashFlow(date=d, amount=-a, kind="plan_payment", origin="plan")
                 for d, a in payments)


def _is_safe(engine: ForecastEngine, ev: EvidenceBundle, asof: date,
             payments: Sequence[tuple[date, Decimal]],
             changes: Sequence[SpendingChange] = ()) -> bool:
    ctx = ForecastContext(ev, asof, _flows(payments), overrides_for(changes))
    return engine.project(ctx).is_safe


def generate(ev: EvidenceBundle, engine: ForecastEngine,
             trace: TraceLike = NULL_TRACE) -> PlanningResult:
    request, profile = ev.request, ev.profile
    asof, requested = request.request_date, request.requested_amount
    elig = assess(profile, request, ev.options)

    base_ctx = ForecastContext(ev, asof)
    base = engine.project(base_ctx)
    safe = safe_amount(engine, base_ctx, requested, base)
    earliest = earliest_full_payment_date(engine, base_ctx, requested, base)

    candidates: list[CandidatePlan] = []
    rejected: list[tuple[str, str]] = []

    def add(method, payments, changes, option_id, status):
        total = sum((a for _, a in payments), ZERO)
        last = max((d for d, _ in payments), default=None)
        candidates.append(CandidatePlan(
            method=method, payments=tuple(payments), changes=tuple(changes),
            source_option_id=option_id, total_paid=total,
            # ">=" not "==": an installment option's total includes its financing fee, so it
            # pays MORE than the requested amount while still completing the request. With "=="
            # every fee-bearing installment plan counted as "never completes" under rule 1.
            completes_full_amount=(total >= requested),
            completes_by_deadline=(total >= requested and last is not None
                                   and last <= request.desired_completion_date),
            implied_status=status))

    def build(changes: Sequence[SpendingChange]) -> None:
        """All method candidates under one (possibly empty) set of spending changes."""
        if changes:
            ctx = ForecastContext(ev, asof, (), overrides_for(changes))
            fc = engine.project(ctx)
            sa = safe_amount(engine, ctx, requested, fc)
            ed = earliest_full_payment_date(engine, ctx, requested, fc)
        else:
            sa, ed = safe, earliest
        status = "affordable_with_plan" if changes else "affordable_now"

        # ---- full payment today
        if elig.accepts("full_payment"):
            if sa.amount == requested and _is_safe(engine, ev, asof,
                                                   [(asof, requested)], changes):
                add("full_payment", [(asof, requested)], changes, None, status)
            elif not changes:
                rejected.append(("full_payment", "not safe on request_date"))

        # ---- wait: the same full payment, later.
        # A wait recommendation must be able to name the date the user should pay on, and that
        # date is reported in earliest_date_for_full_payment - which the specification defines
        # as the date reached WITHOUT optional spending changes. So a wait is only offerable
        # when such a date exists; otherwise the row would say "wait" while also reporting that
        # the full amount never becomes affordable, which is self-contradictory.
        if elig.accepts("wait") and ed is not None and ed > asof and earliest is not None:
            if ed <= request.desired_completion_date and _is_safe(
                    engine, ev, asof, [(ed, requested)], changes):
                add("wait", [(ed, requested)], changes,
                    None, "affordable_with_plan" if changes else "affordable_later")
            elif not changes:
                rejected.append(("wait", "full payment not safe on or before the deadline"))

        # ---- partial payment: exactly two payments summing to the request
        ok, why = elig.partial_allowed(sa.amount, ed)
        if ok and ed is not None:
            payments = [(asof, sa.amount), (ed, requested - sa.amount)]
            if _is_safe(engine, ev, asof, payments, changes):
                add("partial_payment", payments, changes, None, "affordable_with_plan")
            elif not changes:
                rejected.append(("partial_payment", "schedule breaches the minimum balance"))
        elif not changes and why:
            rejected.append(("partial_payment", why))

        # ---- installments: exactly as supplied, never modified
        for verdict in elig.verdicts:
            option = verdict.option
            if option.payment_method != "installments":
                continue
            if not verdict.usable:
                if not changes:
                    rejected.append((option.payment_option_id, verdict.reason or "ineligible"))
                continue
            payments = list(option.schedule)
            if _is_safe(engine, ev, asof, payments, changes):
                add("installments", payments, changes, option.payment_option_id,
                    "affordable_with_plan")
            elif not changes:
                rejected.append((option.payment_option_id,
                                 "schedule breaches the minimum balance"))

    build(())

    if not any(c.completes_by_deadline for c in candidates):
        allowed = permitted_changes(profile, ev.streams)
        for combo in change_sets(allowed):
            before = len(candidates)
            build(combo)
            if trace and len(candidates) > before:
                trace.section("plans").add(
                    "spending_change_candidate",
                    changes=[f"{c.kind}:{c.event_id}" for c in combo],
                    produced=len(candidates) - before)

    # not_recommended is always available as the fallback.
    add("not_recommended", [], (), None, "not_affordable")

    if trace:
        section = trace.section("plans")
        section.add("capacity", safe=str(safe.amount), earliest=earliest,
                    trough=str(base.trough), trough_date=base.trough_date,
                    reserve=str(safe.reserve))
        for c in candidates:
            section.add("candidate", method=c.method, option=c.source_option_id,
                        payments=[[d.isoformat(), str(a)] for d, a in c.payments],
                        changes=[f"{x.kind}:{x.event_id}" for x in c.changes],
                        total=str(c.total_paid), completes=c.completes_by_deadline)
        for what, why in rejected:
            section.add("rejected", what=what, reason=why)

    return PlanningResult(tuple(candidates), base, safe, earliest, elig, tuple(rejected))
