"""Deterministic 90-day cash-flow forecast.

    starting_balance = profile.current_available_balance   (the request-date snapshot)

Settled history is never replayed - it is already inside that snapshot. Only explicit future
obligations (pending and scheduled rows) and projected recurring flows move the balance.

Sub-monthly streams accrue continuously. A weekly grocery habit is a *rate*, not a dated
commitment: charging whole cycles over-reserves on long windows and under-reserves on short ones,
which the forensic phase measured as a clean two-sided bias. Monthly streams stay discrete on their
calendar day.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from types import MappingProxyType
from decimal import Decimal
from typing import Mapping, Sequence

from bow.config import ForecastConfig
from bow.models import (
    EvidenceBundle,
    ForecastResult,
    ProjectedCashFlow,
    RecurringStream,
    StreamKind,
)
from bow.money import quantize
from bow.recurrence import daily_rate, occurrences
from bow.trace import NULL_TRACE, TraceLike

ZERO = Decimal(0)


@dataclass(frozen=True, slots=True)
class ForecastContext:
    """Inputs to one projection. `extra_outflows` is how a candidate plan is tested."""

    evidence: EvidenceBundle
    asof: date
    extra_outflows: tuple[ProjectedCashFlow, ...] = ()
    stream_overrides: Mapping[tuple[StreamKind, str], Decimal | None] = MappingProxyType({})

    @property
    def opening_balance(self) -> Decimal:
        return self.evidence.profile.current_available_balance

    @property
    def minimum_balance(self) -> Decimal:
        return self.evidence.profile.minimum_balance_to_keep

    def overrides(self) -> dict[tuple[StreamKind, str], Decimal | None]:
        return dict(self.stream_overrides)


@dataclass(frozen=True, slots=True)
class DayPoint:
    day: date
    low: Decimal        # the balance the minimum-balance rule is tested against
    close: Decimal


@dataclass(frozen=True, slots=True)
class Forecast:
    opening_balance: Decimal
    minimum_balance: Decimal
    flows: tuple[ProjectedCashFlow, ...]
    path: tuple[DayPoint, ...]
    trough: Decimal
    trough_date: date
    daily_accrual: Decimal
    config_fingerprint: str

    @property
    def reserve(self) -> Decimal:
        """How much of the opening balance the forecast consumes before its worst point."""
        return self.opening_balance - self.trough

    @property
    def breaches(self) -> tuple[tuple[date, Decimal], ...]:
        return tuple((p.day, p.low) for p in self.path if p.low < self.minimum_balance)

    @property
    def is_safe(self) -> bool:
        return not self.breaches

    def suffix_minimum(self, start: date) -> Decimal:
        """Lowest balance from `start` to the horizon, using each day's intraday low."""
        points = [p.low for p in self.path if p.day >= start]
        return min(points) if points else self.path[-1].close

    def suffix_minimum_for_payment(self, start: date) -> Decimal:
        """Lowest balance available to someone paying a discretionary amount on `start`.

        On the payment day itself the payer chooses when to settle, so they pay after that day's
        credits have landed - the day's *closing* balance is what is available. Every later day is
        outside their control and uses the intraday low. Without this distinction a payment on
        payday is judged against the pre-salary balance, which pushes every date one day late.
        """
        points = [p for p in self.path if p.day >= start]
        if not points:
            return self.path[-1].close
        return min([points[0].close] + [p.low for p in points[1:]])

    def prefix_minimum(self, before: date) -> Decimal:
        points = [p.low for p in self.path if p.day < before]
        return min(points) if points else self.opening_balance

    def contributors(self, upto: date | None = None) -> tuple[ProjectedCashFlow, ...]:
        """Flows landing on or before the trough - the events that created it."""
        limit = upto or self.trough_date
        return tuple(f for f in self.flows if f.date <= limit)

    def to_result(self) -> ForecastResult:
        return ForecastResult(
            opening_balance=self.opening_balance,
            minimum_balance=self.minimum_balance,
            flows=self.flows,
            path=tuple((p.day, p.close) for p in self.path),
            trough=self.trough,
            trough_date=self.trough_date,
            breaches=self.breaches,
            config_fingerprint=self.config_fingerprint,
        )


class ForecastEngine:
    def __init__(self, config: ForecastConfig | None = None) -> None:
        self.config = config or ForecastConfig()

    # -------------------------------------------------------------- flows

    def _stream_amount(self, stream: RecurringStream,
                       overrides: Mapping) -> Decimal | None:
        if stream.key in overrides:
            return overrides[stream.key]          # None means the stream was stopped
        return stream.amount

    def flows_for(self, ctx: ForecastContext) -> tuple[tuple[ProjectedCashFlow, ...], Decimal]:
        cfg = self.config
        end = ctx.asof + timedelta(days=cfg.horizon_days)
        overrides = ctx.overrides()
        flows: list[ProjectedCashFlow] = []
        accrual = ZERO

        # 1. explicit future obligations, straight from the canonical layer.
        # `explicit` keeps each flow beside its category so supersession can match streams.
        explicit: list[tuple[ProjectedCashFlow, str]] = []
        for e in ctx.evidence.events:
            if not e.counts_for_cash or e.effective_date is None:
                continue
            if e.effective_date < ctx.asof or e.effective_date > end:
                continue
            amount = e.signed_amount
            if amount is None:
                continue
            explicit.append((ProjectedCashFlow(
                date=e.effective_date, amount=amount, kind="explicit",
                origin=e.economic_id, note=e.lifecycle_role), e.category))
        flows.extend(f for f, _ in explicit)

        # 2. recurring streams
        for stream in ctx.evidence.streams:
            amount = self._stream_amount(stream, overrides)
            if amount is None or amount == 0:
                continue
            sign = 1 if stream.kind == "income" else -1
            scaled = stream if amount == stream.amount else _rescaled(stream, amount)
            if scaled.cadence_class == "sub_monthly" and cfg.submonthly_mode != "discrete":
                if cfg.submonthly_mode == "accrual":
                    accrual += daily_rate(scaled) * (1 if sign < 0 else -1)
                    continue
                if cfg.submonthly_mode == "cycle_ceiling":
                    accrual += daily_rate(scaled) * (1 if sign < 0 else -1)
                    flows.append(ProjectedCashFlow(
                        date=ctx.asof, amount=ZERO, kind="accrual",
                        origin=f"{scaled.kind}:{scaled.group}", note="ceiling"))
                    continue
                # cycle_upfront: a habit already in progress costs a full cycle before it
                # renews, so reserve one cycle immediately rather than prorating it.
                day = ctx.asof
                while day <= end:
                    if day >= ctx.asof:
                        flows.append(ProjectedCashFlow(
                            date=day, amount=sign * scaled.amount, kind="recurring",
                            origin=f"{scaled.kind}:{scaled.group}", note="cycle_upfront"))
                    day = day + timedelta(days=scaled.cadence_days)
                continue
            for day in occurrences(scaled, ctx.asof, end, cfg):
                if self._superseded(scaled, day, explicit, sign):
                    continue
                flows.append(ProjectedCashFlow(
                    date=day, amount=sign * scaled.amount, kind="recurring",
                    origin=f"{scaled.kind}:{scaled.group}", note=scaled.estimator))

        flows.extend(ctx.extra_outflows)
        flows.sort(key=lambda f: (f.date, 0 if f.amount < 0 else 1, f.origin))
        return tuple(flows), accrual

    def _superseded(self, stream: RecurringStream, day: date,
                    explicit: Sequence[tuple[ProjectedCashFlow, str]], sign: int) -> bool:
        """An explicit dated obligation beats an inferred occurrence of the same stream.

        The dataset shows this directly: a user with monthly insurance of 2 510 on the 6th also
        carries a *scheduled* insurance payment of 1 830 on the 11th. Those are one January
        obligation stated twice, not two payments. The explicit row wins because a confirmed
        amount is better evidence than an average - the same precedence the problem statement
        gives a settled event over an estimate.

        Scoped to discrete streams and to a half-cadence window, so at most one occurrence per
        explicit row can be suppressed, and continuously-accruing habits are untouched.
        """
        if stream.cadence_class != "monthly":
            return False                      # accruing habits are never superseded
        window = max(1, stream.cadence_days // 2)
        for flow, category in explicit:
            if (flow.amount < 0) != (sign < 0):
                continue
            if flow.note not in {"scheduled_obligation", "retry", "pending_debit"}:
                continue
            if category != stream.group:      # same economic stream only
                continue
            if abs((flow.date - day).days) <= window:
                return True
        return False

    # -------------------------------------------------------------- projection

    def project(self, ctx: ForecastContext) -> Forecast:
        cfg = self.config
        flows, accrual = self.flows_for(ctx)
        end = ctx.asof + timedelta(days=cfg.horizon_days)

        by_day: dict[date, list[ProjectedCashFlow]] = {}
        for f in flows:
            by_day.setdefault(f.date, []).append(f)

        balance = ctx.opening_balance
        path: list[DayPoint] = []
        day = ctx.asof
        first = True
        while day <= end:
            if not first:
                balance -= accrual          # no accrual on the snapshot date itself
            first = False
            todays = by_day.get(day, ())
            debits = sum((f.amount for f in todays if f.amount < 0), ZERO)
            credits = sum((f.amount for f in todays if f.amount > 0), ZERO)
            if cfg.same_day_order == "debits_first":
                balance += debits
                low = balance
                balance += credits
            else:
                balance += credits
                low = balance
                balance += debits
                low = min(low, balance)
            path.append(DayPoint(day, quantize(low, cfg.quantum),
                                 quantize(balance, cfg.quantum)))
            day += timedelta(days=1)

        trough_point = min(path, key=lambda p: (p.low, p.day))
        forecast = Forecast(
            opening_balance=ctx.opening_balance,
            minimum_balance=ctx.minimum_balance,
            flows=flows,
            path=tuple(path),
            trough=trough_point.low,
            trough_date=trough_point.day,
            daily_accrual=quantize(accrual, cfg.quantum),
            config_fingerprint=cfg.fingerprint(),
        )
        return forecast

    def project_traced(self, ctx: ForecastContext, trace: TraceLike = NULL_TRACE) -> Forecast:
        forecast = self.project(ctx)
        if trace:
            section = trace.section("forecast")
            section.add("summary", opening=str(forecast.opening_balance),
                        minimum=str(forecast.minimum_balance),
                        horizon_days=self.config.horizon_days,
                        daily_accrual=str(forecast.daily_accrual),
                        trough=str(forecast.trough), trough_date=forecast.trough_date,
                        reserve=str(forecast.reserve), safe=forecast.is_safe)
            for f in forecast.contributors():
                section.add("flow", date=f.date, amount=str(f.amount), kind=f.kind,
                            origin=f.origin, note=f.note)
        return forecast


def _rescaled(stream: RecurringStream, amount: Decimal) -> RecurringStream:
    from dataclasses import replace
    return replace(stream, amount=amount)


