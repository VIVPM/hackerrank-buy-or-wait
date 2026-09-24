"""`amount_safe_to_pay` and `earliest_date_for_full_payment`.

Both are pure measures of financial capacity. Neither consults the user's payment-method
preferences: a user who will not consider `full_payment` can still have an
`earliest_date_for_full_payment` of `request_date`, which the solved samples confirm.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Sequence

from bow.forecast import Forecast, ForecastContext, ForecastEngine
from bow.models import ProjectedCashFlow
from bow.money import quantize
from bow.trace import NULL_TRACE, TraceLike

ZERO = Decimal(0)


@dataclass(frozen=True, slots=True)
class SafeAmount:
    """Everything needed to explain, audit or debug the number."""

    amount: Decimal                  # the reported amount_safe_to_pay
    uncapped: Decimal                # before clamping to [0, requested]
    headroom: Decimal                # balance - minimum
    reserve: Decimal                 # balance - trough
    trough: Decimal
    trough_date: date
    minimum_balance: Decimal
    opening_balance: Decimal
    requested_amount: Decimal
    limiting_flows: tuple[ProjectedCashFlow, ...]

    @property
    def is_capped(self) -> bool:
        return self.amount == self.requested_amount

    def as_dict(self) -> dict[str, object]:
        return {
            "amount_safe_to_pay": str(self.amount),
            "uncapped": str(self.uncapped),
            "headroom": str(self.headroom),
            "forecast_reserve": str(self.reserve),
            "trough": str(self.trough),
            "trough_date": self.trough_date.isoformat(),
            "minimum_balance": str(self.minimum_balance),
            "opening_balance": str(self.opening_balance),
            "limiting_flows": [
                {"date": f.date.isoformat(), "amount": str(f.amount),
                 "kind": f.kind, "origin": f.origin, "note": f.note}
                for f in self.limiting_flows
            ],
        }


def safe_amount(engine: ForecastEngine, ctx: ForecastContext, requested: Decimal,
                forecast: Forecast | None = None,
                trace: TraceLike = NULL_TRACE) -> SafeAmount:
    """Largest amount payable on `asof` that keeps the whole horizon above the minimum.

    Paying X today shifts the entire projected path down by X, so the binding constraint is the
    trough:  X <= trough - minimum. Equivalently X <= headroom - reserve; both forms are computed
    and cross-checked.
    """
    fc = forecast or engine.project(ctx)
    headroom = fc.opening_balance - fc.minimum_balance
    # The binding floor for a payment made today, with the same intraday convention as
    # earliest_full_payment_date so the two measures cannot disagree.
    floor = fc.suffix_minimum_for_payment(ctx.asof)
    reserve = fc.opening_balance - floor
    uncapped = floor - fc.minimum_balance
    assert uncapped == headroom - reserve, "trough and headroom forms disagree"

    amount = quantize(max(ZERO, min(uncapped, requested)), engine.config.quantum)
    limiting = tuple(f for f in fc.flows if f.date <= fc.trough_date)
    result = SafeAmount(
        amount=amount, uncapped=quantize(uncapped, engine.config.quantum),
        headroom=headroom, reserve=reserve, trough=fc.trough, trough_date=fc.trough_date,
        minimum_balance=fc.minimum_balance, opening_balance=fc.opening_balance,
        requested_amount=requested, limiting_flows=limiting[-12:],
    )
    if trace:
        trace.section("safe_amount").add("result", **{k: v for k, v in result.as_dict().items()
                                                      if k != "limiting_flows"})
    assert ZERO <= result.amount <= requested, "amount_safe_to_pay left its bounds"
    return result


def earliest_full_payment_date(engine: ForecastEngine, ctx: ForecastContext,
                               requested: Decimal, forecast: Forecast | None = None,
                               trace: TraceLike = NULL_TRACE) -> date | None:
    """First date on which paying `requested` in full keeps the rest of the horizon safe.

    Paying on day d lowers the path from d onward by `requested` and leaves everything before d
    untouched, so the test is a suffix minimum of the *base* forecast:

        suffix_min(d) - requested >= minimum_balance

    That makes the search exact and linear - no re-projection per candidate date. Returns None
    when no date in the horizon qualifies, which is what an empty output field means.
    """
    fc = forecast or engine.project(ctx)
    cfg = engine.config
    end = ctx.asof + timedelta(days=cfg.horizon_days)
    day = ctx.asof
    while day <= end:
        if fc.suffix_minimum_for_payment(day) - requested >= fc.minimum_balance:
            if trace:
                trace.section("safe_amount").add(
                    "earliest_full_payment", date=day,
                    suffix_minimum=str(fc.suffix_minimum_for_payment(day)),
                    requested=str(requested))
            return day
        day += timedelta(days=1)
    if trace:
        trace.section("safe_amount").add(
            "earliest_full_payment", date=None,
            reason="full amount never safe inside the horizon",
            best_suffix_minimum=str(max(fc.suffix_minimum_for_payment(d)
                                        for d in _days(ctx.asof, end))))
    return None


def _days(start: date, end: date) -> Sequence[date]:
    out, day = [], start
    while day <= end:
        out.append(day)
        day += timedelta(days=1)
    return out
