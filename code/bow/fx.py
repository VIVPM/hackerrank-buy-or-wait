"""Deterministic exchange-rate service.

Only `dataset/exchange_rates.csv` is consulted. No live FX, ever.

RESOLUTION RULE (documented; see ARCHITECTURE.md section 12)
-----------------------------------------------------------
For a conversion of `from_ccy -> to_ccy` on date `d`, try in order and stop at the first hit:

  1. identity              from_ccy == to_ccy                      -> rate 1
  2. exact_date            supplied row (from, to, d)
  3. inverse_exact         supplied row (to, from, d), inverted
  4. carry_forward         latest supplied (from, to, x) with x <= d
  5. inverse_carry_forward latest supplied (to, from, x) with x <= d, inverted
  6. MissingRateError

Why this order:

* The problem statement says to "use the row for its settlement date and the stated from_currency
  to to_currency direction", so an exact, correctly-directed row is always preferred (step 2).
* Same-date information beats same-direction information, so an exact inverse (step 3) is
  preferred over a stale direct rate (step 4). The supplied pairs are not exact reciprocals
  (EUR->USD is 1.09 while USD->EUR is 0.92), so inversion is a genuine approximation and is always
  recorded in `ConvertedMoney.rate_source`.
* Steps 4 and 5 carry the last *published* rate forward. They never read a rate dated after `d`:
  using a future rate would be hindsight the user could not have had. This is the standard
  last-known-rate convention.
* Step 6 fails loudly. A missing rate is never silently treated as 1.0.

On this dataset steps 3-5 are defensive only: all 140 foreign-currency events resolve at step 2,
every required ordered pair is supplied directly, and each pair has a single constant rate.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Iterable, Sequence

from bow.errors import MissingRateError
from bow.money import CENT, ConvertedMoney, Money, RateSource, convert

ONE = Decimal(1)


@dataclass(frozen=True, slots=True)
class RateRow:
    rate_date: date
    from_currency: str
    to_currency: str
    rate: Decimal


class ExchangeRateService:
    """Immutable, deterministic. Build once per run and share."""

    def __init__(self, rows: Iterable[RateRow]) -> None:
        exact: dict[tuple[str, str, date], Decimal] = {}
        series: dict[tuple[str, str], list[tuple[date, Decimal]]] = {}
        for row in rows:
            key = (row.from_currency, row.to_currency)
            exact[(row.from_currency, row.to_currency, row.rate_date)] = row.rate
            series.setdefault(key, []).append((row.rate_date, row.rate))
        self._exact = exact
        self._series = {k: sorted(v) for k, v in series.items()}
        self._dates = {k: [d for d, _ in v] for k, v in self._series.items()}

    # ------------------------------------------------------------------ lookup

    def _carry_forward(self, frm: str, to: str, on: date) -> tuple[Decimal, date] | None:
        dates = self._dates.get((frm, to))
        if not dates:
            return None
        i = bisect.bisect_right(dates, on) - 1
        if i < 0:
            return None
        return self._series[(frm, to)][i][1], dates[i]

    def rate_for(self, frm: str, to: str, on: date) -> tuple[Decimal, date | None, RateSource]:
        """Resolve a rate. Raises MissingRateError rather than guessing."""
        if frm == to:
            return ONE, None, "identity"

        direct = self._exact.get((frm, to, on))
        if direct is not None:
            return direct, on, "exact_date"

        inverse = self._exact.get((to, frm, on))
        if inverse is not None:
            if inverse == 0:
                raise MissingRateError(f"zero rate for {to}->{frm} on {on}; cannot invert")
            return ONE / inverse, on, "inverse_exact"

        carried = self._carry_forward(frm, to, on)
        if carried is not None:
            return carried[0], carried[1], "carry_forward"

        carried_inv = self._carry_forward(to, frm, on)
        if carried_inv is not None:
            if carried_inv[0] == 0:
                raise MissingRateError(f"zero rate for {to}->{frm} at {carried_inv[1]}")
            return ONE / carried_inv[0], carried_inv[1], "inverse_carry_forward"

        raise MissingRateError(
            f"no supplied rate for {frm}->{to} on or before {on} "
            f"(and none for the inverse pair); refusing to assume a rate"
        )

    # ------------------------------------------------------------------ convert

    def to_home(self, amount: Money, home_currency: str, on: date,
                quantum: Decimal = CENT) -> ConvertedMoney:
        rate, rate_date, source = self.rate_for(amount.currency, home_currency, on)
        return convert(amount, home_currency, rate, rate_date, source, quantum)

    # ------------------------------------------------------------------ info

    @property
    def pairs(self) -> Sequence[tuple[str, str]]:
        return tuple(sorted(self._series))

    def __len__(self) -> int:
        return len(self._exact)
