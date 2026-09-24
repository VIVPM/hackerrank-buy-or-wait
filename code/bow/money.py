"""Decimal money. Binary floats are never used for monetary values anywhere in `bow`."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import Literal

CENT = Decimal("0.01")

# How a conversion rate was obtained. Ordered from most to least authoritative; see fx.py.
RateSource = Literal[
    "identity",                # same currency, rate 1
    "exact_date",              # supplied row for (from, to, date)
    "inverse_exact",           # supplied row for (to, from, date), inverted
    "carry_forward",           # latest supplied (from, to, d) with d <= date
    "inverse_carry_forward",   # latest supplied (to, from, d) with d <= date, inverted
]


def to_decimal(value: object) -> Decimal:
    """Parse a CSV cell into Decimal. Rejects float to keep binary error out of the system."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):  # ponytail: refuse rather than silently round-trip through binary
        raise TypeError("construct Money from str/int/Decimal, not float")
    if isinstance(value, int):
        return Decimal(value)
    text = str(value).strip()
    if not text:
        raise ValueError("empty monetary value")
    return Decimal(text)


def quantize(value: Decimal, quantum: Decimal = CENT) -> Decimal:
    return value.quantize(quantum, rounding=ROUND_HALF_UP)


@dataclass(frozen=True, slots=True)
class Money:
    amount: Decimal
    currency: str

    def __post_init__(self) -> None:
        if not isinstance(self.amount, Decimal):
            raise TypeError("Money.amount must be Decimal")
        if not self.currency or len(self.currency) != 3:
            raise ValueError(f"bad currency: {self.currency!r}")

    def __add__(self, other: "Money") -> "Money":
        self._same_currency(other)
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: "Money") -> "Money":
        self._same_currency(other)
        return Money(self.amount - other.amount, self.currency)

    def _same_currency(self, other: "Money") -> None:
        if self.currency != other.currency:
            raise ValueError(f"currency mismatch: {self.currency} vs {other.currency}")


@dataclass(frozen=True, slots=True)
class ConvertedMoney:
    """A home-currency amount that remembers exactly how it was derived."""

    source: Money
    target_currency: str
    rate: Decimal
    rate_date: date | None
    rate_source: RateSource
    converted: Decimal

    @property
    def is_converted(self) -> bool:
        return self.source.currency != self.target_currency


def convert(source: Money, target_currency: str, rate: Decimal,
            rate_date: date | None, rate_source: RateSource,
            quantum: Decimal = CENT) -> ConvertedMoney:
    return ConvertedMoney(
        source=source,
        target_currency=target_currency,
        rate=rate,
        rate_date=rate_date,
        rate_source=rate_source,
        converted=quantize(source.amount * rate, quantum),
    )


def format_amount(value: Decimal) -> str:
    """Render for output.csv: drop a trailing .00, otherwise keep two decimals.

    Matches the solved samples, which show both `2024-03-03:25256` and `2026-01-03:620.40`.
    """
    q = quantize(value)
    return str(q.to_integral_value()) if q == q.to_integral_value() else f"{q:.2f}"


def _demo() -> None:
    usd = Money(to_decimal("1800"), "USD")
    idr = convert(usd, "IDR", Decimal("15833.33"), date(2024, 3, 15), "exact_date")
    assert idr.converted == Decimal("28499994.00"), idr.converted
    assert idr.is_converted
    assert format_amount(Decimal("25256.00")) == "25256"
    assert format_amount(Decimal("620.4")) == "620.40"
    assert quantize(Decimal("0.005")) == Decimal("0.01")
    try:
        Money(0.1, "EUR")  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("float must be rejected")
    try:
        _ = Money(to_decimal("1"), "EUR") + Money(to_decimal("1"), "USD")
    except ValueError:
        pass
    else:
        raise AssertionError("currency mismatch must be rejected")
    print("money: ok")


if __name__ == "__main__":
    _demo()
