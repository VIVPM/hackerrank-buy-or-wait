"""Typed loading and indexing of the eight participant CSVs.

Input files are opened read-only and never written. Every monetary field becomes Decimal and every
date becomes `datetime.date`; a malformed required field raises `DatasetError` naming the file,
row and column rather than being coerced to a default.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from bow.config import RunConfig
from bow.errors import DatasetError
from bow.fx import ExchangeRateService, RateRow
from bow.models import (
    ExpectedOutput,
    FinancialProfile,
    PaymentOption,
    RawFinancialEvent,
    RawImage,
    RawMessage,
    Request,
)
from bow.money import Money

# ---------------------------------------------------------------- field parsers


def _require(row: Mapping[str, str], col: str, file: str, line: int) -> str:
    if col not in row:
        raise DatasetError(f"{file}:{line}: missing required column {col!r}")
    value = (row[col] or "").strip()
    if not value:
        raise DatasetError(f"{file}:{line}: required column {col!r} is empty")
    return value


def _optional(row: Mapping[str, str], col: str) -> str | None:
    value = (row.get(col) or "").strip()
    return value or None


def _parse_date(text: str, file: str, line: int, col: str) -> date:
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise DatasetError(f"{file}:{line}: column {col!r} is not YYYY-MM-DD: {text!r}") from exc


def _parse_decimal(text: str, file: str, line: int, col: str) -> Decimal:
    try:
        return Decimal(text)
    except InvalidOperation as exc:
        raise DatasetError(f"{file}:{line}: column {col!r} is not numeric: {text!r}") from exc


def _parse_bool(text: str, file: str, line: int, col: str) -> bool:
    lowered = text.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise DatasetError(f"{file}:{line}: column {col!r} is not a boolean: {text!r}")


def _split_set(text: str | None) -> frozenset[str]:
    return frozenset(p.strip() for p in text.split("|") if p.strip()) if text else frozenset()


def _split_tuple(text: str | None) -> tuple[str, ...]:
    return tuple(p.strip() for p in text.split("|") if p.strip()) if text else ()


def _rows(path: Path, required: Sequence[str]) -> Iterator[tuple[int, dict[str, str]]]:
    if not path.is_file():
        raise DatasetError(f"missing dataset file: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DatasetError(f"{path.name}: file has no header row")
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise DatasetError(f"{path.name}: missing columns {missing}")
        for line, row in enumerate(reader, start=2):
            yield line, row


def _check_unique(ids: Sequence[str], file: str, col: str) -> None:
    seen: set[str] = set()
    for value in ids:
        if value in seen:
            raise DatasetError(f"{file}: duplicate {col} {value!r}")
        seen.add(value)


# ---------------------------------------------------------------- loaders


def load_profiles(path: Path) -> tuple[FinancialProfile, ...]:
    f = path.name
    out: list[FinancialProfile] = []
    cols = ("user_id", "home_currency", "current_available_balance", "minimum_balance_to_keep")
    for line, row in _rows(path, cols):
        months = _optional(row, "max_installment_months")
        out.append(
            FinancialProfile(
                user_id=_require(row, "user_id", f, line),
                home_currency=_require(row, "home_currency", f, line),
                current_available_balance=_parse_decimal(
                    _require(row, "current_available_balance", f, line), f, line,
                    "current_available_balance"),
                minimum_balance_to_keep=_parse_decimal(
                    _require(row, "minimum_balance_to_keep", f, line), f, line,
                    "minimum_balance_to_keep"),
                financial_priorities=_split_tuple(_optional(row, "financial_priorities")),
                protected_categories=_split_set(_optional(row, "expense_categories_to_protect")),
                reducible_categories=_split_set(
                    _optional(row, "expense_categories_user_is_willing_to_reduce")),
                stoppable_categories=_split_set(
                    _optional(row, "expense_categories_user_is_willing_to_stop")),
                methods=_split_set(_optional(row, "payment_methods_user_will_consider")),
                max_installment_months=int(Decimal(months)) if months else None,
            )
        )
    _check_unique([p.user_id for p in out], f, "user_id")
    return tuple(out)


def _request_from_row(row: Mapping[str, str], f: str, line: int) -> Request:
    return Request(
        request_id=_require(row, "request_id", f, line),
        user_id=_require(row, "user_id", f, line),
        request_date=_parse_date(_require(row, "request_date", f, line), f, line, "request_date"),
        request_type=_require(row, "request_type", f, line),
        requested_amount=_parse_decimal(
            _require(row, "requested_amount", f, line), f, line, "requested_amount"),
        desired_completion_date=_parse_date(
            _require(row, "desired_completion_date", f, line), f, line,
            "desired_completion_date"),
        allows_partial_payment=_parse_bool(
            _require(row, "allows_partial_payment", f, line), f, line, "allows_partial_payment"),
        request_text=row.get("request_text", "") or "",
    )


REQUEST_COLS = (
    "request_id", "user_id", "request_date", "request_type", "requested_amount",
    "desired_completion_date", "allows_partial_payment",
)


def load_requests(path: Path) -> tuple[Request, ...]:
    out = [_request_from_row(row, path.name, line) for line, row in _rows(path, REQUEST_COLS)]
    _check_unique([r.request_id for r in out], path.name, "request_id")
    return tuple(out)


def load_samples(path: Path) -> tuple[tuple[Request, ...], tuple[ExpectedOutput, ...]]:
    """sample_requests.csv = the same input columns plus the solved output columns."""
    f = path.name
    requests: list[Request] = []
    expected: list[ExpectedOutput] = []
    for line, row in _rows(path, REQUEST_COLS + ("amount_safe_to_pay", "affordability_status")):
        requests.append(_request_from_row(row, f, line))
        expected.append(
            ExpectedOutput(
                request_id=row["request_id"].strip(),
                amount_safe_to_pay=_parse_decimal(
                    _require(row, "amount_safe_to_pay", f, line), f, line, "amount_safe_to_pay"),
                affordability_status=_require(row, "affordability_status", f, line),
                recommended_payment_method=_require(row, "recommended_payment_method", f, line),
                payment_plan=(row.get("payment_plan") or "").strip(),
                earliest_date_for_full_payment=(
                    row.get("earliest_date_for_full_payment") or "").strip(),
                spending_changes_needed=(row.get("spending_changes_needed") or "").strip(),
                decision_explanation=(row.get("decision_explanation") or "").strip(),
            )
        )
    _check_unique([r.request_id for r in requests], f, "request_id")
    return tuple(requests), tuple(expected)


def load_events(path: Path) -> tuple[RawFinancialEvent, ...]:
    f = path.name
    out: list[RawFinancialEvent] = []
    cols = ("event_id", "user_id", "event_type", "category", "direction", "currency",
            "event_date", "status")
    for line, row in _rows(path, cols):
        currency = _require(row, "currency", f, line)
        raw_amount = _optional(row, "amount")
        # A blank amount is legitimate only when an image supplies the value. Never zero.
        amount = (
            Money(_parse_decimal(raw_amount, f, line, "amount"), currency)
            if raw_amount is not None else None
        )
        settlement = _optional(row, "settlement_date")
        minimum = _optional(row, "minimum_allowed_amount")
        direction = _require(row, "direction", f, line)
        if direction not in {"debit", "credit", "non_cash"}:
            raise DatasetError(f"{f}:{line}: unknown direction {direction!r}")
        status = _require(row, "status", f, line)
        if status not in {"settled", "pending", "scheduled", "cancelled", "failed", "unrealized"}:
            raise DatasetError(f"{f}:{line}: unknown status {status!r}")
        flexibility = _optional(row, "flexibility") or "fixed"
        if flexibility not in {"fixed", "reducible", "stoppable", "reducible_or_stoppable"}:
            raise DatasetError(f"{f}:{line}: unknown flexibility {flexibility!r}")
        out.append(
            RawFinancialEvent(
                event_id=_require(row, "event_id", f, line),
                user_id=_require(row, "user_id", f, line),
                event_type=_require(row, "event_type", f, line),
                description=(row.get("description") or "").strip(),
                category=_require(row, "category", f, line),
                direction=direction,                                    # type: ignore[arg-type]
                amount=amount,
                event_date=_parse_date(
                    _require(row, "event_date", f, line), f, line, "event_date"),
                settlement_date=(
                    _parse_date(settlement, f, line, "settlement_date") if settlement else None),
                status=status,                                          # type: ignore[arg-type]
                linked_event_id=_optional(row, "linked_event_id"),
                flexibility=flexibility,                                # type: ignore[arg-type]
                minimum_allowed_amount=(
                    _parse_decimal(minimum, f, line, "minimum_allowed_amount")
                    if minimum else None),
                source_row=line,
            )
        )
    _check_unique([e.event_id for e in out], f, "event_id")
    return tuple(out)


def load_options(path: Path) -> tuple[PaymentOption, ...]:
    f = path.name
    out: list[PaymentOption] = []
    cols = ("payment_option_id", "request_id", "payment_method", "payment_amount",
            "number_of_payments", "first_payment_date", "total_payable_amount")
    for line, row in _rows(path, cols):
        method = _require(row, "payment_method", f, line)
        if method not in {"full_payment", "installments"}:
            raise DatasetError(f"{f}:{line}: unknown payment_method {method!r}")
        amount = _parse_decimal(
            _require(row, "payment_amount", f, line), f, line, "payment_amount")
        count = int(_parse_decimal(
            _require(row, "number_of_payments", f, line), f, line, "number_of_payments"))
        if count < 1:
            raise DatasetError(f"{f}:{line}: number_of_payments must be >= 1")
        first = _parse_date(
            _require(row, "first_payment_date", f, line), f, line, "first_payment_date")
        freq_raw = _optional(row, "payment_frequency_days")
        freq = int(_parse_decimal(freq_raw, f, line, "payment_frequency_days")) if freq_raw else None
        if count > 1 and freq is None:
            raise DatasetError(f"{f}:{line}: multi-payment option without payment_frequency_days")
        schedule = tuple(
            (first + _days(i * (freq or 0)), amount) for i in range(count)
        )
        fee_raw = _optional(row, "financing_fee")
        out.append(
            PaymentOption(
                payment_option_id=_require(row, "payment_option_id", f, line),
                request_id=_require(row, "request_id", f, line),
                payment_method=method,                                  # type: ignore[arg-type]
                payment_amount=amount,
                number_of_payments=count,
                first_payment_date=first,
                payment_frequency_days=freq,
                financing_fee=(
                    _parse_decimal(fee_raw, f, line, "financing_fee") if fee_raw else Decimal(0)),
                total_payable_amount=_parse_decimal(
                    _require(row, "total_payable_amount", f, line), f, line,
                    "total_payable_amount"),
                schedule=schedule,
                last_payment_date=schedule[-1][0],
            )
        )
    _check_unique([o.payment_option_id for o in out], f, "payment_option_id")
    return tuple(out)


def _days(n: int):
    from datetime import timedelta
    return timedelta(days=n)


def load_messages(path: Path) -> tuple[RawMessage, ...]:
    f = path.name
    out = [
        RawMessage(
            message_id=_require(row, "message_id", f, line),
            user_id=_require(row, "user_id", f, line),
            request_id=_optional(row, "request_id"),
            related_event_id=_optional(row, "related_event_id"),
            sent_at=(row.get("sent_at") or "").strip(),
            source_type=(row.get("source_type") or "").strip(),
            message_text=row.get("message_text", "") or "",
        )
        for line, row in _rows(path, ("message_id", "user_id", "message_text"))
    ]
    _check_unique([m.message_id for m in out], f, "message_id")
    return tuple(out)


def load_images(path: Path, media_dir: Path) -> tuple[RawImage, ...]:
    f = path.name
    out = [
        RawImage(
            image_id=_require(row, "image_id", f, line),
            user_id=_require(row, "user_id", f, line),
            request_id=_optional(row, "request_id"),
            related_event_id=_require(row, "related_event_id", f, line),
            path=str(media_dir / f"{_require(row, 'image_id', f, line)}.png"),
        )
        for line, row in _rows(path, ("image_id", "user_id", "related_event_id"))
    ]
    _check_unique([i.image_id for i in out], f, "image_id")
    return tuple(out)


def load_rates(path: Path) -> tuple[RateRow, ...]:
    f = path.name
    out: list[RateRow] = []
    seen: set[tuple[str, str, date]] = set()
    for line, row in _rows(path, ("rate_date", "from_currency", "to_currency", "rate")):
        rd = _parse_date(_require(row, "rate_date", f, line), f, line, "rate_date")
        frm = _require(row, "from_currency", f, line)
        to = _require(row, "to_currency", f, line)
        if (frm, to, rd) in seen:
            raise DatasetError(f"{f}:{line}: duplicate rate for {frm}->{to} on {rd}")
        seen.add((frm, to, rd))
        rate = _parse_decimal(_require(row, "rate", f, line), f, line, "rate")
        if rate <= 0:
            raise DatasetError(f"{f}:{line}: non-positive rate {rate}")
        out.append(RateRow(rate_date=rd, from_currency=frm, to_currency=to, rate=rate))
    return tuple(out)


# ---------------------------------------------------------------- request context


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Everything one request needs, and nothing else. No LLM involvement."""

    request: Request
    profile: FinancialProfile
    events: tuple[RawFinancialEvent, ...]
    messages: tuple[RawMessage, ...]
    images: tuple[RawImage, ...]
    options: tuple[PaymentOption, ...]
    fx: ExchangeRateService
    _event_by_id: Mapping[str, RawFinancialEvent]

    @property
    def home_currency(self) -> str:
        return self.profile.home_currency

    def event(self, event_id: str) -> RawFinancialEvent | None:
        return self._event_by_id.get(event_id)

    def linked(self, event: RawFinancialEvent) -> RawFinancialEvent | None:
        return self.event(event.linked_event_id) if event.linked_event_id else None

    def messages_for_event(self, event_id: str) -> tuple[RawMessage, ...]:
        return tuple(m for m in self.messages if m.related_event_id == event_id)

    def images_for_event(self, event_id: str) -> tuple[RawImage, ...]:
        return tuple(i for i in self.images if i.related_event_id == event_id)

    def to_home(self, amount: Money, on: date):
        return self.fx.to_home(amount, self.profile.home_currency, on)


# ---------------------------------------------------------------- dataset


@dataclass(frozen=True, slots=True)
class Dataset:
    profiles: Mapping[str, FinancialProfile]
    requests: Mapping[str, Request]                  # eval + sample, one namespace
    expected: Mapping[str, ExpectedOutput]           # solved samples only
    eval_request_ids: tuple[str, ...]
    sample_request_ids: tuple[str, ...]
    events_by_user: Mapping[str, tuple[RawFinancialEvent, ...]]
    event_by_id: Mapping[str, RawFinancialEvent]
    messages_by_user: Mapping[str, tuple[RawMessage, ...]]
    messages_by_request: Mapping[str, tuple[RawMessage, ...]]
    messages_by_event: Mapping[str, tuple[RawMessage, ...]]
    images_by_user: Mapping[str, tuple[RawImage, ...]]
    images_by_request: Mapping[str, tuple[RawImage, ...]]
    images_by_event: Mapping[str, tuple[RawImage, ...]]
    options_by_request: Mapping[str, tuple[PaymentOption, ...]]
    fx: ExchangeRateService

    # -------------------------------------------------------------- access

    @property
    def requests_by_id(self) -> Mapping[str, Request]:
        return self.requests

    @property
    def profile_by_user(self) -> Mapping[str, FinancialProfile]:
        return self.profiles

    def context_for(self, request_id: str) -> RequestContext:
        request = self.requests.get(request_id)
        if request is None:
            raise DatasetError(f"unknown request_id {request_id!r}")
        profile = self.profiles.get(request.user_id)
        if profile is None:
            raise DatasetError(f"request {request_id!r} references unknown user {request.user_id!r}")
        user_messages = self.messages_by_user.get(request.user_id, ())
        user_images = self.images_by_user.get(request.user_id, ())
        return RequestContext(
            request=request,
            profile=profile,
            events=self.events_by_user.get(request.user_id, ()),
            # user-level evidence plus anything scoped to this request; a message scoped to a
            # different request of the same user cannot exist (one request per user) but the
            # filter keeps that guarantee local rather than assumed.
            messages=tuple(
                m for m in user_messages
                if m.request_id in (None, request_id)
            ),
            images=tuple(
                i for i in user_images
                if i.request_id in (None, request_id)
            ),
            options=self.options_by_request.get(request_id, ()),
            fx=self.fx,
            _event_by_id=self.event_by_id,
        )

    # -------------------------------------------------------------- loading

    @classmethod
    def load(cls, config: RunConfig | None = None) -> "Dataset":
        cfg = config or RunConfig()
        root = cfg.dataset_dir
        profiles = load_profiles(root / "financial_profiles.csv")
        eval_requests = load_requests(root / "requests.csv")
        sample_requests, expected = load_samples(root / "sample_requests.csv")
        events = load_events(root / "financial_events.csv")
        options = load_options(root / "request_payment_options.csv")
        messages = load_messages(root / "messages.csv")
        images = load_images(root / "images.csv", root / "media" / "images")
        rates = load_rates(root / "exchange_rates.csv")

        overlap = {r.request_id for r in eval_requests} & {r.request_id for r in sample_requests}
        if overlap:
            raise DatasetError(f"request_id present in both requests and samples: {sorted(overlap)}")

        by_user: dict[str, list[RawFinancialEvent]] = {}
        for event in events:
            by_user.setdefault(event.user_id, []).append(event)
        for user_events in by_user.values():
            user_events.sort(key=lambda e: (e.settlement_date or e.event_date, e.event_id))

        return cls(
            profiles={p.user_id: p for p in profiles},
            requests={r.request_id: r for r in (*eval_requests, *sample_requests)},
            expected={e.request_id: e for e in expected},
            eval_request_ids=tuple(r.request_id for r in eval_requests),
            sample_request_ids=tuple(r.request_id for r in sample_requests),
            events_by_user={u: tuple(v) for u, v in by_user.items()},
            event_by_id={e.event_id: e for e in events},
            messages_by_user=_group(messages, lambda m: m.user_id),
            messages_by_request=_group(messages, lambda m: m.request_id),
            messages_by_event=_group(messages, lambda m: m.related_event_id),
            images_by_user=_group(images, lambda i: i.user_id),
            images_by_request=_group(images, lambda i: i.request_id),
            images_by_event=_group(images, lambda i: i.related_event_id),
            options_by_request=_group(options, lambda o: o.request_id),
            fx=ExchangeRateService(rates),
        )


def _group(items, key):
    out: dict[str, list] = {}
    for item in items:
        k = key(item)
        if k is not None:
            out.setdefault(k, []).append(item)
    return {k: tuple(v) for k, v in out.items()}
