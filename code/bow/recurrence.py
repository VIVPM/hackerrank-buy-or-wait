"""Recurring-series inference.

Two rules carry most of the weight, both established in the forensic phase (see log.txt):

1. Expenses group by `category`; income groups by `description`. Lumping every salary-category
   credit together merges base pay with commissions and inflates projected income enormously.
2. Income is projected only when the description says the employment is continuing. Gig payouts,
   freelance invoices, commissions, bonuses, arrears and prizes are never projected forward, and a
   terminal payroll row ends projection entirely. Anything unrecognised is not projected, because
   inventing income is the one error the problem statement forbids outright.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, replace
from datetime import date, timedelta
from decimal import Decimal
from typing import Iterable, Literal, Sequence

from bow.config import ForecastConfig
from bow.models import (
    CanonicalFinancialEvent,
    EvidenceBundle,
    ImageFact,
    MessageFact,
    RecurringStream,
    StreamKind,
)
from bow.money import quantize
from bow.resolve import ResolutionResult, StreamDirective
from bow.trace import NULL_TRACE, TraceLike

# ---------------------------------------------------------------- income taxonomy

IncomeClass = Literal["continuing", "terminal", "one_off", "irregular", "unknown"]

#: Token tests, in priority order. Matching on tokens rather than exact strings means an unseen
#: description degrades to "unknown" (never projected) instead of being silently treated as salary.
_TERMINAL_TOKENS = ("final employer", "previous employer", "before leave")
_ONE_OFF_TOKENS = ("bonus", "commission", "arrears", "prize", "reimbursement")
_IRREGULAR_TOKENS = (
    "payout", "app earnings", "marketplace", "invoice payment", "contract payment",
    "project payment", "milestone", "retainer", "independent work", "seasonal",
    "peak-season", "temporary assignment",
)
_EMPLOYMENT_TOKENS = ("payroll", "salary", "household income", "wages")


#: Label used to group income events into streams.
#:
#: Payroll descriptions are meaningful - "Base salary" and "Monthly sales commission" are
#: genuinely different streams - so continuing income keeps its description. Freelance and gig
#: labels vary per payment the way grocery descriptions do ("Freelance milestone payment",
#: "Client retainer payment", "Website project payment"), so they are one stream. Grouping them
#: by description hides a perfectly regular income behind ten one-observation series.
IRREGULAR_GROUP = "irregular income"


def _income_group(description: str) -> str | None:
    klass = classify_income(description)
    if klass == "irregular":
        return IRREGULAR_GROUP
    if klass in {"one_off", "unknown"}:
        return None            # bonuses, commissions, arrears and prizes never form a stream
    return description


def _group_class(group: str) -> IncomeClass:
    return "irregular" if group == IRREGULAR_GROUP else classify_income(group)


def classify_income(description: str) -> IncomeClass:
    text = description.strip().lower()
    if any(t in text for t in _TERMINAL_TOKENS):
        return "terminal"
    if any(t in text for t in _ONE_OFF_TOKENS):
        return "one_off"
    if any(t in text for t in _IRREGULAR_TOKENS):
        return "irregular"
    if any(t in text for t in _EMPLOYMENT_TOKENS):
        return "continuing"
    return "unknown"


# ---------------------------------------------------------------- estimators


def estimate_amount(values: Sequence[Decimal], config: ForecastConfig,
                    dates: Sequence[date] | None = None,
                    asof: date | None = None) -> Decimal:
    if not values:
        raise ValueError("cannot estimate from an empty series")
    tail = list(values)
    mode = config.amount_estimator
    if mode == "trailing_horizon":
        if dates and asof:
            cutoff = asof - timedelta(days=config.horizon_days)
            recent = [v for v, d in zip(values, dates) if d >= cutoff]
            if recent:
                tail = recent
        return quantize(sum(tail, Decimal(0)) / Decimal(len(tail)), config.quantum)
    if mode == "last":
        result = tail[-1]
    elif mode == "mean_3":
        result = sum(tail[-3:], Decimal(0)) / Decimal(len(tail[-3:]))
    elif mode == "mean_6":
        result = sum(tail[-6:], Decimal(0)) / Decimal(len(tail[-6:]))
    elif mode == "median_6":
        result = Decimal(str(statistics.median(tail[-6:])))
    else:  # mean_all - the forensic default
        result = sum(tail, Decimal(0)) / Decimal(len(tail))
    return quantize(result, config.quantum)


def _cadence(dates: Sequence[date], config: ForecastConfig) -> int:
    gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
    if not gaps:
        return 30
    value = statistics.mean(gaps) if config.cadence_rule == "mean_gap" else statistics.median(gaps)
    return max(1, int(round(value)))


def add_month(anchor: date, months: int, day_of_month: int) -> date:
    """Calendar-month step that does not drift. Clamps to the end of short months."""
    total = anchor.month - 1 + months
    year = anchor.year + total // 12
    month = total % 12 + 1
    last = [31, 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28,
            31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
    return date(year, month, min(day_of_month, last))


# ---------------------------------------------------------------- inference


@dataclass(frozen=True, slots=True)
class StreamSet:
    streams: tuple[RecurringStream, ...]
    skipped: tuple[tuple[str, str], ...]      # (group, why) - for the trace

    def by_key(self, key: tuple[StreamKind, str]) -> RecurringStream | None:
        for s in self.streams:
            if s.key == key:
                return s
        return None

    @property
    def active(self) -> tuple[RecurringStream, ...]:
        return tuple(s for s in self.streams if not s.is_stale)


def infer_streams(events: Sequence[CanonicalFinancialEvent], asof: date,
                  config: ForecastConfig, trace: TraceLike = NULL_TRACE) -> StreamSet:
    seeds = [e for e in events if e.counts_for_recurrence
             and e.cash_effect is not None and e.effective_date is not None]
    groups: dict[tuple[StreamKind, str], list[CanonicalFinancialEvent]] = {}
    for e in seeds:
        kind: StreamKind = "income" if e.direction == "credit" else "expense"
        group = _income_group(e.description) if kind == "income" else e.category
        if group is None:
            continue
        groups.setdefault((kind, group), []).append(e)

    # Employment continuity is decided once per user, from the most recent employment row.
    employment = [e for e in seeds if e.direction == "credit"
                  and classify_income(e.description) in {"continuing", "terminal"}]
    employment.sort(key=lambda e: (e.effective_date or asof, e.economic_id))
    latest_employment = employment[-1] if employment else None
    employment_ended = (latest_employment is not None
                        and classify_income(latest_employment.description) == "terminal")

    streams: list[RecurringStream] = []
    skipped: list[tuple[str, str]] = []

    for key, members in sorted(groups.items(), key=lambda kv: kv[0]):
        kind, group = key
        members.sort(key=lambda e: (e.effective_date or asof, e.economic_id))
        dates: list[date] = [e.effective_date for e in members if e.effective_date]

        if kind == "income":
            klass = _group_class(group)
            if employment_ended and klass == "continuing":
                skipped.append((group, "employment ended; no income projected"))
                continue
            if klass not in {"continuing", "irregular"}:
                skipped.append((group, f"income class {klass!r} is never projected forward"))
                continue
            recent = (latest_employment.effective_date
                      if latest_employment and klass == "continuing" else None)
            if recent is not None and (recent - dates[-1]).days > 45:
                skipped.append((group, "superseded by a more recent employment stream"))
                continue

        if len(dates) < config.min_observations:
            skipped.append((group, f"only {len(dates)} observations; recurrence not supported"))
            continue

        cadence = _cadence(dates, config)
        amount = estimate_amount([e.cash_effect.converted for e in members], config,
                                 dates, asof)
        monthly = cadence >= config.monthly_threshold_days
        stale_after = Decimal(cadence) * config.income_staleness_cycles
        is_stale = Decimal((asof - dates[-1]).days) > stale_after

        streams.append(RecurringStream(
            key=key, kind=kind, group=group,
            cadence_days=cadence,
            cadence_class="monthly" if monthly else "sub_monthly",
            anchor_date=dates[-1],
            anchor_day_of_month=dates[-1].day if monthly else None,
            amount=amount,
            estimator=config.amount_estimator,
            latest_event_id=members[-1].economic_id,
            flexibility=members[-1].flexibility,
            minimum_allowed_amount=members[-1].minimum_allowed_amount,
            observations=len(members),
            is_stale=is_stale,
            income_class=_group_class(group) if kind == "income" else "",
        ))
        if is_stale:
            skipped.append((group, f"stale: last seen {dates[-1]}, "
                                   f"{(asof - dates[-1]).days}d ago vs cadence {cadence}d"))

    streams, anchor_note = _anchor_on_confirmed_income(streams, events, asof, config)
    if anchor_note:
        skipped.append(anchor_note)

    result = StreamSet(tuple(streams), tuple(skipped))
    if trace:
        section = trace.section("recurrence")
        section.add("summary", streams=len(streams), active=len(result.active),
                    employment_ended=employment_ended,
                    latest_employment=(latest_employment.description
                                       if latest_employment else None))
        for s in streams:
            section.add("stream", key=list(s.key), cadence=s.cadence_days,
                        cadence_class=s.cadence_class, amount=str(s.amount),
                        anchor=s.anchor_date, observations=s.observations, stale=s.is_stale,
                        estimator=s.estimator)
        for group, why in skipped:
            section.add("skipped", group=group, reason=why)
    return result


def _anchor_on_confirmed_income(streams: list[RecurringStream],
                                events: Sequence[CanonicalFinancialEvent], asof: date,
                                config: ForecastConfig):
    """An explicitly scheduled future salary anchors the income projection.

    `Next confirmed salary` is the employer stating the next amount and date outright, which
    beats an average over history - the same precedence the problem statement gives a settled
    event over an estimate. Anchoring on it also removes a double-count: the explicit row is the
    first occurrence, so the projected series starts one cadence later and never lands on it.

    Where a user has no usable salary history at all, this is the only evidence income continues,
    so it creates the stream outright rather than leaving the user with no projected income.
    """
    confirmed = [e for e in events
                 if e.lifecycle_role == "confirmed_income" and e.counts_for_cash
                 and e.cash_effect is not None and e.effective_date is not None]
    if not confirmed:
        return streams, None
    confirmed.sort(key=lambda e: e.effective_date)   # type: ignore[arg-type,return-value]
    anchor = confirmed[0]
    assert anchor.effective_date is not None and anchor.cash_effect is not None
    amount = quantize(anchor.cash_effect.converted, config.quantum)

    income = [s for s in streams if s.kind == "income"]
    if income:
        target = max(income, key=lambda s: s.anchor_date)
        rebuilt = replace(
            target, anchor_date=anchor.effective_date,
            anchor_day_of_month=anchor.effective_date.day,
            amount=amount, estimator=f"{target.estimator}+confirmed",
            latest_event_id=anchor.economic_id, cadence_class="monthly",
            income_class=target.income_class or "continuing",
            cadence_days=max(target.cadence_days, config.monthly_threshold_days),
            is_stale=False)
        return ([rebuilt if s.key == target.key else s for s in streams],
                (target.group, f"anchored on confirmed income {anchor.economic_id}"))

    created = RecurringStream(
        key=("income", anchor.description), kind="income", group=anchor.description,
        cadence_days=30, cadence_class="monthly",
        anchor_date=anchor.effective_date, anchor_day_of_month=anchor.effective_date.day,
        amount=amount, estimator="confirmed", latest_event_id=anchor.economic_id,
        flexibility="fixed", minimum_allowed_amount=None, observations=0, is_stale=False,
        income_class="continuing")
    return (streams + [created],
            (anchor.description, f"income stream created from confirmed {anchor.economic_id}"))


# ---------------------------------------------------------------- override hooks
# Prompt 6 supplies the facts; these are the application points, exercised by synthetic facts
# in the tests so the wiring is proven before any model call.


def apply_image_facts(events: Sequence[CanonicalFinancialEvent],
                      facts: Sequence[ImageFact],
                      home_currency: str, fx) -> tuple[CanonicalFinancialEvent, ...]:
    """Replace unresolved amounts with image-extracted values.

    A resolved amount restores `counts_for_cash` for a future obligation. It never restores
    `counts_for_recurrence`: an image-sourced one-off must not enter a series.
    """
    by_event = {f.related_event_id: f for f in facts}
    out: list[CanonicalFinancialEvent] = []
    for e in events:
        fact = by_event.get(e.source_event_ids[0])
        if fact is None or e.amount_status != "unresolved_image":
            out.append(e)
            continue
        converted = fx.to_home(fact.chosen_amount, home_currency,
                              e.effective_date or fact.chosen_amount)
        restores_cash = e.status in {"pending", "scheduled"}
        out.append(replace(
            e,
            cash_effect=converted,
            amount_status="known",
            counts_for_cash=restores_cash,
            counts_for_recurrence=False,
            exclusion_reason=None if restores_cash else "image_derived",
            resolution_note=(f"amount {fact.chosen_amount.amount} supplied by {fact.image_id} "
                             f"as {fact.semantic_label}"),
            evidence_ids=e.evidence_ids + (fact.image_id,),
        ))
    return tuple(out)


def apply_stream_directives(streams: StreamSet, directives: Sequence[StreamDirective],
                            trace: TraceLike = NULL_TRACE) -> StreamSet:
    """Apply message-derived stream edits. Unquantified facts change nothing, by design."""
    current = list(streams.streams)
    notes = list(streams.skipped)
    for directive in directives:
        fact = directive.fact
        op = directive.op
        target = _match_stream(current, fact)
        if op in {"employment_end", "income_unconfirmed"}:
            classes = _suppressed_classes(fact.subject)
            current = [s for s in current
                       if not (s.kind == "income" and s.income_class in classes)]
            notes.append((fact.subject,
                          f"{op} from {fact.message_id}: {'/'.join(sorted(classes))} "
                          f"income no longer projected"))
        elif op == "salary_change" and fact.amount is not None and target is not None:
            current = _replace_stream(current, target, amount=fact.amount.amount,
                                      first_occurrence=fact.effective_date)
        elif (op == "salary_change" and fact.amount is not None
                and fact.effective_date is not None and target is None):
            # A quantified, dated salary for a user with no active income stream IS the
            # evidence that income starts - e.g. a first salary, or pay resuming after leave,
            # where the history holds only a terminal "payroll before leave" row.
            current = current + [_synthetic_income(fact)]
            notes.append((fact.subject, f"income created from {fact.message_id}"))
        elif op == "amend_date" and fact.effective_date is not None and target is not None:
            current = _replace_stream(current, target, first_occurrence=fact.effective_date)
        elif op == "amend_amount" and target is not None:
            if fact.amount is not None:
                current = _replace_stream(current, target, amount=fact.amount.amount)
            elif fact.multiplier is not None:
                current = _replace_stream(
                    current, target, amount=quantize(target.amount * fact.multiplier))
        elif op == "recurrence_stop" and target is not None:
            current = [s for s in current if s.key != target.key]
        elif op == "recurrence_start" and fact.amount is not None and fact.effective_date:
            current = current + [_synthetic_income(fact)]
        else:
            # Includes every unquantified fact: recorded, never priced.
            notes.append((fact.subject, f"{op} from {fact.message_id} not applied "
                                        f"(quantified={fact.quantified})"))
        if trace:
            trace.section("recurrence").add(
                "directive", op=op, message=fact.message_id,
                target=list(target.key) if target else None, quantified=fact.quantified)
    return StreamSet(tuple(current), tuple(notes))


#: Which income class a suppression message is about. A note about a gig payout says nothing
#: about payroll, and vice versa.
def _suppressed_classes(subject: str) -> set[str]:
    if (subject or "").lower() in {"gig_payout", "invoice"}:
        return {"irregular"}
    return {"continuing", "irregular"}


def _match_stream(streams: Sequence[RecurringStream], fact: MessageFact) -> RecurringStream | None:
    subject = (fact.subject or "").lower()
    income = [s for s in streams if s.kind == "income"]
    if subject in {"salary", "income", "payroll"} or fact.fact_type.startswith("salary"):
        return max(income, key=lambda s: s.anchor_date) if income else None
    for s in streams:
        if s.group.lower() == subject:
            return s
    return None


def anchor_for_first(first: date, stream: RecurringStream) -> date:
    """Anchor such that `occurrences()` yields `first` as its first occurrence.

    `occurrences()` starts one cadence *after* the anchor, because an anchor is a payment that
    already happened. A message saying "salary is X from 2025-08-15" is different: that payment
    has NOT happened and must be projected. Without this, the effective-date payment is silently
    dropped and the user appears to lose a month of income.
    """
    if stream.cadence_class == "monthly":
        return add_month(first, -1, first.day)
    return first - timedelta(days=stream.cadence_days)


def _replace_stream(streams: Sequence[RecurringStream], target: RecurringStream,
                    amount: Decimal | None = None,
                    first_occurrence: date | None = None) -> list[RecurringStream]:
    anchor = anchor_for_first(first_occurrence, target) if first_occurrence else None
    changed = replace(
        target,
        amount=quantize(amount) if amount is not None else target.amount,
        anchor_date=anchor or target.anchor_date,
        anchor_day_of_month=(first_occurrence.day
                             if first_occurrence and target.cadence_class == "monthly"
                             else target.anchor_day_of_month),
        estimator=f"{target.estimator}+message",
        is_stale=False,
    )
    return [changed if s.key == target.key else s for s in streams]


def _synthetic_income(fact: MessageFact) -> RecurringStream:
    assert fact.amount is not None and fact.effective_date is not None
    # The confirmed date is the FIRST payment, so anchor one month earlier.
    anchor = add_month(fact.effective_date, -1, fact.effective_date.day)
    return RecurringStream(
        key=("income", fact.subject or "confirmed income"), kind="income",
        group=fact.subject or "confirmed income", cadence_days=30, cadence_class="monthly",
        anchor_date=anchor, anchor_day_of_month=fact.effective_date.day,
        amount=quantize(fact.amount.amount), estimator="message",
        latest_event_id=fact.message_id, flexibility="fixed", minimum_allowed_amount=None,
        observations=0, is_stale=False, income_class="continuing", amendments=(fact,))


# ---------------------------------------------------------------- assembly


def build_evidence(ctx, resolution: ResolutionResult, config: ForecastConfig,
                   image_facts: Sequence[ImageFact] = (),
                   trace: TraceLike = NULL_TRACE) -> EvidenceBundle:
    events = apply_image_facts(resolution.events, image_facts,
                               ctx.profile.home_currency, ctx.fx)
    streams = infer_streams(events, ctx.request.request_date, config, trace)
    streams = apply_stream_directives(streams, resolution.directives, trace)
    return EvidenceBundle(
        profile=ctx.profile,
        request=ctx.request,
        events=tuple(events),
        streams=streams.active,
        options=ctx.options,
        message_facts=tuple(d.fact for d in resolution.directives),
        image_facts=tuple(image_facts),
        unquantified=tuple(d.fact for d in resolution.directives if not d.fact.quantified),
    )


def occurrences(stream: RecurringStream, start: date, end: date,
                config: ForecastConfig) -> Iterable[date]:
    """Discrete occurrence dates in (start, end]. Monthly streams follow the calendar day."""
    if stream.cadence_class == "monthly" and config.monthly_anchor == "day_of_month":
        day = stream.anchor_day_of_month or stream.anchor_date.day
        step = 1
        while True:
            nxt = add_month(stream.anchor_date, step, day)
            if nxt > end:
                return
            if nxt >= start:
                yield nxt
            step += 1
    else:
        nxt = stream.anchor_date
        while True:
            nxt = nxt + timedelta(days=stream.cadence_days)
            if nxt > end:
                return
            if nxt >= start:
                yield nxt


def daily_rate(stream: RecurringStream) -> Decimal:
    return stream.amount / Decimal(stream.cadence_days)
