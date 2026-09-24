"""Canonical event resolution.

Turns one user's raw rows into economic events, deciding two *independent* questions per event:

    counts_for_cash        does this move money inside the forecast window?
    counts_for_recurrence  may this row be used to infer a repeating pattern?

Keeping them independent is what prevents both failure modes: double-counting an authorization
alongside its settlement, and letting a one-off document poison a recurring series.

Classification uses `linked_event_id`, `event_type`, `status` and `direction` only. Descriptions
are free text and are never matched on, so a renamed description cannot change a financial answer.
Request IDs and event IDs are never special-cased.

CASH POLICY (see prompt section 3)
----------------------------------
`current_available_balance` is the request-date snapshot, so a settled historical event is already
inside it and must not be replayed. It remains the primary *recurrence* evidence. Only explicit
future obligations - pending and scheduled rows - contribute forecast cash.

The dataset satisfies this partition exactly: all 25 148 settled rows fall strictly before their
user's request date, and all 71 pending and 70 scheduled rows fall strictly after it.
`assert_cash_partition` re-checks that at runtime rather than trusting it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Iterable, Mapping, Sequence

from bow.errors import UnresolvedEvidenceError
from bow.models import (
    AmountStatus,
    CanonicalFinancialEvent,
    ExclusionReason,
    FactType,
    LifecycleRole,
    MessageFact,
    OverrideOp,
    RawFinancialEvent,
)
from bow.money import ConvertedMoney
from bow.trace import NULL_TRACE, TraceLike

# ---------------------------------------------------------------- message override interface
# Prompt 6 will populate MessageFact objects from GLM-5.3. The mapping below is the whole
# contract between extraction and resolution: extraction only has to emit a FactType.

#: Facts that modify a specific canonical event (require `target_event_id`).
EVENT_OPERATIONS: Mapping[FactType, OverrideOp] = {
    "event_confirmed": "confirm",
    "event_cancelled": "cancel",
    "event_delayed": "delay",
    "refund_completed": "confirm",
    "refund_pending": "cancel",            # keep the credit out of cash until it settles
    "duplicate_under_investigation": "cancel",
    "internal_transfer": "cancel",         # both legs, when a pair genuinely exists
    "non_cash_valuation": "cancel",
}

#: Facts that modify a recurring stream rather than a single event. Consumed by `recurrence`
#: in the next phase; declared here so both sides agree on the vocabulary.
STREAM_DIRECTIVES: Mapping[FactType, OverrideOp] = {
    "salary_amount_change": "salary_change",
    "salary_date_change": "amend_date",
    "salary_one_off_adjustment": "amend_amount",
    "first_salary_confirmed": "recurrence_start",
    "income_ended": "employment_end",
    "income_unconfirmed": "income_unconfirmed",
    "invoice_approved": "recurrence_start",
    "recurring_expense_change": "amend_amount",
    "new_recurring_obligation_unquantified": "none",   # never priced; no number is invented
    "foreign_currency_note": "none",
    "no_financial_effect": "none",
}


@dataclass(frozen=True, slots=True)
class StreamDirective:
    """A stream-level instruction extracted from a message, for `recurrence` to apply."""

    op: OverrideOp
    fact: MessageFact


# ---------------------------------------------------------------- results


@dataclass(frozen=True, slots=True)
class LifecycleGroup:
    """A compact, human-readable account of one resolved lifecycle."""

    group_id: str
    kind: str
    member_ids: tuple[str, ...]
    survivor_ids: tuple[str, ...]
    ignored_ids: tuple[str, ...]
    reason: str

    def describe(self) -> str:
        kept = ", ".join(self.survivor_ids) or "(none)"
        dropped = ", ".join(self.ignored_ids) or "(none)"
        return f"[{self.kind}] kept={kept} ignored={dropped} :: {self.reason}"


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    events: tuple[CanonicalFinancialEvent, ...]
    groups: tuple[LifecycleGroup, ...]
    directives: tuple[StreamDirective, ...]
    unresolved: tuple[str, ...]

    @property
    def cash_events(self) -> tuple[CanonicalFinancialEvent, ...]:
        return tuple(e for e in self.events if e.counts_for_cash)

    @property
    def recurrence_events(self) -> tuple[CanonicalFinancialEvent, ...]:
        return tuple(e for e in self.events if e.counts_for_recurrence)

    def by_id(self, event_id: str) -> CanonicalFinancialEvent | None:
        for e in self.events:
            if event_id in e.source_event_ids:
                return e
        return None

    def trace_lines(self) -> tuple[str, ...]:
        return tuple(g.describe() for g in self.groups)


# ---------------------------------------------------------------- lifecycle classification


def _pair_kind(parent: RawFinancialEvent, child: RawFinancialEvent) -> str | None:
    """Name the lifecycle for a linked (parent, child) pair from structure alone."""
    if child.event_type == "investment_valuation" or child.direction == "non_cash":
        return "investment_valuation"
    if child.event_type == "investment_sale":
        return "investment_sale"
    if child.event_type == "refund" or (
        parent.direction == "debit" and child.direction == "credit"
    ):
        return "offsetting_credit"
    if parent.status == "cancelled" and child.status == "settled":
        return "authorization_settlement"
    if parent.status == "failed" and child.status == "scheduled":
        return "failed_retry"
    if (parent.status == "settled" and child.status == "pending"
            and parent.direction == child.direction == "debit"
            and parent.amount is not None and child.amount is not None
            and parent.amount.amount == child.amount.amount):
        return "duplicate_charge"
    return None


def _is_reimbursement(parent: RawFinancialEvent) -> bool:
    """An offsetting credit against a *work expense* is a reimbursement, not a refund.

    Distinguished by the parent's category rather than its wording.
    """
    return parent.category == "work_expense"


# ---------------------------------------------------------------- resolver


class EventResolver:
    def __init__(self, home_currency: str, asof: date, fx, image_event_ids: Iterable[str] = (),
                 trace: TraceLike = NULL_TRACE) -> None:
        self.home = home_currency
        self.asof = asof
        self.fx = fx
        self.image_event_ids = frozenset(image_event_ids)
        self.trace = trace

    # -------------------------------------------------------------- helpers

    def _amount(self, raw: RawFinancialEvent) -> tuple[ConvertedMoney | None, AmountStatus]:
        if raw.direction == "non_cash":
            # A valuation carries a number, but it is not money the user can spend.
            return None, "not_applicable"
        if raw.amount is None:
            # Blank amount. NEVER zero. Stays unresolved until an ImageFact supplies it.
            return None, "unresolved_image"
        on = raw.settlement_date or raw.event_date
        return self.fx.to_home(raw.amount, self.home, on), "known"

    def _sign(self, raw: RawFinancialEvent) -> int:
        return {"credit": 1, "debit": -1, "non_cash": 0}[raw.direction]

    def _make(self, raw: RawFinancialEvent, *, role: LifecycleRole, group: str | None,
              cash: bool, recurrence: bool, reason: ExclusionReason | None,
              note: str, evidence: tuple[str, ...] = ()) -> CanonicalFinancialEvent:
        money, status = self._amount(raw)
        if cash and status == "unresolved_image":
            # Fail closed: a cash-affecting event with no amount cannot be silently treated as 0.
            cash = False
            reason, note = "unresolved_amount", (
                f"{note}; amount unresolved (awaiting image extraction) so excluded from cash "
                f"rather than assumed zero")
        return CanonicalFinancialEvent(
            economic_id=raw.event_id,
            source_event_ids=(raw.event_id,),
            user_id=raw.user_id,
            effective_date=raw.settlement_date or raw.event_date,
            direction=raw.direction,
            sign=self._sign(raw),
            cash_effect=money,
            amount_status=status,
            status=raw.status,
            lifecycle_role=role,
            lifecycle_group=group,
            counts_for_cash=cash,
            counts_for_recurrence=recurrence,
            exclusion_reason=reason,
            resolution_note=note,
            category=raw.category,
            description=raw.description,
            event_type=raw.event_type,
            flexibility=raw.flexibility,
            minimum_allowed_amount=raw.minimum_allowed_amount,
            evidence_ids=evidence,
        )

    def _future(self, raw: RawFinancialEvent) -> bool:
        return (raw.settlement_date or raw.event_date) >= self.asof

    def _seeds_recurrence(self, raw: RawFinancialEvent) -> bool:
        """Only settled history seeds recurrence, and only when its amount is genuinely observed.

        An image-linked row is a one-off document whose amount arrives from outside the ledger;
        averaging it into a series is exactly how a 41 272 'grocery' poisons a series whose true
        level is 8 700.
        """
        if raw.status != "settled":
            return False
        if raw.amount is None or raw.event_id in self.image_event_ids:
            return False
        return raw.direction != "non_cash"

    # -------------------------------------------------------------- main entry

    def resolve(self, events: Sequence[RawFinancialEvent],
                facts: Sequence[MessageFact] = ()) -> ResolutionResult:
        by_id = {e.event_id: e for e in events}
        children: dict[str, list[RawFinancialEvent]] = {}
        for e in events:
            if e.linked_event_id:
                children.setdefault(e.linked_event_id, []).append(e)

        canonical: list[CanonicalFinancialEvent] = []
        groups: list[LifecycleGroup] = []
        unresolved: list[str] = []
        handled: set[str] = set()

        # ---- 1. linked lifecycles
        for parent_id, kids in sorted(children.items()):
            parent = by_id.get(parent_id)
            if parent is None:
                # A link we cannot follow: fail closed on the child rather than guess.
                for child in kids:
                    unresolved.append(child.event_id)
                    canonical.append(self._make(
                        child, role="standalone", group=None, cash=False, recurrence=False,
                        reason="unresolved_evidence",
                        note=f"linked_event_id {parent_id!r} not found for this user"))
                    handled.add(child.event_id)
                continue
            for child in kids:
                kind = _pair_kind(parent, child)
                if kind is None:
                    unresolved.append(child.event_id)
                    continue
                group_id = f"{kind}:{parent.event_id}"
                produced, survivors, ignored, reason = self._resolve_pair(
                    kind, parent, child, group_id)
                canonical.extend(produced)
                handled.update({parent.event_id, child.event_id})
                groups.append(LifecycleGroup(
                    group_id=group_id, kind=kind,
                    member_ids=(parent.event_id, child.event_id),
                    survivor_ids=survivors, ignored_ids=ignored, reason=reason))

        # ---- 2. standalone events
        for raw in events:
            if raw.event_id in handled:
                continue
            canonical.append(self._standalone(raw))

        canonical.sort(key=lambda e: (e.effective_date or self.asof, e.economic_id))
        directives = tuple(
            StreamDirective(op=STREAM_DIRECTIVES[f.fact_type], fact=f)
            for f in facts
            if f.fact_type in STREAM_DIRECTIVES and STREAM_DIRECTIVES[f.fact_type] != "none"
        )
        result = ResolutionResult(
            events=tuple(canonical), groups=tuple(groups),
            directives=directives, unresolved=tuple(unresolved))

        if self.trace:
            section = self.trace.section("resolve")
            section.add("summary", raw=len(events), canonical=len(canonical),
                        cash=len(result.cash_events), recurrence=len(result.recurrence_events),
                        lifecycles=len(groups), unresolved=list(unresolved))
            for g in groups:
                section.add("lifecycle", kind=g.kind, members=list(g.member_ids),
                            kept=list(g.survivor_ids), ignored=list(g.ignored_ids),
                            reason=g.reason)
        return result

    # -------------------------------------------------------------- pair handling

    def _resolve_pair(self, kind: str, parent: RawFinancialEvent, child: RawFinancialEvent,
                      group_id: str):
        """Return (canonical events, survivor ids, ignored ids, human reason)."""
        make = self._make

        if kind == "investment_valuation":
            return (
                [make(parent, role="investment_purchase", group=group_id,
                      cash=False, recurrence=False, reason="already_in_opening_balance",
                      note="settled investment contribution; already inside the opening balance"),
                 make(child, role="valuation", group=group_id, cash=False, recurrence=False,
                      reason="non_cash",
                      note="unrealized market value; not spendable cash and never recurring")],
                (), (parent.event_id, child.event_id),
                "unrealized valuation is not cash; the contribution is already in the balance",
            )

        if kind == "investment_sale":
            cash = self._future(child) and child.status == "settled"
            return (
                [make(parent, role="investment_purchase", group=group_id,
                      cash=False, recurrence=False, reason="already_in_opening_balance",
                      note="settled contribution; inside the opening balance"),
                 make(child, role="sale", group=group_id, cash=cash, recurrence=False,
                      reason=None if cash else "already_in_opening_balance",
                      note="realized sale proceeds; a one-off, so it never seeds recurrence")],
                (child.event_id,), (parent.event_id,),
                "sale proceeds are real cash but one-off; the contribution is historical",
            )

        if kind == "offsetting_credit":
            reimb = _is_reimbursement(parent)
            role = "reimbursement" if reimb else "refund"
            if child.status == "pending":
                return (
                    [make(parent, role="refunded_purchase", group=group_id, cash=False,
                          recurrence=False, reason="already_in_opening_balance",
                          note="settled purchase; already inside the opening balance"),
                     make(child, role=role, group=group_id, cash=False, recurrence=False,
                          reason="pending_credit",
                          note="credit initiated but not received; pending credits are never "
                               "counted until they settle")],
                    (), (parent.event_id, child.event_id),
                    "pending credit excluded; the debit already sits in the opening balance",
                )
            return (
                [make(parent, role="refunded_purchase", group=group_id, cash=False,
                      recurrence=False, reason="already_in_opening_balance",
                      note="settled purchase, offset by a settled credit; net zero historically"),
                 make(child, role=role, group=group_id, cash=False, recurrence=False,
                      reason="already_in_opening_balance",
                      note="settled offsetting credit; both legs are inside the opening balance")],
                (), (parent.event_id, child.event_id),
                f"settled {role} offsets the purchase; neither leg seeds recurrence",
            )

        if kind == "authorization_settlement":
            cash = self._future(child)
            return (
                [make(parent, role="authorization", group=group_id, cash=False, recurrence=False,
                      reason="superseded_by_settlement",
                      note="cancelled authorization replaced by its settlement; counting both "
                           "would double-charge the user"),
                 make(child, role="settlement", group=group_id, cash=cash,
                      recurrence=False,
                      reason=None if cash else "already_in_opening_balance",
                      note="the settlement is the single real economic debit")],
                (child.event_id,), (parent.event_id,),
                "authorization superseded by settlement; charged exactly once",
            )

        if kind == "failed_retry":
            cash = self._future(child)
            return (
                [make(parent, role="failed_attempt", group=group_id, cash=False, recurrence=False,
                      reason="failed",
                      note="the debit failed, so no money left the account"),
                 make(child, role="retry", group=group_id, cash=cash, recurrence=False,
                      reason=None if cash else "already_in_opening_balance",
                      note="scheduled retry is the real future obligation")],
                (child.event_id,), (parent.event_id,),
                "failed attempt ignored; the scheduled retry is the single economic debit",
            )

        if kind == "duplicate_charge":
            return (
                [make(parent, role="original_charge", group=group_id, cash=False,
                      recurrence=False, reason="already_in_opening_balance",
                      note="the genuine charge; already inside the opening balance"),
                 make(child, role="duplicate", group=group_id, cash=False, recurrence=False,
                      reason="duplicate",
                      note="same amount, linked to the original; a duplicate is not a second "
                           "economic event")],
                (parent.event_id,), (child.event_id,),
                "duplicate charge ignored; the original is the only economic debit",
            )

        raise UnresolvedEvidenceError(f"unhandled lifecycle kind {kind!r}")

    # -------------------------------------------------------------- standalone handling

    def _standalone(self, raw: RawFinancialEvent) -> CanonicalFinancialEvent:
        recurrence = self._seeds_recurrence(raw)

        if raw.direction == "non_cash" or raw.status == "unrealized":
            return self._make(raw, role="valuation", group=None, cash=False, recurrence=False,
                              reason="non_cash",
                              note="non-cash valuation; never spendable")
        if raw.status == "cancelled":
            return self._make(raw, role="authorization", group=None, cash=False, recurrence=False,
                              reason="cancelled", note="cancelled; no money moved")
        if raw.status == "failed":
            return self._make(raw, role="failed_attempt", group=None, cash=False, recurrence=False,
                              reason="failed", note="failed; no money left the account")
        if raw.status == "pending":
            if raw.direction == "credit":
                return self._make(raw, role="refund", group=None, cash=False, recurrence=False,
                                  reason="pending_credit",
                                  note="pending credit; not counted until it settles")
            return self._make(raw, role="pending_debit", group=None,
                              cash=self._future(raw), recurrence=False, reason=None,
                              note="pending debit reserved on its settlement date")
        if raw.status == "scheduled":
            role: LifecycleRole = "confirmed_income" if raw.direction == "credit" \
                else "scheduled_obligation"
            return self._make(raw, role=role, group=None, cash=self._future(raw),
                              recurrence=False, reason=None,
                              note="explicit dated future commitment")

        # settled
        if self._future(raw):
            return self._make(raw, role="standalone", group=None, cash=True,
                              recurrence=recurrence, reason=None,
                              note="settled on or after the request date")
        note = "settled history: recurrence evidence, already inside the opening balance"
        if not recurrence and raw.event_id in self.image_event_ids:
            note = ("settled one-off whose amount comes from an image; "
                    "excluded from recurrence so it cannot distort the series")
        elif not recurrence and raw.amount is None:
            note = "settled row with no amount; excluded from recurrence, never treated as zero"
        return self._make(raw, role="standalone", group=None, cash=False, recurrence=recurrence,
                          reason="already_in_opening_balance" if recurrence else "one_off",
                          note=note)


# ---------------------------------------------------------------- convenience


def resolve_context(ctx, facts: Sequence[MessageFact] = (),
                    trace: TraceLike = NULL_TRACE) -> ResolutionResult:
    """Resolve one request's user from a `dataset.RequestContext`."""
    resolver = EventResolver(
        home_currency=ctx.profile.home_currency,
        asof=ctx.request.request_date,
        fx=ctx.fx,
        image_event_ids={i.related_event_id for i in ctx.images},
        trace=trace,
    )
    return resolver.resolve(ctx.events, facts)


def assert_cash_partition(result: ResolutionResult, asof: date) -> None:
    """Guard the cash policy: nothing settled before `asof` may contribute forecast cash."""
    for e in result.cash_events:
        if e.effective_date is None or e.effective_date < asof:
            raise UnresolvedEvidenceError(
                f"{e.economic_id} would replay cash dated {e.effective_date} before {asof}")
        if e.cash_effect is None:
            raise UnresolvedEvidenceError(
                f"{e.economic_id} counts for cash but has no resolved amount")


def total_future_cash(result: ResolutionResult) -> Decimal:
    """Signed sum of explicit future obligations. Recurring flows are added later, by `forecast`."""
    return sum((e.signed_amount or Decimal(0) for e in result.cash_events), Decimal(0))
