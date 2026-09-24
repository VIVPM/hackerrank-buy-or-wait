"""Typed internal representations. Pure data — no logic, no I/O, no imports above layer 0."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Literal, Mapping, Sequence

from bow.money import ConvertedMoney, Money

# ---------------------------------------------------------------- enums / aliases

AffordabilityStatus = Literal[
    "affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"
]
PaymentMethod = Literal[
    "full_payment", "partial_payment", "installments", "wait", "not_recommended"
]
Flexibility = Literal["fixed", "reducible", "stoppable", "reducible_or_stoppable"]
EventStatus = Literal["settled", "pending", "scheduled", "cancelled", "failed", "unrealized"]
StreamKind = Literal["expense", "income"]
CadenceClass = Literal["monthly", "sub_monthly"]

# Why a canonical event was dropped from cash flow and/or recurrence inference.
ExclusionReason = Literal[
    "superseded_by_settlement",       # a cancelled authorization replaced by its settlement
    "failed",                         # the debit never left the account
    "cancelled",                      # explicitly cancelled
    "duplicate",                      # a second charge linked to the original
    "pending_credit",                 # money not yet received; never counted
    "non_cash",                       # unrealized valuation
    "internal_transfer",              # both legs of a transfer between the user's own accounts
    "one_off",                        # real, but not evidence of a repeating pattern
    "amended_by_message",             # superseded by a message amendment
    "image_derived",                  # amount supplied by an image; never seeds recurrence
    "unresolved_amount",              # blank amount with no ImageFact yet; NOT zero
    "already_in_opening_balance",     # settled before request_date; inside the snapshot already
    "unresolved_evidence",            # contradictory or unrecognised; fails closed
]

LifecycleRole = Literal[
    "standalone",              # no lifecycle partner
    "authorization",           # cancelled authorization superseded by a settlement
    "settlement",              # the settlement that superseded an authorization
    "failed_attempt",          # a debit that failed
    "retry",                   # the scheduled retry of a failed debit
    "original_charge",         # the genuine charge that a duplicate was linked to
    "duplicate",               # the duplicate
    "refunded_purchase",       # a purchase that was later refunded or reversed
    "refund",                  # the offsetting credit
    "reimbursement",           # employer reimbursement of a work expense
    "investment_purchase",     # cash contribution into an investment
    "valuation",               # unrealized, non-cash
    "sale",                    # realized investment proceeds
    "pending_debit",           # an authorised debit that has not settled yet
    "scheduled_obligation",    # a dated future liability
    "confirmed_income",        # an explicitly scheduled future credit
    "amendment_target",        # an event a message amends
]

# Whether the home-currency amount is actually known.
AmountStatus = Literal["known", "unresolved_image", "not_applicable"]

# Operations a MessageFact may perform. See resolve.EVENT_OPERATIONS / STREAM_DIRECTIVES.
OverrideOp = Literal[
    "confirm", "cancel", "amend_amount", "amend_date", "delay",
    "salary_change", "employment_end", "income_unconfirmed",
    "recurrence_stop", "recurrence_start", "none",
]

FactType = Literal[
    "salary_amount_change", "salary_date_change", "salary_one_off_adjustment",
    "first_salary_confirmed", "income_ended", "income_unconfirmed", "invoice_approved",
    "recurring_expense_change", "new_recurring_obligation_unquantified",
    "refund_pending", "refund_completed", "internal_transfer", "non_cash_valuation",
    "duplicate_under_investigation", "event_confirmed", "event_cancelled", "event_delayed",
    "foreign_currency_note", "no_financial_effect",
]

ImageLabel = Literal[
    "net_pay", "balance_due", "grand_total", "total_paid", "amount_due_by_date", "item_total",
]

FlowKind = Literal["recurring", "accrual", "explicit", "plan_payment"]

# ---------------------------------------------------------------- inputs


@dataclass(frozen=True, slots=True)
class FinancialProfile:
    user_id: str
    home_currency: str
    current_available_balance: Decimal
    minimum_balance_to_keep: Decimal
    financial_priorities: tuple[str, ...]
    protected_categories: frozenset[str]
    reducible_categories: frozenset[str]
    stoppable_categories: frozenset[str]
    methods: frozenset[str]
    max_installment_months: int | None


@dataclass(frozen=True, slots=True)
class Request:
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: Decimal
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str


@dataclass(frozen=True, slots=True)
class RawFinancialEvent:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: Literal["debit", "credit", "non_cash"]
    amount: Money | None          # None only when an image supplies the value
    event_date: date
    settlement_date: date | None
    status: EventStatus
    linked_event_id: str | None
    flexibility: Flexibility
    minimum_allowed_amount: Decimal | None
    source_row: int


@dataclass(frozen=True, slots=True)
class RawMessage:
    message_id: str
    user_id: str
    request_id: str | None
    related_event_id: str | None
    sent_at: str
    source_type: str
    message_text: str


@dataclass(frozen=True, slots=True)
class RawImage:
    image_id: str
    user_id: str
    request_id: str | None
    related_event_id: str
    path: str                     # dataset/media/images/<image_id>.png


@dataclass(frozen=True, slots=True)
class ExpectedOutput:
    """The six scored fields of a solved sample. Used only by the regression harness."""

    request_id: str
    amount_safe_to_pay: Decimal
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: str
    spending_changes_needed: str
    decision_explanation: str


@dataclass(frozen=True, slots=True)
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: Literal["full_payment", "installments"]
    payment_amount: Decimal
    number_of_payments: int
    first_payment_date: date
    payment_frequency_days: int | None
    financing_fee: Decimal
    total_payable_amount: Decimal
    schedule: tuple[tuple[date, Decimal], ...]
    last_payment_date: date


# ---------------------------------------------------------------- AI-derived evidence


@dataclass(frozen=True, slots=True)
class MessageFact:
    message_id: str
    user_id: str
    fact_type: FactType
    subject: str
    target_event_id: str | None
    effective_date: date | None
    amount: Money | None
    multiplier: Decimal | None
    quantified: bool
    confidence: float
    evidence_span: str


@dataclass(frozen=True, slots=True)
class ImageFact:
    image_id: str
    related_event_id: str
    chosen_amount: Money
    semantic_label: ImageLabel
    rejected_candidates: tuple[tuple[str, Money], ...]
    confidence: float
    notes: str


# ---------------------------------------------------------------- resolved / inferred


@dataclass(frozen=True, slots=True)
class CanonicalFinancialEvent:
    """One economic event. `counts_for_cash` and `counts_for_recurrence` are independent:

    a pending debit moves future cash but proves nothing about a repeating pattern, while a
    settled historical rent payment is already inside `current_available_balance` yet is the
    primary evidence that rent recurs.
    """

    economic_id: str
    source_event_ids: tuple[str, ...]
    user_id: str
    effective_date: date | None                 # settlement_date, else event_date
    direction: Literal["debit", "credit", "non_cash"]
    sign: int                                   # +1 credit, -1 debit, 0 non-cash
    cash_effect: ConvertedMoney | None          # home currency; None when unresolved
    amount_status: AmountStatus
    status: EventStatus
    lifecycle_role: LifecycleRole
    lifecycle_group: str | None                 # id shared by members of one lifecycle
    counts_for_cash: bool
    counts_for_recurrence: bool
    exclusion_reason: ExclusionReason | None
    resolution_note: str
    category: str
    description: str
    event_type: str
    flexibility: Flexibility
    minimum_allowed_amount: Decimal | None
    evidence_ids: tuple[str, ...] = ()          # message_id / image_id that touched this event

    @property
    def signed_amount(self) -> Decimal | None:
        """Signed home-currency cash effect, or None when the amount is unresolved."""
        if self.cash_effect is None:
            return None
        return self.cash_effect.converted * self.sign


@dataclass(frozen=True, slots=True)
class RecurringStream:
    key: tuple[StreamKind, str]                 # ("expense", category) | ("income", description)
    kind: StreamKind
    group: str
    cadence_days: int
    cadence_class: CadenceClass
    anchor_date: date
    anchor_day_of_month: int | None
    amount: Decimal                             # home currency
    estimator: str
    latest_event_id: str
    flexibility: Flexibility
    minimum_allowed_amount: Decimal | None
    observations: int
    is_stale: bool
    income_class: str = ""          # "continuing" | "irregular" for income streams
    amendments: tuple[MessageFact, ...] = ()


@dataclass(frozen=True, slots=True)
class RawEvidence:
    """Everything one request needs, before lifecycle resolution. Produced by `evidence`."""

    profile: FinancialProfile
    request: Request
    raw_events: tuple[RawFinancialEvent, ...]
    options: tuple[PaymentOption, ...]
    message_facts: tuple[MessageFact, ...]
    image_facts: tuple[ImageFact, ...]


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """Resolved evidence. Produced by `recurrence`, consumed by `forecast` and below."""

    profile: FinancialProfile
    request: Request
    events: tuple[CanonicalFinancialEvent, ...]
    streams: tuple[RecurringStream, ...]
    options: tuple[PaymentOption, ...]
    message_facts: tuple[MessageFact, ...]
    image_facts: tuple[ImageFact, ...]
    unquantified: tuple[MessageFact, ...] = ()  # obligations asserted without an amount


# ---------------------------------------------------------------- forecasting


@dataclass(frozen=True, slots=True)
class ProjectedCashFlow:
    date: date
    amount: Decimal                             # signed, home currency
    kind: FlowKind
    origin: str                                 # stream key or event_id
    note: str = ""


@dataclass(frozen=True, slots=True)
class ForecastContext:
    evidence: EvidenceBundle
    asof: date
    extra_outflows: tuple[ProjectedCashFlow, ...] = ()
    stream_overrides: Mapping[tuple[StreamKind, str], Decimal | None] = field(
        default_factory=dict
    )


@dataclass(frozen=True, slots=True)
class ForecastResult:
    opening_balance: Decimal
    minimum_balance: Decimal
    flows: tuple[ProjectedCashFlow, ...]
    path: tuple[tuple[date, Decimal], ...]
    trough: Decimal
    trough_date: date
    breaches: tuple[tuple[date, Decimal], ...]
    config_fingerprint: str

    @property
    def is_safe(self) -> bool:
        return not self.breaches


# ---------------------------------------------------------------- planning


@dataclass(frozen=True, slots=True)
class SpendingChange:
    kind: Literal["stop", "reduce_to"]
    stream_key: tuple[StreamKind, str]
    event_id: str
    new_amount: Decimal | None
    monthly_saving: Decimal


@dataclass(frozen=True, slots=True)
class CandidatePlan:
    method: PaymentMethod
    payments: tuple[tuple[date, Decimal], ...]
    changes: tuple[SpendingChange, ...]
    source_option_id: str | None
    total_paid: Decimal
    completes_full_amount: bool
    completes_by_deadline: bool
    implied_status: AffordabilityStatus


@dataclass(frozen=True, slots=True)
class Violation:
    code: str                                   # "V7"
    message: str


@dataclass(frozen=True, slots=True)
class PredictionResult:
    request_id: str
    amount_safe_to_pay: Decimal
    affordability_status: AffordabilityStatus
    recommended_payment_method: PaymentMethod
    payment_plan: str
    earliest_date_for_full_payment: str          # "" when never safe in the horizon
    spending_changes_needed: str
    decision_explanation: str
    trace_id: str | None = None
    validator_notes: tuple[str, ...] = ()
    fallback_used: bool = False


# ---------------------------------------------------------------- accounting


@dataclass(frozen=True, slots=True)
class UsageRecord:
    provider: str
    model: str
    call_type: Literal["message_extract", "image_extract", "agent_step"]
    source_id: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    estimated_cost_usd: Decimal
    cache: Literal["hit", "miss"]
    retries: int
    timestamp: str


OUTPUT_COLUMNS: Sequence[str] = (
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
)
