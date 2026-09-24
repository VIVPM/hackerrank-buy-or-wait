"""Which payment methods and supplied options this user and request actually permit.

Shared by the planner and the validator so both apply one definition. The validator importing
this rather than re-implementing it is deliberate: the rules must not drift apart, while the
validator's *safety* checks stay independent of how a candidate was built.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Sequence

from bow.models import FinancialProfile, PaymentOption, PaymentMethod, Request

IMMEDIATE_METHODS = ("full_payment", "partial_payment", "installments")


@dataclass(frozen=True, slots=True)
class OptionVerdict:
    option: PaymentOption
    usable: bool
    reason: str | None


@dataclass(frozen=True, slots=True)
class Eligibility:
    profile: FinancialProfile
    request: Request
    verdicts: tuple[OptionVerdict, ...]

    def accepts(self, method: str) -> bool:
        if method == "wait":
            # Waiting is just paying in full later, so it needs the full_payment preference.
            return "full_payment" in self.profile.methods
        if method == "not_recommended":
            return True
        return method in self.profile.methods

    @property
    def usable_options(self) -> tuple[PaymentOption, ...]:
        return tuple(v.option for v in self.verdicts if v.usable)

    @property
    def full_option(self) -> PaymentOption | None:
        for v in self.verdicts:
            if v.option.payment_method == "full_payment":
                return v.option
        return None

    def partial_allowed(self, safe: Decimal, earliest: date | None) -> tuple[bool, str | None]:
        """Every condition the specification places on partial payment."""
        if not self.request.allows_partial_payment:
            return False, "request_disallows_partial"
        if not self.accepts("partial_payment"):
            return False, "method_not_accepted"
        if not (Decimal(0) < safe < self.request.requested_amount):
            return False, "safe_amount_not_strictly_between_zero_and_requested"
        if earliest is None:
            return False, "no_earliest_full_payment_date"
        if earliest > self.request.desired_completion_date:
            return False, "earliest_after_deadline"
        if earliest <= self.request.request_date:
            return False, "full_payment_already_safe_today"
        return True, None


def assess(profile: FinancialProfile, request: Request,
           options: Sequence[PaymentOption]) -> Eligibility:
    verdicts = []
    for option in sorted(options, key=lambda o: o.payment_option_id):
        verdicts.append(OptionVerdict(option, *_judge(profile, request, option)))
    return Eligibility(profile=profile, request=request, verdicts=tuple(verdicts))


def _judge(profile: FinancialProfile, request: Request,
           option: PaymentOption) -> tuple[bool, str | None]:
    if option.payment_method == "full_payment":
        return ("full_payment" in profile.methods,
                None if "full_payment" in profile.methods else "method_not_accepted")
    if "installments" not in profile.methods:
        return False, "method_not_accepted"
    if profile.max_installment_months is None:
        return False, "user_considers_no_installments"
    if option.number_of_payments > profile.max_installment_months:
        return False, "exceeds_max_installment_months"
    if option.last_payment_date > request.desired_completion_date:
        # 434 of 515 supplied installment options fail exactly here.
        return False, "schedule_ends_after_desired_completion_date"
    expected = option.payment_amount * option.number_of_payments
    if expected != option.total_payable_amount:
        return False, "option_totals_inconsistent"
    return True, None
