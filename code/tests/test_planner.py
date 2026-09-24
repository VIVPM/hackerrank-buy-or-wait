"""Planner, validator, ranker and serialisation tests.

Assertions are about contract compliance, never about a solved sample's answer.
"""

from __future__ import annotations

import dataclasses
import sys
import unittest
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bow.ai.facts import load_facts                                          # noqa: E402
from bow.config import ForecastConfig                                        # noqa: E402
from bow.dataset import Dataset                                              # noqa: E402
from bow.eligibility import assess                                           # noqa: E402
from bow.forecast import ForecastEngine                                      # noqa: E402
from bow.models import CandidatePlan, SpendingChange                         # noqa: E402
from bow.outputs import serialise_changes, serialise_plan                    # noqa: E402
from bow.predict import predict, status_for                                  # noqa: E402
from bow.rank import rank, sort_key                                          # noqa: E402
from bow.recurrence import build_evidence                                    # noqa: E402
from bow.resolve import resolve_context                                      # noqa: E402
from bow.spending import change_sets, permitted_changes                      # noqa: E402
from bow.validate import validate                                            # noqa: E402
from bow.plans import generate                                               # noqa: E402

DS = Dataset.load()
STORE = load_facts(DS)
CFG = ForecastConfig()
ENGINE = ForecastEngine(CFG)


def context(request_id: str):
    ctx = DS.context_for(request_id)
    m, i = STORE.for_user(ctx.request.user_id)
    res = resolve_context(ctx, m)
    ev = build_evidence(ctx, res, CFG, i)
    return ctx, ev


def planned(request_id: str):
    ctx, ev = context(request_id)
    return ctx, ev, generate(ev, ENGINE)


ALL = tuple(DS.sample_request_ids)


# ---------------------------------------------------------------- contract invariants


class TestOutputInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {}
        for rid in ALL:
            ctx = DS.context_for(rid)
            cls.results[rid] = (ctx, predict(ctx, STORE).result)

    def test_amount_bounds(self):
        for rid, (ctx, r) in self.results.items():
            self.assertGreaterEqual(r.amount_safe_to_pay, Decimal(0), rid)
            self.assertLessEqual(r.amount_safe_to_pay, ctx.request.requested_amount, rid)

    def test_enum_values(self):
        statuses = {"affordable_now", "affordable_with_plan", "affordable_later",
                    "not_affordable"}
        methods = {"full_payment", "partial_payment", "installments", "wait",
                   "not_recommended"}
        for rid, (_, r) in self.results.items():
            self.assertIn(r.affordability_status, statuses, rid)
            self.assertIn(r.recommended_payment_method, methods, rid)

    def test_status_method_agree(self):
        pairs = {("affordable_now", "full_payment"),
                 ("affordable_later", "wait"),
                 ("not_affordable", "not_recommended"),
                 ("affordable_with_plan", "full_payment"),
                 ("affordable_with_plan", "partial_payment"),
                 ("affordable_with_plan", "installments")}
        for rid, (_, r) in self.results.items():
            self.assertIn((r.affordability_status, r.recommended_payment_method), pairs, rid)

    def test_plan_parses_and_sums(self):
        for rid, (ctx, r) in self.results.items():
            if r.payment_plan == "none":
                self.assertEqual(r.recommended_payment_method, "not_recommended", rid)
                continue
            parts = [p.split(":") for p in r.payment_plan.split("|")]
            dates = [date.fromisoformat(p[0]) for p in parts]
            amounts = [Decimal(p[1]) for p in parts]
            self.assertEqual(dates, sorted(dates), rid)
            self.assertTrue(all(a > 0 for a in amounts), rid)
            if r.recommended_payment_method in {"full_payment", "partial_payment", "wait"}:
                self.assertEqual(sum(amounts), ctx.request.requested_amount, rid)

    def test_affordable_now_has_earliest_equal_to_request_date(self):
        for rid, (ctx, r) in self.results.items():
            if r.affordability_status == "affordable_now":
                self.assertEqual(r.earliest_date_for_full_payment,
                                 ctx.request.request_date.isoformat(), rid)

    def test_earliest_is_a_capacity_field_not_a_plan_field(self):
        """Empty exactly when no date in the horizon makes the full amount safe.

        It is NOT tied to the chosen method: a request can be `not_affordable` because the
        full amount only becomes safe after `desired_completion_date`, and the spec still wants
        that capacity date reported.
        """
        from bow.forecast import ForecastContext
        from bow.safeamount import earliest_full_payment_date
        for rid, (ctx, r) in self.results.items():
            _, ev = context(rid)
            expected = earliest_full_payment_date(
                ENGINE, ForecastContext(ev, ctx.request.request_date),
                ctx.request.requested_amount)
            self.assertEqual(r.earliest_date_for_full_payment,
                             expected.isoformat() if expected else "", rid)

    def test_partial_payment_shape(self):
        for rid, (ctx, r) in self.results.items():
            if r.recommended_payment_method != "partial_payment":
                continue
            parts = r.payment_plan.split("|")
            self.assertEqual(len(parts), 2, rid)
            first_date, first_amount = parts[0].split(":")
            self.assertEqual(first_date, ctx.request.request_date.isoformat(), rid)
            self.assertEqual(Decimal(first_amount), r.amount_safe_to_pay, rid)
            self.assertEqual(parts[1].split(":")[0], r.earliest_date_for_full_payment, rid)

    def test_installments_match_a_supplied_option(self):
        for rid, (ctx, r) in self.results.items():
            if r.recommended_payment_method != "installments":
                continue
            schedules = {serialise_plan(o.schedule)
                         for o in DS.options_by_request[rid]
                         if o.payment_method == "installments"}
            self.assertIn(r.payment_plan, schedules, rid)

    def test_spending_changes_are_permitted(self):
        for rid, (ctx, r) in self.results.items():
            if r.spending_changes_needed == "none":
                continue
            _, ev = context(rid)
            allowed = {f"{c.kind}:{c.event_id}" for c in permitted_changes(ctx.profile,
                                                                          ev.streams)}
            for part in r.spending_changes_needed.split("|"):
                bits = part.split(":")
                key = f"{bits[0]}:{bits[1]}"
                self.assertIn(key, allowed, f"{rid}: {part}")
            self.assertLessEqual(len(r.spending_changes_needed.split("|")), 3, rid)

    def test_every_row_is_produced_without_fallback(self):
        for rid, (_, r) in self.results.items():
            self.assertFalse(r.fallback_used, rid)


# ---------------------------------------------------------------- eligibility


class TestEligibility(unittest.TestCase):
    def test_wait_requires_full_payment_preference(self):
        ctx, ev = context("request_12")            # partial|installments, no full_payment
        elig = assess(ctx.profile, ctx.request, ev.options)
        self.assertFalse(elig.accepts("wait"))
        self.assertFalse(elig.accepts("full_payment"))

    def test_options_past_the_deadline_are_rejected(self):
        """No usable option may finish after desired_completion_date - the hard invariant.

        Scanned across the whole dataset. The deadline reason itself fires only 14 times
        because the user-preference gates are checked first and catch most options; what
        matters is that nothing survives with a schedule past the deadline.
        """
        seen = {}
        survivors_past_deadline = []
        for rid, request in DS.requests.items():
            profile = DS.profiles[request.user_id]
            for v in assess(profile, request, DS.options_by_request[rid]).verdicts:
                if not v.usable:
                    seen[v.reason] = seen.get(v.reason, 0) + 1
                elif v.option.last_payment_date > request.desired_completion_date:
                    survivors_past_deadline.append(v.option.payment_option_id)
        self.assertIn("schedule_ends_after_desired_completion_date", seen)
        self.assertEqual(survivors_past_deadline, [])

    def test_blank_max_installments_blocks_installments(self):
        ctx, ev = context("request_01")
        self.assertIsNone(ctx.profile.max_installment_months)
        elig = assess(ctx.profile, ctx.request, ev.options)
        for v in elig.verdicts:
            if v.option.payment_method == "installments":
                self.assertFalse(v.usable)

    def test_partial_conditions_enumerated(self):
        ctx, ev = context("request_02")            # allows_partial_payment is false
        elig = assess(ctx.profile, ctx.request, ev.options)
        ok, why = elig.partial_allowed(Decimal("100"), ctx.request.desired_completion_date)
        self.assertFalse(ok)
        self.assertEqual(why, "request_disallows_partial")


# ---------------------------------------------------------------- spending


class TestSpending(unittest.TestCase):
    def test_only_flexible_and_permitted_streams(self):
        for rid in ALL:
            ctx, ev = context(rid)
            for change in permitted_changes(ctx.profile, ev.streams):
                stream = next(s for s in ev.streams if s.key == change.stream_key)
                if change.kind == "stop":
                    self.assertIn(stream.flexibility,
                                  {"stoppable", "reducible_or_stoppable"})
                    self.assertIn(stream.group, ctx.profile.stoppable_categories)
                else:
                    self.assertIn(stream.flexibility,
                                  {"reducible", "reducible_or_stoppable"})
                    self.assertIn(stream.group, ctx.profile.reducible_categories)
                    self.assertEqual(change.new_amount, stream.minimum_allowed_amount)

    def test_change_sets_are_bounded_and_distinct(self):
        ctx, ev = context("request_21")
        allowed = permitted_changes(ctx.profile, ev.streams)
        for combo in change_sets(allowed):
            self.assertLessEqual(len(combo), 3)
            keys = [c.stream_key for c in combo]
            self.assertEqual(len(set(keys)), len(keys))

    def test_changes_cite_the_latest_occurrence(self):
        ctx, ev = context("request_21")
        for change in permitted_changes(ctx.profile, ev.streams):
            stream = next(s for s in ev.streams if s.key == change.stream_key)
            self.assertEqual(change.event_id, stream.latest_event_id)


# ---------------------------------------------------------------- validator


class TestValidator(unittest.TestCase):
    def test_accepts_every_selected_plan(self):
        for rid in ALL:
            ctx, ev, planning = planned(rid)
            chosen = predict(ctx, STORE)
            method = chosen.result.recommended_payment_method
            plan = next((c for c in planning.candidates if c.method == method
                         and serialise_plan(c.payments) == chosen.result.payment_plan), None)
            if plan is None:
                continue
            self.assertEqual(validate(plan, ev, ENGINE, planning.safe.amount,
                                      planning.earliest), (), rid)

    def test_rejects_a_tampered_installment_schedule(self):
        for rid in ALL:
            ctx, ev, planning = planned(rid)
            inst = next((c for c in planning.candidates if c.method == "installments"), None)
            if inst is None:
                continue
            bad = dataclasses.replace(
                inst, payments=tuple((d, a + Decimal("1")) for d, a in inst.payments))
            codes = {v.code for v in validate(bad, ev, ENGINE, planning.safe.amount,
                                              planning.earliest)}
            self.assertIn("V7", codes, rid)
            return
        self.skipTest("no installment candidate in the sample set")

    def test_rejects_an_unsafe_plan(self):
        ctx, ev, planning = planned("request_01")
        huge = dataclasses.replace(
            planning.candidates[0],
            payments=((ctx.request.request_date, ctx.request.requested_amount * 100),),
            total_paid=ctx.request.requested_amount * 100)
        codes = {v.code for v in validate(huge, ev, ENGINE, planning.safe.amount,
                                          planning.earliest)}
        self.assertIn("V2", codes)

    def test_rejects_stop_and_reduce_on_one_stream(self):
        ctx, ev, planning = planned("request_21")
        allowed = permitted_changes(ctx.profile, ev.streams)
        if len(allowed) < 2:
            self.skipTest("needs two changes")
        duplicate = SpendingChange(kind="reduce_to", stream_key=allowed[0].stream_key,
                                   event_id=allowed[0].event_id,
                                   new_amount=Decimal("1"), monthly_saving=Decimal("1"))
        bad = dataclasses.replace(planning.candidates[0], changes=(allowed[0], duplicate))
        codes = {v.code for v in validate(bad, ev, ENGINE, planning.safe.amount,
                                          planning.earliest)}
        self.assertIn("V11", codes)

    def test_rejects_more_than_three_changes(self):
        ctx, ev, planning = planned("request_21")
        fake = tuple(
            SpendingChange(kind="stop", stream_key=("expense", f"x{i}"), event_id=f"e{i}",
                           new_amount=None, monthly_saving=Decimal(0)) for i in range(4))
        bad = dataclasses.replace(planning.candidates[0], changes=fake)
        codes = {v.code for v in validate(bad, ev, ENGINE, planning.safe.amount,
                                          planning.earliest)}
        self.assertIn("V10", codes)

    def test_rejects_out_of_order_payments(self):
        ctx, ev, planning = planned("request_01")
        asof = ctx.request.request_date
        bad = dataclasses.replace(
            planning.candidates[0], method="partial_payment",
            payments=((asof + timedelta(days=5), Decimal("10")), (asof, Decimal("10"))))
        codes = {v.code for v in validate(bad, ev, ENGINE, planning.safe.amount,
                                          planning.earliest)}
        self.assertIn("V13", codes)


# ---------------------------------------------------------------- ranking


class TestCompletion(unittest.TestCase):
    def test_fee_bearing_installments_complete_by_deadline(self):
        """A financing fee makes an installment plan pay MORE than requested; it still completes.

        Regression: completion used `total == requested`, so every fee-bearing installment plan
        counted as "never completes" under ranking rule 1.
        """
        ds = Dataset.load()
        checked = 0
        for rid in ds.eval_request_ids:
            ctx, ev, planning = planned(rid)
            for c in planning.candidates:
                if (c.method == "installments" and c.total_paid > ctx.request.requested_amount
                        and max(d for d, _ in c.payments) <= ctx.request.desired_completion_date):
                    self.assertTrue(c.completes_by_deadline, f"{rid} {c.source_option_id}")
                    checked += 1
            if checked >= 5:
                break
        self.assertGreater(checked, 0, "no fee-bearing installment plan found to check")


class TestRanking(unittest.TestCase):
    def _plan(self, **kw):
        base = dict(method="installments", payments=((date(2024, 1, 1), Decimal("10")),),
                    changes=(), source_option_id="payment_option_01",
                    total_paid=Decimal("10"), completes_full_amount=True,
                    completes_by_deadline=True, implied_status="affordable_with_plan")
        base.update(kw)
        return CandidatePlan(**base)

    def test_rule1_completion_beats_everything(self):
        complete = self._plan(total_paid=Decimal("100"))
        incomplete = self._plan(completes_by_deadline=False, total_paid=Decimal("1"))
        self.assertIs(rank([incomplete, complete])[0], complete)

    def test_rule2_no_changes_preferred(self):
        clean = self._plan(total_paid=Decimal("100"))
        changed = self._plan(total_paid=Decimal("1"), changes=(
            SpendingChange("stop", ("expense", "gym"), "e1", None, Decimal(0)),))
        self.assertIs(rank([changed, clean])[0], clean)

    def test_rule2_is_boolean_not_a_change_count(self):
        """The spec states rule 2 as "require no spending changes", not "fewer changes".

        Plans that both require changes tie on rule 2 and are separated by rule 3, so a
        cheaper two-change plan must beat a dearer one-change plan.
        """
        one_change = self._plan(total_paid=Decimal("100"), changes=(
            SpendingChange("stop", ("expense", "gym"), "e1", None, Decimal(0)),))
        two_changes_cheaper = self._plan(total_paid=Decimal("90"), changes=(
            SpendingChange("stop", ("expense", "gym"), "e1", None, Decimal(0)),
            SpendingChange("stop", ("expense", "dining"), "e2", None, Decimal(0))))
        self.assertIs(rank([one_change, two_changes_cheaper])[0], two_changes_cheaper)

    def test_rule3_minimise_total_paid(self):
        cheap = self._plan(total_paid=Decimal("100"))
        dear = self._plan(total_paid=Decimal("120"))
        self.assertIs(rank([dear, cheap])[0], cheap)

    def test_rule4_start_earlier(self):
        early = self._plan(payments=((date(2024, 1, 1), Decimal("10")),))
        late = self._plan(payments=((date(2024, 2, 1), Decimal("10")),))
        self.assertIs(rank([late, early])[0], early)

    def test_rule5_fewer_payments(self):
        one = self._plan(payments=((date(2024, 1, 1), Decimal("10")),))
        two = self._plan(payments=((date(2024, 1, 1), Decimal("5")),
                                   (date(2024, 1, 2), Decimal("5"))))
        self.assertIs(rank([two, one])[0], one)

    def test_rule6_lowest_option_id(self):
        a = self._plan(source_option_id="payment_option_01")
        b = self._plan(source_option_id="payment_option_02")
        self.assertIs(rank([b, a])[0], a)

    def test_fallback_is_always_last(self):
        real = self._plan()
        fallback = self._plan(method="not_recommended", payments=(), total_paid=Decimal(0),
                              completes_by_deadline=False, source_option_id=None)
        self.assertIs(rank([fallback, real])[-1], fallback)


# ---------------------------------------------------------------- status and output


class TestStatusMapping(unittest.TestCase):
    def test_mapping(self):
        def plan(method, changes=()):
            return CandidatePlan(method=method, payments=(), changes=changes,
                                 source_option_id=None, total_paid=Decimal(0),
                                 completes_full_amount=True, completes_by_deadline=True,
                                 implied_status="affordable_now")
        self.assertEqual(status_for(plan("full_payment")), "affordable_now")
        self.assertEqual(status_for(plan("wait")), "affordable_later")
        self.assertEqual(status_for(plan("not_recommended")), "not_affordable")
        self.assertEqual(status_for(plan("installments")), "affordable_with_plan")
        self.assertEqual(status_for(plan("partial_payment")), "affordable_with_plan")
        change = SpendingChange("stop", ("expense", "gym"), "e1", None, Decimal(0))
        self.assertEqual(status_for(plan("full_payment", (change,))), "affordable_with_plan")


class TestSerialisation(unittest.TestCase):
    def test_plan_format(self):
        self.assertEqual(serialise_plan(()), "none")
        self.assertEqual(
            serialise_plan(((date(2024, 9, 4), Decimal("28820")),
                            (date(2024, 9, 15), Decimal("10840")))),
            "2024-09-04:28820|2024-09-15:10840")
        self.assertEqual(serialise_plan(((date(2026, 1, 3), Decimal("620.40")),)),
                         "2026-01-03:620.40")

    def test_changes_format(self):
        self.assertEqual(serialise_changes(()), "none")
        changes = (SpendingChange("stop", ("expense", "cloud_storage"), "event_1815", None,
                                  Decimal(0)),
                   SpendingChange("reduce_to", ("expense", "streaming"), "event_1816",
                                  Decimal("23.50"), Decimal(0)))
        self.assertEqual(serialise_changes(changes),
                         "stop:event_1815|reduce_to:event_1816:23.50")


if __name__ == "__main__":
    unittest.main(verbosity=2)
