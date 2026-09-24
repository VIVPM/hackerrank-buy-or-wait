"""Recurrence inference, forecasting and safe-amount tests.

Assertions are about financial mechanics, never about a solved sample's answer.

    python -m unittest discover -s code/tests -t code -v
"""

from __future__ import annotations

import sys
import unittest
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bow.config import ForecastConfig                                        # noqa: E402
from bow.dataset import Dataset                                              # noqa: E402
from bow.forecast import ForecastContext, ForecastEngine                     # noqa: E402
from bow.models import ImageFact, MessageFact                                # noqa: E402
from bow.money import Money                                                  # noqa: E402
from bow.recurrence import (                                                 # noqa: E402
    add_month, apply_image_facts, apply_stream_directives, build_evidence,
    classify_income, estimate_amount, infer_streams, occurrences,
)
from bow.resolve import StreamDirective, resolve_context                     # noqa: E402
from bow.safeamount import earliest_full_payment_date, safe_amount           # noqa: E402
from bow.trace import Trace                                                  # noqa: E402

DS = Dataset.load()
CFG = ForecastConfig()
ENGINE = ForecastEngine(CFG)


def pipeline(request_id: str, config: ForecastConfig = CFG):
    ctx = DS.context_for(request_id)
    res = resolve_context(ctx)
    ev = build_evidence(ctx, res, config)
    fctx = ForecastContext(ev, ctx.request.request_date)
    return ctx, ev, fctx, ForecastEngine(config).project(fctx)


def request_for_user(user_id: str) -> str:
    return next(r for r, q in DS.requests.items() if q.user_id == user_id)


# ---------------------------------------------------------------- income taxonomy


class TestIncomeTaxonomy(unittest.TestCase):
    def test_continuing_families(self):
        for d in ("Payroll credit", "Base salary", "International employer payroll",
                  "First-job payroll", "Primary household salary", "Second household income",
                  "Payroll after returning from leave", "Prorated first salary"):
            self.assertEqual(classify_income(d), "continuing", d)

    def test_terminal_families(self):
        for d in ("Final employer payroll", "Previous employer payroll", "Payroll before leave"):
            self.assertEqual(classify_income(d), "terminal", d)

    def test_never_projected_families(self):
        for d in ("Quarterly performance bonus", "Performance commission",
                  "Monthly sales commission", "Promotion arrears payment", "Prize proceeds"):
            self.assertEqual(classify_income(d), "one_off", d)
        for d in ("Driver platform payout", "Delivery platform payout", "Weekly app earnings",
                  "Task marketplace payout", "Consulting invoice payment",
                  "Content contract payment", "Website project payment",
                  "Freelance milestone payment", "Client retainer payment",
                  "Independent work payment", "Seasonal contract payment", "Peak-season wages",
                  "Temporary assignment pay"):
            self.assertEqual(classify_income(d), "irregular", d)

    def test_unknown_description_is_not_projected(self):
        self.assertEqual(classify_income("Mystery windfall"), "unknown")

    def test_every_income_description_in_the_dataset_is_classified(self):
        seen = {e.description for e in DS.event_by_id.values() if e.event_type == "income"}
        for d in seen:
            self.assertIn(classify_income(d),
                          {"continuing", "terminal", "one_off", "irregular"}, d)


class TestIncomeStreams(unittest.TestCase):
    def test_commissions_do_not_inflate_base_salary(self):
        """user_11 has Base salary plus two commissions, all in the salary category."""
        _, ev, _, _ = pipeline(request_for_user("user_11"))
        income = [s for s in ev.streams if s.kind == "income"]
        self.assertEqual([s.group for s in income], ["Base salary"])

    def test_terminated_employment_projects_no_income(self):
        """user_05's last income row is a Final employer payroll."""
        _, ev, _, _ = pipeline(request_for_user("user_05"))
        self.assertEqual([s for s in ev.streams if s.kind == "income"], [])

    def test_gig_income_forms_one_stream_not_many(self):
        """Gig and freelance labels vary per payment the way grocery descriptions do.

        Grouping them by description hides a perfectly regular income behind a dozen
        one-observation series and leaves the user with no projected income at all.
        """
        _, ev, _, _ = pipeline(request_for_user("user_10"))
        income = [s for s in ev.streams if s.kind == "income"]
        self.assertEqual(len(income), 1)
        self.assertEqual(income[0].income_class, "irregular")
        self.assertGreater(income[0].observations, 5)

    def test_a_message_about_gig_income_suppresses_it(self):
        """Projected from history, but not when a message says the payout is unconfirmed."""
        from bow.ai.facts import load_facts
        store = load_facts(DS)
        ctx = DS.context_for(request_for_user("user_10"))
        m, i = store.for_user(ctx.request.user_id)
        ev = build_evidence(ctx, resolve_context(ctx, m), CFG, i)
        self.assertEqual([s for s in ev.streams if s.kind == "income"], [])

    def test_confirmed_future_salary_anchors_the_projection(self):
        """user_01 has one prorated salary plus a scheduled Next confirmed salary."""
        ctx, ev, fctx, fc = pipeline("request_01")
        income = [s for s in ev.streams if s.kind == "income"]
        self.assertEqual(len(income), 1)
        self.assertEqual(income[0].amount, Decimal("23320.00"))
        self.assertEqual(income[0].anchor_date, date(2024, 3, 15))
        # The explicit row is the first occurrence; the series must start a month later.
        salary_days = sorted(f.date for f in fc.flows if f.amount > 0)
        self.assertEqual(salary_days, [date(2024, 3, 15), date(2024, 4, 15), date(2024, 5, 15)])

    def test_confirmed_salary_is_not_double_counted(self):
        ctx, ev, fctx, fc = pipeline("request_21")
        on_anchor = [f for f in fc.flows if f.date == date(2026, 4, 15) and f.amount > 0]
        self.assertEqual(len(on_anchor), 1, "salary counted twice on its confirmed date")

    def test_stale_income_stream_is_dropped(self):
        """user_13's second household income stops in January; the request is in March."""
        ctx, ev, _, _ = pipeline(request_for_user("user_13"))
        self.assertNotIn("Second household income", [s.group for s in ev.streams])


# ---------------------------------------------------------------- cadence and estimators


class TestCadence(unittest.TestCase):
    def test_calendar_month_does_not_drift(self):
        self.assertEqual(add_month(date(2024, 1, 31), 1, 31), date(2024, 2, 29))
        self.assertEqual(add_month(date(2024, 1, 31), 2, 31), date(2024, 3, 31))
        self.assertEqual(add_month(date(2023, 1, 31), 1, 31), date(2023, 2, 28))
        # Twelve monthly steps land on the same day of month, unlike twelve +30d steps.
        start = date(2024, 1, 15)
        self.assertEqual(add_month(start, 12, 15), date(2025, 1, 15))
        self.assertNotEqual(start + timedelta(days=360), date(2025, 1, 15))

    def test_monthly_occurrences_hold_the_calendar_day(self):
        _, ev, _, _ = pipeline("request_01")
        rent = next(s for s in ev.streams if s.group == "rent")
        days = list(occurrences(rent, date(2024, 3, 3), date(2024, 6, 1), CFG))
        self.assertEqual(days, [date(2024, 4, 2), date(2024, 5, 2)])
        self.assertEqual({d.day for d in days}, {2})

    def test_submonthly_streams_are_detected(self):
        _, ev, _, _ = pipeline("request_01")
        kinds = {s.group: s.cadence_class for s in ev.streams}
        self.assertEqual(kinds["groceries"], "sub_monthly")
        self.assertEqual(kinds["rent"], "monthly")

    def test_estimator_families(self):
        vals = [Decimal("10"), Decimal("20"), Decimal("60")]
        self.assertEqual(estimate_amount(vals, ForecastConfig(amount_estimator="mean_all")),
                         Decimal("30.00"))
        self.assertEqual(estimate_amount(vals, ForecastConfig(amount_estimator="last")),
                         Decimal("60.00"))
        self.assertEqual(estimate_amount(vals, ForecastConfig(amount_estimator="mean_3")),
                         Decimal("30.00"))

    def test_trailing_horizon_uses_a_time_window_not_a_count(self):
        """The next 90 days are estimated from the last 90, so the window adapts to cadence."""
        asof = date(2026, 7, 1)
        vals = [Decimal("100"), Decimal("100"), Decimal("40"), Decimal("40")]
        dates = [date(2025, 1, 1), date(2026, 1, 1),      # outside the window
                 date(2026, 6, 1), date(2026, 6, 15)]     # inside it
        cfg = ForecastConfig(amount_estimator="trailing_horizon")
        self.assertEqual(estimate_amount(vals, cfg, dates, asof), Decimal("40.00"))
        # With no dates supplied it degrades to the full-history mean rather than failing.
        self.assertEqual(estimate_amount(vals, cfg), Decimal("70.00"))

    def test_trailing_horizon_never_empties_the_window(self):
        """A stream whose last occurrence predates the window still yields an estimate."""
        cfg = ForecastConfig(amount_estimator="trailing_horizon")
        vals = [Decimal("10"), Decimal("20")]
        dates = [date(2020, 1, 1), date(2020, 2, 1)]
        self.assertEqual(estimate_amount(vals, cfg, dates, date(2026, 7, 1)), Decimal("15.00"))


# ---------------------------------------------------------------- forecasting


class TestForecast(unittest.TestCase):
    def test_starting_balance_is_the_snapshot(self):
        ctx, ev, fctx, fc = pipeline("request_01")
        self.assertEqual(fc.opening_balance, ctx.profile.current_available_balance)
        self.assertEqual(fc.path[0].day, ctx.request.request_date)

    def test_history_is_never_replayed(self):
        for rid in DS.sample_request_ids:
            ctx, ev, fctx, fc = pipeline(rid)
            for f in fc.flows:
                self.assertGreaterEqual(f.date, ctx.request.request_date, f"{rid} {f.origin}")

    def test_horizon_is_ninety_days(self):
        ctx, ev, fctx, fc = pipeline("request_01")
        self.assertEqual((fc.path[-1].day - fc.path[0].day).days, 90)
        self.assertEqual(len(fc.path), 91)

    def test_pending_credits_never_appear(self):
        ctx, ev, fctx, fc = pipeline(request_for_user("user_20"))
        origins = {f.origin for f in fc.flows}
        self.assertNotIn("event_1785", origins)          # pending merchant refund

    def test_unrealized_valuation_never_appears(self):
        ctx, ev, fctx, fc = pipeline(request_for_user("user_21"))
        self.assertNotIn("event_1856", {f.origin for f in fc.flows})

    def test_accrual_replaces_discrete_submonthly_flows(self):
        ctx, ev, fctx, fc = pipeline("request_01")
        self.assertGreater(fc.daily_accrual, 0)
        self.assertEqual([f for f in fc.flows if f.origin == "expense:groceries"], [])

    def test_discrete_mode_is_configurable(self):
        cfg = ForecastConfig(submonthly_mode="discrete")
        ctx, ev, fctx, fc = pipeline("request_01", cfg)
        self.assertEqual(fc.daily_accrual, 0)
        self.assertTrue([f for f in fc.flows if f.origin == "expense:groceries"])

    def test_everything_is_decimal(self):
        ctx, ev, fctx, fc = pipeline("request_01")
        for value in (fc.opening_balance, fc.trough, fc.reserve, fc.daily_accrual):
            self.assertIsInstance(value, Decimal)
        for f in fc.flows:
            self.assertIsInstance(f.amount, Decimal)

    def test_trough_is_the_minimum_of_the_path(self):
        for rid in DS.sample_request_ids:
            _, _, _, fc = pipeline(rid)
            self.assertEqual(fc.trough, min(p.low for p in fc.path))

    def test_end_of_day_evaluation_ignores_intraday_posting_order(self):
        """Payday carries both a salary and a bill; the floor must not be the pre-salary dip."""
        _, _, _, eod = pipeline("request_19", ForecastConfig(same_day_order="credits_first"))
        _, _, _, intraday = pipeline("request_19", ForecastConfig(same_day_order="debits_first"))
        self.assertGreater(eod.trough, intraday.trough)
        payday = date(2024, 9, 15)
        self.assertEqual(intraday.trough_date, payday)
        self.assertLess(eod.trough_date, payday)


class TestExplicitVersusInferred(unittest.TestCase):
    def test_scheduled_obligation_supersedes_the_inferred_occurrence(self):
        """user_24: monthly insurance of 2510 on the 6th AND a scheduled 1830 on the 11th.

        One January obligation stated twice. Counting both would double-charge insurance.
        """
        ctx, ev, fctx, fc = pipeline(request_for_user("user_24"))
        january = [f for f in fc.flows
                   if f.date.month == 1 and f.date.year == 2026
                   and ("insurance" in f.origin or f.note == "scheduled_obligation")]
        explicit = [f for f in january if f.kind == "explicit"]
        inferred = [f for f in january if f.kind == "recurring"]
        self.assertEqual(len(explicit), 1)
        self.assertEqual(inferred, [], "inferred insurance survived alongside the scheduled row")
        self.assertEqual(explicit[0].amount, Decimal("-1830.00"))

    def test_supersession_only_removes_one_cycle(self):
        ctx, ev, fctx, fc = pipeline(request_for_user("user_24"))
        later = [f for f in fc.flows if f.origin == "expense:insurance"]
        self.assertTrue(later, "later insurance cycles must still be projected")

    def test_accruing_streams_are_never_superseded(self):
        """user_21's pending fuel authorization must not cancel the transport habit."""
        ctx, ev, fctx, fc = pipeline(request_for_user("user_21"))
        transport = next(s for s in ev.streams if s.group == "transport")
        self.assertEqual(transport.cadence_class, "sub_monthly")
        self.assertGreater(fc.daily_accrual, 0)
        self.assertIn("event_1857", {f.origin for f in fc.flows})


# ---------------------------------------------------------------- safe amount


class TestSafeAmount(unittest.TestCase):
    def test_bounds_always_hold(self):
        for rid in DS.sample_request_ids:
            ctx, ev, fctx, fc = pipeline(rid)
            sa = safe_amount(ENGINE, fctx, ctx.request.requested_amount, fc)
            self.assertGreaterEqual(sa.amount, Decimal(0), rid)
            self.assertLessEqual(sa.amount, ctx.request.requested_amount, rid)

    def test_headroom_and_trough_forms_agree(self):
        for rid in DS.sample_request_ids:
            ctx, ev, fctx, fc = pipeline(rid)
            sa = safe_amount(ENGINE, fctx, ctx.request.requested_amount, fc)
            self.assertEqual(sa.uncapped, sa.headroom - sa.reserve)
            self.assertEqual(sa.headroom,
                             ctx.profile.current_available_balance
                             - ctx.profile.minimum_balance_to_keep)

    def test_diagnostics_are_complete(self):
        ctx, ev, fctx, fc = pipeline("request_01")
        d = safe_amount(ENGINE, fctx, ctx.request.requested_amount, fc).as_dict()
        for key in ("amount_safe_to_pay", "headroom", "forecast_reserve", "trough",
                    "trough_date", "minimum_balance", "opening_balance", "limiting_flows"):
            self.assertIn(key, d)

    def test_paying_the_safe_amount_keeps_the_horizon_safe(self):
        """The defining property: pay it and nothing breaches the minimum."""
        from bow.models import ProjectedCashFlow
        for rid in DS.sample_request_ids:
            ctx, ev, fctx, fc = pipeline(rid)
            sa = safe_amount(ENGINE, fctx, ctx.request.requested_amount, fc)
            if sa.amount == 0:
                continue
            payment = ProjectedCashFlow(date=ctx.request.request_date, amount=-sa.amount,
                                        kind="plan_payment", origin="test")
            after = ENGINE.project(ForecastContext(ev, ctx.request.request_date, (payment,)))
            self.assertTrue(after.is_safe, f"{rid} breaches after paying {sa.amount}")

    def test_one_more_unit_would_breach(self):
        """The safe amount is maximal, not merely sufficient."""
        from bow.models import ProjectedCashFlow
        checked = 0
        for rid in DS.sample_request_ids:
            ctx, ev, fctx, fc = pipeline(rid)
            sa = safe_amount(ENGINE, fctx, ctx.request.requested_amount, fc)
            if sa.is_capped or sa.amount == 0:
                continue
            payment = ProjectedCashFlow(date=ctx.request.request_date,
                                        amount=-(sa.amount + Decimal("1.00")),
                                        kind="plan_payment", origin="test")
            after = ENGINE.project(ForecastContext(ev, ctx.request.request_date, (payment,)))
            self.assertFalse(after.is_safe, f"{rid} could have paid more than {sa.amount}")
            checked += 1
        self.assertGreater(checked, 5)


class TestEarliestFullPayment(unittest.TestCase):
    def test_equals_request_date_when_affordable_today(self):
        ctx, ev, fctx, fc = pipeline("request_01")
        got = earliest_full_payment_date(ENGINE, fctx, ctx.request.requested_amount, fc)
        self.assertEqual(got, ctx.request.request_date)

    def test_is_independent_of_method_preferences(self):
        """user_12 does not accept full_payment, yet retains a financial-capacity date."""
        ctx, ev, fctx, fc = pipeline("request_12")
        self.assertNotIn("full_payment", ctx.profile.methods)
        got = earliest_full_payment_date(ENGINE, fctx, ctx.request.requested_amount, fc)
        self.assertTrue(got is None or isinstance(got, date))   # computed regardless

    def test_none_when_never_affordable_in_horizon(self):
        """Uses the full evidence path: this user's gig income is suppressed by a message."""
        from bow.ai.facts import load_facts
        store = load_facts(DS)
        ctx = DS.context_for("request_10")
        m, i = store.for_user(ctx.request.user_id)
        ev = build_evidence(ctx, resolve_context(ctx, m), CFG, i)
        fctx = ForecastContext(ev, ctx.request.request_date)
        self.assertIsNone(earliest_full_payment_date(ENGINE, fctx,
                                                     ctx.request.requested_amount))

    def test_result_is_actually_safe_and_earlier_dates_are_not(self):
        from bow.models import ProjectedCashFlow
        checked = 0
        for rid in DS.sample_request_ids:
            ctx, ev, fctx, fc = pipeline(rid)
            req = ctx.request.requested_amount
            got = earliest_full_payment_date(ENGINE, fctx, req, fc)
            if got is None:
                continue
            pay = ProjectedCashFlow(date=got, amount=-req, kind="plan_payment", origin="t")
            self.assertTrue(ENGINE.project(ForecastContext(ev, ctx.request.request_date,
                                                           (pay,))).is_safe, rid)
            if got > ctx.request.request_date:
                earlier = ProjectedCashFlow(date=got - timedelta(days=1), amount=-req,
                                            kind="plan_payment", origin="t")
                self.assertFalse(ENGINE.project(ForecastContext(
                    ev, ctx.request.request_date, (earlier,))).is_safe, rid)
            checked += 1
        self.assertGreater(checked, 10)


# ---------------------------------------------------------------- override hooks


class TestOverrideHooks(unittest.TestCase):
    def _fact(self, **kw):
        base = dict(message_id="m", user_id="user_01", fact_type="salary_amount_change",
                    subject="salary", target_event_id=None, effective_date=None, amount=None,
                    multiplier=None, quantified=True, confidence=1.0, evidence_span="")
        base.update(kw)
        return MessageFact(**base)

    def test_salary_amount_change(self):
        _, ev, _, _ = pipeline("request_01")
        from bow.recurrence import StreamSet
        streams = StreamSet(ev.streams, ())
        fact = self._fact(amount=Money(Decimal("40000"), "ZAR"),
                          effective_date=date(2024, 4, 15))
        out = apply_stream_directives(streams, [StreamDirective("salary_change", fact)])
        income = next(s for s in out.streams if s.kind == "income")
        self.assertEqual(income.amount, Decimal("40000.00"))
        # The effective date is the FIRST payment, so it must be projected, not consumed
        # as an anchor that already happened.
        first = next(iter(occurrences(income, date(2024, 3, 3), date(2024, 6, 1), CFG)))
        self.assertEqual(first, date(2024, 4, 15))

    def test_employment_end_removes_all_income(self):
        _, ev, _, _ = pipeline("request_01")
        from bow.recurrence import StreamSet
        fact = self._fact(fact_type="income_ended", quantified=False)
        out = apply_stream_directives(StreamSet(ev.streams, ()),
                                      [StreamDirective("employment_end", fact)])
        self.assertEqual([s for s in out.streams if s.kind == "income"], [])

    def test_suppression_is_scoped_to_the_income_class(self):
        """A note about a gig payout must not terminate payroll, and vice versa."""
        from bow.recurrence import _suppressed_classes
        self.assertEqual(_suppressed_classes("gig_payout"), {"irregular"})
        self.assertEqual(_suppressed_classes("invoice"), {"irregular"})
        self.assertEqual(_suppressed_classes("salary"), {"continuing", "irregular"})

    def test_recurring_expense_multiplier(self):
        _, ev, _, _ = pipeline(request_for_user("user_16"))
        from bow.recurrence import StreamSet
        rent = next(s for s in ev.streams if s.group == "rent")
        fact = self._fact(fact_type="recurring_expense_change", subject="rent",
                          multiplier=Decimal("1.12"))
        out = apply_stream_directives(StreamSet(ev.streams, ()),
                                      [StreamDirective("amend_amount", fact)])
        self.assertEqual(next(s for s in out.streams if s.group == "rent").amount,
                         (rent.amount * Decimal("1.12")).quantize(Decimal("0.01")))

    def test_recurrence_start_creates_income(self):
        _, ev, _, _ = pipeline(request_for_user("user_15"))
        from bow.recurrence import StreamSet
        self.assertEqual([s for s in ev.streams if s.kind == "income"], [])
        fact = self._fact(fact_type="first_salary_confirmed", subject="salary",
                          amount=Money(Decimal("1661"), "EUR"),
                          effective_date=date(2026, 1, 15))
        out = apply_stream_directives(StreamSet(ev.streams, ()),
                                      [StreamDirective("recurrence_start", fact)])
        income = [s for s in out.streams if s.kind == "income"]
        self.assertEqual(len(income), 1)
        self.assertEqual(income[0].amount, Decimal("1661.00"))

    def test_unquantified_fact_changes_nothing(self):
        _, ev, _, _ = pipeline("request_01")
        from bow.recurrence import StreamSet
        before = StreamSet(ev.streams, ())
        fact = self._fact(fact_type="new_recurring_obligation_unquantified",
                          subject="childcare", quantified=False)
        out = apply_stream_directives(before, [StreamDirective("none", fact)])
        self.assertEqual(out.streams, before.streams)

    def test_image_fact_resolves_a_future_obligation(self):
        """user_16's scheduled outstanding rent has no amount until the image supplies it."""
        ctx = DS.context_for(request_for_user("user_16"))
        res = resolve_context(ctx)
        before = res.by_id("event_1442")
        self.assertFalse(before.counts_for_cash)
        self.assertEqual(before.amount_status, "unresolved_image")

        fact = ImageFact(image_id="image_02", related_event_id="event_1442",
                         chosen_amount=Money(Decimal("100000"), "INR"),
                         semantic_label="balance_due", rejected_candidates=(), confidence=0.95,
                         notes="")
        patched = apply_image_facts(res.events, [fact], ctx.profile.home_currency, ctx.fx)
        after = next(e for e in patched if e.economic_id == "event_1442")
        self.assertTrue(after.counts_for_cash)
        self.assertEqual(after.amount_status, "known")
        self.assertEqual(after.signed_amount, Decimal("-100000.00"))
        self.assertFalse(after.counts_for_recurrence)     # a document is never a series
        self.assertIn("image_02", after.evidence_ids)

    def test_image_fact_does_not_resurrect_recurrence(self):
        ctx = DS.context_for(request_for_user("user_17"))
        res = resolve_context(ctx)
        fact = ImageFact(image_id="image_03", related_event_id="event_1545",
                         chosen_amount=Money(Decimal("41272"), "INR"),
                         semantic_label="grand_total", rejected_candidates=(), confidence=0.9,
                         notes="")
        patched = apply_image_facts(res.events, [fact], ctx.profile.home_currency, ctx.fx)
        after = next(e for e in patched if e.economic_id == "event_1545")
        self.assertFalse(after.counts_for_recurrence)
        self.assertFalse(after.counts_for_cash)           # settled history stays historical


class TestTraceability(unittest.TestCase):
    def test_forecast_trace_records_streams_and_flows(self):
        ctx = DS.context_for("request_01")
        trace = Trace("request_01")
        res = resolve_context(ctx, trace=trace)
        ev = build_evidence(ctx, res, CFG, trace=trace)
        fctx = ForecastContext(ev, ctx.request.request_date)
        fc = ENGINE.project_traced(fctx, trace)
        safe_amount(ENGINE, fctx, ctx.request.requested_amount, fc, trace)
        earliest_full_payment_date(ENGINE, fctx, ctx.request.requested_amount, fc, trace)
        stages = {s.stage for s in trace.steps}
        self.assertEqual(stages, {"resolve", "recurrence", "forecast", "safe_amount"})
        labels = {s.label for s in trace.steps}
        self.assertIn("stream", labels)
        self.assertIn("flow", labels)
        self.assertIn("earliest_full_payment", labels)


if __name__ == "__main__":
    unittest.main(verbosity=2)
