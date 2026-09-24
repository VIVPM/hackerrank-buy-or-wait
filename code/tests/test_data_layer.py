"""Data-layer tests. Real dataset rows wherever a real row proves the point.

    python -m unittest discover -s code/tests -t code -v
"""

from __future__ import annotations

import sys
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bow.dataset import Dataset, _parse_bool, _parse_date, _parse_decimal, load_rates  # noqa: E402
from bow.errors import DatasetError, MissingRateError                                  # noqa: E402
from bow.fx import ExchangeRateService, RateRow                                        # noqa: E402
from bow.money import Money, format_amount, quantize, to_decimal                       # noqa: E402

DS = Dataset.load()


# ---------------------------------------------------------------- parsing


class TestParsing(unittest.TestCase):
    def test_decimal_is_exact(self):
        self.assertEqual(to_decimal("0.1") + to_decimal("0.2"), Decimal("0.3"))
        self.assertEqual(_parse_decimal("15833.33", "f", 1, "rate"), Decimal("15833.33"))

    def test_float_rejected(self):
        with self.assertRaises(TypeError):
            to_decimal(0.1)

    def test_bad_decimal_names_the_cell(self):
        with self.assertRaises(DatasetError) as cm:
            _parse_decimal("12,34", "financial_events.csv", 99, "amount")
        self.assertIn("financial_events.csv:99", str(cm.exception))
        self.assertIn("amount", str(cm.exception))

    def test_date_parsing(self):
        self.assertEqual(_parse_date("2024-03-03", "f", 1, "d"), date(2024, 3, 3))
        for bad in ("03/03/2024", "2024-13-01", "2024-3-3x", ""):
            with self.assertRaises(DatasetError):
                _parse_date(bad, "f", 1, "d")

    def test_bool_parsing(self):
        self.assertTrue(_parse_bool("True", "f", 1, "c"))
        self.assertFalse(_parse_bool("false", "f", 1, "c"))
        with self.assertRaises(DatasetError):
            _parse_bool("maybe", "f", 1, "c")

    def test_output_amount_formatting(self):
        # Both forms appear in the solved samples.
        self.assertEqual(format_amount(Decimal("25256.00")), "25256")
        self.assertEqual(format_amount(Decimal("620.4")), "620.40")
        self.assertEqual(format_amount(Decimal("15952906.666")), "15952906.67")
        self.assertEqual(format_amount(Decimal("0")), "0")
        self.assertEqual(quantize(Decimal("2.345")), Decimal("2.35"))


# ---------------------------------------------------------------- loading


class TestLoading(unittest.TestCase):
    def test_row_counts(self):
        self.assertEqual(len(DS.profiles), 275)
        self.assertEqual(len(DS.eval_request_ids), 250)
        self.assertEqual(len(DS.sample_request_ids), 25)
        self.assertEqual(len(DS.requests), 275)
        self.assertEqual(len(DS.expected), 25)
        self.assertEqual(len(DS.event_by_id), 25342)
        self.assertEqual(sum(len(v) for v in DS.options_by_request.values()), 790)
        self.assertEqual(sum(len(v) for v in DS.messages_by_user.values()), 215)
        self.assertEqual(sum(len(v) for v in DS.images_by_user.values()), 16)
        self.assertEqual(len(DS.fx), 134)

    def test_money_is_decimal_not_float(self):
        event = DS.event_by_id["event_01"]
        self.assertIsInstance(event.amount.amount, Decimal)
        self.assertEqual(event.amount, Money(Decimal("5148"), "ZAR"))
        self.assertIsInstance(DS.profiles["user_01"].current_available_balance, Decimal)

    def test_blank_amount_is_none_not_zero(self):
        """16 events have no amount; an image supplies it. None must never become 0."""
        blanks = [e for e in DS.event_by_id.values() if e.amount is None]
        self.assertEqual(len(blanks), 16)
        self.assertIsNone(DS.event_by_id["event_253"].amount)
        self.assertTrue(all(i.related_event_id in DS.event_by_id
                            for v in DS.images_by_user.values() for i in v))

    def test_dates_are_typed(self):
        event = DS.event_by_id["event_102"]
        self.assertEqual(event.event_date, date(2024, 3, 2))
        self.assertEqual(event.settlement_date, date(2024, 3, 5))
        self.assertEqual(DS.requests["request_01"].request_date, date(2024, 3, 3))

    def test_unrealized_events_have_no_settlement_date(self):
        no_settle = [e for e in DS.event_by_id.values() if e.settlement_date is None]
        self.assertEqual(len(no_settle), 10)
        self.assertTrue(all(e.status == "unrealized" for e in no_settle))

    def test_profile_list_fields(self):
        p = DS.profiles["user_01"]
        self.assertEqual(p.home_currency, "ZAR")
        self.assertIn("rent", p.protected_categories)
        self.assertEqual(p.methods, frozenset({"full_payment"}))
        self.assertIsNone(p.max_installment_months)          # blank => no installments
        self.assertEqual(DS.profiles["user_02"].max_installment_months, 7)

    def test_option_schedule_derived(self):
        (opt,) = [o for o in DS.options_by_request["request_02"]
                  if o.payment_option_id == "payment_option_05"]
        self.assertEqual(opt.number_of_payments, 3)
        self.assertEqual([d for d, _ in opt.schedule],
                         [date(2025, 8, 8), date(2025, 9, 7), date(2025, 10, 7)])
        self.assertEqual(opt.last_payment_date, date(2025, 10, 7))
        self.assertEqual(opt.payment_amount * 3, opt.total_payable_amount)

    def test_duplicate_id_rejected(self):
        with self.assertRaises(DatasetError):
            from bow.dataset import _check_unique
            _check_unique(["a", "b", "a"], "f.csv", "id")

    def test_missing_file_rejected(self):
        with self.assertRaises(DatasetError):
            load_rates(Path("dataset") / "does_not_exist.csv")


# ---------------------------------------------------------------- joins


class TestJoins(unittest.TestCase):
    def test_request_profile_event_join(self):
        ctx = DS.context_for("request_01")
        self.assertEqual(ctx.profile.user_id, "user_01")
        self.assertEqual(ctx.home_currency, "ZAR")
        self.assertEqual(len(ctx.events), 103)
        self.assertTrue(all(e.user_id == "user_01" for e in ctx.events))

    def test_events_sorted_by_cash_date(self):
        ctx = DS.context_for("request_01")
        keys = [(e.settlement_date or e.event_date) for e in ctx.events]
        self.assertEqual(keys, sorted(keys))

    def test_linked_event_lookup(self):
        # event_99 is the settled reversal of event_98.
        refund = DS.event_by_id["event_99"]
        self.assertEqual(refund.linked_event_id, "event_98")
        ctx = DS.context_for("request_01")
        original = ctx.linked(refund)
        self.assertIsNotNone(original)
        self.assertEqual(original.event_id, "event_98")
        self.assertEqual(original.amount.amount, refund.amount.amount)   # nets to zero

    def test_no_dangling_links(self):
        dangling = [e.event_id for e in DS.event_by_id.values()
                    if e.linked_event_id and e.linked_event_id not in DS.event_by_id]
        self.assertEqual(dangling, [])

    def test_message_event_join(self):
        # message_14 describes event_1785, the pending refund for user_20.
        msgs = DS.messages_by_event["event_1785"]
        self.assertEqual([m.message_id for m in msgs], ["message_14"])
        self.assertEqual(msgs[0].user_id, DS.event_by_id["event_1785"].user_id)

    def test_message_indexes_agree(self):
        for user_id, msgs in DS.messages_by_user.items():
            self.assertTrue(all(m.user_id == user_id for m in msgs))
        total = sum(len(v) for v in DS.messages_by_user.values())
        self.assertEqual(total, 215)
        self.assertEqual(sum(len(v) for v in DS.messages_by_request.values()), 128)

    def test_image_event_join_and_path(self):
        img = DS.images_by_event["event_253"][0]
        self.assertEqual(img.image_id, "image_01")
        self.assertEqual(img.related_event_id, "event_253")
        self.assertTrue(Path(img.path).is_file(), img.path)
        self.assertTrue(img.path.endswith("image_01.png"))

    def test_every_image_file_exists(self):
        for images in DS.images_by_user.values():
            for img in images:
                self.assertTrue(Path(img.path).is_file(), img.path)

    def test_payment_option_retrieval(self):
        opts = DS.options_by_request["request_01"]
        self.assertEqual(len(opts), 4)
        full = [o for o in opts if o.payment_method == "full_payment"]
        self.assertEqual(len(full), 1)
        self.assertEqual(full[0].payment_amount, DS.requests["request_01"].requested_amount)
        self.assertEqual(full[0].first_payment_date, DS.requests["request_01"].request_date)

    def test_context_is_request_scoped(self):
        ctx = DS.context_for("request_20")
        self.assertTrue(all(m.request_id in (None, "request_20") for m in ctx.messages))
        self.assertTrue(all(i.request_id in (None, "request_20") for i in ctx.images))
        self.assertEqual(len(ctx.images), 1)

    def test_unknown_request_fails_loudly(self):
        with self.assertRaises(DatasetError):
            DS.context_for("request_99999")

    def test_sample_and_eval_share_one_namespace(self):
        self.assertEqual(set(DS.eval_request_ids) & set(DS.sample_request_ids), set())
        self.assertIn("request_01", DS.requests)     # sample
        self.assertIn("request_26", DS.requests)     # eval
        self.assertNotIn("request_26", DS.expected)


# ---------------------------------------------------------------- fx


class TestExchangeRates(unittest.TestCase):
    def test_identity(self):
        m = Money(Decimal("100"), "ZAR")
        got = DS.fx.to_home(m, "ZAR", date(2024, 3, 3))
        self.assertEqual(got.converted, Decimal("100.00"))
        self.assertEqual(got.rate_source, "identity")
        self.assertFalse(got.is_converted)

    def test_real_foreign_payroll_usd_to_idr(self):
        """event_2288: user_25 is an IDR profile paid USD 1800 on 2024-03-15."""
        event = DS.event_by_id["event_2288"]
        self.assertEqual(event.amount, Money(Decimal("1800"), "USD"))
        got = DS.fx.to_home(event.amount, "IDR", event.settlement_date)
        self.assertEqual(got.rate, Decimal("15833.33"))
        self.assertEqual(got.rate_source, "exact_date")
        self.assertEqual(got.rate_date, date(2024, 3, 15))
        self.assertEqual(got.converted, Decimal("28499994.00"))

    def test_real_foreign_expense_usd_to_inr_on_first_of_month(self):
        """event_7307: USD taxi fare for an INR profile settling 2025-10-01, not a 15th."""
        event = DS.event_by_id["event_7307"]
        self.assertIsNone(event.amount)                       # image supplies the value
        rate, rate_date, source = DS.fx.rate_for("USD", "INR", date(2025, 10, 1))
        self.assertEqual(source, "exact_date")
        self.assertEqual(rate_date, date(2025, 10, 1))
        self.assertEqual(rate, Decimal("83.33"))

    def test_eur_to_zar(self):
        got = DS.fx.to_home(Money(Decimal("1804"), "EUR"), "ZAR", date(2025, 3, 15))
        self.assertEqual(got.rate, Decimal("20"))
        self.assertEqual(got.converted, Decimal("36080.00"))

    def test_every_foreign_event_resolves_exactly(self):
        """Every cross-currency row must hit step 2 of the documented rule.

        140 rows are cross-currency. 139 carry an amount; the 140th (event_7307, a USD taxi
        fare on an INR profile) has a blank amount pending image extraction, so it is counted
        separately here and its rate is asserted in the test above.
        """
        with_amount, blank = 0, []
        for user_id, events in DS.events_by_user.items():
            home = DS.profiles[user_id].home_currency
            for e in events:
                currency = e.amount.currency if e.amount else None
                on = e.settlement_date or e.event_date
                if e.amount is None:
                    # Currency is on the row even when the amount is not.
                    raw = DS.event_by_id[e.event_id]
                    if raw.amount is None and e.category and e.event_id == "event_7307":
                        blank.append(e.event_id)
                    continue
                if currency == home:
                    continue
                _, _, source = DS.fx.rate_for(currency, home, on)
                self.assertEqual(source, "exact_date", f"{e.event_id} fell back to {source}")
                with_amount += 1
        self.assertEqual(with_amount, 139)
        self.assertEqual(blank, ["event_7307"])

    def test_carry_forward_never_reads_the_future(self):
        svc = ExchangeRateService([
            RateRow(date(2024, 1, 15), "USD", "INR", Decimal("80")),
            RateRow(date(2024, 2, 15), "USD", "INR", Decimal("90")),
        ])
        rate, rate_date, source = svc.rate_for("USD", "INR", date(2024, 2, 10))
        self.assertEqual((rate, rate_date, source), (Decimal("80"), date(2024, 1, 15),
                                                     "carry_forward"))

    def test_inverse_exact_beats_stale_direct(self):
        svc = ExchangeRateService([
            RateRow(date(2024, 1, 15), "USD", "EUR", Decimal("0.90")),
            RateRow(date(2024, 2, 15), "EUR", "USD", Decimal("1.10")),
        ])
        rate, rate_date, source = svc.rate_for("USD", "EUR", date(2024, 2, 15))
        self.assertEqual(source, "inverse_exact")
        self.assertEqual(rate_date, date(2024, 2, 15))
        self.assertAlmostEqual(float(rate), 1 / 1.10, places=10)

    def test_missing_rate_fails_closed(self):
        svc = ExchangeRateService([RateRow(date(2024, 1, 15), "USD", "INR", Decimal("80"))])
        with self.assertRaises(MissingRateError):
            svc.rate_for("ZAR", "IDR", date(2024, 1, 15))          # pair absent entirely
        with self.assertRaises(MissingRateError):
            svc.rate_for("USD", "INR", date(2023, 12, 31))         # before the first quote

    def test_conversion_keeps_provenance(self):
        got = DS.fx.to_home(Money(Decimal("1800"), "USD"), "IDR", date(2024, 3, 15))
        self.assertEqual(got.source.amount, Decimal("1800"))
        self.assertEqual(got.source.currency, "USD")
        self.assertEqual(got.target_currency, "IDR")
        self.assertIsNotNone(got.rate_date)
        self.assertTrue(got.is_converted)


# ---------------------------------------------------------------- harness utilities


class TestRegressionUtilities(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
        import regress
        self.r = regress

    def test_parse_plan(self):
        self.assertEqual(self.r.parse_plan("none"), ())
        self.assertEqual(self.r.parse_plan(""), ())
        self.assertEqual(
            self.r.parse_plan("2024-09-04:28820|2024-09-15:10840"),
            ((date(2024, 9, 4), Decimal("28820")), (date(2024, 9, 15), Decimal("10840"))))

    def test_plan_compare_ignores_trailing_zero(self):
        self.assertEqual(self.r.parse_plan("2026-01-03:620.40"),
                         self.r.parse_plan("2026-01-03:620.4"))

    def test_parse_changes_order_insensitive(self):
        a = self.r.parse_changes("stop:event_1815|reduce_to:event_1816:23.50")
        b = self.r.parse_changes("reduce_to:event_1816:23.5|stop:event_1815")
        self.assertEqual(a, b)
        self.assertEqual(self.r.parse_changes("none"), frozenset())

    def test_norm_date(self):
        for blank in ("", "none", "nan", "  "):
            self.assertEqual(self.r.norm_date(blank), "")
        self.assertEqual(self.r.norm_date("2024-03-03"), "2024-03-03")

    def test_oracle_scores_perfect_and_null_scores_zero(self):
        perfect = self.r.run(self.r._oracle(DS), DS)
        self.assertEqual(len(perfect.scored), 25)
        self.assertTrue(all(s.all_correct for s in perfect.scored))
        empty = self.r.run(self.r._null, DS)
        self.assertFalse(any(s.all_correct for s in empty.scored))

    def test_broken_predictor_is_survived_not_raised(self):
        def boom(_ctx):
            raise RuntimeError("nope")
        report = self.r.run(boom, DS, ["request_01"])
        self.assertEqual(len(report.scored), 0)
        self.assertIn("RuntimeError", report.scores[0].error)

    def test_single_request_selection(self):
        report = self.r.run(self.r._oracle(DS), DS, ["request_07"])
        self.assertEqual([s.request_id for s in report.scores], ["request_07"])

    def test_error_metrics(self):
        report = self.r.run(self.r._oracle(DS), DS, ["request_02"])
        score = report.scores[0]
        self.assertEqual(score.abs_error, Decimal(0))
        self.assertEqual(score.rel_error, Decimal(0))
        self.assertEqual(score.error_over_requested, Decimal(0))

    def test_capped_flag(self):
        report = self.r.run(self.r._oracle(DS), DS)
        capped = {s.request_id for s in report.scores if s.is_capped}
        # safe == requested for exactly these four; only three are affordable_now.
        self.assertEqual(capped, {"request_01", "request_09", "request_12", "request_16"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
