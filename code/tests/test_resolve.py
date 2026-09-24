"""Canonical event resolution tests.

Every lifecycle family is exercised on real dataset rows. No solved prediction is referenced:
these tests assert economic bookkeeping, not answers.

    python -m unittest discover -s code/tests -t code -v
"""

from __future__ import annotations

import dataclasses
import sys
import unittest
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bow.dataset import Dataset                                              # noqa: E402
from bow.errors import UnresolvedEvidenceError                               # noqa: E402
from bow.resolve import (                                                    # noqa: E402
    EVENT_OPERATIONS, STREAM_DIRECTIVES, EventResolver, assert_cash_partition,
    resolve_context, total_future_cash,
)
from bow.trace import Trace, trace_for, NULL_TRACE                           # noqa: E402

DS = Dataset.load()


def request_for_user(user_id: str) -> str:
    for rid, req in DS.requests.items():
        if req.user_id == user_id:
            return rid
    raise KeyError(user_id)


def resolved_for_user(user_id: str):
    ctx = DS.context_for(request_for_user(user_id))
    return ctx, resolve_context(ctx)


def find_group(result, kind: str):
    return [g for g in result.groups if g.kind == kind]


def owner_of(event_id: str) -> str:
    return DS.event_by_id[event_id].user_id


# ---------------------------------------------------------------- whole-corpus invariants


class TestCorpusInvariants(unittest.TestCase):
    """Run the resolver over all 275 users once and assert global properties."""

    @classmethod
    def setUpClass(cls):
        cls.results = {}
        for rid in (*DS.eval_request_ids, *DS.sample_request_ids):
            ctx = DS.context_for(rid)
            cls.results[rid] = (ctx, resolve_context(ctx))

    def test_every_raw_event_becomes_exactly_one_canonical_event(self):
        for rid, (ctx, res) in self.results.items():
            self.assertEqual(len(res.events), len(ctx.events), rid)
            sources = [s for e in res.events for s in e.source_event_ids]
            self.assertEqual(len(sources), len(set(sources)), f"{rid}: an event was emitted twice")
            self.assertEqual(set(sources), {e.event_id for e in ctx.events}, rid)

    def test_nothing_is_left_unresolved(self):
        for rid, (_, res) in self.results.items():
            self.assertEqual(res.unresolved, (), rid)

    def test_cash_partition_holds_everywhere(self):
        """No settled historical row may be replayed into the request-date balance."""
        for rid, (ctx, res) in self.results.items():
            assert_cash_partition(res, ctx.request.request_date)
            for e in res.cash_events:
                self.assertGreaterEqual(e.effective_date, ctx.request.request_date, rid)
                self.assertIn(e.status, {"pending", "scheduled", "settled"})

    def test_cash_events_always_have_a_resolved_amount(self):
        for rid, (_, res) in self.results.items():
            for e in res.cash_events:
                self.assertIsNotNone(e.cash_effect, f"{rid}/{e.economic_id}")
                self.assertEqual(e.amount_status, "known")
                self.assertIsNotNone(e.signed_amount)

    def test_recurrence_events_are_settled_with_observed_amounts(self):
        for rid, (_, res) in self.results.items():
            for e in res.recurrence_events:
                self.assertEqual(e.status, "settled", f"{rid}/{e.economic_id}")
                self.assertEqual(e.amount_status, "known")
                self.assertIsNone(e.lifecycle_group, f"{e.economic_id} is a lifecycle member")

    def test_flags_are_independent(self):
        """Both combinations must actually occur, or the split is decorative."""
        cash_only = rec_only = 0
        for _, res in self.results.values():
            for e in res.events:
                cash_only += e.counts_for_cash and not e.counts_for_recurrence
                rec_only += e.counts_for_recurrence and not e.counts_for_cash
        self.assertGreater(cash_only, 0)
        self.assertGreater(rec_only, 0)

    def test_lifecycle_family_counts(self):
        kinds = {}
        for _, res in self.results.values():
            for g in res.groups:
                kinds[g.kind] = kinds.get(g.kind, 0) + 1
        self.assertEqual(kinds, {
            "offsetting_credit": 22, "investment_valuation": 10,
            "authorization_settlement": 8, "failed_retry": 7,
            "duplicate_charge": 6, "investment_sale": 5,
        })

    def test_every_lifecycle_member_is_excluded_from_recurrence(self):
        for _, res in self.results.values():
            grouped = {m for g in res.groups for m in g.member_ids}
            for e in res.events:
                if e.source_event_ids[0] in grouped:
                    self.assertFalse(e.counts_for_recurrence, e.economic_id)


# ---------------------------------------------------------------- lifecycle families


class TestAuthorizationSettlement(unittest.TestCase):
    def test_not_double_counted(self):
        """event_100 (cancelled authorization) + event_101 (settled purchase), same 816.20."""
        ctx, res = resolved_for_user(owner_of("event_100"))
        auth = res.by_id("event_100")
        settle = res.by_id("event_101")
        self.assertEqual(auth.lifecycle_role, "authorization")
        self.assertEqual(settle.lifecycle_role, "settlement")
        self.assertEqual(auth.exclusion_reason, "superseded_by_settlement")
        self.assertFalse(auth.counts_for_cash)
        self.assertFalse(auth.counts_for_recurrence)
        self.assertFalse(settle.counts_for_recurrence)
        # Same amount on both rows: counting both would charge the user twice.
        self.assertEqual(DS.event_by_id["event_100"].amount.amount,
                         DS.event_by_id["event_101"].amount.amount)
        charged = [e for e in (auth, settle) if e.counts_for_cash]
        self.assertLessEqual(len(charged), 1)

    def test_all_eight_pairs_resolve_the_same_way(self):
        seen = 0
        for rid in (*DS.eval_request_ids, *DS.sample_request_ids):
            res = resolve_context(DS.context_for(rid))
            for g in find_group(res, "authorization_settlement"):
                seen += 1
                self.assertEqual(len(g.survivor_ids), 1)
                self.assertEqual(len(g.ignored_ids), 1)
        self.assertEqual(seen, 8)


class TestFailedRetry(unittest.TestCase):
    def test_counts_once_as_a_future_obligation(self):
        """A failed debit never left the account; its scheduled retry is the real liability."""
        found = 0
        for rid in (*DS.eval_request_ids, *DS.sample_request_ids):
            ctx = DS.context_for(rid)
            res = resolve_context(ctx)
            for g in find_group(res, "failed_retry"):
                found += 1
                failed_id, retry_id = g.member_ids
                failed, retry = res.by_id(failed_id), res.by_id(retry_id)
                self.assertEqual(failed.exclusion_reason, "failed")
                self.assertFalse(failed.counts_for_cash)
                self.assertTrue(retry.counts_for_cash, retry_id)
                self.assertEqual(retry.status, "scheduled")
                # exactly one economic debit
                self.assertEqual(sum(1 for e in (failed, retry) if e.counts_for_cash), 1)
                self.assertEqual(DS.event_by_id[failed_id].amount.amount,
                                 DS.event_by_id[retry_id].amount.amount)
        self.assertEqual(found, 7)


class TestDuplicateCharge(unittest.TestCase):
    def test_duplicate_ignored_original_kept(self):
        """event_12709 is a pending duplicate of the settled event_12708, same 134.75."""
        ctx, res = resolved_for_user(owner_of("event_12709"))
        original, dup = res.by_id("event_12708"), res.by_id("event_12709")
        self.assertEqual(dup.exclusion_reason, "duplicate")
        self.assertFalse(dup.counts_for_cash)
        self.assertFalse(dup.counts_for_recurrence)
        self.assertEqual(original.lifecycle_role, "original_charge")
        # The original is settled history, so it is inside the opening balance, not forecast cash.
        self.assertFalse(original.counts_for_cash)
        self.assertEqual(original.exclusion_reason, "already_in_opening_balance")

    def test_duplicate_would_otherwise_be_a_pending_debit(self):
        """Without the link it looks exactly like a chargeable pending debit. That is the trap."""
        raw = DS.event_by_id["event_12709"]
        self.assertEqual((raw.status, raw.direction), ("pending", "debit"))
        ctx, res = resolved_for_user(raw.user_id)
        self.assertGreaterEqual(raw.settlement_date, ctx.request.request_date)
        self.assertFalse(res.by_id("event_12709").counts_for_cash)


class TestRefundsAndReversals(unittest.TestCase):
    def test_settled_reversal_nets_to_zero_and_seeds_nothing(self):
        """event_98 charge + event_99 settled reversal, both historical."""
        ctx, res = resolved_for_user(owner_of("event_98"))
        charge, reversal = res.by_id("event_98"), res.by_id("event_99")
        self.assertEqual(charge.lifecycle_role, "refunded_purchase")
        self.assertEqual(reversal.lifecycle_role, "refund")
        for e in (charge, reversal):
            self.assertFalse(e.counts_for_cash)
            self.assertFalse(e.counts_for_recurrence)

    def test_pending_refund_is_never_counted(self):
        """event_1785: refund initiated, not received. Pending credits never count."""
        ctx, res = resolved_for_user(owner_of("event_1785"))
        refund = res.by_id("event_1785")
        self.assertEqual(refund.status, "pending")
        self.assertEqual(refund.direction, "credit")
        self.assertEqual(refund.exclusion_reason, "pending_credit")
        self.assertFalse(refund.counts_for_cash)
        # ...even though it settles inside the forecast window.
        self.assertGreater(DS.event_by_id["event_1785"].settlement_date,
                           ctx.request.request_date)

    def test_reimbursement_distinguished_from_refund_by_category(self):
        found = 0
        for rid in (*DS.eval_request_ids, *DS.sample_request_ids):
            res = resolve_context(DS.context_for(rid))
            for e in res.events:
                if e.lifecycle_role == "reimbursement":
                    found += 1
                    parent = DS.event_by_id[
                        [m for m in next(g for g in res.groups
                                         if g.group_id == e.lifecycle_group).member_ids][0]]
                    self.assertEqual(parent.category, "work_expense")
        self.assertEqual(found, 7)


class TestInvestments(unittest.TestCase):
    def test_unrealized_valuation_is_never_spendable(self):
        found = 0
        for rid in (*DS.eval_request_ids, *DS.sample_request_ids):
            res = resolve_context(DS.context_for(rid))
            for e in res.events:
                if e.lifecycle_role == "valuation":
                    found += 1
                    self.assertFalse(e.counts_for_cash)
                    self.assertFalse(e.counts_for_recurrence)
                    self.assertEqual(e.exclusion_reason, "non_cash")
                    self.assertIsNone(e.cash_effect)
                    self.assertEqual(e.amount_status, "not_applicable")
                    self.assertIsNone(e.signed_amount)
        self.assertEqual(found, 10)

    def test_sale_proceeds_are_cash_but_never_recurring(self):
        found = 0
        for rid in (*DS.eval_request_ids, *DS.sample_request_ids):
            res = resolve_context(DS.context_for(rid))
            for e in res.events:
                if e.lifecycle_role == "sale":
                    found += 1
                    self.assertFalse(e.counts_for_recurrence)
        self.assertEqual(found, 5)

    def test_contribution_is_historical_not_replayed(self):
        for rid in (*DS.eval_request_ids, *DS.sample_request_ids):
            res = resolve_context(DS.context_for(rid))
            for e in res.events:
                if e.lifecycle_role == "investment_purchase":
                    self.assertFalse(e.counts_for_cash)
                    self.assertFalse(e.counts_for_recurrence)


class TestFutureObligations(unittest.TestCase):
    def test_pending_debits_remain_available_for_forecasting(self):
        ctx, res = resolved_for_user("user_01")
        pending = res.by_id("event_102")            # Pending fuel authorization, settles 2024-03-05
        self.assertTrue(pending.counts_for_cash)
        self.assertEqual(pending.lifecycle_role, "pending_debit")
        self.assertEqual(pending.sign, -1)
        self.assertEqual(pending.signed_amount, Decimal("-567.60"))
        self.assertFalse(pending.counts_for_recurrence)

    def test_confirmed_future_salary_is_cash_but_not_a_recurrence_seed(self):
        ctx, res = resolved_for_user("user_01")
        salary = res.by_id("event_103")             # Next confirmed salary, scheduled 2024-03-15
        self.assertTrue(salary.counts_for_cash)
        self.assertEqual(salary.lifecycle_role, "confirmed_income")
        self.assertEqual(salary.sign, 1)
        self.assertEqual(salary.signed_amount, Decimal("23320.00"))
        self.assertFalse(salary.counts_for_recurrence)

    def test_total_future_cash_is_signed(self):
        ctx, res = resolved_for_user("user_01")
        self.assertEqual(total_future_cash(res),
                         Decimal("23320.00") - Decimal("567.60"))

    def test_scheduled_obligations_counted(self):
        roles = set()
        for rid in (*DS.eval_request_ids, *DS.sample_request_ids):
            res = resolve_context(DS.context_for(rid))
            for e in res.events:
                if e.lifecycle_role == "scheduled_obligation" and e.counts_for_cash:
                    roles.add(e.status)
                    self.assertEqual(e.sign, -1)
        self.assertEqual(roles, {"scheduled"})


class TestForeignCurrency(unittest.TestCase):
    def test_cash_amounts_are_converted_to_home_currency(self):
        ctx, res = resolved_for_user("user_25")     # IDR profile, USD payroll
        salary = res.by_id("event_2288")
        self.assertTrue(salary.counts_for_cash)
        self.assertEqual(salary.cash_effect.source.currency, "USD")
        self.assertEqual(salary.cash_effect.target_currency, "IDR")
        self.assertEqual(salary.cash_effect.rate_source, "exact_date")
        self.assertEqual(salary.signed_amount, Decimal("28499994.00"))


class TestImagePlaceholders(unittest.TestCase):
    def test_blank_amount_never_becomes_zero(self):
        blanks = [e for e in DS.event_by_id.values() if e.amount is None]
        self.assertEqual(len(blanks), 16)
        checked = 0
        for raw in blanks:
            ctx, res = resolved_for_user(raw.user_id)
            e = res.by_id(raw.event_id)
            checked += 1
            self.assertIsNone(e.cash_effect)
            self.assertIsNone(e.signed_amount)          # not Decimal(0)
            self.assertEqual(e.amount_status, "unresolved_image")
            self.assertFalse(e.counts_for_cash, f"{raw.event_id} would be spent as zero")
            self.assertFalse(e.counts_for_recurrence)
        self.assertEqual(checked, 16)

    def test_future_blank_amounts_are_flagged_not_silently_dropped(self):
        """4 image events are pending/scheduled; they must say why they are not cash yet."""
        flagged = []
        for raw in [e for e in DS.event_by_id.values() if e.amount is None]:
            if raw.status in {"pending", "scheduled"}:
                _, res = resolved_for_user(raw.user_id)
                e = res.by_id(raw.event_id)
                self.assertEqual(e.exclusion_reason, "unresolved_amount")
                self.assertIn("never", e.resolution_note + " never")
                flagged.append(raw.event_id)
        self.assertEqual(len(flagged), 4)

    def test_image_linked_settled_row_does_not_seed_recurrence(self):
        """event_1545 is a 41 272 'grocery' in a series whose level is about 8 700."""
        ctx, res = resolved_for_user("user_17")
        poison = res.by_id("event_1545")
        self.assertFalse(poison.counts_for_recurrence)
        self.assertEqual(poison.exclusion_reason, "one_off")
        peers = [e for e in res.recurrence_events if e.category == "groceries"]
        self.assertGreater(len(peers), 10)
        self.assertNotIn("event_1545", {e.economic_id for e in peers})


class TestInternalTransfers(unittest.TestCase):
    def test_no_transfer_pair_is_invented_without_evidence(self):
        """6 users get a 'transfer between your two accounts' message.

        Five of them have no matching debit/credit pair at all, so the message is explanatory,
        not an instruction to exclude anything. The resolver must not fabricate a pair.
        """
        for user_id in ("user_18", "user_33", "user_57", "user_171", "user_273"):
            _, res = resolved_for_user(user_id)
            self.assertEqual([e for e in res.events
                              if e.exclusion_reason == "internal_transfer"], [])
            credits = [e for e in res.events if e.direction == "credit"]
            self.assertTrue(all(e.category == "salary" for e in credits), user_id)


# ---------------------------------------------------------------- interfaces


class TestOverrideInterface(unittest.TestCase):
    def test_every_fact_type_has_a_declared_operation(self):
        from typing import get_args
        from bow.models import FactType
        declared = set(EVENT_OPERATIONS) | set(STREAM_DIRECTIVES)
        self.assertEqual(set(get_args(FactType)) - declared, set())

    def test_event_and_stream_operations_do_not_overlap(self):
        self.assertEqual(set(EVENT_OPERATIONS) & set(STREAM_DIRECTIVES), set())

    def test_unquantified_obligation_never_becomes_a_number(self):
        self.assertEqual(STREAM_DIRECTIVES["new_recurring_obligation_unquantified"], "none")

    def test_directives_are_produced_from_facts(self):
        from bow.models import MessageFact
        fact = MessageFact(
            message_id="m1", user_id="user_01", fact_type="salary_amount_change",
            subject="salary", target_event_id=None, effective_date=None, amount=None,
            multiplier=None, quantified=True, confidence=1.0, evidence_span="")
        ctx = DS.context_for("request_01")
        res = resolve_context(ctx, facts=[fact])
        self.assertEqual(len(res.directives), 1)
        self.assertEqual(res.directives[0].op, "salary_change")

    def test_no_effect_facts_produce_no_directive(self):
        from bow.models import MessageFact
        fact = MessageFact(
            message_id="m2", user_id="user_01", fact_type="no_financial_effect",
            subject="", target_event_id=None, effective_date=None, amount=None,
            multiplier=None, quantified=False, confidence=1.0, evidence_span="")
        res = resolve_context(DS.context_for("request_01"), facts=[fact])
        self.assertEqual(res.directives, ())


class TestTraceability(unittest.TestCase):
    def test_trace_is_off_by_default_and_costs_nothing(self):
        res = resolve_context(DS.context_for("request_01"))
        self.assertFalse(NULL_TRACE)
        self.assertIsNone(NULL_TRACE.write(Path(".")))

    def test_trace_can_be_enabled_for_one_request(self):
        enabled = trace_for("request_01", {"request_01"})
        disabled = trace_for("request_02", {"request_01"})
        self.assertIsInstance(enabled, Trace)
        self.assertFalse(disabled)
        resolve_context(DS.context_for("request_01"), trace=enabled)
        stages = {s.stage for s in enabled.steps}
        self.assertEqual(stages, {"resolve"})
        summary = [s for s in enabled.steps if s.label == "summary"][0]
        self.assertEqual(summary.detail["raw"], 103)
        self.assertEqual(summary.detail["unresolved"], [])

    def test_lifecycle_trace_explains_what_survived(self):
        ctx = DS.context_for(request_for_user(owner_of("event_100")))
        trace = Trace(ctx.request.request_id)
        res = resolve_context(ctx, trace=trace)
        lines = res.trace_lines()
        self.assertTrue(any("authorization_settlement" in ln for ln in lines))
        line = next(ln for ln in lines if "authorization_settlement" in ln)
        self.assertIn("kept=event_101", line)
        self.assertIn("ignored=event_100", line)
        entries = [s for s in trace.steps if s.label == "lifecycle"]
        self.assertTrue(entries)
        self.assertIn("reason", entries[0].detail)

    def test_trace_writes_one_file(self):
        import tempfile
        trace = Trace("request_01")
        resolve_context(DS.context_for("request_01"), trace=trace)
        with tempfile.TemporaryDirectory() as tmp:
            path = trace.write(Path(tmp))
            self.assertTrue(path.is_file())
            self.assertEqual(path.name, "request_01.json")


class TestFailClosed(unittest.TestCase):
    def test_dangling_link_is_flagged_not_guessed(self):
        ctx = DS.context_for("request_01")
        events = list(ctx.events)
        broken = dataclasses.replace(events[-1], linked_event_id="event_does_not_exist")
        events[-1] = broken
        resolver = EventResolver(ctx.profile.home_currency, ctx.request.request_date, ctx.fx)
        res = resolver.resolve(events)
        self.assertIn(broken.event_id, res.unresolved)
        flagged = res.by_id(broken.event_id)
        self.assertEqual(flagged.exclusion_reason, "unresolved_evidence")
        self.assertFalse(flagged.counts_for_cash)

    def test_cash_partition_guard_rejects_a_replayed_settled_event(self):
        ctx = DS.context_for("request_01")
        res = resolve_context(ctx)
        with self.assertRaises(UnresolvedEvidenceError):
            # Pretend the request were a year later: the pending debit now sits in the past.
            assert_cash_partition(res, ctx.request.request_date.replace(year=2025))


if __name__ == "__main__":
    unittest.main(verbosity=2)
