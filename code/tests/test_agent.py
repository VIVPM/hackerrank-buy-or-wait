"""Agent loop mechanics, offline: a scripted client stands in for the model."""

from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from bow.agent import MAX_STEPS, run_agent
from bow.ai.client import ChatResult
from bow.ai.facts import load_facts
from bow.config import RunConfig
from bow.dataset import Dataset
from bow.errors import ExtractionError
from bow.predict import predict

DS = Dataset.load()
STORE = load_facts(DS)


class Scripted:
    """Replays tool calls; `pick` chooses the finish plan from the latest list_plans."""

    text_model = "scripted"

    def __init__(self, calls, pick=None, fail=False):
        self.calls, self.pick, self.fail, self.n = list(calls), pick, fail, 0

    def complete_json(self, *, user_content, **_):
        if self.fail:
            raise ExtractionError("provider down")
        self.n += 1
        tool, arg = self.calls.pop(0) if self.calls else ("get_request", None)
        plan_id = None
        if tool == "finish":
            plans = json.loads(user_content.rsplit("Result: ", 1)[1].split("\n\nNext")[0])
            plan_id = self.pick(plans) if self.pick else arg
        call = {"thought": "", "tool": tool, "evidence_id": None if tool == "finish" else arg,
                "plan_id": plan_id}
        return call, ChatResult("", 10, 5, 1, "scripted", "test", Decimal(0)), 0


def cfg(tmp):
    return dataclasses.replace(RunConfig(), cache_dir=Path(tmp), usage_path=Path(tmp) / "u.jsonl")


def rid_with_evidence():
    return next(r for r in DS.eval_request_ids
                if STORE.for_user(DS.context_for(r).request.user_id)[0])


class TestAgentLoop(unittest.TestCase):
    def run_script(self, calls, pick=None, fail=False, rid="request_01"):
        with tempfile.TemporaryDirectory() as tmp:
            client = Scripted(calls, pick, fail)
            return run_agent(DS.context_for(rid), STORE, run_cfg=cfg(tmp), client=client), client

    def test_happy_path_uses_a_validated_plan(self):
        run, _ = self.run_script(
            [("get_request", None), ("list_plans", None), ("finish", None)],
            pick=lambda plans: plans[0]["plan_id"])
        self.assertIsNone(run.fallback)
        self.assertIsNotNone(run.agrees_with_ranker)

    def test_unread_evidence_is_excluded(self):
        rid = rid_with_evidence()
        run, _ = self.run_script([("list_plans", None), ("finish", None)],
                                 pick=lambda p: p[0]["plan_id"], rid=rid)
        self.assertEqual(run.read, [])

    def test_read_evidence_is_included(self):
        rid = rid_with_evidence()
        msg = STORE.for_user(DS.context_for(rid).request.user_id)[0][0].message_id
        run, _ = self.run_script([("read_message", msg), ("list_plans", None),
                                  ("finish", None)], pick=lambda p: p[0]["plan_id"], rid=rid)
        self.assertEqual(run.read, [msg])

    def test_bad_tool_argument_is_recoverable(self):
        run, _ = self.run_script(
            [("read_message", "no_such_id"), ("list_plans", None), ("finish", None)],
            pick=lambda p: p[0]["plan_id"])
        self.assertTrue(run.steps[0]["result"].startswith("ERROR"))
        self.assertIsNone(run.fallback)

    def test_step_cap_falls_back_to_pipeline(self):
        run, client = self.run_script([])           # never finishes
        self.assertEqual(client.n, MAX_STEPS)
        self.assertIn("step cap", run.fallback)
        self.assertEqual(run.result, predict(DS.context_for("request_01"), STORE).result)

    def test_unknown_plan_falls_back(self):
        run, _ = self.run_script([("list_plans", None), ("finish", None)],
                                 pick=lambda p: "P999")
        self.assertIn("unknown plan_id", run.fallback)
        self.assertEqual(run.result, predict(DS.context_for("request_01"), STORE).result)

    def test_provider_failure_falls_back(self):
        run, _ = self.run_script([], fail=True)
        self.assertIn("ExtractionError", run.fallback)
        self.assertEqual(run.result, predict(DS.context_for("request_01"), STORE).result)

    def test_argument_only_kept_for_tools_that_take_one(self):
        run, _ = self.run_script([("get_request", "get_request"), ("list_plans", "junk"),
                                  ("finish", None)], pick=lambda p: p[0]["plan_id"])
        self.assertIsNone(run.steps[0]["call"]["evidence_id"])
        self.assertFalse(run.steps[0]["result"].startswith("ERROR"))

    def test_steps_are_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = [("get_request", None), ("list_plans", None), ("finish", None)]
            first = Scripted(calls, pick=lambda p: p[0]["plan_id"])
            run_agent(DS.context_for("request_01"), STORE, run_cfg=cfg(tmp), client=first)
            second = Scripted([], fail=True)            # would fail if it were called
            again = run_agent(DS.context_for("request_01"), STORE, run_cfg=cfg(tmp),
                              client=second)
            self.assertIsNone(again.fallback)


if __name__ == "__main__":
    unittest.main()
