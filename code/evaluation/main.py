"""Reproducible evaluation entry point.

Uses the SAME production predictor and validator as `code/main.py`. It contains no financial
algorithm of its own.

    python code/evaluation/main.py                 # validate output.csv + score the 25 samples
    python code/evaluation/main.py --samples-only  # ground-truth metrics for the samples
    python code/evaluation/main.py --output-only   # hard validation of output.csv only

Exits non-zero on any hard invariant failure.

For `sample_requests.csv` ground truth exists, so accuracy is reported. For the 250 evaluation
requests the ground truth is hidden, so NO accuracy figure is invented - only structural and
financial invariants are checked, plus aggregate distributions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bow.ai.facts import load_facts                                   # noqa: E402
from bow.config import RunConfig                                      # noqa: E402
from bow.dataset import Dataset                                       # noqa: E402
from bow.eligibility import assess                                    # noqa: E402
from bow.forecast import ForecastContext, ForecastEngine              # noqa: E402
from bow.models import OUTPUT_COLUMNS                                 # noqa: E402
from bow.outputs import serialise_plan                                # noqa: E402
from bow.predict import predict                                       # noqa: E402
from bow.recurrence import build_evidence                             # noqa: E402
from bow.resolve import resolve_context                               # noqa: E402
from bow.spending import REDUCIBLE, STOPPABLE, permitted_changes      # noqa: E402

STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}
PAIRS = {("affordable_now", "full_payment"), ("affordable_later", "wait"),
         ("not_affordable", "not_recommended"), ("affordable_with_plan", "full_payment"),
         ("affordable_with_plan", "partial_payment"), ("affordable_with_plan", "installments")}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_plan(text: str):
    if text.strip().lower() == "none":
        return []
    out = []
    for part in text.split("|"):
        day, _, amount = part.rpartition(":")
        out.append((date.fromisoformat(day), Decimal(amount)))
    return out


# ---------------------------------------------------------------- output validation


def validate_output(path: Path, ds: Dataset, store, cfg: RunConfig) -> list[str]:
    problems: list[str] = []
    if not path.is_file():
        return [f"missing {path}"]

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != list(OUTPUT_COLUMNS):
            problems.append(f"column order is {reader.fieldnames}, expected {list(OUTPUT_COLUMNS)}")
        rows = list(reader)

    expected_ids = list(ds.eval_request_ids)
    ids = [r["request_id"] for r in rows]
    if len(rows) != len(expected_ids):
        problems.append(f"{len(rows)} data rows, expected {len(expected_ids)}")
    if len(set(ids)) != len(ids):
        dupes = [k for k, v in Counter(ids).items() if v > 1]
        problems.append(f"duplicate request_ids: {dupes[:5]}")
    missing = set(expected_ids) - set(ids)
    extra = set(ids) - set(expected_ids)
    if missing:
        problems.append(f"missing request_ids: {sorted(missing)[:5]}")
    if extra:
        problems.append(f"unexpected request_ids (sample leakage?): {sorted(extra)[:5]}")

    engine = ForecastEngine(cfg.forecast)
    for row in rows:
        rid = row["request_id"]
        if rid not in ds.requests:
            continue
        problems.extend(_check_row(row, rid, ds, store, engine, cfg))
    return problems


def _check_row(row, rid, ds: Dataset, store, engine, cfg) -> list[str]:
    bad: list[str] = []
    def fail(msg): bad.append(f"{rid}: {msg}")

    ctx = ds.context_for(rid)
    request, profile = ctx.request, ctx.profile

    raw = row["amount_safe_to_pay"]
    for token in ("nan", "inf", "none", "null", "e+", "e-"):
        if token in raw.lower():
            fail(f"malformed amount {raw!r}")
            return bad
    try:
        safe = Decimal(raw)
    except InvalidOperation:
        fail(f"amount_safe_to_pay not numeric: {raw!r}")
        return bad
    if not (Decimal(0) <= safe <= request.requested_amount):
        fail(f"amount_safe_to_pay {safe} outside [0, {request.requested_amount}]")

    status, method = row["affordability_status"], row["recommended_payment_method"]
    if status not in STATUSES:
        fail(f"invalid affordability_status {status!r}")
    if method not in METHODS:
        fail(f"invalid recommended_payment_method {method!r}")
    if (status, method) in PAIRS or status not in STATUSES or method not in METHODS:
        pass
    else:
        fail(f"inconsistent status/method pair ({status}, {method})")

    earliest = row["earliest_date_for_full_payment"].strip()
    if earliest:
        try:
            earliest_date = date.fromisoformat(earliest)
        except ValueError:
            fail(f"earliest date not YYYY-MM-DD: {earliest!r}")
            earliest_date = None
    else:
        earliest_date = None
    if status == "affordable_now" and earliest != request.request_date.isoformat():
        fail("affordable_now must report earliest == request_date")

    plan_text = row["payment_plan"]
    try:
        payments = parse_plan(plan_text)
    except Exception as exc:
        fail(f"unparseable payment_plan {plan_text!r}: {exc}")
        return bad
    if not payments and method != "not_recommended":
        fail(f"{method} with an empty plan")
    if payments and method == "not_recommended":
        fail("not_recommended must have payment_plan 'none'")
    dates = [d for d, _ in payments]
    if dates != sorted(dates):
        fail("payments are not chronological")
    if any(a <= 0 for _, a in payments):
        fail("a payment amount is not positive")
    total = sum((a for _, a in payments), Decimal(0))

    if method in {"full_payment", "partial_payment", "wait"} and total != request.requested_amount:
        fail(f"payments total {total}, expected {request.requested_amount}")
    if method != "not_recommended" and dates and max(dates) > request.desired_completion_date:
        fail(f"last payment {max(dates)} after deadline {request.desired_completion_date}")

    if method == "full_payment" and "full_payment" not in profile.methods:
        fail("full_payment not accepted by the user")
    if method == "wait":
        if "full_payment" not in profile.methods:
            fail("wait requires the full_payment preference")
        if earliest_date is None:
            fail("wait without an earliest full-payment date")
    if method == "partial_payment":
        if not request.allows_partial_payment:
            fail("partial payment on a request that disallows it")
        if "partial_payment" not in profile.methods:
            fail("partial_payment not accepted by the user")
        if not (Decimal(0) < safe < request.requested_amount):
            fail("partial payment requires 0 < safe < requested")
        if len(payments) != 2:
            fail(f"partial payment must have exactly 2 payments, got {len(payments)}")
        else:
            (d1, a1), (d2, a2) = payments
            if d1 != request.request_date:
                fail("first partial payment not on request_date")
            if a1 != safe:
                fail(f"first partial payment {a1} != amount_safe_to_pay {safe}")
            if a2 != request.requested_amount - safe:
                fail("second partial payment is not the remainder")
            if earliest_date is None or d2 != earliest_date:
                fail("second partial payment not on earliest_date_for_full_payment")
            if earliest_date and earliest_date > request.desired_completion_date:
                fail("earliest date after the deadline")
    if method == "installments":
        if "installments" not in profile.methods:
            fail("installments not accepted by the user")
        options = [o for o in ds.options_by_request[rid] if o.payment_method == "installments"]
        match = next((o for o in options if serialise_plan(o.schedule) == plan_text), None)
        if match is None:
            fail("installment schedule does not match any supplied option exactly")
        else:
            if total != match.total_payable_amount:
                fail(f"total {total} != option total {match.total_payable_amount}")
            if len(payments) != match.number_of_payments:
                fail("payment count differs from the supplied option")
            verdict = next(v for v in assess(profile, request,
                                             ds.options_by_request[rid]).verdicts
                           if v.option.payment_option_id == match.payment_option_id)
            if not verdict.usable:
                fail(f"chosen option is ineligible: {verdict.reason}")

    # ---- spending changes + independent 90-day safety replay
    messages, images = store.for_user(request.user_id)
    ev = build_evidence(ctx, resolve_context(ctx, messages), cfg.forecast, images)
    changes_text = row["spending_changes_needed"].strip()
    overrides = {}
    if changes_text.lower() != "none":
        parts = changes_text.split("|")
        if len(parts) > 3:
            fail(f"{len(parts)} spending changes exceeds 3")
        allowed = {c.event_id: c for c in permitted_changes(profile, ev.streams)}
        seen_events = set()
        for part in parts:
            bits = part.split(":")
            kind, event_id = bits[0], bits[1] if len(bits) > 1 else ""
            if event_id in seen_events:
                fail(f"two changes target the same stream ({event_id})")
            seen_events.add(event_id)
            stream = next((s for s in ev.streams if s.latest_event_id == event_id), None)
            if stream is None:
                fail(f"change targets an unknown stream ({event_id})")
                continue
            if kind == "stop":
                if stream.flexibility not in STOPPABLE:
                    fail(f"{stream.group} is not stoppable")
                if stream.group not in profile.stoppable_categories:
                    fail(f"user will not stop {stream.group}")
                overrides[stream.key] = None
            elif kind == "reduce_to":
                if stream.flexibility not in REDUCIBLE:
                    fail(f"{stream.group} is not reducible")
                if stream.group not in profile.reducible_categories:
                    fail(f"user will not reduce {stream.group}")
                if Decimal(bits[2]) != stream.minimum_allowed_amount:
                    fail(f"reduce_to {bits[2]} != minimum_allowed_amount "
                         f"{stream.minimum_allowed_amount}")
                overrides[stream.key] = Decimal(bits[2])
            else:
                fail(f"unknown spending change kind {kind!r}")
            if event_id not in allowed:
                fail(f"change {part!r} is not a permitted change for this user")

    if method != "not_recommended":
        from bow.models import ProjectedCashFlow
        flows = tuple(ProjectedCashFlow(date=d, amount=-a, kind="plan_payment", origin="check")
                      for d, a in payments)
        forecast = engine.project(ForecastContext(ev, request.request_date, flows, overrides))
        if not forecast.is_safe:
            when, value = forecast.breaches[0]
            fail(f"90-day safety replay fails: balance {value} on {when} below "
                 f"{profile.minimum_balance_to_keep}")
    return bad


# ---------------------------------------------------------------- sample scoring


def score_samples(ds: Dataset, store, cfg: RunConfig) -> tuple[dict, list[str]]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import regress

    def predictor(ctx):
        return predict(ctx, store, cfg.forecast).result

    report = regress.run(predictor, ds, None, None)
    fields = report.field_accuracy()
    errs = sorted(s.error_over_requested for s in report.scored)
    n = len(errs)
    metrics = {
        "samples": n,
        "complete_rows": sum(s.all_correct for s in report.scored),
        "fields": {k: f"{v[0]}/{v[1]}" for k, v in fields.items()},
        "amount_mean_error_pct": round(float(sum(errs) / n) * 100, 2),
        "amount_median_error_pct": round(float(errs[n // 2]) * 100, 2),
        "amount_max_error_pct": round(float(errs[-1]) * 100, 2),
        "within_1pct": sum(1 for e in errs if e < Decimal("0.01")),
        "within_5pct": sum(1 for e in errs if e < Decimal("0.05")),
        "within_10pct": sum(1 for e in errs if e < Decimal("0.10")),
        "exact": sum(1 for s in report.scored if s.abs_error == 0),
    }
    return metrics, [s.request_id for s in report.scores if s.error]


# ---------------------------------------------------------------- distributions


def distributions(path: Path, ds: Dataset) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    status = Counter(r["affordability_status"] for r in rows)
    method = Counter(r["recommended_payment_method"] for r in rows)
    by_type, by_ccy = Counter(), Counter()
    zero = capped = partial = 0
    e_req = e_later = e_empty = 0
    changes = 0
    for r in rows:
        rid = r["request_id"]
        request = ds.requests[rid]
        profile = ds.profiles[request.user_id]
        by_type[(request.request_type, r["affordability_status"])] += 1
        by_ccy[(profile.home_currency, r["affordability_status"])] += 1
        amount = Decimal(r["amount_safe_to_pay"])
        if amount == 0:
            zero += 1
        elif amount == request.requested_amount:
            capped += 1
        else:
            partial += 1
        earliest = r["earliest_date_for_full_payment"].strip()
        if not earliest:
            e_empty += 1
        elif earliest == request.request_date.isoformat():
            e_req += 1
        else:
            e_later += 1
        if r["spending_changes_needed"].strip().lower() != "none":
            changes += 1
    return {"rows": len(rows), "status": dict(status), "method": dict(method),
            "by_request_type": {f"{a}|{b}": c for (a, b), c in sorted(by_type.items())},
            "by_currency": {f"{a}|{b}": c for (a, b), c in sorted(by_ccy.items())},
            "amount_zero": zero, "amount_capped": capped, "amount_partial": partial,
            "earliest_request_date": e_req, "earliest_later": e_later,
            "earliest_empty": e_empty, "with_spending_changes": changes}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Buy or Wait? evaluation and validation")
    ap.add_argument("--samples-only", action="store_true")
    ap.add_argument("--output-only", action="store_true")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = RunConfig()
    ds = Dataset.load(cfg)
    store = load_facts(ds, cfg, strict=True)
    print(f"evidence: {store.coverage}")

    failed = False

    if not args.output_only:
        metrics, errored = score_samples(ds, store, cfg)
        print("\n--- solved samples (ground truth available) ---")
        print(json.dumps(metrics, indent=2))
        if errored:
            print(f"ERROR: samples raised: {errored}")
            failed = True

    if not args.samples_only:
        target = args.output or cfg.output_path
        print(f"\n--- evaluation output: {target} ---")
        problems = validate_output(target, ds, store, cfg)
        if problems:
            print(f"HARD FAILURES: {len(problems)}")
            for p in problems[:40]:
                print(f"  {p}")
            failed = True
        else:
            print("all invariants hold for every row "
                  "(no accuracy is reported: ground truth is hidden)")
            print(f"sha256: {sha256(target)}")
            print(json.dumps(distributions(target, ds), indent=2))

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
