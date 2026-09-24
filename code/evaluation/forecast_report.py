"""Phase-5 diagnostic: run the forecast layer over the solved samples.

Evaluates only what exists at this stage - amount_safe_to_pay and earliest_date_for_full_payment.
Recommendation method and payment plan are NOT judged here. No sample ID is special-cased.

    python code/evaluation/forecast_report.py
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bow.config import ForecastConfig                       # noqa: E402
from bow.dataset import Dataset                             # noqa: E402
from bow.forecast import ForecastContext, ForecastEngine     # noqa: E402
from bow.recurrence import build_evidence                    # noqa: E402
from bow.resolve import resolve_context                      # noqa: E402
from bow.safeamount import earliest_full_payment_date, safe_amount   # noqa: E402
from bow.ai.facts import load_facts                        # noqa: E402


def run(config: ForecastConfig | None = None, use_ai: bool = True):
    cfg = config or ForecastConfig()
    engine = ForecastEngine(cfg)
    ds = Dataset.load()
    store = load_facts(ds) if use_ai else None
    rows = []
    for rid in ds.sample_request_ids:
        ctx = ds.context_for(rid)
        mfacts, ifacts = store.for_user(ctx.request.user_id) if store else ((), ())
        res = resolve_context(ctx, mfacts)
        ev = build_evidence(ctx, res, cfg, ifacts)
        fctx = ForecastContext(ev, ctx.request.request_date)
        fc = engine.project(fctx)
        req = ctx.request.requested_amount
        sa = safe_amount(engine, fctx, req, fc)
        earliest = earliest_full_payment_date(engine, fctx, req, fc)
        exp = ds.expected[rid]
        implied_reserve = (ctx.profile.current_available_balance
                           - ctx.profile.minimum_balance_to_keep - exp.amount_safe_to_pay)
        rows.append(dict(
            rid=rid, expected=exp.amount_safe_to_pay, predicted=sa.amount,
            requested=req, implied_reserve=implied_reserve, reserve=sa.reserve,
            trough=sa.trough, trough_date=sa.trough_date,
            exp_earliest=exp.earliest_date_for_full_payment.strip(),
            got_earliest=earliest.isoformat() if earliest else "",
            capped=exp.amount_safe_to_pay == req, daily=fc.daily_accrual,
            streams=len(ev.streams), flows=len(fc.flows)))
    return rows


def report(rows):
    print(f"{'request':<12}{'expected':>15}{'predicted':>15}{'err/req':>9}"
          f"{'impl.reserve':>16}{'pred.reserve':>16}{'res err':>9}  earliest exp/got")
    print("-" * 128)
    errs, res_errs, e_ok, e_total = [], [], 0, 0
    for r in rows:
        err = abs(r["predicted"] - r["expected"]) / r["requested"]
        errs.append(err)
        res_err = ((r["reserve"] - r["implied_reserve"]) / abs(r["implied_reserve"])
                   if r["implied_reserve"] else Decimal(0))
        if not r["capped"]:
            res_errs.append(abs(res_err))
        e_total += 1
        match = r["exp_earliest"] == r["got_earliest"]
        e_ok += match
        print(f"{r['rid']:<12}{r['expected']:>15,.2f}{r['predicted']:>15,.2f}"
              f"{float(err)*100:>8.1f}%{r['implied_reserve']:>16,.2f}{r['reserve']:>16,.2f}"
              f"{float(res_err)*100:>8.1f}%  {r['exp_earliest'] or '(none)':<12}"
              f"{r['got_earliest'] or '(none)':<12}{'OK' if match else 'X'}"
              f"{'  CAP' if r['capped'] else ''}")
    n = len(errs)
    mean = sum(errs) / n
    srt = sorted(errs)
    print(f"\namount_safe_to_pay error / requested_amount:")
    print(f"  mean {float(mean)*100:.2f}%   median {float(srt[n//2])*100:.2f}%")
    for t in ("0.01", "0.05", "0.10"):
        print(f"  within {float(t)*100:>4.0f}% : {sum(1 for e in errs if e < Decimal(t)):>2}/{n}")
    print(f"  exact      : {sum(1 for r in rows if r['predicted'] == r['expected']):>2}/{n}")
    caps = [r for r in rows if r["capped"]]
    print(f"  capped rows correct: {sum(1 for r in caps if r['predicted'] == r['expected'])}/{len(caps)}")
    if res_errs:
        rs = sorted(res_errs)
        print(f"\nforecast reserve error (uncapped rows only, n={len(rs)}):")
        print(f"  mean {float(sum(rs)/len(rs))*100:.2f}%   median {float(rs[len(rs)//2])*100:.2f}%")
    print(f"\nearliest_date_for_full_payment exact matches: {e_ok}/{e_total}")


if __name__ == "__main__":
    report(run())
