"""Root-cause tracing for the 25-sample suite. Reads cached AI facts; makes no model calls.

    python code/evaluation/rca.py            # summary of every sample
    python code/evaluation/rca.py request_11 # full backward trace for one
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bow.ai.facts import load_facts                              # noqa: E402
from bow.config import ForecastConfig                            # noqa: E402
from bow.dataset import Dataset                                  # noqa: E402
from bow.forecast import ForecastContext, ForecastEngine         # noqa: E402
from bow.predict import predict                                  # noqa: E402
from bow.recurrence import build_evidence                        # noqa: E402
from bow.resolve import resolve_context                          # noqa: E402

DS = Dataset.load()
STORE = load_facts(DS)


def analyse(rid: str, cfg: ForecastConfig | None = None):
    cfg = cfg or ForecastConfig()
    engine = ForecastEngine(cfg)
    ctx = DS.context_for(rid)
    m, i = STORE.for_user(ctx.request.user_id)
    ev = build_evidence(ctx, resolve_context(ctx, m), cfg, i)
    fc = engine.project(ForecastContext(ev, ctx.request.request_date))
    out = predict(ctx, STORE, cfg).result
    exp = DS.expected[rid]
    p = ctx.profile
    capped = exp.amount_safe_to_pay == ctx.request.requested_amount
    implied_trough = p.minimum_balance_to_keep + exp.amount_safe_to_pay
    floor = fc.suffix_minimum_for_payment(ctx.request.request_date)
    return dict(rid=rid, ctx=ctx, ev=ev, fc=fc, out=out, exp=exp, capped=capped,
                implied_trough=None if capped else implied_trough,
                predicted_floor=floor,
                gap=None if capped else floor - implied_trough)


def summary():
    print(f"{'req':<12}{'fields missed':<62}{'impl.trough':>16}{'pred.floor':>16}{'gap':>14}")
    print("-" * 122)
    for rid in DS.sample_request_ids:
        a = analyse(rid)
        missed = []
        e, o = a["exp"], a["out"]
        if o.amount_safe_to_pay != e.amount_safe_to_pay:
            missed.append("safe")
        if o.affordability_status != e.affordability_status:
            missed.append("status")
        if o.recommended_payment_method != e.recommended_payment_method:
            missed.append("method")
        if o.payment_plan.replace(".00", "") != e.payment_plan.replace(".00", ""):
            missed.append("plan")
        if o.earliest_date_for_full_payment != (e.earliest_date_for_full_payment or ""):
            missed.append("earliest")
        if o.spending_changes_needed != (e.spending_changes_needed or "none"):
            missed.append("changes")
        it = "capped" if a["capped"] else f"{a['implied_trough']:,.2f}"
        gap = "-" if a["gap"] is None else f"{a['gap']:,.2f}"
        print(f"{rid:<12}{','.join(missed) or 'ALL OK':<62}{it:>16}"
              f"{a['predicted_floor']:>16,.2f}{gap:>14}")


def detail(rid: str):
    a = analyse(rid)
    ctx, ev, fc, out, exp = a["ctx"], a["ev"], a["fc"], a["out"], a["exp"]
    p = ctx.profile
    print(f"===== {rid} ({ctx.request.user_id}) =====")
    print(f"request   : {ctx.request.request_date} type={ctx.request.request_type} "
          f"amount={ctx.request.requested_amount} deadline={ctx.request.desired_completion_date} "
          f"partial_ok={ctx.request.allows_partial_payment}")
    print(f"profile   : {p.home_currency} balance={p.current_available_balance} "
          f"min={p.minimum_balance_to_keep} methods={sorted(p.methods)} "
          f"max_inst={p.max_installment_months}")
    print(f"expected  : safe={exp.amount_safe_to_pay} {exp.affordability_status}/"
          f"{exp.recommended_payment_method} plan={exp.payment_plan} "
          f"earliest={exp.earliest_date_for_full_payment} changes={exp.spending_changes_needed}")
    print(f"predicted : safe={out.amount_safe_to_pay} {out.affordability_status}/"
          f"{out.recommended_payment_method} plan={out.payment_plan} "
          f"earliest={out.earliest_date_for_full_payment} changes={out.spending_changes_needed}")
    if not a["capped"]:
        print(f"trough    : implied={a['implied_trough']:,.2f} predicted_floor="
              f"{a['predicted_floor']:,.2f} gap={a['gap']:,.2f}  "
              f"(positive gap = we reserve too little)")
    print(f"\nfacts     : {[(f.fact_type, str(f.amount.amount) if f.amount else None, str(f.effective_date)) for f in ev.message_facts]}")
    print(f"images    : {[(f.image_id, str(f.chosen_amount.amount), f.semantic_label) for f in ev.image_facts]}")
    print(f"\nstreams ({len(ev.streams)}):")
    for s in sorted(ev.streams, key=lambda s: (s.kind, s.group)):
        print(f"   {s.kind:<8}{s.group:<22}{s.cadence_class:<12}gap={s.cadence_days:>3} "
              f"amt={s.amount:>14,.2f} anchor={s.anchor_date} n={s.observations} "
              f"est={s.estimator}")
    print(f"\ndaily accrual: {fc.daily_accrual}")
    print(f"trough: {fc.trough} on {fc.trough_date}")
    print("\npath to the floor:")
    for point in fc.path:
        flows = [f for f in fc.flows if f.date == point.day]
        if flows or point.low == a["predicted_floor"]:
            tag = "  <== FLOOR" if point.low == a["predicted_floor"] else ""
            detail = "; ".join(f"{f.origin}={f.amount}" for f in flows)
            print(f"   {point.day}  low={point.low:>16,.2f} close={point.close:>16,.2f}"
                  f"  {detail}{tag}")
        if point.day > fc.trough_date and point.day > ctx.request.request_date:
            if (point.day - ctx.request.request_date).days > 45:
                break


if __name__ == "__main__":
    if len(sys.argv) > 1:
        detail(sys.argv[1])
    else:
        summary()
