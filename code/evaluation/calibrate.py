"""Forecast calibration analysis. Cache-only; makes no model calls."""
from __future__ import annotations
import statistics, sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bow.ai.facts import load_facts
from bow.config import ForecastConfig
from bow.dataset import Dataset
from bow.forecast import ForecastContext, ForecastEngine
from bow.recurrence import build_evidence
from bow.resolve import resolve_context
from bow.safeamount import safe_amount

DS = Dataset.load(); STORE = load_facts(DS)

def evidence(rid, cfg):
    ctx = DS.context_for(rid); m, i = STORE.for_user(ctx.request.user_id)
    return ctx, build_evidence(ctx, resolve_context(ctx, m), cfg, i)

def run_one(rid, cfg):
    ctx, ev = evidence(rid, cfg)
    eng = ForecastEngine(cfg)
    fctx = ForecastContext(ev, ctx.request.request_date)
    fc = eng.project(fctx)
    sa = safe_amount(eng, fctx, ctx.request.requested_amount, fc)
    exp = DS.expected[rid]
    capped = exp.amount_safe_to_pay == ctx.request.requested_amount
    implied = (ctx.profile.current_available_balance
               - ctx.profile.minimum_balance_to_keep - exp.amount_safe_to_pay)
    return dict(rid=rid, ctx=ctx, ev=ev, fc=fc, sa=sa, exp=exp, capped=capped,
                implied_reserve=implied, residual=sa.reserve - implied)

def decompose(rid, cfg=None):
    cfg = cfg or ForecastConfig()
    r = run_one(rid, cfg)
    fc, ctx = r["fc"], r["ctx"]
    upto = fc.trough_date
    per = defaultdict(Decimal)
    for f in fc.flows:
        if f.date <= upto:
            per[f.origin] += f.amount
    days = (upto - ctx.request.request_date).days
    accr = fc.daily_accrual * days
    return r, per, accr, days

def stream_stats(rid, cfg=None):
    cfg = cfg or ForecastConfig()
    ctx, ev = evidence(rid, cfg)
    res = resolve_context(ctx, STORE.for_user(ctx.request.user_id)[0])
    per = defaultdict(list)
    for e in res.events:
        if e.counts_for_recurrence and e.cash_effect is not None and e.direction == "debit":
            per[e.category].append(e.cash_effect.converted)
    out = []
    for s in ev.streams:
        if s.kind != "expense":
            continue
        vals = per.get(s.group, [])
        if len(vals) < 2:
            continue
        f = [float(v) for v in vals]
        mean = statistics.mean(f); sd = statistics.pstdev(f)
        out.append(dict(cat=s.group, cadence=s.cadence_days, cls=s.cadence_class,
                        n=len(f), mean=mean, median=statistics.median(f),
                        last=f[-1], m3=statistics.mean(f[-3:]), m6=statistics.mean(f[-6:]),
                        mn=min(f), mx=max(f), sd=sd, cv=sd / mean if mean else 0.0,
                        used=float(s.amount)))
    return out
