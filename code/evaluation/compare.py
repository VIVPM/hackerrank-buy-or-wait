"""Baseline and configuration comparison on the 25 solved samples.

    python code/evaluation/compare.py                 # all configurations
    python code/evaluation/compare.py --only rules_only pipeline   # no model calls

Configurations, all scored by the same `regress.run` on the same data:

* rules_only    - deterministic engine, no AI evidence at all (the no-model baseline)
* pipeline      - the submitted system: fixed order, every AI fact used
* agent_guided  - bounded agent, tool descriptions + ranking rules + strategy guidance
* agent_brief   - bounded agent, tool descriptions only (prompt ablation)

Agent steps are cached, so a re-run is free and reproduces the same numbers. Writes
`evaluation/comparison.md`.
"""

from __future__ import annotations

import argparse
import sys
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from bow.agent import MAX_STEPS, run_agent                         # noqa: E402
from bow.ai.client import ChatClient                               # noqa: E402
from bow.ai.facts import load_facts                                # noqa: E402
from bow.config import RunConfig                                   # noqa: E402
from bow.dataset import Dataset                                    # noqa: E402
from bow.predict import predict                                    # noqa: E402
from bow.usage import UsageLedger, read_ledger                     # noqa: E402
from regress import SCORED_FIELDS, run                             # noqa: E402

CONFIGS = ("rules_only", "pipeline", "agent_guided", "agent_brief")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=CONFIGS)
    args = ap.parse_args(argv)
    chosen = args.only or CONFIGS

    cfg = RunConfig()
    ds = Dataset.load(cfg)
    store = load_facts(ds, cfg, strict=True)
    client = ChatClient(cfg.models)
    ledger = UsageLedger(cfg.usage_path, cfg.prices, cfg.budget_ceiling_usd)
    relevant = {u: {m.message_id for m in ms if m.fact_type != "no_financial_effect"}
                | {i.image_id for i in store.for_user(u)[1]}
                for u, ms in store.messages_by_user.items()}

    rows, agent_stats = {}, {}
    for name in chosen:
        runs = {}
        if name == "rules_only":
            predictor = lambda ctx: predict(ctx, None, cfg.forecast).result      # noqa: E731
        elif name == "pipeline":
            predictor = lambda ctx: predict(ctx, store, cfg.forecast).result     # noqa: E731
        else:
            prompt = name.removeprefix("agent_")

            def predictor(ctx, prompt=prompt, runs=runs):
                r = run_agent(ctx, store, run_cfg=cfg, prompt=prompt, client=client,
                              ledger=ledger, trace_dir=cfg.debug_dir / "agent" / prompt)
                runs[ctx.request.request_id] = (r, ctx.request.user_id)
                return r.result
        print(f"running {name} ...", flush=True)
        rows[name] = run(predictor, ds)
        if runs:
            agent_stats[name] = runs

    lines = ["# Baseline and configuration comparison", "",
             "25 solved samples, same scorer (`evaluation/regress.py`) for every row. "
             "Agent steps are cached, so these numbers reproduce offline.", "",
             "| Field | " + " | ".join(rows) + " |",
             "|---|" + "---:|" * len(rows)]
    for f in SCORED_FIELDS:
        lines.append(f"| `{f}` | " + " | ".join(
            f"{r.field_accuracy()[f][0]}/25" for r in rows.values()) + " |")
    lines.append("| all five structured fields | " + " | ".join(
        f"{sum(r.field_accuracy()[f][0] for f in SCORED_FIELDS if f != 'amount_safe_to_pay')}"
        f"/{25 * (len(SCORED_FIELDS) - 1)}" for r in rows.values()) + " |")
    for label, fn in (("amount: median error", lambda e: e[len(e) // 2]),
                      ("amount: mean error", lambda e: sum(e) / len(e))):
        vals = []
        for r in rows.values():
            e = sorted(s.error_over_requested for s in r.scored)
            vals.append(f"{float(fn(e)) * 100:.2f}%")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    if agent_stats:
        spend = {}
        for rec in read_ledger(cfg.usage_path):
            if rec.call_type == "agent_step" and rec.cache == "miss":
                prompt = rec.source_id.rsplit(":", 1)[-1]
                s = spend.setdefault(prompt, [0, 0, 0, Decimal(0)])
                s[0] += 1
                s[1] += rec.input_tokens
                s[2] += rec.output_tokens
                s[3] += rec.estimated_cost_usd
        lines += ["", "## Agent behaviour", "",
                  "| | " + " | ".join(agent_stats) + " |", "|---|" + "---:|" * len(agent_stats)]

        def row(label, fn):
            lines.append(f"| {label} | " + " | ".join(fn(v) for v in agent_stats.values()) + " |")

        def paid(idx):
            return lambda: " | ".join(
                str(spend.get(n.removeprefix("agent_"), [0, 0, 0, Decimal(0)])[idx])
                for n in agent_stats)

        row("mean tool calls per request", lambda v: f"{sum(len(r.steps) for r, _ in v.values()) / len(v):.1f}")
        row(f"hit the {MAX_STEPS}-step cap", lambda v: str(sum(1 for r, _ in v.values() if r.fallback and 'cap' in r.fallback)))
        row("fell back to pipeline (any reason)", lambda v: str(sum(1 for r, _ in v.values() if r.fallback)))
        row("chose same plan as deterministic ranker",
            lambda v: f"{sum(1 for r, _ in v.values() if r.agrees_with_ranker)}"
                      f"/{sum(1 for r, _ in v.values() if r.agrees_with_ranker is not None)}")

        def recall(v):
            want = sum(len(relevant.get(u, ())) for _, u in v.values())
            got = sum(len(set(r.read) & relevant.get(u, set())) for r, u in v.values())
            return f"{got}/{want}"
        row("relevant evidence read", recall)
        row("tool errors (recovered)", lambda v: str(sum(
            1 for r, _ in v.values() for s in r.steps if str(s["result"]).startswith("ERROR"))))
        # Ledger totals across every paid run of this configuration (cached steps are free),
        # so after a re-run these exceed the cost of the table's own run.
        for label, idx in (("model calls, all paid runs", 0), ("input tokens, all paid runs", 1),
                           ("output tokens, all paid runs", 2), ("cost USD, all paid runs", 3)):
            lines.append(f"| {label} | {paid(idx)()} |")
        lines += ["", "Traces: `code/.debug/agent/<prompt>/<request_id>.json` (every tool call, "
                  "its result, evidence read, and any fallback reason)."]

    out = HERE / "comparison.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
