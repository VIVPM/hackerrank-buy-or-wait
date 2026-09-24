"""Token/cost ledger with a hard budget ceiling.

Every model call appends one JSONL record, cache hits included, so `usage_report.md` can state
what the scored run actually consumed rather than what a re-run would have cost.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Sequence

from bow.config import PriceTable, RunConfig
from bow.errors import BudgetExceeded
from bow.models import UsageRecord


class UsageLedger:
    """Append-only. Thread-safe because the extractors may fan out later."""

    def __init__(self, path: Path, prices: PriceTable, ceiling_usd: Decimal) -> None:
        self.path = path
        self.prices = prices
        self.ceiling = ceiling_usd
        self._lock = threading.Lock()
        self._records: list[UsageRecord] = []
        if path.exists():
            self._records = list(read_ledger(path))

    # ------------------------------------------------------------------ spend

    @property
    def spent(self) -> Decimal:
        return sum((r.estimated_cost_usd for r in self._records if r.cache == "miss"),
                   Decimal(0))

    def check_budget(self, projected: Decimal = Decimal(0)) -> None:
        if self.spent + projected > self.ceiling:
            raise BudgetExceeded(
                f"spent {self.spent:.4f} USD + projected {projected:.4f} would cross the "
                f"{self.ceiling:.2f} USD ceiling")

    # ------------------------------------------------------------------ record

    def record(self, *, provider: str, model: str, call_type: str, source_id: str,
               input_tokens: int, output_tokens: int, latency_ms: int,
               cache: str, retries: int,
               provider_cost: Decimal | None = None) -> UsageRecord:
        # The router reports what it actually charged; that beats a local price table.
        cost = (Decimal(0) if cache == "hit"
                else provider_cost if provider_cost is not None
                else self.prices.cost(model, input_tokens, output_tokens))
        record = UsageRecord(
            provider=provider, model=model, call_type=call_type,      # type: ignore[arg-type]
            source_id=source_id, input_tokens=input_tokens, output_tokens=output_tokens,
            latency_ms=latency_ms, estimated_cost_usd=cost,
            cache=cache, retries=retries,                             # type: ignore[arg-type]
            timestamp=datetime.now(timezone.utc).isoformat())
        with self._lock:
            self._records.append(record)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                payload = asdict(record)
                payload["estimated_cost_usd"] = str(record.estimated_cost_usd)
                handle.write(json.dumps(payload) + "\n")
        return record

    @property
    def records(self) -> Sequence[UsageRecord]:
        return tuple(self._records)


def read_ledger(path: Path) -> Iterable[UsageRecord]:
    if not path.exists():
        return ()
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        raw["estimated_cost_usd"] = Decimal(raw["estimated_cost_usd"])
        out.append(UsageRecord(**raw))
    return out


def summarise(records: Sequence[UsageRecord], requests: int) -> str:
    """Render evaluation/usage_report.md content."""
    by_model: dict[str, list[UsageRecord]] = {}
    for r in records:
        by_model.setdefault(r.model, []).append(r)

    lines = [
        "# Token Usage and Cost Report",
        "",
        "Covers the run that produced the submitted `output.csv`.",
        "Models are used only for evidence extraction; every financial calculation is "
        "deterministic Python.",
        "",
        "## Per model",
        "",
        "| Provider | Model | Calls | Cache hits | Input tokens | Output tokens | "
        "Total tokens | Cost (USD) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    total_in = total_out = 0
    total_cost = Decimal(0)
    for model, rs in sorted(by_model.items()):
        tin = sum(r.input_tokens for r in rs)
        tout = sum(r.output_tokens for r in rs)
        cost = sum((r.estimated_cost_usd for r in rs), Decimal(0))
        hits = sum(1 for r in rs if r.cache == "hit")
        total_in += tin
        total_out += tout
        total_cost += cost
        lines.append(f"| {rs[0].provider} | {model} | {len(rs)} | {hits} | {tin:,} | "
                     f"{tout:,} | {tin + tout:,} | {cost:.4f} |")

    calls = len(records)
    total_tokens = total_in + total_out
    lines += [
        "",
        "## Overall",
        "",
        f"- Model calls: **{calls}**",
        f"- Cache hits: **{sum(1 for r in records if r.cache == 'hit')}**",
        f"- Retries: **{sum(r.retries for r in records)}**",
        f"- Input tokens: **{total_in:,}**",
        f"- Output tokens: **{total_out:,}**",
        f"- Total tokens: **{total_tokens:,}**",
        f"- Requests processed: **{requests}**",
        f"- Average tokens per request: **{total_tokens / requests:,.1f}**"
        if requests else "- Average tokens per request: n/a",
        f"- Estimated total cost: **USD {total_cost:.4f}**",
        f"- Estimated cost per request: **USD {total_cost / requests:.6f}**"
        if requests else "- Estimated cost per request: n/a",
        "",
        "Extraction is one call per message and one per image for the whole dataset, cached by "
        "content hash, so the per-request figures are amortised rather than incurred per request.",
    ]
    return "\n".join(lines) + "\n"


def write_report(config: RunConfig, requests: int) -> Path:
    records = list(read_ledger(config.usage_path))
    target = config.usage_path.parent / "usage_report.md"
    target.write_text(summarise(records, requests), encoding="utf-8")
    return target
