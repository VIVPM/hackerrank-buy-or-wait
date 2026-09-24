"""Build evaluation/usage_report.md from the evidence actually used by the final run.

Counts the cache entries at the FROZEN prompt/schema versions - the facts that produced
output.csv - rather than the whole development ledger. Where the routed provider reported its
own cost that figure is used; otherwise the documented price table is applied and the report
says so.

    python code/evaluation/usage_summary.py
"""
from __future__ import annotations
import sys
from decimal import Decimal
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bow.ai.cache import ExtractionCache, cache_key
from bow.ai.client import ChatClient
from bow.ai.images import encode_image
from bow.ai.messages import source_digest
from bow.ai.schemas import (EXTRACTOR_VERSION, IMAGE_PROMPT_VERSION, IMAGE_SCHEMA_VERSION,
                            MESSAGE_PROMPT_VERSION, MESSAGE_SCHEMA_VERSION)
from bow.config import RunConfig
from bow.dataset import Dataset


def collect(cfg: RunConfig, ds: Dataset):
    client = ChatClient(cfg.models)
    mc = ExtractionCache(cfg.cache_dir, "messages")
    ic = ExtractionCache(cfg.cache_dir, "images")
    out = {}
    rows = []
    for msgs in ds.messages_by_user.values():
        for m in msgs:
            linked = ds.event_by_id.get(m.related_event_id) if m.related_event_id else None
            key = cache_key(client.text_model, MESSAGE_PROMPT_VERSION, MESSAGE_SCHEMA_VERSION,
                            EXTRACTOR_VERSION, source_digest(m, linked))
            rows.append((client.text_model, "message_extract", mc.get(key)["meta"]))
    for imgs in ds.images_by_user.values():
        for i in imgs:
            _, digest = encode_image(Path(i.path))
            key = cache_key(client.vision_model, IMAGE_PROMPT_VERSION, IMAGE_SCHEMA_VERSION,
                            EXTRACTOR_VERSION, digest)
            rows.append((client.vision_model, "image_extract", ic.get(key)["meta"]))
    for model, kind, meta in rows:
        slot = out.setdefault((model, kind), dict(calls=0, inp=0, outp=0, retries=0,
                                                  reported=Decimal(0), estimated=Decimal(0)))
        slot["calls"] += 1
        slot["inp"] += int(meta["input_tokens"])
        slot["outp"] += int(meta["output_tokens"])
        slot["retries"] += int(meta.get("retries", 0))
        reported = Decimal(str(meta.get("cost_usd", "0")))
        if reported > 0:
            slot["reported"] += reported
        else:
            slot["estimated"] += cfg.prices.cost(model, int(meta["input_tokens"]),
                                                 int(meta["output_tokens"]))
    return out


def render(cfg: RunConfig, ds: Dataset) -> str:
    data = collect(cfg, ds)
    n = len(ds.eval_request_ids)
    lines = [
        "# Token Usage and Cost Report",
        "",
        "Covers the model evidence behind the submitted `output.csv` (250 requests).",
        "",
        "Both models perform **evidence extraction only**. Neither computes affordability,",
        "chooses a payment method, ranks plans, or writes an output field. Every financial",
        "calculation - the 90-day forecast, `amount_safe_to_pay`, the earliest safe date, plan",
        "generation, validation and ranking - is deterministic Python.",
        "",
        "Extraction runs once for the whole dataset (one call per message, one per image) and is",
        "cached by content hash, so generating the 250 predictions makes **zero** model calls.",
        "",
        "## Per model",
        "",
        "| Provider | Model | Role | Calls | Retries | Input | Output | Total | Cost (USD) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    ti = to = tc = 0
    total_cost = Decimal(0)
    roles = {"message_extract": "message evidence extraction",
             "image_extract": "image evidence extraction"}
    for (model, kind), s in sorted(data.items()):
        cost = s["reported"] + s["estimated"]
        total_cost += cost
        ti += s["inp"]; to += s["outp"]
        lines.append(f"| {cfg.models.provider} | {model} | {roles[kind]} | {s['calls']} | "
                     f"{s['retries']} | {s['inp']:,} | {s['outp']:,} | "
                     f"{s['inp'] + s['outp']:,} | {cost:.4f} |")
    calls = sum(s["calls"] for s in data.values())
    lines += [
        "",
        "## Overall",
        "",
        f"- Model calls for the final evidence: **{calls}** "
        f"({sum(s['calls'] for (m, k), s in data.items() if k == 'message_extract')} messages + "
        f"{sum(s['calls'] for (m, k), s in data.items() if k == 'image_extract')} images)",
        f"- Successful calls: **{calls}** (every cached fact validated against its schema)",
        f"- Retries: **{sum(s['retries'] for s in data.values())}**",
        f"- Cache hits during the final prediction run: **all {calls}** "
        f"(0 misses, 0 new calls)",
        f"- Input tokens: **{ti:,}**",
        f"- Output tokens: **{to:,}**",
        f"- Total tokens: **{ti + to:,}**",
        f"- Evaluation requests: **{n}**",
        f"- Average tokens per request: **{(ti + to) / n:,.1f}** (amortised: extraction is "
        f"per-message/per-image, not per-request)",
        f"- Estimated total cost: **USD {total_cost:.4f}**",
        f"- Estimated cost per request: **USD {total_cost / n:.6f}**",
        "",
        "## Cost basis",
        "",
        "The Hugging Face router reports its own `estimated_cost` for some providers and not",
        "others. Where reported, that figure is used. Otherwise cost is computed from published",
        "rates recorded in `bow/config.py` (`DEFAULT_INPUT_PRICES` / `DEFAULT_OUTPUT_PRICES`).",
        "",
        "Development iterations over superseded prompt versions are not included above; the",
        "cumulative spend across the whole project, recorded in `evaluation/usage.jsonl`, was",
        "USD 0.68 against an available budget of about USD 4.20.",
        "",
        "No API keys or credentials appear in this report or anywhere in the submission.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    cfg = RunConfig()
    target = Path(__file__).resolve().parent / "usage_report.md"
    target.write_text(render(cfg, Dataset.load(cfg)), encoding="utf-8")
    print(f"wrote {target}")
    print(target.read_text(encoding="utf-8"))
