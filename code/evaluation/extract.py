"""One-off AI evidence extraction over the whole dataset.

    python code/evaluation/extract.py --dry-run    # projected cost, zero calls
    python code/evaluation/extract.py              # extract everything, cached

One call per message and one per image for the entire dataset - never per request, never per
re-run. A second invocation with unchanged sources, prompts and schemas makes zero model calls.
"""

from __future__ import annotations

import argparse
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bow.ai.cache import ExtractionCache                       # noqa: E402
from bow.ai.client import ChatClient                           # noqa: E402
from bow.ai.images import extract_image                        # noqa: E402
from bow.ai.images import low_confidence as image_low          # noqa: E402
from bow.ai.messages import extract_message                    # noqa: E402
from bow.ai.messages import low_confidence as message_low      # noqa: E402
from bow.config import RunConfig                               # noqa: E402
from bow.dataset import Dataset                                # noqa: E402
from bow.errors import ExtractionError                         # noqa: E402
from bow.usage import UsageLedger, write_report                # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report projected cost, make no calls")
    ap.add_argument("--limit", type=int, default=0, help="cap the number of messages (testing)")
    args = ap.parse_args(argv)

    cfg = RunConfig()
    ds = Dataset.load(cfg)
    client = ChatClient(cfg.models)
    client.validate()
    ledger = UsageLedger(cfg.usage_path, cfg.prices, cfg.budget_ceiling_usd)
    msg_cache = ExtractionCache(cfg.cache_dir, "messages")
    img_cache = ExtractionCache(cfg.cache_dir, "images")

    messages = [m for ms in ds.messages_by_user.values() for m in ms]
    images = [i for xs in ds.images_by_user.values() for i in xs]
    if args.limit:
        messages = messages[:args.limit]

    print(f"client      : {client.base_url}")
    print(f"text model  : {client.text_model}")
    print(f"vision model: {client.vision_model}")
    print(f"messages    : {len(messages)}   images: {len(images)}")
    print(f"cache       : messages={msg_cache.entries()} images={img_cache.entries()}")
    print(f"spent so far: USD {ledger.spent:.5f} / ceiling {cfg.budget_ceiling_usd}")

    projected = Decimal("0.00065") * len(messages) + Decimal("0.0015") * len(images)
    print(f"projected   : USD {projected:.4f}")
    if args.dry_run:
        ledger.check_budget(projected)
        print("dry run: budget check passed, no calls made")
        return 0

    ledger.check_budget(projected)
    facts, failures = [], []
    for n, m in enumerate(messages, 1):
        linked = ds.event_by_id.get(m.related_event_id) if m.related_event_id else None
        try:
            fact, hit = extract_message(m, linked, client, msg_cache, ledger)
            facts.append(fact)
        except ExtractionError as exc:
            failures.append((m.message_id, str(exc)[:120]))
            continue
        if n % 25 == 0 or n == len(messages):
            print(f"  messages {n}/{len(messages)}  spent USD {ledger.spent:.4f}")

    image_facts = []
    for i in images:
        linked = ds.event_by_id.get(i.related_event_id)
        try:
            fact, hit = extract_image(i, linked, client, img_cache, ledger)
            image_facts.append(fact)
        except ExtractionError as exc:
            failures.append((i.image_id, str(exc)[:120]))
    print(f"  images {len(image_facts)}/{len(images)}  spent USD {ledger.spent:.4f}")

    print(f"\nmessage facts: {len(facts)}   image facts: {len(image_facts)}")
    print(f"failures     : {len(failures)}")
    for sid, why in failures:
        print(f"   {sid}: {why}")
    lowm, lowi = message_low(facts), image_low(image_facts)
    print(f"low-confidence messages: {len(lowm)}  images: {len(lowi)}")
    for f in lowi:
        print(f"   {f.image_id}: {f.chosen_amount.amount} as {f.semantic_label} "
              f"conf={f.confidence}")
    print(f"\ntotal spend  : USD {ledger.spent:.5f}")
    print("report       :", write_report(cfg, requests=len(ds.eval_request_ids)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
