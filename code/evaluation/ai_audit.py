"""Audit of the cached AI evidence: image table, message-fact categories, spot checks.

Reads only the cache - makes no model calls.

    python code/evaluation/ai_audit.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bow.ai.cache import ExtractionCache, cache_key             # noqa: E402
from bow.ai.client import ChatClient                            # noqa: E402
from bow.ai.images import SUSPECT_SEMANTICS, encode_image, to_fact as image_fact  # noqa: E402
from bow.ai.messages import source_digest, to_fact as message_fact               # noqa: E402
from bow.ai.messages import _linked_summary                     # noqa: E402
from bow.ai.schemas import (                                    # noqa: E402
    EXTRACTOR_VERSION, IMAGE_PROMPT_VERSION, IMAGE_SCHEMA_VERSION,
    MESSAGE_PROMPT_VERSION, MESSAGE_SCHEMA_VERSION,
)
from bow.config import RunConfig                                # noqa: E402
from bow.dataset import Dataset                                 # noqa: E402


def load_all(ds: Dataset, cfg: RunConfig):
    client = ChatClient(cfg.models)
    mc = ExtractionCache(cfg.cache_dir, "messages")
    ic = ExtractionCache(cfg.cache_dir, "images")
    msgs, imgs = [], []
    for ms in ds.messages_by_user.values():
        for m in ms:
            linked = ds.event_by_id.get(m.related_event_id) if m.related_event_id else None
            key = cache_key(client.text_model, MESSAGE_PROMPT_VERSION,
                            MESSAGE_SCHEMA_VERSION, EXTRACTOR_VERSION,
                            source_digest(m, linked))
            hit = mc.get(key)
            if hit:
                msgs.append((m, message_fact(m, hit["fact"]), hit["fact"]))
    for xs in ds.images_by_user.values():
        for i in xs:
            _, digest = encode_image(Path(i.path))
            key = cache_key(client.vision_model, IMAGE_PROMPT_VERSION,
                            IMAGE_SCHEMA_VERSION, EXTRACTOR_VERSION, digest)
            hit = ic.get(key)
            if hit:
                linked = ds.event_by_id.get(i.related_event_id)
                cur = linked.amount.currency if linked and linked.amount else None
                imgs.append((i, image_fact(i, hit["fact"], cur), hit["fact"]))
    return msgs, imgs


def main() -> int:
    cfg = RunConfig()
    ds = Dataset.load(cfg)
    msgs, imgs = load_all(ds, cfg)

    print(f"cached message facts: {len(msgs)}/215     cached image facts: {len(imgs)}/16")

    print("\n================ IMAGE EXTRACTION TABLE ================")
    print(f"{'image':<10}{'event':<13}{'amount':>16}{'ccy':>5}  {'semantics':<20}"
          f"{'doc type':<20}{'status':<10}{'conf':>5}")
    print("-" * 110)
    for i, f, raw in sorted(imgs, key=lambda t: t[0].image_id):
        print(f"{i.image_id:<10}{i.related_event_id:<13}{f.chosen_amount.amount:>16,.2f}"
              f"{f.chosen_amount.currency:>5}  {raw['amount_semantics']:<20}"
              f"{raw['document_type']:<20}{raw['payment_status']:<10}{f.confidence:>5.2f}")

    print("\n---------------- semantic traps ----------------")
    for i, f, raw in sorted(imgs, key=lambda t: t[0].image_id):
        others = raw.get("other_amounts") or []
        if not others:
            continue
        rejected = ", ".join(f"{o['label']}={o['value']:,.2f}" for o in others)
        print(f"{i.image_id}: chose {raw['amount_semantics']}={f.chosen_amount.amount:,.2f} "
              f"over [{rejected}]")
    suspect = [i.image_id for i, f, raw in imgs if raw["amount_semantics"] in SUSPECT_SEMANTICS]
    print(f"\nchose a semantically-invalid figure: {suspect or 'none'}")
    low = [(i.image_id, f.confidence) for i, f, _ in imgs if f.confidence < 0.6]
    print(f"low confidence (<0.60): {low or 'none'}")

    print("\n================ MESSAGE FACT CATEGORIES ================")
    ops = Counter(raw["operation"] for _, _, raw in msgs)
    for op, n in ops.most_common():
        print(f"  {op:<28}{n:>4}")
    print(f"\n  quantified (amount AND confirmed): "
          f"{sum(1 for _, f, _ in msgs if f.quantified)}")
    print(f"  unconfirmed money mentioned      : "
          f"{sum(1 for _, _, r in msgs if r.get('amount') and not r['confirmed'])}")
    print(f"  amount absent entirely           : "
          f"{sum(1 for _, _, r in msgs if not r.get('amount'))}")
    rec = Counter(r["recurrence_effect"] for _, _, r in msgs)
    print(f"  recurrence effects               : {dict(rec)}")
    lowm = [(m.message_id, f.confidence) for m, f, _ in msgs if f.confidence < 0.5]
    print(f"  low confidence (<0.50)           : {lowm or 'none'}")

    print("\n---------------- spot checks ----------------")
    wanted = ("message_01", "message_03", "message_07", "message_10", "message_12",
              "message_13", "message_14", "message_15", "message_17", "message_20")
    for m, f, raw in sorted(msgs, key=lambda t: t[0].message_id):
        if m.message_id not in wanted:
            continue
        print(f"\n{m.message_id} [{m.source_type}] {m.message_text[:96]}...")
        print(f"   -> op={raw['operation']} target={raw['target']} amount={raw.get('amount')} "
              f"{raw.get('currency') or ''} eff={raw.get('effective_date')} "
              f"new_date={raw.get('new_date')} confirmed={raw['confirmed']} "
              f"rec={raw['recurrence_effect']} conf={f.confidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
