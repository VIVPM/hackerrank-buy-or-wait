"""Load cached AI facts for the deterministic engine. Never makes a model call.

The engine consumes typed facts from disk. Extraction (`evaluation/extract.py`) is a separate,
one-off step, so a prediction run is fully deterministic and costs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from bow.ai.cache import ExtractionCache, cache_key
from bow.ai.client import ChatClient
from bow.ai.images import encode_image, to_fact as image_to_fact
from bow.ai.messages import source_digest, to_fact as message_to_fact
from bow.ai.schemas import (
    EXTRACTOR_VERSION, IMAGE_PROMPT_VERSION, IMAGE_SCHEMA_VERSION,
    MESSAGE_PROMPT_VERSION, MESSAGE_SCHEMA_VERSION,
)
from bow.config import RunConfig
from bow.dataset import Dataset
from bow.errors import ExtractionError
from bow.models import ImageFact, MessageFact


@dataclass(frozen=True, slots=True)
class FactStore:
    messages_by_user: Mapping[str, tuple[MessageFact, ...]]
    images_by_user: Mapping[str, tuple[ImageFact, ...]]
    missing: tuple[str, ...]

    def for_user(self, user_id: str) -> tuple[tuple[MessageFact, ...], tuple[ImageFact, ...]]:
        return (self.messages_by_user.get(user_id, ()), self.images_by_user.get(user_id, ()))

    @property
    def coverage(self) -> str:
        m = sum(len(v) for v in self.messages_by_user.values())
        i = sum(len(v) for v in self.images_by_user.values())
        return f"{m} message facts, {i} image facts, {len(self.missing)} uncached"


def load_facts(ds: Dataset, cfg: RunConfig | None = None,
               strict: bool = False) -> FactStore:
    cfg = cfg or RunConfig()
    client = ChatClient(cfg.models)
    msg_cache = ExtractionCache(cfg.cache_dir, "messages")
    img_cache = ExtractionCache(cfg.cache_dir, "images")
    by_user_m: dict[str, list[MessageFact]] = {}
    by_user_i: dict[str, list[ImageFact]] = {}
    missing: list[str] = []

    for messages in ds.messages_by_user.values():
        for message in messages:
            linked = (ds.event_by_id.get(message.related_event_id)
                      if message.related_event_id else None)
            key = cache_key(client.text_model, MESSAGE_PROMPT_VERSION,
                            MESSAGE_SCHEMA_VERSION, EXTRACTOR_VERSION,
                            source_digest(message, linked))
            entry = msg_cache.get(key)
            if entry is None:
                missing.append(message.message_id)
                continue
            by_user_m.setdefault(message.user_id, []).append(
                message_to_fact(message, entry["fact"]))

    for images in ds.images_by_user.values():
        for image in images:
            linked = ds.event_by_id.get(image.related_event_id)
            _, digest = encode_image(Path(image.path))
            key = cache_key(client.vision_model, IMAGE_PROMPT_VERSION,
                            IMAGE_SCHEMA_VERSION, EXTRACTOR_VERSION, digest)
            entry = img_cache.get(key)
            if entry is None:
                missing.append(image.image_id)
                continue
            currency = linked.amount.currency if linked and linked.amount else None
            by_user_i.setdefault(image.user_id, []).append(
                image_to_fact(image, entry["fact"], currency))

    if strict and missing:
        raise ExtractionError(f"{len(missing)} sources are not cached: {missing[:5]}")
    return FactStore({k: tuple(v) for k, v in by_user_m.items()},
                     {k: tuple(v) for k, v in by_user_i.items()},
                     tuple(missing))
