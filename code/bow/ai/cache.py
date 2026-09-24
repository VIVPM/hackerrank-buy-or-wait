"""Content-addressed extraction cache.

Stores the *structured fact*, not model prose, so a cached run is byte-identical and reviewable.
The key binds the model, the prompt, the schema and the extractor version to a hash of the source
content, so editing one prompt invalidates only the entries that prompt produced.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bow.errors import CacheCorruptionError


def content_hash(*parts: str | bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8") if isinstance(part, str) else part)
        digest.update(b"\x1f")
    return digest.hexdigest()


def cache_key(model: str, prompt_version: str, schema_version: str,
              extractor_version: str, source_digest: str) -> str:
    return content_hash(model, prompt_version, schema_version, extractor_version,
                        source_digest)[:40]


@dataclass(frozen=True, slots=True)
class ExtractionCache:
    root: Path
    kind: str                     # "messages" | "images"

    @property
    def directory(self) -> Path:
        return self.root / self.kind

    def path_for(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self.path_for(key)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CacheCorruptionError(f"{path} is not valid JSON") from exc
        if not isinstance(payload, dict) or "fact" not in payload:
            raise CacheCorruptionError(f"{path} has no 'fact' entry")
        return payload

    def put(self, key: str, *, fact: dict[str, Any], meta: dict[str, Any]) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(key)
        path.write_text(json.dumps({"key": key, "fact": fact, "meta": meta}, indent=2,
                                   sort_keys=True), encoding="utf-8")
        return path

    def entries(self) -> int:
        return len(list(self.directory.glob("*.json"))) if self.directory.is_dir() else 0
