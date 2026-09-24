"""Selective per-request debug trace.

Disabled tracing costs nothing: `NULL_TRACE` implements the same surface as no-ops, so call sites
never need a conditional and a full run does not accumulate thousands of irrelevant rows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class TraceStep:
    stage: str
    label: str
    detail: dict[str, Any]


class _NullSection:
    """Mirrors `_Section.add(label, **detail)` so call sites need no conditional."""

    def add(self, label: str, **detail: Any) -> None:
        return None


_NULL_SECTION = _NullSection()


class NullTrace:
    """No-op. Shared singleton; never accumulates anything."""

    enabled = False

    def add(self, stage: str, label: str, **detail: Any) -> None:
        return None

    def section(self, stage: str) -> _NullSection:
        return _NULL_SECTION

    def write(self, directory: Path) -> Path | None:
        return None

    def __bool__(self) -> bool:
        return False


NULL_TRACE = NullTrace()


@dataclass
class Trace:
    request_id: str
    steps: list[TraceStep] = field(default_factory=list)
    enabled = True

    def add(self, stage: str, label: str, **detail: Any) -> None:
        self.steps.append(TraceStep(stage=stage, label=label, detail=detail))

    def section(self, stage: str) -> "_Section":
        return _Section(self, stage)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "steps": [
                {"stage": s.stage, "label": s.label, **_jsonable(s.detail)} for s in self.steps
            ],
        }

    def write(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.request_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def __bool__(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class _Section:
    trace: "Trace"
    stage: str

    def add(self, label: str, **detail: Any) -> None:
        self.trace.add(self.stage, label, **detail)


TraceLike = Trace | NullTrace


def trace_for(request_id: str, enabled_ids: Iterable[str]) -> TraceLike:
    """Enable tracing for a selected request only."""
    return Trace(request_id) if request_id in set(enabled_ids) else NULL_TRACE


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
