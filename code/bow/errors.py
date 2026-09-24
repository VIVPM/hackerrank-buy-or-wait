"""Exception taxonomy. Every one of these means "stop and be explicit", never "guess a number"."""

from __future__ import annotations


class BowError(Exception):
    """Base for everything raised by the pipeline."""


class DatasetError(BowError):
    """Malformed or unparseable input CSV. Raised at load time with file/row/column."""


class MissingRateError(BowError):
    """No exchange rate for the ordered pair. Never fall back to 1.0."""


class ExtractionError(BowError):
    """Model returned unusable output after the repair retry."""


class CacheCorruptionError(BowError):
    """Cached payload failed its checksum; treat as a miss and re-extract."""


class BudgetExceeded(BowError):
    """The next model call would cross RunConfig.budget_ceiling_usd."""


class InvariantError(BowError):
    """An arithmetic invariant broke. The candidate is discarded, never emitted."""


class UnresolvedEvidenceError(BowError):
    """Contradictory evidence that the precedence rules could not settle."""
