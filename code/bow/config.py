"""Frozen configuration. Secrets come from the environment only — never from a file in the repo."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path | None = None) -> None:
    """Populate os.environ from the repo-root .env. Real environment always wins.

    ponytail: six lines beat a python-dotenv dependency for KEY=value.
    """
    env_path = path or REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


@dataclass(frozen=True, slots=True)
class ForecastConfig:
    """Every knob the forensic phase showed to be load-bearing.

    Defaults are chosen for a financial reason, never because they fit the 25 solved samples.
    See ARCHITECTURE.md section 6 for the justification of each.
    """

    horizon_days: int = 90                       # stated in problem_statement.md
    horizon_inclusive: bool = True
    #   mean_all          - mean of every observation
    #   trailing_horizon  - mean of observations inside the last `horizon_days`; estimates the
    #                       next 90 days from the last 90, so the window adapts to cadence
    #                       (~3 points for a monthly bill, ~13 for a weekly habit)
    #   mean_6 / mean_3 / median_6 / last - fixed observation counts
    amount_estimator: Literal[
        "mean_all", "trailing_horizon", "mean_6", "mean_3", "median_6", "last"
    ] = "trailing_horizon"
    cadence_rule: Literal["median_gap", "mean_gap"] = "median_gap"
    monthly_threshold_days: int = 26
    monthly_anchor: Literal["day_of_month", "fixed_interval"] = "day_of_month"
    # How habitual sub-monthly spending is projected.
    #   accrual        - a daily rate (unbiased over long windows, under-reserves short ones)
    #   discrete       - occurrences on their inferred dates
    #   cycle_upfront  - one full cycle reserved immediately, then discrete at cadence
    #   cycle_ceiling  - accrual rounded up to whole cycles
    submonthly_mode: Literal[
        "accrual", "discrete", "cycle_upfront", "cycle_ceiling"
    ] = "accrual"
    income_staleness_cycles: Decimal = Decimal("1.5")
    min_observations: int = 3
    # How a day carrying both income and bills is evaluated. `minimum_balance_to_keep` is a
    # balance the user wants to KEEP, which is an end-of-day position, not an artefact of the
    # bank's intraday posting order. "debits_first" measures the pre-salary intraday dip on
    # payday and reports a floor the user never actually experiences.
    same_day_order: Literal["debits_first", "credits_first"] = "credits_first"
    quantum: Decimal = Decimal("0.01")
    earliest_horizon: Literal["fixed_from_asof", "sliding"] = "fixed_from_asof"

    def fingerprint(self) -> str:
        parts = [f"{f.name}={getattr(self, f.name)}" for f in self.__dataclass_fields__.values()]
        return "|".join(parts)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    text_model: str = "zai-org/GLM-5.3"
    vision_model: str = "Qwen/Qwen3-VL-235B-A22B-Instruct"
    provider: str = "huggingface-routed"
    temperature: float = 0.0
    timeout_s: int = 90
    max_retries: int = 2

    @property
    def api_token(self) -> str | None:
        return os.environ.get("HF_TOKEN")


#: USD per 1M tokens, from the providers' published rates. Only used when the router does not
#: report its own `estimated_cost`: GLM-5.3 does report it, the vision provider does not, so the
#: vision figures below are the documented fallback rather than a measured charge.
DEFAULT_INPUT_PRICES: dict[str, Decimal] = {
    "Qwen/Qwen3-VL-235B-A22B-Instruct": Decimal("0.30"),
    "qwen/qwen3-vl-235b-a22b-instruct": Decimal("0.30"),
    "zai-org/GLM-5.3": Decimal("0.60"),
    "zai-org/glm-5.3": Decimal("0.60"),
}
DEFAULT_OUTPUT_PRICES: dict[str, Decimal] = {
    "Qwen/Qwen3-VL-235B-A22B-Instruct": Decimal("1.20"),
    "qwen/qwen3-vl-235b-a22b-instruct": Decimal("1.20"),
    "zai-org/GLM-5.3": Decimal("2.20"),
    "zai-org/glm-5.3": Decimal("2.20"),
}


@dataclass(frozen=True, slots=True)
class PriceTable:
    """USD per 1M tokens. Fallback only; a provider-reported cost always wins."""

    input_per_mtok: dict[str, Decimal] = field(
        default_factory=lambda: dict(DEFAULT_INPUT_PRICES))
    output_per_mtok: dict[str, Decimal] = field(
        default_factory=lambda: dict(DEFAULT_OUTPUT_PRICES))

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> Decimal:
        million = Decimal(1_000_000)
        cin = self.input_per_mtok.get(model, Decimal(0))
        cout = self.output_per_mtok.get(model, Decimal(0))
        return (Decimal(input_tokens) * cin + Decimal(output_tokens) * cout) / million


@dataclass(frozen=True, slots=True)
class RunConfig:
    dataset_dir: Path = REPO_ROOT / "dataset"
    output_path: Path = REPO_ROOT / "output.csv"
    # Deliberately NOT a dotfile: some zip tools silently skip hidden entries, which would
    # produce an archive that looks complete and fails on the grader's machine.
    cache_dir: Path = REPO_ROOT / "code" / "ai_cache"
    debug_dir: Path = REPO_ROOT / "code" / ".debug"
    usage_path: Path = REPO_ROOT / "code" / "evaluation" / "usage.jsonl"
    budget_ceiling_usd: Decimal = Decimal("3.00")   # hard stop, below the ~4.20 allowance
    trace_request_ids: frozenset[str] = frozenset()
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    prices: PriceTable = field(default_factory=PriceTable)
