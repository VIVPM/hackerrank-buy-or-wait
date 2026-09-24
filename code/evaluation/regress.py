"""Regression harness over the 25 solved samples.

The harness calls the SAME prediction interface that will later process requests.csv. There is no
sample-specific production logic anywhere: the only thing the harness knows that the production run
does not is the expected output, which it uses solely to score.

    python code/evaluation/regress.py --self-test
    python code/evaluation/regress.py --request request_07
    python code/evaluation/regress.py                    # all 25, once a predictor is wired
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Protocol, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bow.config import RunConfig                                          # noqa: E402
from bow.dataset import Dataset, RequestContext                           # noqa: E402
from bow.models import ExpectedOutput, PredictionResult                   # noqa: E402
from bow.money import format_amount                                       # noqa: E402

SCORED_FIELDS = (
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
)


class Predictor(Protocol):
    """The single prediction interface. `predict.predict_request` will satisfy this."""

    def __call__(self, ctx: RequestContext) -> PredictionResult: ...


# ---------------------------------------------------------------- normalisation


def parse_plan(text: str) -> tuple[tuple[date, Decimal], ...]:
    """`2024-09-04:28820|2024-09-15:10840` -> ((date, Decimal), ...). `none`/blank -> ()."""
    cleaned = (text or "").strip()
    if not cleaned or cleaned.lower() == "none":
        return ()
    out: list[tuple[date, Decimal]] = []
    for part in cleaned.split("|"):
        day, _, amount = part.strip().rpartition(":")
        if not day:
            raise ValueError(f"malformed payment_plan entry: {part!r}")
        out.append((datetime.strptime(day.strip(), "%Y-%m-%d").date(), Decimal(amount.strip())))
    return tuple(out)


def parse_changes(text: str) -> frozenset[str]:
    """Order-insensitive canonical form; `reduce_to:e:23.50` == `reduce_to:e:23.5`."""
    cleaned = (text or "").strip()
    if not cleaned or cleaned.lower() == "none":
        return frozenset()
    out: set[str] = set()
    for part in cleaned.split("|"):
        bits = part.strip().split(":")
        if bits[0] == "reduce_to" and len(bits) == 3:
            out.add(f"reduce_to:{bits[1]}:{Decimal(bits[2]).normalize()}")
        else:
            out.add(part.strip())
    return frozenset(out)


def norm_date(text: str) -> str:
    cleaned = (text or "").strip()
    return "" if cleaned.lower() in {"", "none", "nan"} else cleaned


def field_matches(name: str, expected: ExpectedOutput, predicted: PredictionResult) -> bool:
    if name == "amount_safe_to_pay":
        return predicted.amount_safe_to_pay == expected.amount_safe_to_pay
    if name == "payment_plan":
        return parse_plan(predicted.payment_plan) == parse_plan(expected.payment_plan)
    if name == "spending_changes_needed":
        return (parse_changes(predicted.spending_changes_needed)
                == parse_changes(expected.spending_changes_needed))
    if name == "earliest_date_for_full_payment":
        return (norm_date(predicted.earliest_date_for_full_payment)
                == norm_date(expected.earliest_date_for_full_payment))
    return getattr(predicted, name) == getattr(expected, name)


# ---------------------------------------------------------------- scoring


@dataclass(frozen=True, slots=True)
class RequestScore:
    request_id: str
    matched: frozenset[str]
    expected: ExpectedOutput
    predicted: PredictionResult
    requested_amount: Decimal
    is_capped: bool
    error: str | None = None

    @property
    def all_correct(self) -> bool:
        return len(self.matched) == len(SCORED_FIELDS)

    @property
    def abs_error(self) -> Decimal:
        return abs(self.predicted.amount_safe_to_pay - self.expected.amount_safe_to_pay)

    @property
    def rel_error(self) -> Decimal | None:
        """Relative to the expected amount; undefined when the expected amount is zero."""
        if self.expected.amount_safe_to_pay == 0:
            return None
        return self.abs_error / self.expected.amount_safe_to_pay

    @property
    def error_over_requested(self) -> Decimal:
        return self.abs_error / self.requested_amount if self.requested_amount else Decimal(0)


@dataclass
class RegressionReport:
    scores: list[RequestScore] = field(default_factory=list)

    @property
    def scored(self) -> list[RequestScore]:
        return [s for s in self.scores if s.error is None]

    def field_accuracy(self) -> dict[str, tuple[int, int]]:
        done = self.scored
        return {f: (sum(1 for s in done if f in s.matched), len(done)) for f in SCORED_FIELDS}

    def summary(self) -> str:
        done, failed = self.scored, [s for s in self.scores if s.error]
        if not done:
            body = "  no requests scored"
        else:
            lines = [f"requests fully correct : {sum(s.all_correct for s in done)}/{len(done)}"]
            for name, (ok, total) in self.field_accuracy().items():
                lines.append(f"  {name:<32} {ok:>3}/{total}")
            errs = sorted(s.error_over_requested for s in done)
            mean = sum(errs) / len(errs)
            median = errs[len(errs) // 2]
            lines += [
                "",
                "amount_safe_to_pay (error / requested_amount)",
                f"  mean   {float(mean) * 100:6.2f}%",
                f"  median {float(median) * 100:6.2f}%",
                f"  within 1%  {sum(1 for e in errs if e < Decimal('0.01')):>3}/{len(errs)}",
                f"  within 5%  {sum(1 for e in errs if e < Decimal('0.05')):>3}/{len(errs)}",
                f"  within 10% {sum(1 for e in errs if e < Decimal('0.10')):>3}/{len(errs)}",
                f"  exact      {sum(1 for s in done if s.abs_error == 0):>3}/{len(errs)}",
            ]
            capped = [s for s in done if s.is_capped]
            if capped:
                lines.append(
                    f"  capped cases correct {sum(1 for s in capped if s.abs_error == 0)}"
                    f"/{len(capped)}   (a miss here flips affordability_status)"
                )
            body = "\n".join(lines)
        tail = f"\n\nerrored: {len(failed)}" + "".join(
            f"\n  {s.request_id}: {s.error}" for s in failed) if failed else ""
        return body + tail

    def table(self) -> str:
        head = (f"{'request':<12}{'expected':>16}{'predicted':>16}{'abs err':>14}"
                f"{'/req':>8}  fields")
        rows = [head, "-" * len(head)]
        for s in self.scores:
            if s.error:
                rows.append(f"{s.request_id:<12}{'ERROR':>16}  {s.error}")
                continue
            missed = [f for f in SCORED_FIELDS if f not in s.matched]
            rows.append(
                f"{s.request_id:<12}"
                f"{format_amount(s.expected.amount_safe_to_pay):>16}"
                f"{format_amount(s.predicted.amount_safe_to_pay):>16}"
                f"{format_amount(s.abs_error):>14}"
                f"{float(s.error_over_requested) * 100:>7.1f}%"
                f"  {'ALL OK' if not missed else ','.join(m[:14] for m in missed)}"
            )
        return "\n".join(rows)


# ---------------------------------------------------------------- runner


def run(predictor: Predictor, dataset: Dataset | None = None,
        request_ids: Sequence[str] | None = None,
        diagnostics_dir: Path | None = None) -> RegressionReport:
    ds = dataset or Dataset.load()
    ids = tuple(request_ids) if request_ids else ds.sample_request_ids
    report = RegressionReport()
    for request_id in ids:
        expected = ds.expected.get(request_id)
        if expected is None:
            raise KeyError(f"{request_id} has no expected output; not a solved sample")
        ctx = ds.context_for(request_id)
        try:
            predicted = predictor(ctx)
        except Exception as exc:                       # harness must survive a broken predictor
            report.scores.append(RequestScore(
                request_id=request_id, matched=frozenset(), expected=expected,
                predicted=_blank(request_id), requested_amount=ctx.request.requested_amount,
                is_capped=False, error=f"{type(exc).__name__}: {exc}"))
            continue
        score = RequestScore(
            request_id=request_id,
            matched=frozenset(f for f in SCORED_FIELDS if field_matches(f, expected, predicted)),
            expected=expected,
            predicted=predicted,
            requested_amount=ctx.request.requested_amount,
            is_capped=expected.amount_safe_to_pay == ctx.request.requested_amount,
        )
        report.scores.append(score)
        if diagnostics_dir is not None:
            _write_diagnostic(diagnostics_dir, ctx, score, _EXTRA.pop(request_id, None))
    return report


#: Filled by the production predictor so diagnostics carry planner internals for Prompt 8.
_EXTRA: dict = {}


def _blank(request_id: str) -> PredictionResult:
    return PredictionResult(
        request_id=request_id, amount_safe_to_pay=Decimal(0),
        affordability_status="not_affordable", recommended_payment_method="not_recommended",
        payment_plan="none", earliest_date_for_full_payment="",
        spending_changes_needed="none", decision_explanation="")


def _write_diagnostic(directory: Path, ctx: RequestContext, score: RequestScore,
                      extra: dict | None = None) -> None:
    """One file per request. Prompt 8 reads these for root-cause analysis."""
    directory.mkdir(parents=True, exist_ok=True)
    profile, expected = ctx.profile, score.expected
    implied_reserve = (profile.current_available_balance
                       - profile.minimum_balance_to_keep
                       - expected.amount_safe_to_pay)
    payload = {
        "request_id": score.request_id,
        "user_id": ctx.request.user_id,
        "request_date": ctx.request.request_date.isoformat(),
        "desired_completion_date": ctx.request.desired_completion_date.isoformat(),
        "requested_amount": str(ctx.request.requested_amount),
        "allows_partial_payment": ctx.request.allows_partial_payment,
        "home_currency": profile.home_currency,
        "opening_balance": str(profile.current_available_balance),
        "minimum_balance": str(profile.minimum_balance_to_keep),
        "methods": sorted(profile.methods),
        "max_installment_months": profile.max_installment_months,
        "is_capped": score.is_capped,
        # Meaningful only when the truth is not clipped at requested_amount.
        "implied_truth_reserve": None if score.is_capped else str(implied_reserve),
        "implied_truth_trough": None if score.is_capped else str(
            profile.current_available_balance - implied_reserve),
        "expected": {f: str(getattr(expected, f)) for f in SCORED_FIELDS},
        "predicted": {f: str(getattr(score.predicted, f)) for f in SCORED_FIELDS},
        "matched": sorted(score.matched),
        "missed": [f for f in SCORED_FIELDS if f not in score.matched],
        "abs_error": str(score.abs_error),
        "rel_error": None if score.rel_error is None else str(score.rel_error),
        "error_over_requested": str(score.error_over_requested),
        "n_events": len(ctx.events),
        "n_messages": len(ctx.messages),
        "n_images": len(ctx.images),
        "options": [
            {"id": o.payment_option_id, "method": o.payment_method,
             "n": o.number_of_payments, "amount": str(o.payment_amount),
             "first": o.first_payment_date.isoformat(),
             "last": o.last_payment_date.isoformat(),
             "total": str(o.total_payable_amount)}
            for o in ctx.options
        ],
        # Populated from ForecastResult / CandidatePlan once those modules land (phases 5 and 7).
        "predicted_trough": None,
        "predicted_trough_date": None,
        "projected_flows": [],
        "streams": [],
        "candidates": [],
        "selected_reason": None,
        "validator_notes": list(score.predicted.validator_notes),
    }
    if extra:
        payload.update(extra)
    (directory / f"{score.request_id}.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------- self-test predictors
# Used ONLY by --self-test to prove the comparator works before a real predictor exists.
# These are not production logic and are never importable by `bow`.


def _oracle(ds: Dataset) -> Predictor:
    def predict(ctx: RequestContext) -> PredictionResult:
        e = ds.expected[ctx.request.request_id]
        return PredictionResult(
            request_id=e.request_id, amount_safe_to_pay=e.amount_safe_to_pay,
            affordability_status=e.affordability_status,                  # type: ignore[arg-type]
            recommended_payment_method=e.recommended_payment_method,      # type: ignore[arg-type]
            payment_plan=e.payment_plan,
            earliest_date_for_full_payment=e.earliest_date_for_full_payment,
            spending_changes_needed=e.spending_changes_needed,
            decision_explanation=e.decision_explanation)
    return predict


def _null(ctx: RequestContext) -> PredictionResult:
    return _blank(ctx.request.request_id)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Buy or Wait? regression harness")
    parser.add_argument("--request", help="score a single request_id")
    parser.add_argument("--self-test", action="store_true",
                        help="verify the comparator with oracle and null predictors")
    parser.add_argument("--diagnostics", action="store_true",
                        help="write per-request diagnostic JSON")
    args = parser.parse_args(argv)

    cfg = RunConfig()
    ds = Dataset.load(cfg)
    ids = [args.request] if args.request else None
    diag = (cfg.debug_dir / "regression") if args.diagnostics else None

    if args.self_test:
        perfect = run(_oracle(ds), ds, ids)
        empty = run(_null, ds, ids)
        n = len(perfect.scored)
        ok = (sum(s.all_correct for s in perfect.scored) == n
              and sum(s.all_correct for s in empty.scored) == 0)
        print(f"oracle predictor : {sum(s.all_correct for s in perfect.scored)}/{n} fully correct")
        print(f"null predictor   : {sum(s.all_correct for s in empty.scored)}/{n} fully correct")
        print("harness self-test:", "ok" if ok else "FAILED")
        return 0 if ok else 1

    from bow.predict import predict, _default_store

    def predictor(ctx):
        out = predict(ctx, _default_store(ctx))
        _EXTRA[ctx.request.request_id] = out.diagnostics
        return out.result

    report = run(predictor, ds, ids, diag)
    print(report.table())
    print()
    print(report.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
