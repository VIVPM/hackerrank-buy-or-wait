"""The orchestrator: one request in, one PredictionResult out.

This is the single prediction interface. The regression harness and the full run both call
`predict_request`; there is no sample-specific path anywhere.

Status mapping is deterministic and follows from the winning plan. `earliest_date_for_full_payment`
is reported from the capacity calculation, never from the chosen method - a user who will not
consider `full_payment` can still have an earliest date of `request_date`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from bow.ai.facts import FactStore
from bow.config import ForecastConfig, RunConfig
from bow.dataset import RequestContext
from bow.explain import build as build_explanation
from bow.forecast import ForecastEngine
from bow.models import AffordabilityStatus, CandidatePlan, PredictionResult
from bow.outputs import plan_fields, serialise_date
from bow.plans import generate
from bow.rank import explain_choice, rank
from bow.recurrence import build_evidence
from bow.resolve import resolve_context
from bow.validate import validate
from bow.trace import NULL_TRACE, TraceLike

ZERO = Decimal(0)


def status_for(plan: CandidatePlan) -> AffordabilityStatus:
    if plan.method == "not_recommended":
        return "not_affordable"
    if plan.method == "wait":
        return "affordable_later"
    if plan.method == "full_payment" and not plan.changes:
        return "affordable_now"
    # full payment made possible only by spending changes, plus every partial and installment
    # plan, completes the request through a plan rather than outright.
    return "affordable_with_plan"


def result_for(ctx: RequestContext, evidence, planning, winner: CandidatePlan,
               rejections: dict[str, list[str]]) -> PredictionResult:
    """One output row for a chosen plan. Shared by the pipeline and the agent."""
    status = status_for(winner)
    plan_text, changes_text = plan_fields(winner)
    earliest = planning.earliest
    if status == "affordable_now":
        # By definition the full amount is safe today, so capacity and plan agree.
        earliest = ctx.request.request_date
    return PredictionResult(
        request_id=ctx.request.request_id,
        amount_safe_to_pay=planning.safe.amount,
        affordability_status=status,
        recommended_payment_method=winner.method,
        payment_plan=plan_text,
        earliest_date_for_full_payment=serialise_date(earliest),
        spending_changes_needed=changes_text,
        decision_explanation=build_explanation(winner, evidence,
                                               planning.safe.amount, earliest),
        validator_notes=tuple(f"{k}: {'; '.join(v)}" for k, v in rejections.items()),
    )


@dataclass(frozen=True, slots=True)
class Prediction:
    result: PredictionResult
    diagnostics: dict


def predict(ctx: RequestContext, store: FactStore | None = None,
            config: ForecastConfig | None = None,
            trace: TraceLike = NULL_TRACE) -> Prediction:
    cfg = config or ForecastConfig()
    engine = ForecastEngine(cfg)
    messages, images = store.for_user(ctx.request.user_id) if store else ((), ())

    resolution = resolve_context(ctx, messages, trace)
    evidence = build_evidence(ctx, resolution, cfg, images, trace)
    planning = generate(evidence, engine, trace)

    ordered = rank(planning.candidates)
    rejections: dict[str, list[str]] = {}
    winner: CandidatePlan | None = None
    for candidate in ordered:
        problems = validate(candidate, evidence, engine,
                            planning.safe.amount, planning.earliest)
        if not problems:
            winner = candidate
            break
        label = (f"{candidate.method}:{candidate.source_option_id or '-'}"
                 f":{len(candidate.changes)}")
        rejections[label] = [f"{v.code} {v.message}" for v in problems]

    if winner is None:                      # fail closed, never emit an invalid row
        winner = next(c for c in ordered if c.method == "not_recommended")

    runners = [c for c in ordered if c is not winner and c.method != "not_recommended"]
    result = result_for(ctx, evidence, planning, winner, rejections)

    diagnostics = {
        "request_id": ctx.request.request_id,
        "user_id": ctx.request.user_id,
        "safe": planning.safe.as_dict(),
        "earliest": serialise_date(planning.earliest),
        "streams": [
            {"key": list(s.key), "cadence": s.cadence_days, "class": s.cadence_class,
             "amount": str(s.amount), "anchor": s.anchor_date.isoformat(),
             "estimator": s.estimator, "observations": s.observations}
            for s in evidence.streams
        ],
        "flows": [
            {"date": f.date.isoformat(), "amount": str(f.amount), "kind": f.kind,
             "origin": f.origin, "note": f.note}
            for f in planning.base_forecast.flows
        ],
        "daily_accrual": str(planning.base_forecast.daily_accrual),
        "candidates": [
            {"method": c.method, "option": c.source_option_id,
             "payments": [[d.isoformat(), str(a)] for d, a in c.payments],
             "changes": [f"{x.kind}:{x.event_id}" for x in c.changes],
             "total": str(c.total_paid), "completes_by_deadline": c.completes_by_deadline,
             "selected": c is winner}
            for c in ordered
        ],
        "selected_reason": explain_choice(winner, runners),
        "rejected_candidates": rejections,
        "rejected_before_validation": [list(r) for r in planning.rejected],
        "message_facts": [
            {"id": f.message_id, "type": f.fact_type, "subject": f.subject,
             "amount": str(f.amount.amount) if f.amount else None,
             "effective_date": f.effective_date.isoformat() if f.effective_date else None,
             "quantified": f.quantified, "confidence": f.confidence}
            for f in messages
        ],
        "image_facts": [
            {"id": f.image_id, "event": f.related_event_id,
             "amount": str(f.chosen_amount.amount), "label": f.semantic_label,
             "confidence": f.confidence}
            for f in images
        ],
        "unquantified": [f.message_id for f in evidence.unquantified],
    }
    return Prediction(result, diagnostics)


def predict_request(ctx: RequestContext) -> PredictionResult:
    """Signature the regression harness expects."""
    return predict(ctx, _default_store(ctx)).result


_STORE: FactStore | None = None


def _default_store(ctx: RequestContext) -> FactStore | None:
    global _STORE
    if _STORE is None:
        from bow.ai.facts import load_facts
        from bow.dataset import Dataset
        _STORE = load_facts(Dataset.load(RunConfig()))
    return _STORE
