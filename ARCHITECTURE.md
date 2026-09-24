# Buy or Wait? — Architecture

Design document for the HackerRank Orchestrate (September 2026) submission.
Status: **design phase**. No forecasting engine, no planner, no evaluation run, no model calls yet.

Grounded in the two completed analysis phases (see `log.txt`). The rules cited below as *confirmed*
were derived from `problem_statement.md` and reproduced against the 25 solved samples.

---

## 1. Directory / package structure

```text
code/
  main.py                     CLI entry point
  bow/
    __init__.py
    config.py                 frozen config objects (run, forecast, model, pricing)
    money.py                  Decimal money, quantisation, arithmetic guards
    models.py                 all typed internal data models
    errors.py                 exception taxonomy
    fx.py                     dated exchange-rate conversion with provenance
    dataset.py                CSV loading + indexing -> Dataset
    trace.py                  per-request debug trace (no-op unless enabled)
    usage.py                  UsageRecord ledger, cost model, usage_report.md
    ai/
      __init__.py
      schemas.py              JSON schemas, prompt text, version constants
      client.py               HF routed inference client, retry, usage capture
      cache.py                content-addressed extraction cache
      messages.py             GLM-5.3 message-fact extractor
      images.py               Qwen3-VL image-fact extractor
    evidence.py               CSV + message facts + image facts -> EvidenceBundle
    resolve.py                canonical event resolver (lifecycles, de-duplication)
    recurrence.py             recurring-series inference
    forecast.py               ForecastEngine protocol + default implementation
    safeamount.py             amount_safe_to_pay, earliest_date_for_full_payment
    eligibility.py            payment-method eligibility rules
    spending.py               spending-change search
    plans.py                  candidate plan generation
    validate.py               independent validator
    rank.py                   deterministic plan ranker
    explain.py                decision_explanation builder
    predict.py                orchestrator: one request -> PredictionResult
    outputs.py                output.csv writer / formatting
  evaluation/
    regress.py                regression harness over sample_requests.csv
    check_layers.py           import-layer guard (no cycles, no upward imports)
    usage_report.md           generated; required by the submission contract
  ai_cache/                   committed extraction cache (deterministic reruns)
  .debug/                     per-request traces, gitignored
```

Modules are small on purpose. `eligibility.py` and `rank.py` are separated from `plans.py`
specifically so `validate.py` can re-apply the same rules without importing the planner — the
validator must be able to reject a plan the planner produced.

### Layering

A module may import only from **strictly lower** layers. Enforced mechanically by
`evaluation/check_layers.py`, which is the authoritative copy of this table.

| Layer | Modules |
|---|---|
| 0 | `errors` |
| 1 | `money`, `config` |
| 2 | `models` |
| 3 | `fx`, `dataset`, `trace`, `usage`, `ai/schemas`, `ai/cache` |
| 4 | `ai/client` |
| 5 | `ai/messages`, `ai/images` |
| 6 | `evidence` |
| 7 | `resolve` |
| 8 | `recurrence` |
| 9 | `forecast` |
| 10 | `safeamount` |
| 11 | `eligibility`, `spending` |
| 12 | `plans` |
| 13 | `validate`, `rank` |
| 14 | `explain` |
| 15 | `predict`, `outputs` |
| — | `main`, `evaluation/*` (entry points, outside the package) |

Layer 0 is not a single tier: `models` depends on `money`, so the ranks are finer-grained than the
conceptual grouping. The evidence path is strictly one-directional —
`evidence` builds a `RawEvidence` (profile, request, raw events, options, facts); `resolve` turns
raw events into canonical ones; `recurrence` adds streams and assembles the final `EvidenceBundle`
that `forecast` and everything below it consume.

---

## 2. Component diagram

```text
dataset/*.csv ──► dataset.py ──┐
                               │
messages.csv ──► ai/messages ──┤  (GLM-5.3, once, cached)
                               ├──► evidence.py ──► resolve.py ──► recurrence.py
media/images ──► ai/images ────┤  (Qwen3-VL, once, cached)        │
                               │                                   ▼
exchange_rates.csv ─► fx.py ───┘                              forecast.py
                                                                   │
                            ┌──────────────────────────────────────┤
                            ▼                                      ▼
                      safeamount.py                          (re-run per candidate)
                            │                                      ▲
                            ▼                                      │
   request_payment_options.csv ─► eligibility.py ─► plans.py ──────┤
                                       ▲                │          │
                                  spending.py ──────────┘          │
                                                        │          │
                                                        ▼          │
                                                   validate.py ────┘
                                                        │
                                                        ▼
                                                     rank.py
                                                        │
                                                        ▼
                                                    explain.py
                                                        │
                                                        ▼
                                        predict.py ──► PredictionResult ──► outputs.py
                                                        │
                                                        └──► trace.py (opt-in per request_id)
```

One logical orchestrator (`predict.py`). No agent swarm. No persistent LLM conversation: the two
model calls happen in a preprocessing pass before any request is evaluated, and their output is
JSON on disk.

---

## 3. Module responsibilities

| Module | Responsibility | Input | Output | Depends on |
|---|---|---|---|---|
| `config` | Frozen configuration objects. No I/O beyond env reads for secrets. | env, CLI flags | `RunConfig`, `ForecastConfig`, `ModelConfig`, `PriceTable` | — |
| `money` | `Decimal` money, quantisation, safe add/mul, currency guards. Binary floats are never used for money. | `Decimal`/`str` | `Money` | — |
| `models` | Every typed internal representation (§4). Pure dataclasses, no logic. | — | — | `money` |
| `errors` | Exception taxonomy used for fail-closed behaviour (§12). | — | — | — |
| `fx` | Convert `Money` to home currency using the rate for the settlement date; record provenance. Never defaults to 1.0 for an unknown pair. | `Money`, target currency, date | `ConvertedMoney` | `models`, `money`, `errors` |
| `dataset` | Parse all eight CSVs into typed rows; build indexes by `user_id`, `request_id`, `event_id`, `related_event_id`. Validates the schema on load. | `dataset/` | `Dataset` | `models`, `money`, `errors` |
| `ai/schemas` | JSON schema + prompt text + `PROMPT_VERSION`, `SCHEMA_VERSION`, `EXTRACTOR_VERSION`. | — | — | — |
| `ai/client` | HF routed inference calls, timeout, bounded retry, temperature 0, emits `UsageRecord`. | prompt, schema | raw JSON, usage | `config`, `usage`, `errors` |
| `ai/cache` | Content-addressed JSON cache; key derivation; corruption detection. | key parts | cached payload / miss | `config`, `errors` |
| `ai/messages` | GLM-5.3 extraction of one message into a `MessageFact` list. One call per message. | `RawMessage` | `list[MessageFact]` | `ai/*`, `models` |
| `ai/images` | Qwen3-VL extraction of one image into an `ImageFact`. One call per image. | image path, linked event | `ImageFact` | `ai/*`, `models` |
| `evidence` | Assemble everything a single user needs into one bundle; attach facts to their targets. | `Dataset`, fact caches, `user_id` | `EvidenceBundle` | `dataset`, `models`, `fx` |
| `resolve` | Group raw events into economic transactions; decide per event whether it counts for cash and whether it counts for recurrence (§5). | `EvidenceBundle` | `list[CanonicalFinancialEvent]` | `models`, `fx`, `errors` |
| `recurrence` | Infer recurring streams from canonical events flagged `counts_for_recurrence`; apply message amendments; staleness test. | canonical events, facts, `ForecastConfig` | `list[RecurringStream]` | `models`, `config` |
| `forecast` | Project cash flows over the horizon and return the balance path and its trough. Configurable (§6). | `ForecastContext` | `ForecastResult` | `recurrence`, `models`, `config` |
| `safeamount` | `amount_safe_to_pay` and `earliest_date_for_full_payment`. | `ForecastContext`, requested amount | `Decimal`, `date \| None` | `forecast` |
| `eligibility` | Which methods this user/request permits; which supplied options are usable. | profile, request, options | `EligibilitySet` | `models` |
| `spending` | Search permitted flexible streams for ≤3 changes that make a plan safe. | streams, profile, target | `list[SpendingChangeSet]` | `models`, `config` |
| `plans` | Generate every candidate plan (full / partial / installments / wait / none), with and without spending changes. | request, options, forecast, eligibility | `list[CandidatePlan]` | `eligibility`, `spending`, `safeamount` |
| `validate` | Independently verify a candidate against every contract rule (§11). Re-derives the forecast; does not trust the planner. | `CandidatePlan`, context | `list[Violation]` | `forecast`, `eligibility` |
| `rank` | Apply the six documented tie-breakers in order. | valid candidates | winner | `models` |
| `explain` | Deterministic templated explanation grounded in the winning plan's numbers. | winner, context | `str` | `models`, `money` |
| `predict` | Orchestrate one request end to end; own the fail-closed fallback. | `request_id` | `PredictionResult` | everything above |
| `outputs` | Write `output.csv` with exact columns/order and amount formatting. | `list[PredictionResult]` | `output.csv` | `models`, `money` |
| `trace` | Opt-in per-request debug record; `NullTrace` otherwise. | — | JSON file | `models` |
| `usage` | Usage ledger (JSONL), cost model, budget ceiling, `usage_report.md`. | `UsageRecord` | report | `config` |
| `evaluation/regress` | Run the production pipeline over `sample_requests.csv` and score it. | — | reports | `predict` |

---

## 4. Internal data models

All monetary values are `decimal.Decimal`. `money.py` exposes a single `Money` type and forbids
implicit float construction.

```python
Money            = (amount: Decimal, currency: str)

ConvertedMoney   = (source: Money, rate: Decimal, rate_date: date | None,
                    rate_source: {"identity","exact_date","nearest_date","inverse_pair"},
                    target_currency: str, converted: Decimal)
```

`ConvertedMoney` is the provenance unit: every home-currency number in the system carries the
source amount, source currency, rate, rate date and how the rate was found.

| Model | Key fields |
|---|---|
| `FinancialProfile` | `user_id`, `home_currency`, `current_available_balance: Decimal`, `minimum_balance_to_keep: Decimal`, `priorities`, `protected`, `reducible`, `stoppable`, `methods: frozenset`, `max_installment_months: int \| None` |
| `Request` | `request_id`, `user_id`, `request_date`, `request_type`, `requested_amount: Decimal`, `desired_completion_date`, `allows_partial_payment: bool`, `request_text` |
| `RawFinancialEvent` | verbatim typed CSV row + `source_row: int` |
| `CanonicalFinancialEvent` | `economic_id`, `source_event_ids: tuple[str,...]`, `cash_effect: ConvertedMoney \| None`, `cash_date: date \| None`, `sign: {+1,-1}`, `status_class`, `lifecycle_role`, `counts_for_cash: bool`, `counts_for_recurrence: bool`, `exclusion_reason: str \| None`, `category`, `description`, `flexibility`, `minimum_allowed_amount: Decimal \| None`, `evidence_ids: tuple[str,...]` |
| `RecurringStream` | `key` (`kind`,`group`), `kind: {"expense","income"}`, `group` (category for expenses, description for income), `cadence_days: int`, `cadence_class: {"monthly","sub_monthly"}`, `anchor_date`, `anchor_day_of_month`, `amount: Decimal`, `estimator: str`, `latest_event_id`, `flexibility`, `minimum_allowed_amount`, `observations: int`, `is_stale: bool`, `amendments: tuple[MessageFact,...]` |
| `MessageFact` | `message_id`, `user_id`, `fact_type` (enum, see §9), `subject` (enum), `target_event_id \| None`, `effective_date \| None`, `amount: Money \| None`, `multiplier: Decimal \| None`, `quantified: bool`, `confidence: float`, `evidence_span: str` |
| `ImageFact` | `image_id`, `related_event_id`, `chosen_amount: Money`, `semantic_label` (enum: `net_pay`,`balance_due`,`grand_total`,`total_paid`,`amount_due_by_date`,`item_total`), `rejected_candidates: tuple[(label, Money),...]`, `confidence: float`, `notes: str` |
| `ProjectedCashFlow` | `date`, `amount: Decimal` (signed, home currency), `kind: {"recurring","accrual","explicit","plan_payment"}`, `origin` (stream key or event id), `note` |
| `ForecastContext` | `evidence`, `asof`, `config`, `extra_outflows: tuple[ProjectedCashFlow,...]`, `stream_overrides: Mapping[key, Decimal \| None]` |
| `ForecastResult` | `opening_balance`, `minimum_balance`, `flows: tuple[ProjectedCashFlow,...]`, `path: tuple[(date, Decimal),...]`, `trough: Decimal`, `trough_date`, `breaches: tuple[(date, Decimal),...]`, `config_fingerprint: str` |
| `PaymentOption` | `payment_option_id`, `request_id`, `method`, `payment_amount: Decimal`, `number_of_payments`, `first_payment_date`, `payment_frequency_days: int \| None`, `financing_fee: Decimal`, `total_payable_amount: Decimal`, derived `schedule: tuple[(date, Decimal),...]`, `last_payment_date` |
| `SpendingChange` | `kind: {"stop","reduce_to"}`, `stream_key`, `event_id`, `new_amount: Decimal \| None`, `monthly_saving: Decimal` |
| `CandidatePlan` | `method`, `payments: tuple[(date, Decimal),...]`, `changes: tuple[SpendingChange,...]`, `source_option_id: str \| None`, `total_paid: Decimal`, `completes_full_amount: bool`, `completes_by_deadline: bool`, `implied_status` |
| `PredictionResult` | the eight output fields + `trace_id`, `validator_notes`, `fallback_used: bool` |
| `UsageRecord` | `provider`, `model`, `call_type`, `source_id`, `input_tokens`, `output_tokens`, `latency_ms`, `estimated_cost_usd: Decimal`, `cache: {"hit","miss"}`, `retries`, `timestamp` |

**Audit chain.** `PredictionResult → CandidatePlan → ForecastResult.flows → ProjectedCashFlow.origin
→ RecurringStream.latest_event_id / CanonicalFinancialEvent.source_event_ids →
ConvertedMoney (rate + rate_date) → raw CSV row / `MessageFact.message_id` / `ImageFact.image_id`.`
Every final number is traceable to source evidence.

---

## 5. Event-resolution architecture

The resolver's job is to make double-counting structurally impossible. It does this with **two
independent booleans per canonical event**, because the two questions are genuinely different:

- `counts_for_cash` — does this move money in the forecast window?
- `counts_for_recurrence` — may this row be used to infer a repeating pattern?

The second flag is what stops a 41,272 one-off "Bulk groceries and pantry purchase" from
contaminating a grocery stream that averages 8,700, and stops a duplicated payslip row from
shifting a salary cadence off the 15th.

### Flow

```text
RawFinancialEvent[]
   ├─ group into EconomicTransaction clusters
   │     key: transitive closure over linked_event_id, plus description-family pairing
   ▼
EconomicTransaction { role-tagged members, resolution_rule, winner(s) }
   ▼
CanonicalFinancialEvent[]  (counts_for_cash, counts_for_recurrence, exclusion_reason)
```

### Lifecycle patterns and how each resolves

| Lifecycle | Members | Resolution |
|---|---|---|
| authorization → settlement | `cancelled` authorization + linked `settled` purchase | settlement counts for cash; authorization excluded (`superseded_by_settlement`) |
| failed debit → retry | `failed` attempt + linked `scheduled` retry | retry counts on its settlement date; failure excluded (`failed`) |
| purchase → refund (settled) | settled debit + linked settled credit | both count for cash, net zero; **neither** counts for recurrence |
| purchase → refund (pending) | settled debit + linked `pending` credit | debit counts; pending credit excluded (`pending_credit`) |
| duplicate → reversal | `Possible duplicate card charge` linked to original | duplicate excluded (`duplicate`) |
| estimate → amended/settled | message amendment targeting an event | amended value wins (`amended_by_message`), amendment recorded on the canonical event |
| investment purchase → valuation → sale | purchase (settled debit), valuation (`unrealized`, `non_cash`), sale (settled credit) | purchase and sale count for cash; valuation never counts (`non_cash`); none count for recurrence |
| scheduled → cancellation | `scheduled` row later `cancelled`, or a message cancelling it | excluded (`cancelled`) |
| pending → settlement | `pending` debit | counts for cash on `settlement_date`; excluded from recurrence |
| internal transfer | matching debit/credit flagged by a bank message | both excluded (`internal_transfer`) |
| blank amount → image | event with null amount + `images.csv` link | amount supplied by `ImageFact`; counts for cash iff its status says so; **never** counts for recurrence |

Conflict precedence, applied in this order (spec §6.3): explicit cancellation/settlement/amendment →
newer record from the same source → a settled event over an estimate → the financially safer
interpretation.

Downstream modules consume only `CanonicalFinancialEvent` and never touch `RawFinancialEvent`.
`recurrence.py` filters on `counts_for_recurrence`; `forecast.py` filters on `counts_for_cash`.
That single rule is what prevents accidental double counting.

---

## 6. Forecaster interface and configuration

```python
class ForecastEngine(Protocol):
    def project(self, ctx: ForecastContext) -> ForecastResult: ...
```

`ForecastContext.extra_outflows` is how candidate plans are tested — the plan's payments are added
as negative flows and the projection is re-run. `stream_overrides` is how spending changes are
tested — a stream is mapped to `None` (stopped) or a reduced amount. Nothing else in the system
needs to know how the projection works.

```python
@dataclass(frozen=True)
class ForecastConfig:
    horizon_days: int = 90
    horizon_inclusive: bool = True
    amount_estimator: Literal["mean_all","mean_6","mean_3","median_6","last"] = "mean_all"
    cadence_rule: Literal["median_gap","mean_gap"] = "median_gap"
    monthly_threshold_days: int = 26
    monthly_anchor: Literal["day_of_month","fixed_interval"] = "day_of_month"
    submonthly_mode: Literal["accrual","discrete"] = "accrual"
    income_staleness_cycles: Decimal = Decimal("1.5")
    min_observations: int = 3
    same_day_order: Literal["debits_first","credits_first"] = "debits_first"
    quantum: Decimal = Decimal("0.01")
    earliest_horizon: Literal["fixed_from_asof","sliding"] = "fixed_from_asof"
```

Every default has a financial justification, not a sample-fit one:

| Setting | Default | Why |
|---|---|---|
| `horizon_days` | 90 | Stated in the problem statement. |
| `amount_estimator` | `mean_all` | "Forecast essential variable spending conservatively"; the mean is the lowest-variance unbiased estimate of a stable spending habit. `last` is affirmatively disqualified — it more than doubles reserve error in the forensic phase. |
| `submonthly_mode` | `accrual` | A weekly grocery habit is a *rate*, not a dated commitment. Discrete replay charges an integer number of cycles, which over-reserves on long windows and under-reserves on short ones; accrual is unbiased in both. |
| `monthly_anchor` | `day_of_month` | Rent, utilities, subscriptions and payroll recur on a calendar day, not every N days. |
| `income_staleness_cycles` | 1.5 | An income stream that has missed a full cycle is no longer evidenced; projecting it would be inventing unsupported income. |
| `same_day_order` | `debits_first` | The spec's final conflict rule is "the financially safer interpretation". |
| `quantum` | `0.01` | Money is not continuous. |
| `earliest_horizon` | `fixed_from_asof` | Every observed `earliest_date_for_full_payment` in the samples lies inside 90 days of `request_date`. |

Settings are swept and *measured* by the regression harness, but a setting is only adopted if it has
an independent financial rationale. A 75-day horizon scored better than 90 in the forensic phase and
was rejected for exactly this reason.

---

## 7. Model-call and caching strategy

Two preprocessing passes, run once for the whole dataset, before any request is evaluated:

```text
messages.csv  ──(215 calls, GLM-5.3)──► ai_cache/messages/<key>.json ──► MessageFact[]
media/images  ──( 16 calls, Qwen3-VL)──► ai_cache/images/<key>.json  ──► ImageFact[]
```

Model context never grows: each call sees exactly one message or one image plus a fixed schema.
There is no conversation, no history, no per-request prompt, and no model involvement in any
financial calculation.

**Cache key** = `sha256(model_id ‖ prompt_version ‖ schema_version ‖ EXTRACTOR_VERSION ‖ content_hash)`
where the prompt and schema versions are the **per-extractor** constants from `bow/ai/schemas.py`
(`MESSAGE_*` for messages, `IMAGE_*` for images) and `content_hash` is over the message text +
source type + sent-at, or the image file bytes.
Changing a prompt invalidates only the entries produced by that prompt; the other extractor is
untouched. Cache entries store the parsed fact, the raw response, and the `UsageRecord`, so a
cached rerun still reports honest cost attribution.

Both extractors run at temperature 0 with a strict JSON schema. Message and image content is
**untrusted data**: the schemas contain only typed enums, dates and amounts — there is no free-text
field that could carry an instruction into the engine, and the deterministic engine never executes
text.

`MessageFact.fact_type` enum (closed set, derived from the ~25 observed template families):
`salary_amount_change`, `salary_date_change`, `salary_one_off_adjustment`, `first_salary_confirmed`,
`income_ended`, `income_unconfirmed`, `invoice_approved`, `recurring_expense_change`,
`new_recurring_obligation_unquantified`, `refund_pending`, `refund_completed`, `internal_transfer`,
`non_cash_valuation`, `duplicate_under_investigation`, `event_confirmed`, `event_cancelled`,
`event_delayed`, `foreign_currency_note`, `no_financial_effect`.

`quantified: bool` is mandatory. When a message asserts an obligation without an amount (the
"a new recurring childcare payment begins" family), the fact is recorded with `quantified=False`
and **no monetary proxy is ever invented**. It surfaces in the trace and in the explanation as an
acknowledged uncertainty; it does not enter the forecast.

---

## 8. Regression harness design

`evaluation/regress.py` calls the **same** `predict.predict_request` used for `requests.csv`. The
only difference is which request table the `Dataset` exposes. There is no sample-specific code path
anywhere in `bow/`, and no request ID is ever referenced by the engine.

Outputs:

1. **Field-level accuracy** — exact-match rate for `affordability_status`,
   `recommended_payment_method`, `payment_plan`, `earliest_date_for_full_payment`,
   `spending_changes_needed`.
2. **Request-level accuracy** — proportion of requests where all six scored fields match.
3. **`amount_safe_to_pay`** — absolute error, relative error, and error as a percentage of
   `requested_amount`; plus mean/median and the counts within 1%/5%/10%.
4. **Capped-case check** — a dedicated counter for rows where the truth is
   `amount_safe_to_pay == requested_amount`, because missing one of those flips
   `affordability_status` to the wrong value. Two of the four are currently missed.

Per-request diagnostic record (JSON, one file per request, for Prompt 8 root-cause work):

```json
{"request_id": "...", "expected": {...}, "predicted": {...},
 "opening_balance": "...", "minimum_balance": "...",
 "predicted_trough": "...", "predicted_trough_date": "...",
 "implied_truth_trough": "...", "implied_truth_reserve": "...", "is_capped": false,
 "projected_flows": [...], "streams": [...],
 "candidates": [{"method": "...", "total_paid": "...", "valid": true, "violations": []}],
 "selected": {"method": "...", "rank_reason": "beat installments on rule 3 (total paid)"}}
```

`implied_truth_trough` = `current_available_balance − amount_safe_to_pay(expected)`, and
`implied_truth_reserve` = `opening − minimum − expected`. Both are marked meaningless when the row
is capped, since the truth is clipped at `requested_amount` there.

`regress.py --sweep configs.json` scores a list of `ForecastConfig`s and prints a comparison table.
It reports; it does not auto-select.

---

## 9. Trace / debug strategy

```python
trace = Trace.for_request(request_id) if request_id in run_config.trace_request_ids else NULL_TRACE
```

`NullTrace` implements the same methods as no-ops, so tracing costs nothing when disabled and the
call sites need no conditionals. Enabled traces write one JSON file to `code/.debug/<request_id>.json`
with an ordered step list mirroring the audit chain:

```text
request → profile → canonical events (+ exclusion reasons) → streams (+ amendments, staleness)
       → projected flows → forecast path & trough → safe amount → earliest date
       → candidates (each with validator verdict) → ranker decision → final row
```

Selective by `request_id` via `--trace request_07,request_21`. The harness enables tracing
automatically for any sample whose prediction misses, so a failing run leaves exactly the traces
needed and nothing else.

---

## 10. Validation architecture

`validate.py` takes a `CandidatePlan` plus the request, profile, options and a fresh
`ForecastEngine`, and returns `list[Violation]`. It **re-derives** the forecast rather than reusing
the planner's result, so a planner bug cannot smuggle an invalid plan through. It shares
`eligibility.py` with the planner but nothing else.

Checks:

| # | Rule |
|---|---|
| V1 | `0 <= amount_safe_to_pay <= requested_amount` |
| V2 | Projected balance never below `minimum_balance_to_keep` anywhere in the 90-day window, with the plan's payments applied |
| V3 | When the status claims completion, the payments sum exactly to `requested_amount` |
| V4 | Final payment on or before `desired_completion_date` where the status requires completion |
| V5 | Method ∈ `payment_methods_user_will_consider` (and `wait` requires `full_payment`) |
| V6 | Partial: request allows it, user accepts it, `0 < safe < requested`, exactly two payments, first = safe on `request_date`, second on `earliest_date_for_full_payment` ≤ deadline, sum = requested |
| V7 | Installments: schedule matches a supplied `payment_option_id` exactly (dates and amounts), and `number_of_payments <= max_installment_months` |
| V8 | Spending changes target only streams whose `flexibility` permits the action |
| V9 | Change categories are in the user's declared reduce/stop lists |
| V10 | At most three changes |
| V11 | No `stop` and `reduce_to` on the same event |
| V12 | `reduce_to` amount equals the stream's `minimum_allowed_amount` |
| V13 | Payments strictly chronological |
| V14 | `earliest_date_for_full_payment` equals `request_date` iff full payment is safe today; empty iff never safe in the horizon |
| V15 | Output field values are in the allowed enums; `payment_plan` format is well-formed |

Any violation disqualifies the candidate before ranking. If every candidate is disqualified, the
result is `not_affordable` / `not_recommended` with `amount_safe_to_pay` still reported.

---

## 11. Cost / usage architecture

- One call per message (215) and one per image (16) — **231 calls total for the whole dataset**, not
  per request, not per rerun.
- Rough estimate at routed-inference prices: messages ≈ 130k tokens total, images ≈ 40k tokens
  total. Expected spend well under **USD 0.50** against the USD 4.20 ceiling.
- `usage.py` appends a `UsageRecord` per call to `evaluation/usage.jsonl` capturing provider, model,
  call type, source id, input/output tokens, latency, estimated cost, cache hit/miss and retry count.
- `RunConfig.budget_ceiling_usd` is a hard stop: the client refuses a call that would cross it and
  raises `BudgetExceeded` rather than silently continuing.
- `usage_report.md` is generated from the ledger, with per-model and overall totals, total and
  average tokens per request, and total and per-request cost. It is regenerated from the ledger of
  the run that produced the submitted `output.csv`; cached entries carry their original usage so the
  report reflects real consumption, and cache hits are reported separately so the numbers are honest.

---

## 12. Error handling

Fail-closed throughout: never invent a financial fact to keep going. The output contract still
requires 250 rows, so `predict.py` catches per-request failures and emits a conservative row rather
than crashing the run or fabricating a number.

| Failure | Behaviour |
|---|---|
| Malformed CSV row | Raise at load time with file/row/column. The dataset is an input we do not control; a silent coercion would corrupt every downstream number. |
| Missing exchange rate for the exact date | Use the nearest date **for the same ordered pair**, record `rate_source="nearest_date"`. |
| Exchange-rate pair absent entirely | `MissingRateError`. The event is excluded from cash flow, flagged in the trace, and the request is marked degraded. Never assume 1.0. |
| Invalid AI JSON | One repair retry with the schema restated. Then mark the fact unusable, proceed on CSV evidence alone, record in trace and usage ledger. |
| AI timeout | Bounded retries with backoff, then treated as unusable as above. |
| Low-confidence image extraction | The blank amount is **never** zero. If the extractor cannot commit to a label, take the most conservative candidate (largest for a debit, smallest for a credit), mark `confidence`, and flag the request. |
| Cache corruption | Checksum mismatch → treat as a miss and re-extract; log it. |
| Missing `related_event_id` target | The fact is retained but unattached; it may still apply at user level. Logged. |
| Contradictory evidence | Apply the documented precedence (cancellation/amendment → newer same-source → settled over estimate → safer interpretation). If still tied, take the safer reading and record both in the trace. |
| Unsupported / unparseable payment option | Option dropped from the candidate set, logged. Never invent a schedule. |
| Arithmetic invariant violation | `InvariantError`. The candidate is discarded; if it was the winner, fall back down the ranking. |
| All candidates invalid | Emit `not_affordable` / `not_recommended`, `payment_plan=none`, empty `earliest_date_for_full_payment`, `amount_safe_to_pay` as computed, explanation stating the constraint. |
| Unhandled per-request exception | Emit the same conservative row with `fallback_used=True`, log the traceback, continue the run. The run summary reports the count; a non-zero count blocks submission. |

---

## 13. Implementation order

| Phase | Scope | Gate |
|---|---|---|
| 3 | `config`, `money`, `models`, `errors`, `fx`, `dataset` | All CSVs load into typed models; FX round-trips with provenance |
| 4 | `resolve`, `recurrence` | Every lifecycle family in §5 classified; stream inference stable across all 275 users |
| 5 | `forecast`, `safeamount`, `evaluation/regress` | Harness runs on the 25 samples with **CSV-only** evidence — no AI spend yet — and reports the §8 metrics |
| 6 | `ai/*` (schemas, client, cache, messages, images) | 231 extractions cached; usage ledger populated; cost confirmed under budget |
| 7 | `eligibility`, `spending`, `plans`, `validate`, `rank`, `explain` | All six scored fields produced; validator rejects malformed candidates |
| 8 | Root-cause analysis on harness misses; config tuning under the §6 justification rule | Documented accuracy on the 25 |
| 9 | Full 250 run, `usage_report.md`, `code.zip` packaging | 250 rows, contract checks pass, zero fallback rows |

Phase 5 deliberately precedes Phase 6 so the forecaster is debugged at zero API cost.

---

## 14. Risks and mitigations

| # | Risk | Mitigation |
|---|---|---|
| R1 | `amount_safe_to_pay` residual (~6% MAE, ~1.4% median) is scored directly | `ForecastConfig` keeps every choice tunable; the harness is a permanent regression suite; Phase 8 is dedicated to it |
| R2 | Missing a capped case flips `affordability_status` (2 of 4 currently missed) | Dedicated capped-case metric in the harness; treat as a release gate, not a statistic |
| R3 | Message extraction quality across English and Indonesian templates | Closed enum schema, temperature 0, validation of every enum/date/amount, unusable facts degrade to CSV-only rather than guessing |
| R4 | Image ambiguity (`image_04` is truncated below "Item Bill"; `image_07` shows both 8,528.10 and 8,528) | Extractor must return the chosen label *and* the rejected candidates; low confidence routes to the conservative branch and is flagged |
| R5 | Prompt injection from untrusted message/image content | Schemas carry no free-text action field; the engine consumes typed facts only and never executes text |
| R6 | FX gaps — rates are dated only on the 15th, but `event_7307` settles 2025-10-01 | Nearest-date rule for the same pair, provenance recorded; absent pair fails closed |
| R7 | Budget overrun | Hard ceiling in the client, committed cache, 231 total calls, no per-request LLM reasoning |
| R8 | Non-determinism between the scored run and the reported usage | Temperature 0, committed cache, usage ledger tied to the run that wrote `output.csv` |
| R9 | Over-fitting the 25 samples | Config changes require an independent financial rationale; the 75-day horizon precedent is recorded as a rejected fit |
| R10 | Only 45/275 requests can use partial payment and 80/275 have a usable installment option — most requests reduce to full/wait/not_recommended | Eligibility is computed once, early, and drives candidate generation, so the common path is cheap and well tested |

---

## Assumptions

1. `current_available_balance` is the balance as at `request_date`, with all settled events already
   reflected. Nothing settled before `request_date` is replayed into the forecast.
2. The 90-day safety window runs from `request_date` inclusive.
3. `max_installment_months` is compared against `number_of_payments` (confirmed: the two encodings
   agree on all 275 users, and blank always coincides with `installments` absent from the methods).
4. An installment option is usable only if its final payment falls on or before
   `desired_completion_date` (confirmed on all 25 samples; 434 of 515 options fail this).
5. `reduce_to` uses the stream's `minimum_allowed_amount` exactly.
6. A spending change names the most recent settled occurrence of the stream.
7. `decision_explanation` is templated deterministically. It is graded on usefulness and consistency,
   and a template grounded in the winning plan's real numbers is both — and costs nothing.
