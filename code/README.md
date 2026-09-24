# Buy or Wait? — solution

An AI-assisted financial decision agent for the HackerRank Orchestrate challenge.

For every request in `dataset/requests.csv` it decides whether the user should pay in full, pay
part now, use an offered installment plan, wait, or not proceed — and reports the largest amount
that is safe to pay today without breaking the user's minimum balance at any point in a 90-day
forecast.

---

## Quick start

```bash
python code/main.py
```

Writes `output.csv` (250 rows + header) to the repository root in about **2 seconds**.

```bash
python code/evaluation/main.py
```

Scores the 25 solved samples against their ground truth, then hard-validates `output.csv`.
Exits non-zero on any invariant failure.

**Requirements: Python 3.10+ and nothing else.** The solution uses only the standard library — no
`pip install`, no virtualenv, no network access, no API key.

---

## How it works

The two models do **evidence extraction only**. Neither computes affordability, picks a payment
method, ranks plans or writes an output field. Every financial calculation is deterministic Python.

| Model | Role |
|---|---|
| `zai-org/GLM-5.3` | reads each message and emits a typed fact (salary change, income ended, rent increase, refund pending, …) |
| `Qwen/Qwen3-VL-235B-A22B-Instruct` | reads each document image and extracts the one amount that represents the transaction's real economic value |

Extraction runs **once for the whole dataset** — 215 messages, 16 images — and the structured
facts are cached by content hash in `code/ai_cache/`. Prediction therefore makes **zero model
calls** and is fully reproducible offline.

```
dataset CSVs ──┐
               ├─► evidence ─► canonical events ─► recurring streams ─► 90-day forecast
cached facts ──┘                                                             │
                                                                             ▼
   candidate plans ─► independent validator ─► deterministic ranker ─► output row
```

### Pipeline stages

1. **Canonical event resolution** — resolves transaction lifecycles (authorization → settlement,
   failed → retry, purchase → refund, duplicate charges, investment valuations) so no economic
   event is counted twice. Each event carries two independent flags: whether it moves cash in the
   forecast window, and whether it may seed a recurring series.
2. **Recurrence inference** — infers streams from settled history. Expenses group by category;
   income groups by class, because freelance and gig labels vary per payment. Income is projected
   only when the evidence supports it continuing.
3. **90-day forecast** — projects from `current_available_balance` at `request_date`. Monthly
   commitments land on their calendar day; sub-monthly habits accrue as a daily rate.
4. **Capacity** — `amount_safe_to_pay` and `earliest_date_for_full_payment`, both measured
   independently of the user's method preferences.
5. **Planning** — generates every financially meaningful candidate, re-forecasting each one from
   scratch, including plans that need permitted spending changes.
6. **Validation** — a separate module re-derives the forecast and applies 14 numbered checks. It
   does not trust the planner.
7. **Ranking** — the specification's six tie-breakers, applied literally.

All money is `decimal.Decimal`. Floats are rejected at construction.

---

## Agent mode (`--agent`)

```bash
python code/main.py --agent --samples     # writes output_agent_samples.csv
python code/evaluation/compare.py         # agent vs pipeline vs no-AI baseline
```

The default run is a fixed-order pipeline. Agent mode (`bow/agent.py`) hands control of the
workflow to GLM-5.3 in a **bounded tool loop**: the model decides which evidence to read, when to
run the forecast, and which validated plan to recommend. The engine still does all arithmetic.

| Tool | What it does |
|---|---|
| `get_request` | request + profile: balance, minimum, accepted methods, deadline, changeable categories |
| `list_evidence` | message/image ids with their extracted fact type (never raw text) |
| `read_message` / `read_image` | read one fact — and **add it to the forecast's evidence** |
| `run_forecast` | 90-day forecast on the evidence read so far |
| `list_plans` | every candidate plan that passed the independent validator |
| `finish` | recommend one listed plan |

Guardrails: at most 10 model calls per request; a bad tool call returns an error the model can
recover from; the budget ceiling is checked before every call; and any failure (step cap,
provider error, budget, an unknown plan id) **falls back to the deterministic pipeline's answer**,
with the reason recorded. Every step is cached, so a re-run is free and reproducible. Per-request
traces: `code/.debug/agent/<prompt>/<request_id>.json`.

Measured results are in `evaluation/comparison.md`.

---

## Re-running the AI extraction (optional)

Not needed to reproduce `output.csv` — the cache is included. If you want to re-extract:

```bash
export HF_TOKEN=...                       # never read from source
python code/evaluation/extract.py --dry-run   # projected cost, makes no calls
python code/evaluation/extract.py             # ~231 calls, roughly USD 0.20
```

Extraction uses Hugging Face routed inference (`router.huggingface.co/v1`). No dedicated
endpoints are created. Cache keys bind the model, prompt version, schema version, extractor
version and a content hash, so editing one prompt invalidates only that extractor's entries.

Message and image content is treated as **untrusted data**: the schemas are closed enums with no
free-text action field, so nothing inside a document can reach the decision engine as an
instruction.

---

## Layout

```
code/
  main.py                  prediction entry point  (--manifest prints the frozen config)
  bow/                     the solution
    config.py              frozen ForecastConfig, model config, prices
    money.py               Decimal money, FX provenance
    models.py              typed data models
    dataset.py             CSV loading and indexing
    fx.py                  dated exchange-rate resolution
    resolve.py             canonical event / lifecycle resolution
    recurrence.py          recurring-stream inference
    forecast.py            90-day projection
    safeamount.py          amount_safe_to_pay, earliest safe date
    eligibility.py         method and option eligibility
    spending.py            permitted spending changes
    plans.py               candidate generation
    validate.py            independent validator
    rank.py                deterministic ranking
    explain.py             templated explanation
    predict.py             orchestrator
    outputs.py             output serialisation
    ai/                    extraction clients, schemas, cache, fact loading
  evaluation/
    main.py                evaluation + validation entry point
    usage_report.md        token and cost report
    extract.py             one-off AI extraction
    regress.py             25-sample regression harness
    usage_summary.py       regenerates usage_report.md
    check_layers.py        import-layer guard
  tests/                   177 unit and integration tests
  ai_cache/                231 cached structured facts (215 messages, 16 images)
```

---

## Tests

```bash
python -m unittest discover -s code/tests -t code
python code/evaluation/check_layers.py
```

177 tests. The layer guard enforces a one-directional import hierarchy so the dependency graph
cannot develop cycles.

---

## Reproducibility

`python code/main.py` is deterministic. Two consecutive runs produce a byte-identical
`output.csv`:

```
sha256  7ec8757424b04be82e2b4f7b5c0c8b2c322579cfec0d9731e231125eb7e94f2a
```

`python code/main.py --manifest` prints the exact configuration, model ids, prompt/schema
versions and git commit behind a run.

---

## Accuracy on the 25 solved samples

Measured with the same code path that processes the evaluation set — there is no sample-specific
logic anywhere.

| Field | Correct |
|---|---|
| `recommended_payment_method` | 22/25 |
| `spending_changes_needed` | 21/25 |
| `payment_plan` | 21/25 |
| `affordability_status` | 20/25 |
| `earliest_date_for_full_payment` | 19/25 |

`amount_safe_to_pay`: median error **1.88%** of the requested amount, 21/25 within 5%, 3 exact.
The residual traces to estimating variable household spending (groceries, transport, dining); no
generalising correction was found that the visible evidence supports, and none was forced.

---

## Cost

231 model calls, 219,655 tokens, **USD 0.20** for the whole dataset. Prediction adds nothing —
it makes no model calls. See `evaluation/usage_report.md`.
