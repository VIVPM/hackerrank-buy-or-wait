# AI Judge Interview Preparation

**Buy or Wait?** — HackerRank Orchestrate, September 2026.

Every number in this document was read from the final repository state, not from memory or earlier
design notes. Where documentation and code disagreed, the code won. Where a constant has no
stronger justification than "it sits safely inside a band that the data makes indifferent", this
document says so rather than inventing a reason.

---

## 60-Second Explanation

> For every one of 250 financial requests, the system decides whether the user should pay in full
> today, pay part now and the rest later, take one of the seller's installment offers, wait, or not
> proceed at all — and it reports the largest amount that is genuinely safe to pay today.
>
> The inputs are mixed. Most of it is structured: 275 financial profiles, 25,342 ledger events,
> 790 payment options, dated exchange rates. But some of the decisive evidence is unstructured —
> 215 customer messages in English and Indonesian, and 16 document images like payslips, hospital
> bills and taxi receipts.
>
> So I used AI for exactly the part that needs semantic understanding. **GLM-5.3 reads each message**
> and emits one typed fact — salary changed to X from this date, employment ended, this refund is
> still pending. **Qwen3-VL reads each image** and picks out the one amount that represents the real
> economic value — net pay rather than gross, balance due rather than invoice total, the fare rather
> than the cash handed over.
>
> Everything financial after that is deterministic Python. I reconstruct the ledger, resolve
> transaction lifecycles so nothing is counted twice, infer which expenses and incomes recur,
> convert currencies, and roll a 90-day balance forecast forward. The largest safe payment is the
> gap between the forecast's lowest point and the user's minimum balance. Then I generate every
> financially meaningful payment plan, re-forecast each one, hand them to a validator that doesn't
> trust the planner, and rank the survivors by the challenge's six tie-breakers.
>
> Extraction runs once for the whole dataset and is cached, so generating all 250 predictions makes
> zero model calls and takes about two seconds — and it's byte-for-byte reproducible.

---

## 3-Minute Technical Explanation

The problem is a safety-constrained planning problem dressed as a personal-finance question. For
each request I need the largest `x` payable today such that a 90-day balance projection never dips
below the user's `minimum_balance_to_keep`, plus a recommended plan that respects the user's stated
payment preferences and the seller's actual offers.

**Where the AI sits.** Two narrow extraction jobs, both one-shot and schema-constrained:

- GLM-5.3, one call per message, returns a closed-enum JSON fact (operation, target, amount,
  currency, effective date, confirmed flag, recurrence effect, confidence).
- Qwen3-VL-235B-A22B-Instruct, one call per image, returns the chosen amount, a semantic label for
  *which* figure it chose, and the other candidate figures it rejected so the choice is auditable.

Neither model sees a balance, computes affordability, picks a method, or writes an output field.

**The deterministic pipeline**, in order:

1. **Canonical event resolution.** Raw ledger rows become economic events. Eight linked lifecycle
   families are resolved — a cancelled authorization superseded by its settlement, a failed debit
   replaced by its scheduled retry, a duplicate charge linked to the original, a purchase offset by
   a refund, an employer reimbursement, an investment contribution with its unrealized valuation
   and later sale. Each event carries two independent booleans: does it move cash inside the
   forecast window, and may it seed a recurring series.
2. **Recurrence inference.** Expenses group by category; income groups by class. Income is only
   projected forward when the evidence supports it continuing.
3. **FX normalisation.** Foreign amounts convert on their settlement date using the supplied table,
   carrying full provenance (rate, rate date, how the rate was found).
4. **90-day forecast** from `current_available_balance` at `request_date`. Monthly commitments land
   on their calendar day; sub-monthly habits accrue as a daily rate.
5. **Capacity**: `amount_safe_to_pay` and `earliest_date_for_full_payment`, both measured without
   reference to the user's method preferences.
6. **Plan generation**: full payment, partial payment, each supplied installment option, wait, and
   the not-recommended fallback — plus variants requiring permitted spending changes. Each
   candidate is re-forecast from scratch.
7. **Independent validation**: 14 numbered checks, re-deriving the forecast rather than trusting
   the planner.
8. **Deterministic ranking** by the challenge's six rules.
9. **Templated explanation** built from the winning plan's own numbers.

All money is `decimal.Decimal`; the `Money` type raises on a float. 28 modules, ~4,800 lines,
177 tests, zero third-party dependencies.

---

## Architecture

```
dataset/*.csv ──────────────┐
                            │
messages.csv ──► GLM-5.3 ───┤  (one call each, cached)
                            ├──► evidence ──► resolve ──► recurrence ──► forecast
media/images ──► Qwen3-VL ──┘         │          │            │             │
                                      │          │            │             ▼
exchange_rates.csv ──► fx ────────────┘          │            │        safeamount
                                                 │            │             │
                                                 ▼            ▼             ▼
                              canonical events, two flags   streams    plans ◄── eligibility
                                                                         │   ◄── spending
                                                                         ▼
                                                                     validate
                                                                         │
                                                                         ▼
                                                                       rank
                                                                         │
                                                                         ▼
                                                                      explain
                                                                         │
                                                                         ▼
                                                                    output.csv
```

### Module responsibilities

| Module | Does | Consumes | Produces | Why deterministic / AI |
|---|---|---|---|---|
| `dataset.py` | typed load + 11 indexes | 8 CSVs | `Dataset`, `RequestContext` | Deterministic — parsing must be exact and fail loudly |
| `fx.py` | dated rate resolution | `Money`, date | `ConvertedMoney` w/ provenance | Deterministic — the rate table is supplied |
| `ai/messages.py` | message → typed fact | one message | `MessageFact` | **AI** — natural language, two languages, implicit meaning |
| `ai/images.py` | image → typed fact | one PNG | `ImageFact` | **AI** — needs to know *which* number matters, not just read it |
| `resolve.py` | lifecycle resolution, de-duplication | raw events | `CanonicalFinancialEvent` | Deterministic — it's bookkeeping with clear rules |
| `recurrence.py` | stream inference, fact application | canonical events, facts | `RecurringStream` | Deterministic — cadence and level are statistics |
| `forecast.py` | 90-day projection | evidence, config | balance path, trough | Deterministic — arithmetic |
| `safeamount.py` | safe amount, earliest date | forecast | `Decimal`, `date` | Deterministic — a constrained maximum |
| `eligibility.py` | method/option permissions | profile, request, options | verdicts + reasons | Deterministic — rules from the spec |
| `spending.py` | permitted changes, enumeration | profile, streams | `SpendingChange` sets | Deterministic — small bounded search |
| `plans.py` | candidate generation | all of the above | `CandidatePlan[]` | Deterministic |
| `validate.py` | independent verification | a candidate | `Violation[]` | Deterministic — defence in depth |
| `rank.py` | the six tie-breakers | valid candidates | winner | Deterministic — the spec fixes the order |
| `explain.py` | templated explanation | winner | `str` | Deterministic — must not drift from the plan |
| `predict.py` | orchestrator, status mapping | `RequestContext` | `PredictionResult` | Deterministic |

**Why separate?** Two reasons that actually earned their keep. First, the validator can only be
meaningful if it's independent of the planner — and it caught a real defect late (see §Independent
Validation). Second, the import graph is enforced one-directional by `check_layers.py`, so the
dependency structure can't quietly rot into a cycle.

---

## Why I Did Not Use a Multi-Agent Swarm

A swarm is the wrong shape for this problem, for five concrete reasons:

**Compounding interpretation error.** If one agent estimates recurring expenses, another projects
income, and a third assembles a plan, each hands the next a *judgement* rather than a number. Errors
multiply instead of being caught. A wrong grocery estimate silently becomes a wrong recommendation
with no point at which anything is checkable.

**Non-determinism.** The submission must be reproducible. My `output.csv` regenerates to the same
SHA256 every run. That is impossible if several models negotiate the answer between themselves.

**Token and context growth.** A typical user has ~92 ledger events. Feeding 250 users' histories
through conversational agents would be enormous and repetitive, for no gain — the arithmetic is
trivial for a computer and error-prone for a model.

**Lost-in-the-middle.** The decisive fact is often one line buried in a long history: a *Final
employer payroll* rather than a *Payroll credit*, or a refund that is `pending` rather than
`settled`. Long contexts are exactly where models miss those.

**Exact schedules can't be negotiated.** An installment plan must match a supplied option to the
cent and the day. A ranking must apply six tie-breakers in order. Those are specification
compliance, not judgement calls.

**The pattern I used instead:** one logical orchestrator (`predict.py`), small isolated AI
extraction calls, all state in Python data structures and on-disk JSON, and deterministic
specialist modules.

### "Is this still an AI agent?"

Yes — and I'd argue it's the more defensible kind. The agent perceives an environment it cannot
parse mechanically (free-text messages in two languages, photographed documents), converts that
perception into structured belief, reasons over that belief with tools, and commits to an action
under constraints. That AI perception is *load-bearing*: without it, four requests have a liability
with no amount at all, and a user's salary change or termination is invisible. What I deliberately
did not do is let a language model perform arithmetic that a computer does exactly. Using AI where
it's better than code, and code where it's better than AI, is engineering judgement, not an absence
of AI.

---

## Avoiding Context-Window and Lost-in-the-Middle Problems

The system never places hundreds of financial events, all messages, all images, or a candidate-plan
set inside one persistent model conversation. There is no chat history at all.

- **Isolated calls.** One message per call; one image per call. A message call's context is that
  message, its metadata, and a one-line summary of its linked event if it has one.
- **Strict JSON.** Closed-enum schemas, `temperature=0.0`, structured-output mode.
- **State lives in Python.** Facts are cached as JSON on disk. Prediction reads them.
- **No 250-request conversation.** Prediction makes zero model calls.

The consequences are measurable: **accuracy** (the model is never asked to find a needle in a
25,000-row haystack), **cost** ($0.20 total instead of per-request reasoning), **determinism**
(same input, same bytes out), and **debuggability** (I can open any single cached fact and compare it
to its source message).

---

## AI Models

### Text — `zai-org/GLM-5.3`

**Used for:** one structured extraction per message → `operation`, `target`, `amount`, `currency`,
`effective_date`, `new_date`, `confirmed`, `recurrence_effect`, `percent_change`, `notes_short`,
`confidence`.

**Explicitly not allowed to:** compute affordability, choose a payment method, rank plans, decide
whether money is spendable, or write any output field.

**Why this model:** the messages are half English, half Indonesian, and the distinction that matters
is often subtle — "your quarterly bonus is still subject to review" versus "your regular salary for
the next payroll is EUR 1452". That calls for strong multilingual instruction-following, and it needs
structured-output mode, which the guardrails depend on.

**Why not smaller:** I prioritised capability because the volume was tiny (215 calls) and the cost
negligible ($0.19), so there was no economic reason to trade accuracy for size. I did **not**
benchmark smaller models and I can't claim they would have failed — the honest statement is that the
downside of a misread message is a materially wrong recommendation and the upside of a smaller model
was a few cents.

### Vision — `Qwen/Qwen3-VL-235B-A22B-Instruct`

**Used for:** one structured extraction per image → document type, currency, the chosen amount, the
semantic label for that amount, payment status, the rejected candidate amounts, confidence.

**Explicitly not allowed to:** recommend a plan or judge affordability.

**Why this model:** the task is not OCR, it's *document comprehension*. A payslip shows Salary
4,500,000, Total Earnings 4,780,800 and Net Pay 4,365,000 — three plausible numbers, one correct
answer. A rent receipt shows Total 200,000, Received 100,000, Balance Due 100,000. Choosing
correctly requires understanding the document's structure. A 235B mixture-of-experts VLM has the
headroom; the images include a handwritten pharmacy bill and a low-contrast thermal receipt.

**Why Instruct rather than a long-reasoning variant:** the job is a constrained extraction with a
fixed schema, not a puzzle. Instruct returns clean JSON immediately. This mattered in practice —
GLM's reasoning mode consumed the entire completion budget thinking and returned empty content
until I disabled it.

**Why two models rather than one:** they're genuinely different tasks. Using a vision model for 215
text-only messages would cost more for no benefit; text-only for images is impossible.

**Why routed inference rather than a dedicated endpoint:** 231 calls total. A dedicated GPU endpoint
bills for provisioned time, not usage — wildly wrong for this shape of workload, plus cold-start and
teardown management. Routed inference is pay-per-token and needs no lifecycle management. The
tradeoff is that the router picks a provider per call and providers disagree about parameters; I
handle that with a variant ladder (below).

---

## Vision Processing

Sixteen images, one extraction each, cached by file content hash. Every image is linked to a ledger
event whose `amount` column is **blank**.

**The core design point: a blank amount is never zero.** It is carried as
`amount_status="unresolved_image"` with `cash_effect=None`. If the amount is still unresolved and
the event would otherwise be spendable cash, the event is excluded from cash *and flagged*, never
silently treated as zero.

**Semantic traps present in this dataset**, all handled correctly:

| Image | Trap | Chosen | Rejected |
|---|---|---:|---|
| `image_01` | gross vs net pay | **4,365,000** net_pay | total_earnings 4,780,800 |
| `image_02` | invoice total vs balance due | **100,000** balance_due | invoice total 200,000 |
| `image_05` | pay-by-date vs late amount vs previous balance | **704.05** | after-date 822.05, previous balance 3,543.54 |
| `image_10` | subtotal vs grand total | **79,679.26** | subtotal 72,045.00 |
| `image_12` | fare vs cash tendered | **33.50** | cash_tendered 40.00 |
| `image_14` | handwritten total | **4,543** | six line items |
| `image_15` | taxable value vs total incl. taxes | **9,968.00** | 9,512 / 9,124 |

All 16 matched an independent manual read of the images. Confidence was ≥0.95 on every one.

**Why not plain OCR:** OCR would return every number on the page. It would not tell you that on a
payslip the economically relevant figure is Net Pay, or that on a part-paid rent receipt it's
Balance Due. That selection is the whole task. I also ask the model to return the *rejected*
candidates, so the choice is auditable rather than trusted — a judge can open any cached fact and
see what it considered.

Four of the sixteen are future obligations (`pending`/`scheduled`) where the extracted amount
directly changes the forecast: −100,000 rent arrears, −704.05 telecom, −79,679.26 grocery invoice,
−3,650 hospital bill. The other twelve are settled history whose value matters mainly because they
must *not* pollute a recurring series — `event_1545` is a 41,272 "grocery" in a series whose true
level is about 8,700.

---

## Message Processing

GLM extracts a fact per message. The engine's vocabulary (`FactType`) includes salary amount change,
salary date change, income ended, income unconfirmed, recurring expense change, event delayed,
refund pending, duplicate under investigation, and no financial effect.

### The important lesson: raw LLM labels were not trusted

This is the single most valuable thing I learned during development. The extractor's `operation`
field was **not reliable**. It labelled a 12% rent increase as `recurrence_start`, and a closed
prize claim as `recurrence_stop`.

Taken at face value, **50 of the 215 messages** — notes about bonuses, prizes, invoices, gig payouts
and utility bills — would have mapped to "income ended" and **zeroed the salary of roughly a fifth
of all users**.

**The fix** was to derive the fact type deterministically from the *combination* of fields that are
stable — `target`, `recurrence_effect`, and whether a number is present — rather than from the label
alone, in `ai/messages.py::derive_fact_type`. It enforces one hard property:

> A message that is not about an income source can never terminate income projection.

Suppression is also **scoped by income class**: a note about a gig payout suppresses only irregular
income, never payroll. And a stated salary *amount* takes precedence over a "stops" flag, because an
employer naming next month's figure is a change, not a termination.

After the fix, spurious `income_ended` fell from 50 to 14 genuine cases.

### Prompt injection

Message and image content is **untrusted data, never instruction**. Three layers:

1. Both system prompts state it explicitly and instruct the model to ignore embedded directives.
2. The schemas are **closed** — fixed enums, dates, numbers. There is no free-text action field, so
   there is no channel through which text could reach the engine as a command.
3. The engine consumes typed facts only and never executes text.

There is also a documented case of evidence that *looks* actionable and isn't: six users receive a
bank message about "a transfer between your two accounts". Five of them have no matching debit/credit
pair at all. The resolver invents nothing, and a test asserts no internal-transfer exclusion is ever
produced for those users.

---

## Event Resolution

### Two independent flags

Every canonical event carries `counts_for_cash` and `counts_for_recurrence` separately, because they
answer genuinely different questions:

- A **pending debit** moves future cash but proves nothing about a repeating pattern.
- A **settled historical rent payment** is already inside the opening balance so it must not be
  replayed, yet it is the primary evidence that rent recurs.

Both combinations occur in the data, and a test fails if either count reaches zero — otherwise the
split would be decorative.

### Lifecycle families (all present in the final code)

| Family | Resolution |
|---|---|
| authorization → settlement | settlement counts; cancelled authorization `superseded_by_settlement` |
| failed debit → retry | failure excluded; the scheduled retry is the real obligation |
| duplicate charge | linked same-amount pending debit excluded as `duplicate` |
| purchase → refund (settled) | both count, netting to zero; neither seeds recurrence |
| purchase → refund (pending) | debit counts; **pending credit never counted** |
| reimbursement | distinguished from refund by the parent's `work_expense` category |
| pending → settlement | reserved on `settlement_date` |
| investment valuation | `unrealized`/`non_cash` — never spendable |
| investment sale proceeds | real cash, but one-off, never recurring |
| scheduled obligation | dated future liability, counts |
| cancelled event | excluded |

Classification uses `event_type`, `status`, `direction` and link structure — **never description
text** — so renaming a description cannot change a financial answer.

### "Why not simply sum `financial_events.csv`?"

Because roughly 90 rows in that file are not independent economic events, and summing would
double-count them. A cancelled card authorization sits next to its settlement for the same amount —
sum both and you charge the user twice. A failed debit sits next to its scheduled retry. A duplicate
charge sits next to the original. A pending refund looks exactly like income that hasn't arrived. An
unrealized portfolio valuation looks like money. And separately, every settled row is *already*
inside `current_available_balance`, so summing history and adding it to the balance counts the
user's entire past twice.

---

## Starting Balance Assumption

`current_available_balance` is the **request-date snapshot**. Settled history is therefore never
replayed into the forecast; it is recurrence evidence. Only explicit future obligations — `pending`
and `scheduled` rows — move the projected balance.

The dataset confirms this partition exactly: all 25,148 settled rows fall strictly *before* their
user's request date, and all 71 pending and 70 scheduled rows fall strictly *after* it.
`resolve.assert_cash_partition` re-checks this at runtime rather than trusting it.

**What would go wrong otherwise:** subtracting settled history again would deduct months of rent,
groceries and salary that the balance already reflects. For a typical user that's tens of thousands
of currency units of phantom spending — every user would look near-insolvent and almost every answer
would collapse to `not_affordable`.

---

## Recurrence Inference

### Expenses group by `category`; income groups by class

This asymmetry is deliberate and was the single highest-impact discovery in development.

**Expenses:** grocery descriptions vary per purchase ("Bulk pantry shop", "Fresh food shop",
"Supermarket basket") but they're one habit. Category is the right key.

**Income:** the opposite problem in two directions.
- *Payroll* descriptions are meaningful — "Base salary" and "Monthly sales commission" are genuinely
  different streams. Grouping them by category merges a stable salary with irregular commissions and
  inflates projected income enormously. One user's projected monthly income nearly doubled.
- *Freelance and gig* labels vary per payment ("Freelance milestone payment", "Client retainer
  payment", "Website project payment") exactly like groceries. Grouping *those* by description hides
  a perfectly regular income behind a dozen one-observation series. **39 of 275 users** were left
  with zero projected income until this was fixed.

So income is grouped by *class*: continuing payroll keeps its description; irregular income collapses
into one stream.

### Income taxonomy

Classified by description tokens in `recurrence.classify_income`. An unrecognised description
degrades to `unknown` and is **never projected** — inventing income is the one error the spec forbids
outright.

| Class | Examples | Projected? |
|---|---|---|
| `continuing` | Payroll credit, Base salary, International employer payroll, Primary household salary | yes |
| `terminal` | **Final employer payroll**, Previous employer payroll, Payroll before leave | **no — ends all income projection** |
| `one_off` | bonus, commission, arrears, prize | no |
| `irregular` | platform payouts, freelance/contract/retainer payments | yes, as one stream, unless a message says it's unconfirmed |
| `unknown` | anything else | no |

`Base salary ≠ commission ≠ Final employer payroll ≠ freelance payment` — four different treatments,
all in the `salary` category in the raw CSV.

### Staleness

A stream whose last occurrence is more than `income_staleness_cycles × cadence` before the request
date is dropped. One user's second household income stops in January against a March request — 47
days on a 30-day cadence — and projecting it would invent income that has demonstrably stopped.

### Confirmed-salary anchoring

An explicitly scheduled `Next confirmed salary` row anchors the income projection at its stated date
and amount, and the projected series starts **one cadence later** so the explicit row is the first
occurrence. This does two things: it gives income to users whose only salary evidence is the
scheduled row, and it removes a double-count where the explicit event and a projected occurrence
would both land on the same day.

### Monthly vs sub-monthly

Cadence ≥ `monthly_threshold_days` (26) → **monthly**, projected on its calendar day-of-month via
`add_month`, which clamps 31 → 28/29/30 and does not drift. Below → **sub-monthly**, accrued as a
daily rate.

Verified empirically rather than assumed from category names — median coefficient of variation
across the sample set:

| Class | Median CV | Categories | Cadence |
|---|---:|---|---|
| A — exactly fixed | **0.000** | rent, housing, debt_repayment, education, insurance, gym, family_support, all subscriptions | monthly |
| B — moderate | 0.049–0.069 | healthcare, utilities, shopping, entertainment | monthly |
| C — variable | 0.156–0.159 | transport, dining, groceries | sub-monthly |

Eleven categories have CV of exactly zero, so the estimator choice cannot affect them at all. The
cadence split coincides with the variance split — which independently validates treating monthly
commitments as dated events and sub-monthly habits as a rate.

### Explicit obligations override inferred occurrences

If an explicit `pending`/`scheduled` debit lands within half a cadence of an inferred occurrence of
the *same category*, the inferred one is suppressed. The data shows why: one user has monthly
insurance of 2,510 on the 6th **and** a scheduled insurance payment of 1,830 on the 11th. Those are
one January obligation stated twice, not two payments. The explicit row wins because a confirmed
amount is better evidence than an average — the same precedence the specification gives a settled
event over an estimate. Scoped to discrete streams only, so accruing habits are untouched.

---

## Constants, Thresholds and Why They Exist

Classification key: **[SPEC]** specified by the challenge · **[DATA]** derived from observed data ·
**[SAFETY]** engineering safety choice · **[API]** model/provider requirement · **[EMPIRICAL]**
empirically selected but stated as a generalised rule.

| Constant | Value | Where | Why | Class | Alternative considered |
|---|---|---|---|---|---|
| `horizon_days` | **90** | `ForecastConfig` | The problem statement mandates a 90-day safety check | **[SPEC]** | 75 scored better on samples — **explicitly rejected** as unjustified |
| `monthly_threshold_days` | **26** | `ForecastConfig` | Separates monthly cadences (28–31 days, varying by calendar month) from sub-monthly (observed 5, 7, 10, 14, 21). Any threshold from 22 to 29 produces identical behaviour on this dataset; 26 sits inside that band with margin for February | **[DATA]** | see flashcard — honest note below |
| `monthly_anchor` | `day_of_month` | `ForecastConfig` | Rent and payroll recur on a calendar day. Fixed +30d drifts ~5 days per year | **[DATA]** | `fixed_interval` |
| `income_staleness_cycles` | **1.5** | `ForecastConfig` | A stream that has missed a full cycle *plus a half-cycle grace period* is no longer evidenced | **[EMPIRICAL]** | 1.0 too aggressive (normal timing jitter), 2.0 keeps demonstrably-ended income |
| `min_observations` | **3** | `ForecastConfig` | You need ≥2 gaps to take a median gap. Three occurrences is the minimum that establishes a pattern rather than a coincidence | **[SAFETY]** | 2 gives a single gap with no median |
| `amount_estimator` | `trailing_horizon` | `ForecastConfig` | Estimate the next 90 days from the last 90; the window adapts to cadence | **[EMPIRICAL]** | measured against 5 alternatives — table below |
| trailing window | `= horizon_days` (90) | `estimate_amount` | Symmetric with what's being forecast, rather than an arbitrary observation count | **[EMPIRICAL]** | `mean_3` scored identically but is a fixed count with weaker rationale |
| `submonthly_mode` | `accrual` | `ForecastConfig` | A weekly grocery habit is a *rate*, not a dated commitment | **[EMPIRICAL]** | `discrete` and `cycle_upfront` both tested and worse |
| `same_day_order` | `credits_first` | `ForecastConfig` | `minimum_balance_to_keep` is a balance the user *keeps* — an end-of-day position, not a bank posting-order artefact | **[DATA]** | `debits_first` creates an unsupported transient trough by assuming debit-before-credit ordering |
| `quantum` | **0.01** | `ForecastConfig` | Currency precision; money is not continuous | **[SAFETY]** | whole units (1) tested — worse everywhere |
| `cadence_rule` | `median_gap` | `ForecastConfig` | Median resists a single irregular gap | **[SAFETY]** | `mean_gap` |
| `earliest_horizon` | `fixed_from_asof` | `ForecastConfig` | Every observed earliest date lies inside 90 days of `request_date` | **[DATA]** | `sliding` |
| `MAX_CHANGES` | **3** | `spending.py` | "up to three" — direct from the spec | **[SPEC]** | none |
| partial-payment count | **2** | `plans.py`, `validate.py` | "exactly two payments" — direct from the spec | **[SPEC]** | none |
| installment schedule | must equal a supplied option | `eligibility.py`, `validate.py` | "Installment plans must exactly match a supplied payment option" | **[SPEC]** | none |
| `max_installment_months` | per user profile | `eligibility.py` | Compared against `number_of_payments`; blank ⇒ user considers no installments | **[SPEC]** | none |
| deadline rule | last payment ≤ `desired_completion_date` | `eligibility.py` | Ranking rule 1 | **[SPEC]** | none |
| six ranking rules | fixed order | `rank.py` | Verbatim from the spec | **[SPEC]** | LLM ranking — rejected |
| `budget_ceiling_usd` | **3.00** | `RunConfig` | ~4.20 was available; a ceiling below it preserves margin for retries and mistakes | **[SAFETY]** | 4.20 leaves no headroom |
| `temperature` | **0.0** | `ModelConfig` | Deterministic structured extraction | **[API]** | none defensible |
| `timeout_s` | **90** | `ModelConfig` | Vision calls on ~1 MB images took up to ~11 s; 90 s is generous without hanging a batch | **[SAFETY]** | — |
| `max_retries` | **2** | `ModelConfig` | Bounded corrective retries; observed retries across the final evidence: **1** | **[SAFETY]** | unbounded risks cost blow-up |
| `LOW_CONFIDENCE` (messages) | **0.5** | `ai/messages.py` | Inspection flag only — does **not** gate any financial decision | **[SAFETY]** | — |
| `LOW_CONFIDENCE` (images) | **0.6** | `ai/images.py` | Same; images held to a higher bar because one image can be a whole liability | **[SAFETY]** | — |
| suspect-semantics clamp | **0.4** | `ai/images.py` | If the model picks a figure that is *never* a transaction's economic value (gross pay, cash tendered, subtotal), confidence is clamped for inspection rather than the number being silently substituted | **[SAFETY]** | auto-correct — rejected as guessing |
| `max_tokens` | 500 msg / 900 img / 700 default | `ai/*.py` | Sized to observed outputs (~100 and ~194 tokens) with generous headroom | **[API]** | — |
| income supersession window | **45 days** | `recurrence.py` | A parallel income stream more than ~1.5 monthly cycles older than the latest employment row is treated as superseded | **[EMPIRICAL]** | — |
| default cadence fallback | **30 days** | `recurrence.py` | Used only when a stream has no computable gap | **[SAFETY]** | — |
| price table | GLM 0.60/2.20, Qwen 0.30/1.20 per Mtok | `config.py` | Published rates; **fallback only** — a provider-reported cost always wins | **[API]** | — |
| `FAR_FUTURE` | 9999-12-31 | `rank.py` | Sort sentinel for a plan with no payments | **[SAFETY]** | — |

### Honest note on `monthly_threshold_days = 26`

I want to be precise here because a judge may press on it. The observed cadences in this dataset are
5, 7, 10, 14, 21 (sub-monthly) and 30, 31 (monthly). **Any threshold from 22 to 29 produces
identical behaviour.** 26 was chosen to sit inside that band with room for February (28/29 days) on
one side and a 21-day cadence on the other. It is not derived from an optimisation, and I would not
claim it is. The data makes the choice indifferent within a wide range, and 26 is comfortably
interior to it.

### Single source of truth for cache versions

All five extraction versions live in **one place**, `code/bow/ai/schemas.py`:
`MESSAGE_PROMPT_VERSION`, `MESSAGE_SCHEMA_VERSION`, `IMAGE_PROMPT_VERSION`, `IMAGE_SCHEMA_VERSION`,
`EXTRACTOR_VERSION`. Every consumer — `ai/messages.py`, `ai/images.py`, `ai/facts.py`, `main.py`,
`evaluation/usage_summary.py`, `evaluation/ai_audit.py` — imports them from there.

An earlier design kept a second, coarser triple in `config.py`. That is what the per-extractor split
replaced, and the superseded constants were removed so the cache key has exactly one definition. If a
judge asks why versions aren't in the config object: because a config value is a *knob*, while these
are a *content identity* that must stay locked to the prompt and schema text they describe. Keeping
them in the same module as that text makes it impossible to bump one without seeing the other.

---

## Financial Mathematics

Let:

- `B₀` = `current_available_balance` at `request_date`
- `M` = `minimum_balance_to_keep`
- `C(t)` = cumulative projected net cash flow from `request_date` to `t` (income minus expenses)

Paying `x` today shifts the entire projected path down by `x`:

```
B(t) = B₀ − x + C(t)
```

Safety requires `B(t) ≥ M` for every `t` in the horizon. Rearranged:

```
x ≤ min over t of ( B₀ + C(t) ) − M
```

The right-hand side is the **trough** of the unmodified forecast minus the minimum. So:

```
amount_safe_to_pay = clamp( trough − M , 0 , requested_amount )
```

Equivalently `headroom − reserve`, where `headroom = B₀ − M` and `reserve = B₀ − trough`. Both forms
are computed and cross-checked with an assertion — if they ever disagreed, that would be a bug.

### `earliest_date_for_full_payment`

The first date `d` at which paying the **entire** requested amount keeps the rest of the horizon
safe. Paying at `d` lowers the path from `d` onward and leaves everything before it untouched, so the
test is a **suffix minimum** of the base forecast:

```
suffix_min(d) − requested_amount ≥ M
```

This makes the search exact and linear — no re-projection per candidate date.

One subtlety that mattered: on the payment day itself the day's **closing** balance is what's
available, because the dataset provides no intraday posting order and the payer is not bound to
settle before that day's credits; every later day uses the intraday low. Without that distinction, a
payment on payday is judged against an assumed pre-salary balance and every date comes out one day
late. Fixing it moved sample accuracy from 9/25 to 17/25 on this field.

### Why it's independent of preference

`earliest_date_for_full_payment` is a measure of **financial capacity**, not of the recommendation.
The spec is explicit, and a solved sample confirms it: a user who does not accept `full_payment` is
recommended installments, yet the field still reports `request_date` because the money was there.

---

## Payment Planning

| Method | Eligibility | Plan shape |
|---|---|---|
| `full_payment` | user accepts it; full amount safe on `request_date` | one payment on `request_date` |
| `partial_payment` | request allows it **and** user accepts it **and** `0 < safe < requested` **and** earliest exists **and** earliest ≤ deadline | exactly two payments |
| `installments` | user accepts; `number_of_payments ≤ max_installment_months`; last payment ≤ deadline; totals consistent | **exactly a supplied option** |
| `wait` | user accepts `full_payment`; a **no-spending-change** earliest date exists and is ≤ deadline | one payment on that date |
| `not_recommended` | always available fallback | `none` |

**Partial payment** specifically: payment 1 is `amount_safe_to_pay` on `request_date`; payment 2 is
`requested_amount − amount_safe_to_pay` on `earliest_date_for_full_payment`; they must sum exactly to
the requested amount; the second date must be on or before `desired_completion_date`.

**Installments** are never invented. The schedule is reconstructed from the supplied option's
`first_payment_date`, `number_of_payments` and `payment_frequency_days`, and the validator re-checks
it matches that option to the cent and the day. A striking dataset fact: **434 of 515** supplied
installment options finish *after* their request's deadline, so most offers are simply not usable —
which is why only 60 of 250 recommendations are installments despite every request having offers.

Every candidate is **re-forecast from scratch** with its own payments applied, so a plan is never
judged against a projection it doesn't actually cause.

**One efficiency note that is not an approximation:** spending-change variants are only generated
when no change-free candidate completes by the deadline. Ranking rule 1 (complete by deadline)
dominates rule 2 (no spending changes), so a change-free candidate that completes on time can never
be beaten by one needing changes. Skipping them there is provably equivalent and far cheaper.

---

## Flexible Spending Changes

Format: `stop:<event_id>` and `reduce_to:<event_id>:<new_amount>`, joined by `|`, or `none`.

Rules enforced in `spending.py` and re-checked in `validate.py`:

- Only **recurring expense streams**, never one-offs.
- The stream's `flexibility` must permit the action (`stoppable`/`reducible_or_stoppable` to stop;
  `reducible`/`reducible_or_stoppable` to reduce).
- The category must be in the user's own `expense_categories_user_is_willing_to_stop` /
  `..._to_reduce` list. Both conditions, not either.
- `reduce_to` uses the stream's `minimum_allowed_amount` — the only reduced value the dataset
  supplies.
- The cited `event_id` is the **latest settled occurrence** of that stream.
- At most **3** changes.
- Never stop and reduce the same stream — structurally impossible, since at most one change per
  stream is generated.

**Search:** exhaustive bounded enumeration of all 1–3 combinations over distinct streams
(`itertools.combinations`), deterministically ordered. Flexible streams per user number at most a
handful, so exhaustive search is cheap and *complete*.

**Why not let the LLM pick the expenses?** Three reasons. It would be non-deterministic. It would be
unverifiable — I'd have to validate its choice against the same rules anyway, so the model adds a
step without removing one. And the search space is small enough that exhaustive enumeration
guarantees I never miss a valid answer, which a heuristic cannot.

---

## Independent Validation

`validate.py` takes a candidate plus the evidence and **re-derives the forecast itself**. It shares
`eligibility.py` with the planner deliberately — the *rules* must not drift apart — but every safety
and arithmetic check is recomputed from source.

**Fourteen** numbered checks: `V1` amount bounds · `V2` minimum balance across the horizon · `V3` exact
payment sums · `V4` completion deadline · `V5` method acceptance · `V6` partial structure · `V7`
installment equality with the supplied option · `V8` stream flexibility · `V9` permitted category ·
`V10` ≤3 changes · `V11` no stop+reduce on one stream · `V12` `reduce_to` equals the allowed minimum
· `V13` chronological order and positive amounts · `V15` fallback shape.

**On the missing `V14`:** the codes run V1–V13 and V15 because V14 was dropped during development and
the rest were not renumbered. That is deliberate — a check code is an identifier that appears in
debug traces and test names, so renumbering would silently change what an existing trace means.
Stable codes are worth more than contiguous ones. If a judge asks, the honest answer is "V14 was
removed; I don't reuse or recycle codes."

### "If your planner already generates valid plans, why a second validator?"

Defence in depth. The planner's job is to be *clever* — enumerate options, search spending changes,
optimise. The validator's job is to be *suspicious*. A bug in clever code must not reach the output.

And it earned its place. During the final 250-request run the validator rejected **four rows** where
the planner had produced a `wait` recommendation with an **empty** `earliest_date_for_full_payment`.
That's a genuine contradiction: `affordable_later` means "the full amount is expected to become safe
later", which an empty earliest date denies. Cause: a `wait` candidate generated only under spending
changes, while the earliest field correctly reports the *no-changes* capacity the spec asks for.

**Generalised fix:** a `wait` candidate requires a no-spending-change earliest date to exist — you
cannot tell someone to wait without being able to name the date.

**The honest trade:** that fix *lowered* visible sample accuracy from 107 to 103 of 125 discrete
fields. I kept it. A spec-invalid row is worse than a lost sample point, and tuning it back to
recover points would have been exactly the overfitting I spent the whole project avoiding.

---

## Candidate Ranking

Applied literally in `rank.py::sort_key`, in this order:

1. Completes the full request by `desired_completion_date`
2. Requires no spending changes
3. Minimises total amount paid
4. Starts payment earlier
5. Uses fewer payments
6. Lowest `payment_option_id`

`not_recommended` always sorts last. Each rule has its own unit test.

**Rule 2 is a boolean, and the code implements it as one** — `1 if plan.changes else 0`. A plan
needing one change does not outrank a plan needing two; both simply "require changes", and rule 3
(lowest total paid) separates them. A dedicated test pins exactly that, so the distinction cannot
silently regress into a change *count*.

**Why not an LLM?** The order is *given* by the specification. There is nothing to infer. A model
could only reproduce it less reliably and non-deterministically.

**A worked example from the solved samples:** one request had both an eligible 2-payment installment
option (total 41,246.40, including a financing fee) and a valid partial-payment plan (total 39,660).
Both complete by the deadline and need no changes, so rules 1 and 2 tie; rule 3 — minimise total paid
— selects partial payment. The ground truth agrees. That's the ranking doing real work, not
decoration.

---

## How the 25 Solved Samples Were Used

They were a **regression suite**, not training data.

- The same `bow.predict` path processes them and the 250. There is no sample-specific branch
  anywhere.
- On a failure I traced *backwards* — output → ranker → validator → candidates → safe amount →
  forecast → streams → canonical events → AI facts → raw evidence — and identified the **first**
  point of divergence before changing anything.
- Only generalised fixes were applied, each stated as a rule that applies to unseen users, each with
  a regression test.
- **All 25 were re-run after every meaningful change**, tracking newly fixed, still failing, and new
  regressions.

### Anti-overfitting safeguards

- No request/user/event/image/message ID appears anywhere in `code/bow/` — verified by grep.
- `bow.predict` has **zero** references to expected outputs; the oracle predictor used to self-test
  the harness lives in `evaluation/` and is unreachable from production.
- A **75-day horizon** scored measurably better (MAE 4.29% vs 6.53% at the time) and was
  **explicitly rejected** — the spec says 90 and there is no financial argument for 75.
- Whole-unit rounding, `cycle_upfront` accrual and `mean_3` were each tested and rejected when the
  gain was isolated or the justification weak.
- When one fix improved two samples with zero regressions but only via a fixed observation count, I
  preferred the time-window form (`trailing_horizon`) because it has a rationale independent of the
  samples.

---

## Major Bugs and Discoveries During Development

### 1. Income taxonomy mattered far more than estimator choice
**Symptom:** one user's predicted reserve was 4,810 against an implied 32,638; another 22,791 against
512,055. **Root cause:** every `salary`-category credit was projected forward, including a *Final
employer payroll* (employment ended), gig payouts, and commissions. **Fix:** classify income by
description into continuing / terminal / one-off / irregular / unknown; only continuing and
(conditionally) irregular project. **Generalises** because it keys on a stated property of the
income, not on who the user is.

### 2. Income grouped by description hid regular freelance income
**Symptom:** 39 of 275 users had *zero* projected income. **Root cause:** freelance labels vary per
payment, so each became a one-observation series below `min_observations`. **Fix:** collapse
irregular income into one stream by class. **Generalises** — it's the same reasoning that groups
groceries by category.

### 3. Same-day ordering created an unsupported transient trough
**Symptom:** one sample's safe amount was 15,232 against an expected 28,820. **Root cause:** the
floor was an assumed *intraday* low on payday — a 12,650 family-support debit ordered before a
131,000 salary on the same date, an ordering the dataset does not supply. **Fix:** evaluate the
minimum-balance rule on the end-of-day position. **Generalises** — `minimum_balance_to_keep` reads as
a balance the user keeps, and end-of-day semantics assume no posting order at all. That one change
took the sample from 34.3% error to 0.1%.

### 4. Earliest date was systematically one day late
**Symptom:** every earliest date landed one day after the expected one. **Root cause:** payment
feasibility was tested against an assumed pre-credit intraday low on the payment day itself. **Fix:**
on the payment day use the closing balance, because the dataset provides no intraday posting order;
using a pre-credit intraday low introduced an unsupported one-day delay. 9/25 → 17/25.

### 5. Raw GLM operation labels were unsafe
**Symptom:** 50 messages about bonuses, prizes and utility bills would have terminated salary.
**Root cause:** trusting the model's `operation` label. **Fix:** derive the fact type from
`target` + `recurrence_effect` + presence of a number, with the invariant that a non-income message
can never stop income. **Generalises** — it relies on the stable fields, not the noisy one.

### 6. A future-dated salary change silently deleted a month of income
**Symptom:** two samples collapsed to zero safe amount after message facts were added. **Root cause:**
a message saying "salary is X from 15 August" set the stream's anchor to 15 August, and occurrences
start one cadence *after* the anchor — so August's payment vanished. **Fix:** `anchor_for_first`
places the anchor one cadence *before* the stated first payment.

### 7. Synthetic income streams had no `income_class`
**Symptom:** a suppression directive silently failed to match. **Root cause:** streams created from a
confirmed-salary row or a message were built without the class field that suppression keys on.
**Fix:** set it at construction. Caught by a test, not by inspection.

### 8. Foreign-currency salary was never converted
**Symptom:** one IDR user's reserve was wrong by 61.8 million. **Root cause:** a USD payroll on an IDR
profile was summed unconverted. **Fix:** convert on the settlement date with full provenance.

### 9. Image amounts must be semantic, not naive totals
**Symptom:** a payslip's largest number is 4,780,800 but the money received is 4,365,000. **Fix:**
ask the VLM for the economically relevant figure *and* a label for which figure it chose, plus the
rejected candidates. Seven of sixteen images contain a trap of this kind.

### 10. `wait` with no earliest date (found in the final run)
Covered above under Independent Validation. Found by the validator on the real 250, not on samples.

### 11. A prompt fix that over-generalised and had to be rolled back
**Symptom:** after a prompt change, discrete accuracy fell from 107 to 96 of 125. **Root cause:** an
instruction to mark pending payouts as "stops" caused the model to stamp *stops* on messages that
**confirmed** a salary — `income_ended` jumped from 9 to 72. **Fix:** diagnosed rather than patched
around; replaced with a narrower instruction that restored and then improved the baseline. Worth
mentioning because it shows the regression suite catching a self-inflicted wound.

---

## Alternatives Considered

| Alternative | Why considered | Why rejected |
|---|---|---|
| Per-request LLM reasoning | Simplest to build; "obviously agentic" | Non-deterministic, ~250× the cost, unverifiable arithmetic, no way to guarantee an installment schedule matches an offer |
| Multi-agent swarm | Fashionable; parallel specialists | Compounding interpretation errors, non-determinism, context growth, exact schedules can't be negotiated |
| One giant persistent context | Fewer moving parts | 25,342 events; lost-in-the-middle on exactly the one-line facts that decide the answer |
| Dedicated HF inference endpoint | Predictable latency | Bills for provisioned time; 231 calls total makes it absurd. Routed inference is pay-per-token |
| LLM ranking of plans | Could "weigh tradeoffs" | The six rules are *given* by the spec — nothing to infer, everything to lose |
| LLM-inferred recurrence | Could spot patterns | Cadence and level are statistics; a model would be slower, costlier and unauditable |
| `last` value estimator | Tracks the latest level | MAE 4.03% vs 2.90% for `mean_all` — high variance on noisy weekly streams |
| `mean_3` | Best discrete score (107/125) | Tied with `trailing_horizon` on discrete but worse MAE (3.29% vs 3.15%), and "last 3 observations" has weaker justification than a time window |
| `mean_6` / `median_6` | Recency with more stability | 105/125 discrete; neither dominated |
| `mean_all` | Lowest MAE (2.90%) | 103/125 discrete — lost on the prioritised fields |
| Whole-unit rounding (`quantum=1`) | The whole-number reserve signature suggested generator rounding | **Worse in every configuration** (discrete 103→96). Cleanly disproved the hypothesis |
| Discrete sub-monthly events | Matches how spending actually happens | MAE 3.37% vs 2.90%, no discrete gain |
| `cycle_upfront` accrual | Explained one sample's residual *exactly* (+56.11 vs −56.13) | Regressed the aggregate (MAE 4.26%, discrete 101). Textbook isolated fit — rejected |
| 75-day horizon | MAE 4.29% vs 6.53% at the time | **No financial justification**; spec says 90. Rejected on principle |
| Income grouped by category | Simpler | Merges base salary with commissions; inflates income enormously |

---

## The `trailing_horizon` Tradeoff

A judge may well notice that I chose an estimator with a *worse* mean error. Here is the measured
comparison:

| Estimator | MAE | median | <1/5/10% | discrete fields |
|---|---:|---:|---|---:|
| `mean_all` | **2.90%** | 1.88% | 10/21/24 | 103/125 |
| **`trailing_horizon`** | 3.15% | 1.88% | 9/21/23 | **107/125** |
| `mean_3` | 3.29% | 1.95% | 9/20/22 | 107/125 |
| `mean_6` | 3.17% | 1.73% | 10/20/22 | 105/125 |
| `last` | 4.03% | 1.97% | 8/20/22 | 103/125 |

**The trade:** +4 discrete fields for +0.25 percentage points of mean amount error.

**Why that's the right call:** five of the structured output fields are categorical or exact-style —
`affordability_status`, `recommended_payment_method`, `payment_plan`,
`earliest_date_for_full_payment`, `spending_changes_needed` — alongside the numeric
`amount_safe_to_pay`, and `decision_explanation` is additionally scored for usefulness and
consistency. An exact-style field is right or wrong; a numeric one degrades gracefully. Trading a
fraction of a percent of numeric error for four structured-field wins is favourable under that shape
of scoring.

**Why `trailing_horizon` over `mean_3`,** which scored the same on discrete: `mean_3` is a fixed
observation count with no rationale beyond "it worked". `trailing_horizon` says *estimate the next 90
days from the last 90 days* — symmetric with what's being forecast, and the window adapts to cadence
automatically (~3 points for a monthly bill, ~13 for a weekly habit). It also had the better MAE of
the two.

**Stability check:** across the 25 samples `trailing_horizon` was better on 2 and **worse on 0**,
with the improvement monotone across `mean_all` → `mean_6` → `mean_3`, which suggests recency
genuinely matters rather than noise.

**What I do not claim:** that it is mathematically exact or that it reconstructs the generator.

---

## Agent Mode (added after the submission)

Judge feedback said the model was "boxed into extraction" while the rest was author-written
routing. `code/bow/agent.py` answers that with a **bounded tool loop**: GLM-5.3 chooses which tool
to call next, and the deterministic engine executes it.

| Tool | Effect |
|---|---|
| `get_request` | request + profile |
| `list_evidence` | message/image ids with extracted fact type — never raw text |
| `read_message` / `read_image` | read one fact **and add it to the forecast's evidence** |
| `run_forecast` | 90-day forecast on the evidence read so far |
| `list_plans` | every candidate that passed the independent validator |
| `finish` | recommend one listed plan |

**Discipline:** max 10 model calls per request; bad tool calls return a recoverable error;
budget checked before every call; any failure (step cap, provider error, budget, unknown plan id)
falls back to the pipeline's answer with the reason traced; every step cached. Evidence selection
is real — unread evidence is excluded — so skipping evidence visibly costs accuracy.
Run it: `python code/main.py --agent --samples`.

**Measured, not assumed** (`code/evaluation/compare.py` → `comparison.md`, 25 samples, same scorer):

| | no AI | pipeline | agent (guided prompt) | agent (brief prompt) |
|---|---:|---:|---:|---:|
| 5 structured fields | 100/125 | 103/125 | 103/125 | 103/125 |
| amount mean error | 8.14% | 3.15% | 3.15% | 3.15% |
| same plan as rule-based ranker | — | — | 25/25 | 25/25 |
| tool errors / fallbacks / step-cap hits | — | — | 0 / 0 / 0 | 0 / 0 / 0 |
| mean tool calls per request | — | — | 5.5 | 5.8 |

- The no-AI baseline shows the evidence extraction earns its place: mean amount error 8.14% → 3.15%.
- The agent reads **13/13** relevant evidence items and now matches the pipeline exactly.
- **The first run did not.** It scored 97/125 with 19 recovered tool errors. Tracing the misses
  found two things. First, **a real engine bug**: `completes_by_deadline` required
  `total == requested`, so every installment plan with a financing fee looked like it never
  completes. The agent applied rule 1 literally and refused correct plans; the pipeline had been
  right only by accident. Fixed to `>=`, with a regression test; 0 of 250 output rows changed.
  Second, **an ambiguous tool argument**: most errors were a tool name placed in the `id` field.
  Renamed to `evidence_id`. The re-run went to 103/125 with zero errors.
- The prompt ablation (guided vs brief) now ties; the guidance mainly buys fewer calls (5.5 vs 5.8).
- Context stays small by design: about 830 input tokens per call and about 5,000 per request,
  a fresh context per request, data kept in Python, and tool results summarised and capped.
- Cost: about USD 0.085 per 25 requests per prompt variant (guided 0.0848, brief 0.0817), roughly
  USD 0.0034 per request; cached re-runs cost nothing.

**Interview line:** the agent now matches the pipeline on the same data, with every step bounded,
traced and cached — and on the way there, by reading my engine's output literally, it found a bug
that 25-sample accuracy had hidden. The pipeline stays the default because it is equal in accuracy
and needs no model calls at prediction time; the agent is the mode that earns the "agent" label.

---

## Known Limitation

**Sub-monthly variable spending estimation — groceries, transport and dining — is the one component
I could not fully resolve.**

Final frozen sample behaviour:

- `amount_safe_to_pay` exact on **3 of 25**
- **21 of 25** within 5% of the requested amount, **23 of 25** within 10%
- median error **1.88%**, mean **3.15%**, max **13.44%**

I decomposed every residual by contributing stream. Fixed streams (rent, debt, insurance,
subscriptions) reconcile to the cent — eleven categories have a coefficient of variation of exactly
zero. The entire residual sits in the three variable categories, and crucially **the residuals have
both signs**: on some requests the accrual over-charges, on others it under-charges. That rules out
a single scaling correction.

I tested six estimator families, three sub-monthly projection modes, and whole-unit rounding at
three granularities. One hypothesis (`cycle_upfront`) explained one sample's residual almost exactly
— and regressed the aggregate, so I rejected it.

A further observation I could not exploit: **16 of 21** non-capped ground-truth implied reserves are
exact whole numbers, while every candidate component carries cents. That suggests the reference
implementation rounds somewhere I couldn't locate — I disproved per-occurrence, per-stream and
per-monthly-total rounding, leaving only final-reserve rounding or a generator that chose the answer
first. Neither is recoverable from the inputs.

**I stopped rather than force it.** Any further movement would have been fitting 25 labels, which
would likely *hurt* on the hidden 250.

---

## Final Evaluation Run

| Metric | Value |
|---|---|
| Requests processed | **250 / 250** |
| Execution time | ~2.3 seconds |
| Model calls during prediction | **0** |
| Independent validation | **250 / 250 pass** |
| Reproducibility | byte-identical across reruns |
| `output.csv` SHA256 | `7ec8757424b04be82e2b4f7b5c0c8b2c322579cfec0d9731e231125eb7e94f2a` |

**Status distribution:** `affordable_with_plan` 74 · `not_affordable` 65 · `affordable_now` 62 ·
`affordable_later` 49

**Method distribution:** `full_payment` 66 · `not_recommended` 65 · `installments` 60 · `wait` 49 ·
`partial_payment` 10

**Plans requiring spending changes:** 24

`amount_safe_to_pay`: 15 zero · 81 capped at the requested amount · 154 strictly between.
`earliest_date_for_full_payment`: 81 = request_date · 105 later · 64 empty.

All nine request types and all five currencies span all four statuses — no degenerate collapse.

**I make no accuracy claim on the 250.** The ground truth is hidden, so `evaluation/main.py` reports
validity, safety and completeness rather than inventing a percentage. That distinction is
deliberate.

---

## AI Usage and Cost

| Model | Role | Calls | Retries | Input | Output | Cost |
|---|---|---:|---:|---:|---:|---:|
| `zai-org/GLM-5.3` | message evidence extraction | 215 | 1 | 143,369 | 48,013 | ~$0.1916 |
| `Qwen/Qwen3-VL-235B-A22B-Instruct` | image evidence extraction | 16 | 0 | 25,167 | 3,106 | ~$0.0113 |
| **Combined** | | **231** | **1** | **168,536** | **51,119** | **~$0.2029** |

- **219,655 total tokens**
- **~878.6 tokens amortised per evaluation request**
- **~USD 0.000812 AI evidence cost per request**

Three categories, kept distinct:

1. **Final evidence generation** — 231 calls, ~$0.2029. This is the usage behind `output.csv`.
2. **Deterministic prediction execution** — **zero** model calls, ~2.3 s for all 250.
3. **Cache reuse** — all 231 facts served from disk; the usage ledger is byte-identical before and
   after a prediction run.

Separately, **cumulative development spend across the whole project was ~USD 0.676**, which includes
superseded prompt versions and probes. I report that separately and never substitute it for the
final-evidence figure.

---

## Caching Strategy

Cache key:

```
sha256( model ‖ prompt_version ‖ schema_version ‖ extractor_version ‖ content_hash )
```

`content_hash` covers the message text, source type and timestamp — or the image file bytes. The
cache stores the **structured fact**, not model prose, so a cached run is reviewable and
byte-reproducible.

**Versions are per-extractor**, which is the point: message and image versions are independent
constants. When I tightened the message schema, **all 215 message facts re-extracted and all 16
image reads stayed cached** — exactly the selective invalidation the design promises, and it saved
re-paying for vision calls three separate times.

Frozen production versions: message prompt **5**, message schema **2**, image prompt **2**, image
schema **1**, extractor **1**. The shipped cache holds exactly 215 + 16 entries at those versions;
645 superseded development entries were pruned, and the `output.csv` SHA256 was unchanged afterwards
— which proves the prune kept precisely the right files.

**Benefits:** cost (one extraction for the whole dataset), determinism (identical reruns), speed
(the 166-test suite runs in seconds), and reduced API-failure exposure (prediction cannot fail
because a provider is down).

---

## Security and Robustness

- **Credentials**: `HF_TOKEN` read from the environment only, via a six-line stdlib `.env` loader.
  Never in source, never in the cache, never in the transcript. Verified by grep across the whole
  submission candidate.
- **Prompt injection**: untrusted-data framing in both system prompts, closed-enum schemas with no
  free-text action field, and an engine that consumes typed facts and never executes text.
- **Fail-closed on malformed AI output**: bounded corrective retry, then the fact is marked unusable
  and the pipeline proceeds on CSV evidence rather than guessing.
- **A blank amount is never zero** — it stays `unresolved_image` and is excluded from cash with a
  reason, not assumed away.
- **No invented income**: an unrecognised income description is never projected; an unquantified
  obligation ("a new recurring childcare payment begins") is recorded and **never priced**.
- **A missing exchange rate never becomes 1.0** — it raises `MissingRateError`.
- **Decimal everywhere**; `Money` raises `TypeError` on a float.
- **Invalid plans are rejected**, and the run fails loudly rather than writing a bad row.
- **Provider resilience**: a four-variant payload ladder handles providers that disagree about
  `strict` and thinking mode, remembering the first that works.

---

## Why Decimal Instead of Float

Binary floating point can't represent most decimal fractions exactly, so `0.1 + 0.2 != 0.3`. In this
problem that's not academic:

- An installment plan must sum to a supplied option's `total_payable_amount` **exactly**. A float
  drift of 1e-15 turns a valid plan into a rejected one.
- A partial payment's two legs must sum exactly to the requested amount.
- `amount_safe_to_pay` is serialised into a CSV that will be compared to a reference value.
- The dataset spans EUR and USD amounts with cents *and* IDR amounts in the tens of millions.
  Float's 53-bit mantissa handles either, but comparisons across that range are where subtle
  equality failures appear.

So `Money` is `Decimal`-backed and **refuses to be constructed from a float**, which turns a whole
class of silent corruption into an immediate error at the boundary.

---

## Evaluation Strategy

The two sets need genuinely different treatment, and conflating them would be dishonest.

**25 solved samples — ground truth exists.** `evaluation/main.py` reports real per-field accuracy
plus absolute, relative and over-requested error for `amount_safe_to_pay`.

**250 evaluation requests — ground truth is hidden.** Reporting an "accuracy" here would be
fabrication. Instead it verifies:

- exactly 250 predictions, 250 unique IDs, every `requests.csv` ID exactly once, no extras, no
  sample leakage
- exact column set and order; no NaN, Infinity, `None`, or scientific notation; dates `YYYY-MM-DD`
- amount bounds, enum validity, status/method consistency
- payment-plan arithmetic and chronology; deadline compliance
- partial-payment exact structure; installment exact option match
- spending-change legality
- an **independent 90-day safety replay** for every recommended plan
- deterministic reproducibility

It exits non-zero on any hard failure. On the final run it exits **0**.

---

# Likely AI Judge Questions and Suggested Answers

### 1. Explain your architecture.
Structured CSVs plus two AI extraction layers feed a deterministic engine. GLM-5.3 turns each message
into a typed fact; Qwen3-VL turns each image into a typed amount with a semantic label. Those facts
join the ledger, which I resolve into canonical economic events with two independent flags — does it
move cash, does it seed recurrence. From there it's recurrence inference, FX normalisation, a 90-day
forecast, the safe amount and earliest safe date, candidate plan generation, an independent
validator, and a deterministic ranker. One orchestrator, no agent swarm, no chat history.

### 2. Why didn't you use multiple agents?
Because financial arithmetic shouldn't be negotiated. If several agents each pass judgements along,
errors compound and nothing is checkable. It would also be non-deterministic, and my output has to
reproduce to the same SHA256. The problem needs exact schedules that match supplied offers to the
cent — that's specification compliance, not a judgement call. I used one orchestrator with small
isolated extraction calls and deterministic specialists.

### 3. Where exactly is AI used?
Two places only: reading 215 messages and reading 16 images. Both are one call per item, schema-
constrained, cached. Neither model ever sees a balance, computes affordability, picks a method, or
writes an output field.

### 4. Why GLM-5.3?
The messages are half English, half Indonesian, and the distinctions are subtle — "your bonus is
still under review" versus "your salary for the next payroll is 1,452". That needs strong
multilingual instruction-following and reliable structured output. At 215 calls the cost was $0.19,
so I prioritised capability over size. I should be straight that I didn't benchmark smaller models —
the reasoning was that a misread message costs a wrong recommendation and a smaller model saves
cents, so the asymmetry made the choice for me.

### 5. Why Qwen3-VL-235B?
Because it's document comprehension, not OCR. A payslip shows three plausible numbers and only one
is the money received. The set also includes a handwritten pharmacy bill and a low-contrast thermal
receipt. I chose a large VLM for the headroom on those, and at 16 images it cost 1.1 cents — I didn't
test a smaller one, but all 16 extractions matched an independent manual read, so it cleared the bar
I actually needed.

### 6. Why two models instead of one?
Different tasks. Running 215 text-only messages through a vision model costs more for no benefit,
and a text model can't read images at all. Separating them also means a change to one prompt
invalidates only that cache.

### 7. Why not let the LLM make the final decision?
Because the final decision is arithmetic plus a fixed rulebook. The ranking order is given by the
spec — there's nothing to infer. And I need the same answer every run. A model would be slower,
costlier, unverifiable, and non-deterministic, with no upside.

### 8. How do you prevent hallucinated financial values?
Several ways. The schemas are closed enums, so there's no free-text channel. A blank amount is never
zero — it stays unresolved and is excluded with a reason. An unrecognised income description is never
projected. An unquantified obligation is recorded and never priced. And I derive the fact type from
the model's stable fields rather than its label, because I measured that the label was wrong often
enough to matter.

### 9. How do you handle prompt injection?
Three layers. Both system prompts state the content is untrusted data and that embedded instructions
must be ignored. The schemas are closed — there's no field through which text could become a command.
And the engine only ever consumes typed facts; it never executes text. There's also a real case in
the data of a message that looks actionable but isn't, and the resolver invents nothing for it.

### 10. Why use Decimal?
Because installment totals and partial-payment legs must sum exactly. `0.1 + 0.2` isn't `0.3` in
binary float, and a 1e-15 drift turns a valid plan into a rejected one. My `Money` type is
Decimal-backed and raises on a float, so the error surfaces at the boundary instead of corrupting a
comparison later.

### 11. Why is the forecast 90 days?
Because the problem statement mandates a 90-day safety check. It's the one constant I didn't choose.
Worth noting: a 75-day horizon actually scored better on my samples, and I rejected it precisely
because there's no justification for it beyond fitting.

### 12. Why is `monthly_threshold_days` 26?
It separates monthly cadences from sub-monthly ones. Observed cadences are 5, 7, 10, 14 and 21 days
on one side and 30–31 on the other. Any threshold from 22 to 29 behaves identically on this data; 26
sits inside that band with margin for February at 28 days. I'd be honest that it isn't optimised —
the data makes the choice indifferent across a wide range.

### 13. Why is staleness 1.5 cycles?
It means a stream that has missed a full cycle plus a half-cycle of grace is no longer evidenced.
One user's second household income stops in January against a March request — 47 days on a 30-day
cadence — and projecting it would invent income that demonstrably stopped. 1.0 would be too
aggressive given normal payment-timing jitter; 2.0 keeps income that has clearly ended.

### 14. Why require 3 observations?
You need at least two gaps to take a median gap, which needs three occurrences. Below that you're
extrapolating from a coincidence rather than detecting a pattern — and the spec says to detect
recurrence only when history supports it.

### 15. Why `trailing_horizon`?
It estimates the next 90 days from the last 90 days — symmetric with what's being forecast — and the
window adapts to cadence automatically. At the estimator-calibration stage it improved the structured
fields from 103 to 107 of 125 with zero regressions. The later `wait`-invariant fix — a correctness
fix, not a tuning one — brought the final frozen score back to 103 of 125, so both numbers are true
of different points in the timeline. `mean_3` scored the same at calibration but is an arbitrary
observation count; `trailing_horizon` had the better MAE and a rationale that doesn't reference the
samples.

### 16. Why continuous accrual for sub-monthly expenses?
Because a weekly grocery habit is a rate, not a dated commitment. Charging whole cycles distorts the
window boundaries — it over-reserves on some horizons and under-reserves on others depending on where
the cycle falls. Accrual gave the strongest aggregate performance of the three projection modes I
tested and avoided those boundary distortions: MAE 2.90% against 3.37% for discrete events and 4.26%
for cycle-upfront. I wouldn't claim it's unbiased — residuals remain in both directions — only that
it was the best of what I measured.

### 17. Why `credits_first`?
Because `minimum_balance_to_keep` reads as a balance the user wants to *keep* — an end-of-day
position. The dataset gives settlement dates but no intraday posting order, so ordering debits first
asserts a sequence the data doesn't support: on one sample it put the floor at a dip where a 12,650
debit was assumed to land before a 131,000 salary on the same day. End-of-day semantics avoid
inventing that ordering, and they matched the solved evidence substantially better — that sample went
from 34% error to 0.1%.

### 18. Why don't historical settled events affect the opening balance again?
Because `current_available_balance` already is the request-date snapshot — the history is baked in.
Subtracting it again would deduct months of rent, groceries and salary the user has already paid,
making everyone look insolvent. The data confirms the split: every settled row precedes its request
date, every pending and scheduled row follows it, and I assert that at runtime.

### 19. How do you handle refunds?
By status. A settled refund linked to its purchase means both legs count and net to zero
historically, and neither seeds recurrence. A *pending* refund is never counted — the specification
is explicit that pending credits don't count until they settle, and there's a case in the data where
a user's refund is initiated but not received.

### 20. How do you handle duplicate card charges?
A pending debit linked to a settled debit of the same amount is a duplicate, and it's excluded. It's
the sharpest trap in the dataset, because structurally it looks exactly like a chargeable pending
liability settling inside the forecast window — only the link to a same-amount original reveals it. I
have a test asserting exactly that.

### 21. How do you handle a failed debit plus retry?
The failed attempt never left the account, so it's excluded. The linked scheduled retry is the real
future obligation and counts on its settlement date. Exactly one economic debit, and a test asserts
that count.

### 22. How do you treat pending credits?
Never counted until settled — that's a direct spec rule and it covers refunds, bonuses, commissions,
prize proceeds and gig payouts. There's a case where a pending refund settles inside the forecast
window and it's still excluded.

### 23. How do you treat investments?
Three different ways. The contribution is a settled cash debit, already in the opening balance. The
unrealized valuation is `non_cash` and never becomes spendable money — that's the trap. Realized sale
proceeds are real cash but one-off, so they never seed a recurring series.

### 24. How do images affect transactions?
Sixteen ledger events have a blank `amount` and are linked to an image. Four of those are pending or
scheduled — future liabilities whose size only the image knows, including a 100,000 rent arrears and
a 79,679.26 grocery invoice. The other twelve are settled history, where the value matters mainly so
it doesn't pollute a recurring series — one is a 41,272 "grocery" in a series whose true level is
about 8,700.

### 25. Why not just use OCR?
OCR reads every number on the page. It won't tell you that on a payslip the money received is Net Pay
rather than Total Earnings, or that on a part-paid rent receipt it's Balance Due rather than the
invoice total. Choosing *which* number matters is the whole task. I also ask the model for the
rejected candidates so the choice is auditable rather than trusted.

### 26. How are messages integrated?
Each becomes one typed fact. Event-level facts confirm, cancel or delay a specific event;
stream-level facts change a salary amount or date, end income, or adjust a recurring expense. The
engine applies them to inferred streams before forecasting. Facts without a quantified amount are
recorded and never priced.

### 27. How do you distinguish payroll from bonus or freelance income?
By description class, not by category — they're all `salary` in the raw CSV. Continuing payroll
projects forward. Terminal rows like *Final employer payroll* stop projection entirely. Bonuses,
commissions, arrears and prizes never project. Freelance and gig income projects as a single stream
unless a message says the payout is unconfirmed. That taxonomy was the highest-impact fix in the
project.

### 28. How do you detect stale salary?
If a stream's last occurrence is more than 1.5 cadences before the request date, it's dropped. It
catches income that has quietly stopped without any message saying so.

### 29. How is `earliest_date_for_full_payment` calculated?
Paying the full amount on a date `d` lowers the projected path from `d` onward and leaves everything
before it untouched, so the test is a suffix minimum of the base forecast: the lowest balance from
`d` to the horizon, minus the requested amount, must stay above the minimum. That makes the search
exact and linear rather than re-projecting for every candidate date.

### 30. How is it different from the recommendation?
It's a capacity measure, not a recommendation. A user who won't consider full payment still gets an
earliest date — one solved sample recommends installments while the field reports `request_date`,
because the money was there. The spec also defines it *without* optional spending changes, which is
what caused the one invariant violation I found in the final run.

### 31. How do partial payments work?
Exactly two payments: the safe amount today, the remainder on the earliest full-payment date. They
must sum exactly to the requested amount, and the second date must be on or before the deadline. It's
only offered when the request allows partial payment, the user accepts it, and the safe amount is
strictly between zero and the requested amount.

### 32. Why can't you invent installment schedules?
Because the spec says an installment plan must exactly match a supplied option. I reconstruct the
schedule from the option's first payment date, count and frequency, and the validator re-checks it
matches to the cent and the day. Interestingly, 434 of 515 supplied options finish after their
deadline, so most offers simply aren't usable.

### 33. How are spending changes selected?
Exhaustive bounded enumeration. I take every stream that is both flexible in the ledger and in a
category the user said they'd change, then enumerate all combinations of up to three on distinct
streams, re-forecasting each. `reduce_to` always uses the stream's stated minimum. Flexible streams
per user number at most a handful, so exhaustive search is cheap and complete.

### 34. Why a maximum of three?
That's the specification — "up to three". Not my choice.

### 35. Why a validator separate from the planner?
Defence in depth. The planner's job is to be clever; the validator's job is to be suspicious. It
re-derives the forecast rather than trusting the planner. And it earned it — on the final 250 run it
caught four rows recommending `wait` with no earliest date, which is self-contradictory. I'd rather
find that than ship it.

### 36. How is ranking implemented?
The six rules from the spec as a sort key, in order: completes by deadline, requires no spending
changes, lowest total paid, starts earliest, fewer payments, lowest option id. Each rule has its own
test. Rule 2 is a boolean, not a count — a one-change plan doesn't outrank a two-change plan; they
tie there and rule 3 separates them. There's a test pinning that specifically, because implementing
it as a count is the easy mistake and it would silently override rule 3.
There's a real case where partial payment beats an eligible installment option purely on rule 3,
because the installment carries a financing fee.

### 37. How did you use the 25 samples without overfitting?
As a regression suite, never as training data. Same production path, no sample-specific branch. On a
failure I traced backwards to the first divergence and only applied fixes I could state as a general
rule with a test. All 25 re-ran after every change. And I explicitly rejected a 75-day horizon that
scored better, because "it fits" isn't a reason.

### 38. What was your biggest bug?
Income taxonomy. I was projecting every salary-category credit forward, including a *Final employer
payroll* — employment that had ended — plus gig payouts and commissions. One user's reserve was off
by an order of magnitude. The related discovery was that grouping income by description hid regular
freelance income behind a dozen one-observation series, leaving 39 of 275 users with zero projected
income.

### 39. What limitation remains?
Estimating variable sub-monthly spending — groceries, transport, dining. Exact amount matches are 3
of 25, though 21 of 25 are within 5% and the median error is 1.88%. The residuals have both signs, so
no single correction fixes them, and I stopped rather than fit 25 labels.

### 40. What would you improve with more time?
A learned or probabilistic model for variable essential spending, with uncertainty intervals rather
than point estimates, so the safe amount could be a conservative quantile. And more labelled examples
for genuine cross-validation — 25 is too few to fit anything without risking exactly the overfitting
I was avoiding.

### 41. How much did AI cost?
About 20 cents. 231 calls — 215 messages and 16 images — 219,655 tokens, roughly $0.2029. That's
about 0.08 cents per request amortised. Cumulative development spend including superseded prompt
versions was about 68 cents, against roughly $4.20 available.

### 42. How do caches work?
The key is a hash of model, prompt version, schema version, extractor version and a content hash of
the source. I cache the structured fact, not model prose, so a cached run is reviewable. Versions are
per-extractor, which paid off — when I tightened the message schema, all 215 messages re-extracted
and all 16 image reads stayed cached.

### 43. How does the solution scale?
Prediction is linear in requests and makes no model calls — 250 in 2.3 seconds. Extraction is linear
in messages and images, not in requests, so doubling the request count with the same evidence adds
zero AI cost. The forecast is O(horizon) per candidate with a small candidate set.

### 44. What happens when the model API fails?
Extraction fails closed — bounded retries with backoff, then the fact is marked unusable and the
pipeline continues on CSV evidence rather than guessing. There's also a variant ladder for providers
that disagree about parameters, which I needed: the router moved me to a provider requiring
`strict: true` and rejecting my thinking flag mid-run. And because prediction reads cached facts, a
provider outage can't affect a prediction run at all.

### 45. Why is your solution deterministic?
Temperature zero, cached facts, no wall-clock or random dependency, sorted iteration and explicit
tie-breakers everywhere. Two consecutive 250-request runs produce a byte-identical file — I verified
the SHA256 matches.

### 46. How do you know the 250 output is valid without ground truth?
I don't check accuracy — I check validity, and I say so rather than inventing a number. Every row
goes through the independent validator: bounds, enums, status/method consistency, payment arithmetic,
deadlines, exact installment matching, spending-change legality, and a full independent 90-day safety
replay. 250 of 250 pass, and the evaluation script exits non-zero if any don't.

### 47. What was the hardest edge case?
The duplicate card charge. It's a pending debit settling inside the forecast window for a plausible
amount — structurally identical to a real liability. The only signal is that it's linked to a settled
charge of exactly the same amount. Get it wrong and you reserve money the user will never spend.

### 48. What did AI contribute compared with ordinary code?
Things ordinary code genuinely can't do. Sixteen financial events have blank amounts and require
image interpretation — four of those are future liabilities that directly move the forecast — and
knowing which figure to take requires understanding that Net Pay beats Total Earnings. And 215
messages change salaries, end employment or flag income as unconfirmed in free-form bilingual prose.
Without AI, four future liabilities are unknown and every salary amendment is invisible.

### 49. What part of the solution contributes most to accuracy?
Income modelling, by a wide margin. Getting the taxonomy and grouping right moved more than every
estimator choice combined — it was the difference between 39 users having zero projected income and
having correct income. Estimator tuning moved things by fractions of a percent; income classification
moved them by orders of magnitude.

### 50. If you had twice the budget, what would you change?
Almost nothing about the architecture — I used only 20 cents of about $4.20, so budget was never the
constraint. I'd spend it on a second extraction pass over messages with a different prompt framing
and reconcile disagreements, since message interpretation is where my remaining AI risk sits. I would
not spend it on per-request LLM reasoning; that would cost determinism for no accuracy gain.

---

## Hard Follow-Up Questions

### "Spending changes are often missing or wrong. Why not make that a first-class step?"
It already is: `spending.py` enumerates every permitted change set, and the validator re-checks
each action. I traced all 4 sample misses on that field, and every one is caused upstream — our
`amount_safe_to_pay` lands on the wrong side of the requested amount. Twice we're too optimistic
(we think no change is needed), twice too pessimistic (we ask for changes that aren't needed, or
the one permitted change isn't enough). In `request_06` the selector was looking at the right
item — streaming is the only category that user lets us stop. So the fix is better variable-spending
estimation, not another selection step.

### "Shouldn't you recompute the safe amount and earliest date after the spending changes?"
No — that would contradict the specification, which defines both *before* optional spending
changes. The ground truth confirms it: one sample recommends full payment today with a reduction,
yet its earliest date is weeks later, because that date is measured without the change. What *is*
derived from the adjusted plan — affordability status, method, plan, explanation — already is, and
the validator checks the fields agree on all 250 rows.

### "You fill in full-payment dates by default."
We don't: 64 of 250 rows are deliberately blank, and on the samples we never fill a date where the
truth is blank. Our 3 date errors go the other way — blank where the truth has a date — because a
too-cautious forecast finds no safe date.

### "You say 26 days means monthly. Why 26 and not 25 or 28?"
Honestly, on this dataset it makes no difference — observed cadences are 5, 7, 10, 14 and 21 on one
side and 30–31 on the other, so anything from 22 to 29 behaves identically. I picked 26 to sit
interior to that band with margin for February at 28 days and for a 21-day cadence below. I wouldn't
claim it's optimised, and if you showed me a 24-day cadence in unseen data I'd want to revisit it.

### "Why should I trust a 1.5-cycle staleness threshold?"
It encodes "missed a full cycle plus a half-cycle of grace", which matters because payment dates
jitter — a salary can land a few days late without having stopped. 1.0 would drop a stream on ordinary
jitter; 2.0 keeps income that has demonstrably ended. The case that drove it is a stream whose last
occurrence was 47 days before the request on a 30-day cadence, where projecting it would invent
income. It's an engineering choice inside a defensible band, not a fitted parameter.

### "You optimised discrete accuracy at the expense of numeric MAE. Isn't that overfitting?"
It's a scoring-function argument, not a fitting argument. Five of the structured output fields are
categorical or exact-style — status, method, plan, earliest date, spending changes — alongside the
numeric `amount_safe_to_pay`, and `decision_explanation` is additionally scored for usefulness and
consistency. An exact-style field is right or wrong; a numeric one degrades gracefully. Trading 0.25
points of mean numeric error for four structured-field wins is favourable under that shape of
scoring. The anti-overfitting evidence is that the change was better on
2 samples and worse on **zero**, the trend was monotone across three estimators, and I chose the
time-window form over the equally-scoring fixed-count form precisely because it has a rationale
independent of the samples.

### "If GLM misclassifies a message, doesn't the whole forecast fail?"
It would if I trusted its labels — and I measured that I couldn't. That's why the fact type is derived
from `target`, `recurrence_effect` and the presence of a number rather than the model's `operation`
label, with a hard invariant that a non-income message can never terminate income. When I checked,
50 of 215 messages would have wrongly zeroed salary under naive trust. After the guardrails, 14
genuine cases remain. The system is designed to fail toward "no change", which is the conservative
direction.

### "Why use a 235B VLM for only 16 images?"
Because those 16 images carry four liabilities worth up to 100,000 currency units each, and seven of
them contain a semantic trap where the largest number on the page is the wrong answer. Getting one
wrong is a materially wrong recommendation. The whole vision spend was 1.1 cents — there was no
saving worth the risk of a weaker model misreading a payslip.

### "Is this really agentic if Python makes the final decision?"
The agent perceives an environment it can't parse mechanically, converts perception to structured
belief, reasons over it with tools, and acts under constraints. The AI perception is load-bearing:
remove it and four liabilities have no amount and every salary amendment is invisible. What I avoided
is letting a language model do arithmetic a computer does exactly. I'd argue choosing the right tool
per subtask is what makes it a well-built agent, not a less agentic one.

### "Why should credits post before debits on the same day?"
Strictly, I can't know that they do — the dataset supplies settlement dates and no intraday posting
order at all. That's precisely the argument. Ordering debits first asserts a sequence the data
doesn't give me, and it manufactures a transient dip: on one sample it put the floor where a 12,650
debit was assumed to land before a 131,000 salary on the same day. Treating the constraint as an
end-of-day position assumes less, and it matches the reading of `minimum_balance_to_keep` as a
balance the user *keeps* rather than a bank's posting artefact. The solved evidence agrees — that
sample went from 34% error to 0.1% — but I'd defend it on the "don't invent an ordering" ground even
without that.

### "How do you know `trailing_horizon` generalises?"
I don't know it with certainty, and I wouldn't claim it. The evidence is: better on 2 samples, worse
on none; a monotone trend across `mean_all` → `mean_6` → `mean_3` suggesting recency genuinely
matters rather than noise; and a rationale that doesn't reference the samples at all — estimate the
next horizon from the last horizon. I deliberately chose it over `mean_3`, which scored identically,
because a fixed observation count would have been the fitted choice.

### "Why didn't you use the LLM to infer recurrence?"
Cadence and level are statistics over dated numbers — a computer does that exactly and instantly. A
model would be slower, cost more, vary between runs, and give me no way to audit why a stream got a
particular amount. The place judgement was actually needed was deciding *which* income continues, and
even there the evidence is a description and a message, which I handle with a taxonomy plus extracted
facts rather than freeform reasoning.

### "Why did visible sample accuracy fall when you fixed the wait invariant?"
Because some samples were getting the right method for the wrong reason. They were predicting `wait`
with spurious spending changes, which matched the status and method but not the changes field. When
I required a genuine no-changes earliest date, those fell through to a different recommendation. I
kept the fix anyway: shipping a row that says "wait until this date" while also reporting that the
amount never becomes affordable is malformed output, and four rows in the real 250 had exactly that.
Correctness over four sample points.

### "Your explanations are templated. Isn't that a weaker use of AI?"
It's a deliberate choice. `decision_explanation` is scored on usefulness and consistency, and a
template built from the winning plan's own numbers cannot drift from the plan it describes — an LLM
rewrite could produce a fluent sentence that contradicts the schedule. It also costs nothing and is
deterministic. If the scoring rewarded prose variety I'd reconsider, but consistency with the
recommendation seemed the higher value.

### "You only got 3 of 25 exactly right on `amount_safe_to_pay`. Isn't that poor?"
On exact matches, yes — and I'm not going to dress it up. The fuller picture is 21 of 25 within 5%
and a median error of 1.88%, and the five structured fields are at 103 of
125. I decomposed every residual to a single component, tested six estimator families and three
projection modes, and found that the residuals have both signs so no single correction fixes them. I
also found that 16 of 21 reference reserves are exact whole numbers while every input carries cents,
which suggests a rounding step I couldn't locate. At that point further movement would have been
fitting 25 labels.

---

## Constant / Threshold Flashcards

**90 days**
- Controls: forecast horizon and the safety window
- Why: mandated by the problem statement
- Source: **[SPEC]**
- If changed: 75 days scored better on samples and was rejected as unjustified; shortening hides
  future obligations, lengthening invents certainty

**26 days**
- Controls: monthly vs sub-monthly stream classification
- Why: separates 28–31 day monthly cadences from observed sub-monthly ones (5, 7, 10, 14, 21)
- Evidence: **[DATA]** — any value 22–29 is behaviourally identical here
- Alternative: 30 would misclassify February; ≤21 would misclassify a 21-day habit as monthly

**1.5 cycles**
- Controls: when a recurring income stream is considered stale and dropped
- Why: a stream that has missed a full cycle plus a half-cycle of grace is no longer evidenced
- Source: **[EMPIRICAL]** — driven by a stream last seen 47 days before a request on a 30-day cadence
- If changed: 1.0 drops streams on ordinary timing jitter; 2.0 keeps income that has clearly stopped

**3 observations**
- Controls: minimum history before a stream is inferred
- Why: you need two gaps to take a median gap, which needs three occurrences
- Source: **[SAFETY]** + the spec's "detect recurrence only when history supports it"
- If changed: 2 gives a single gap with no median and no robustness

**`trailing_horizon` (90-day window)**
- Controls: the amount estimated for each recurring stream
- Why: estimate the next 90 days from the last 90; the window adapts to cadence
- Source: **[EMPIRICAL]**, measured against 5 alternatives
- If changed: `mean_all` gives better MAE (2.90% vs 3.15%) but worse discrete (103 vs 107 of 125)

**`accrual` (sub-monthly)**
- Controls: how weekly/fortnightly habits enter the forecast
- Why: a habit is a rate, not a dated commitment; discrete charging produced larger boundary errors
  in both directions in the tested samples
- Source: **[EMPIRICAL]** — `discrete` 3.37% MAE, `cycle_upfront` 4.26%, `accrual` 2.90%
- If changed: short windows under-reserve, long windows over-reserve

**`credits_first`**
- Controls: how a day carrying both income and bills is evaluated
- Why: the minimum balance is an end-of-day position, not a posting-order artefact
- Source: **[DATA]** — one sample moved from 34% error to 0.1%
- If changed: creates intraday troughs on payday that never occur

**0.01 (`quantum`)**
- Controls: monetary rounding precision
- Why: currency precision; money isn't continuous
- Source: **[SAFETY]**
- If changed: whole units tested and worse in every configuration (discrete 103 → 96)

**3 spending changes**
- Controls: maximum changes in `spending_changes_needed`
- Why: "up to three" in the specification
- Source: **[SPEC]**
- If changed: invalid output

**2 partial payments**
- Controls: partial-payment plan shape
- Why: "exactly two payments" in the specification
- Source: **[SPEC]**
- If changed: invalid output

**USD 3.00 budget ceiling**
- Controls: hard stop before a model call
- Why: ~$4.20 available; the ceiling preserves margin for retries and mistakes
- Source: **[SAFETY]** — actual final evidence spend was $0.2029
- If changed: raising removes the safety margin; lowering could block legitimate extraction

**temperature 0.0**
- Controls: model sampling
- Why: deterministic structured extraction; the run must reproduce
- Source: **[API]**
- If changed: identical inputs would give different facts and break reproducibility

**max_retries 2 / timeout 90 s**
- Controls: bounded corrective retries and per-call timeout
- Why: recover from transient failures without unbounded cost; vision calls took up to ~11 s
- Source: **[SAFETY]** — observed retries in the final evidence: 1
- If changed: unbounded retries risk cost blow-up; a short timeout fails healthy vision calls

**confidence 0.5 / 0.6 / 0.4 clamp**
- Controls: inspection flags on extracted facts
- Why: surfaces facts worth a human look; images held to a higher bar since one image can be a whole
  liability. The 0.4 clamp applies when the model picks a figure that is never a transaction's
  economic value
- Source: **[SAFETY]**
- Important: these **do not gate any financial decision** — they are reporting thresholds only

**45-day income supersession window**
- Controls: dropping a parallel income stream much older than the latest employment row
- Why: ~1.5 monthly cycles; a stream that stale alongside a fresher one is superseded
- Source: **[EMPIRICAL]**

---

## What I Would Improve With More Time

1. **A probabilistic model for variable sub-monthly spending.** This is my one unresolved component.
   Rather than a point estimate, I'd model each variable stream as a distribution and take a
   conservative quantile — which would also let the safe amount carry a confidence level instead of
   implying false precision.
2. **Uncertainty intervals on the forecast.** Report the safe amount as a range, and let the plan
   ranker prefer plans that stay safe across the interval rather than only at the point estimate.
3. **Genuine cross-validation.** With 25 labelled examples I could not hold out a meaningful
   validation set. More labels would let me select rules on held-out data instead of reasoning about
   whether a rule generalises.
4. **Confidence-aware fallback.** Right now a low-confidence extraction is flagged but still used. I'd
   route low-confidence image facts to a conservative branch, or a second extraction with a different
   prompt, and reconcile.
5. **Broader provider resilience.** The variant ladder handles parameter disagreements; I'd extend it
   to explicit provider pinning with health checks.

**What I would not do:** replace deterministic validation or ranking with LLM reasoning. That would
trade exactness and reproducibility for nothing.

---

## What AI Actually Did

**AI (GLM-5.3 and Qwen3-VL) did the semantic work:**

- Understood 215 free-form messages in English and Indonesian, and extracted what each one asserts
  about the user's finances.
- Interpreted 16 document images *semantically* — not just reading numbers, but deciding which
  number represents the transaction's real economic value.

**Deterministic Python did everything numerical:**

- Reconstructed financial state from 25,342 ledger events
- Resolved transaction lifecycles so nothing is double-counted
- Converted currencies on dated rates with provenance
- Inferred which expenses and incomes recur, at what cadence and level
- Projected a 90-day balance path
- Computed `amount_safe_to_pay` and `earliest_date_for_full_payment`
- Generated, validated and ranked payment plans
- Optimised spending changes by exhaustive search
- Serialised the output

**The principle:** AI where semantic understanding beats code, deterministic code where numerical
exactness and reproducibility matter. Sixteen financial events have blank amounts that only the
vision model can resolve — four of them future liabilities that move the forecast — and every salary
amendment is invisible without the text model. The AI is load-bearing, not decorative. But no model
is ever asked to add two numbers.

---

## Where to Find Things in the Code

| Topic | File / function |
|---|---|
| Prediction entry point | `code/main.py` → `main()` |
| Frozen run manifest | `code/main.py` → `manifest()` (`--manifest`) |
| Dataset loading and indexes | `code/bow/dataset.py` → `Dataset.load`, `Dataset.context_for` |
| FX conversion | `code/bow/fx.py` → `ExchangeRateService.rate_for`, `.to_home` |
| Event lifecycle resolution | `code/bow/resolve.py` → `EventResolver.resolve`, `_pair_kind`, `_standalone` |
| Cash-partition guard | `code/bow/resolve.py` → `assert_cash_partition` |
| Income taxonomy | `code/bow/recurrence.py` → `classify_income`, `_income_group` |
| Recurrence inference | `code/bow/recurrence.py` → `infer_streams` |
| Confirmed-salary anchoring | `code/bow/recurrence.py` → `_anchor_on_confirmed_income` |
| Message fact application | `code/bow/recurrence.py` → `apply_stream_directives` |
| Image fact application | `code/bow/recurrence.py` → `apply_image_facts` |
| Calendar-month stepping | `code/bow/recurrence.py` → `add_month`, `occurrences` |
| 90-day forecast | `code/bow/forecast.py` → `ForecastEngine.project`, `flows_for` |
| Explicit-vs-inferred precedence | `code/bow/forecast.py` → `_superseded` |
| `amount_safe_to_pay` | `code/bow/safeamount.py` → `safe_amount` |
| `earliest_date_for_full_payment` | `code/bow/safeamount.py` → `earliest_full_payment_date` |
| Method / option eligibility | `code/bow/eligibility.py` → `assess`, `_judge`, `partial_allowed` |
| Spending-change enumeration | `code/bow/spending.py` → `permitted_changes`, `change_sets` |
| Candidate plan generation | `code/bow/plans.py` → `generate` |
| Independent validator | `code/bow/validate.py` → `validate` (V1–V15) |
| Ranking | `code/bow/rank.py` → `sort_key`, `rank`, `explain_choice` |
| Explanation | `code/bow/explain.py` → `build` |
| Status mapping | `code/bow/predict.py` → `status_for` |
| Orchestrator | `code/bow/predict.py` → `predict` |
| Bounded tool-using agent | `code/bow/agent.py` → `run_agent`, `_Session.t_*` tools |
| Agent vs baselines | `code/evaluation/compare.py` → `comparison.md` |
| Output serialisation | `code/bow/outputs.py` → `serialise_plan`, `write_output` |
| Message extraction + guardrails | `code/bow/ai/messages.py` → `extract_message`, `derive_fact_type` |
| Image extraction | `code/bow/ai/images.py` → `extract_image`, `to_fact` |
| Prompts and schemas | `code/bow/ai/schemas.py` |
| Provider client + variant ladder | `code/bow/ai/client.py` → `ChatClient._post`, `_VARIANTS` |
| Cache keys | `code/bow/ai/cache.py` → `cache_key`, `ExtractionCache` |
| Fact loading (no model calls) | `code/bow/ai/facts.py` → `load_facts` |
| Usage ledger and budget guard | `code/bow/usage.py` → `UsageLedger` |
| Configuration / constants | `code/bow/config.py` → `ForecastConfig`, `ModelConfig`, `RunConfig` |
| Evaluation entry point | `code/evaluation/main.py` → `main`, `validate_output` |
| 25-sample regression harness | `code/evaluation/regress.py` → `run` |
| One-off AI extraction | `code/evaluation/extract.py` |
| Usage report generation | `code/evaluation/usage_summary.py` |
| Import-layer guard | `code/evaluation/check_layers.py` |
| Tests | `code/tests/` (4 test modules, 177 tests) |

---

## Numbers Worth Remembering

**Dataset**
- 250 evaluation requests · 25 solved samples · 275 profiles (one request per user)
- 25,342 financial events · 790 payment options · 215 messages · 16 images · 134 exchange rates
- 434 of 515 installment options finish after their deadline

**AI usage (final evidence behind `output.csv`)**
- 231 calls: 215 message + 16 image · 1 retry
- GLM-5.3: 143,369 in / 48,013 out / ~$0.1916
- Qwen3-VL: 25,167 in / 3,106 out / ~$0.0113
- Combined: 219,655 tokens · ~$0.2029 · ~878.6 tokens and ~$0.000812 per request
- Development spend (separate): ~$0.676 of ~$4.20 available

**Final 250 run**
- 250/250 predictions · 250/250 independent validation · ~2.3 s · 0 model calls
- SHA256 `7ec8757424b04be82e2b4f7b5c0c8b2c322579cfec0d9731e231125eb7e94f2a`, byte-identical on rerun
- Status: 74 with_plan / 65 not_affordable / 62 now / 49 later
- Method: 66 full / 65 not_recommended / 60 installments / 49 wait / 10 partial
- 24 plans use spending changes

**Final 25-sample metrics (frozen)**
- `recommended_payment_method` 22/25 · `payment_plan` 21/25 · `spending_changes_needed` 21/25
- `affordability_status` 20/25 · `earliest_date_for_full_payment` 19/25
- `amount_safe_to_pay`: 3 exact, 9 within 1%, 21 within 5%, 23 within 10%
- mean error 3.15%, median 1.88%, max 13.44% · complete rows 3/25

**Code**
- 28 modules, ~4,800 lines in `bow/` · 177 tests in 4 modules · zero third-party dependencies
- 14 validator checks (V1–V13, V15) · 6 ranking rules · 8 lifecycle families · 5 income classes

---

# Last-Minute 5-Minute Review

**Architecture in 9 lines**
1. Load 8 CSVs into typed models; Decimal money everywhere
2. GLM-5.3 → one typed fact per message (cached); Qwen3-VL → one typed amount per image (cached)
3. Resolve raw events into canonical economic events with two flags: counts_for_cash, counts_for_recurrence
4. Infer recurring streams — expenses by category, income by class
5. Convert foreign amounts on their settlement date with provenance
6. Project a 90-day balance path; monthly = dated, sub-monthly = accrued rate
7. amount_safe_to_pay = trough − minimum, clamped to [0, requested]
8. Generate every candidate plan, re-forecast each, validate independently, rank by the six rules
9. Write output.csv — deterministic, byte-identical on rerun

**Model roles** — GLM-5.3: messages only. Qwen3-VL: images only. Python: all finance.

**Top 10 constants** — 90-day horizon [SPEC] · 26-day monthly threshold [DATA, indifferent 22–29] ·
1.5-cycle staleness [EMPIRICAL] · 3 observations [SAFETY] · trailing_horizon estimator [EMPIRICAL] ·
accrual sub-monthly [EMPIRICAL] · credits_first [DATA] · quantum 0.01 [SAFETY] · max 3 changes
[SPEC] · exactly 2 partial payments [SPEC]

**Top 10 edge cases** — cancelled authorization + settlement · failed debit + retry · duplicate card
charge · pending refund (never counted) · employer reimbursement · unrealized investment valuation ·
investment sale proceeds · Final employer payroll (income stops) · stale income stream · blank
amount awaiting an image (never zero)

**Final cost** — $0.2029 for the evidence; $0 for prediction; ~$0.68 total development

**Known limitation** — variable sub-monthly spending estimation; 3/25 exact, 21/25 within 5%, median
error 1.88%; residuals have both signs so no single correction fixes them

**Biggest tradeoff** — chose `trailing_horizon` over `mean_all`: +4 discrete fields for +0.25pp MAE,
because five structured fields are exact-match while the numeric one degrades gracefully

**Biggest bug** — income taxonomy: projecting every salary-category credit including terminated
employment, gig payouts and commissions; plus grouping income by description, which left 39 of 275
users with zero projected income

**Strongest point** — every financial number is deterministic, independently validated and
reproducible to the byte, with AI used exactly where code cannot substitute

**Why not multi-agent** — compounding interpretation errors, non-determinism, context growth, and
exact schedules that must match supplied offers to the cent. One orchestrator, isolated extraction,
deterministic specialists.

**Why a deterministic financial engine** — the arithmetic is exact and the rulebook is given. A model
could only reproduce it less reliably, more expensively, and differently each run.
