# Baseline and configuration comparison

25 solved samples, same scorer (`evaluation/regress.py`) for every row. Agent steps are cached, so these numbers reproduce offline.

| Field | rules_only | pipeline | agent_guided | agent_brief |
|---|---:|---:|---:|---:|
| `amount_safe_to_pay` | 3/25 | 3/25 | 3/25 | 3/25 |
| `affordability_status` | 20/25 | 20/25 | 20/25 | 20/25 |
| `recommended_payment_method` | 21/25 | 22/25 | 22/25 | 22/25 |
| `payment_plan` | 20/25 | 21/25 | 21/25 | 21/25 |
| `earliest_date_for_full_payment` | 18/25 | 19/25 | 19/25 | 19/25 |
| `spending_changes_needed` | 21/25 | 21/25 | 21/25 | 21/25 |
| all five structured fields | 100/125 | 103/125 | 103/125 | 103/125 |
| amount: median error | 2.25% | 1.88% | 1.88% | 1.88% |
| amount: mean error | 8.14% | 3.15% | 3.15% | 3.15% |

## Agent behaviour

| | agent_guided | agent_brief |
|---|---:|---:|
| mean tool calls per request | 5.5 | 5.8 |
| hit the 10-step cap | 0 | 0 |
| fell back to pipeline (any reason) | 0 | 0 |
| chose same plan as deterministic ranker | 25/25 | 25/25 |
| relevant evidence read | 13/13 | 13/13 |
| tool errors (recovered) | 0 | 0 |
| model calls, all paid runs | 291 | 322 |
| input tokens, all paid runs | 240940 | 229846 |
| output tokens, all paid runs | 17771 | 22909 |
| cost USD, all paid runs | 0.1836602 | 0.1883074 |

Traces: `code/.debug/agent/<prompt>/<request_id>.json` (every tool call, its result, evidence read, and any fallback reason).

## History: the first run, before two fixes

The first measurement (agent prompt v1) scored the agents at **97/125 (guided)** and **100/125
(brief)**, with 19 and 38 recovered tool errors, 1 and 3 step-cap hits, and 21/23 and 21/22
agreement with the ranker. Tracing the misses found:

- **An engine bug the agent exposed.** `completes_by_deadline` used `total == requested`, so every
  installment plan with a financing fee counted as "never completes". The agent applied rule 1
  literally and refused plans that were correct (request_02, request_22). Fixed to `>=` with a
  regression test; 0 of 250 `output.csv` rows changed, because the pipeline had been right by
  accident (`not_recommended` always sorts last).
- **An ambiguous tool argument.** 38 of 57 errors were a tool name placed in the `id` field. Renamed
  to `evidence_id`, and malformed calls are cleaned before being echoed back.

The table above is the re-run with both fixes (agent prompt v2). Its own run cost about USD 0.17;
the "all paid runs" cost rows also include the first run.
