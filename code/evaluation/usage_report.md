# Token Usage and Cost Report

Covers the model evidence behind the submitted `output.csv` (250 requests).

Both models perform **evidence extraction only**. Neither computes affordability,
chooses a payment method, ranks plans, or writes an output field. Every financial
calculation - the 90-day forecast, `amount_safe_to_pay`, the earliest safe date, plan
generation, validation and ranking - is deterministic Python.

Extraction runs once for the whole dataset (one call per message, one per image) and is
cached by content hash, so generating the 250 predictions makes **zero** model calls.

## Per model

| Provider | Model | Role | Calls | Retries | Input | Output | Total | Cost (USD) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| huggingface-routed | Qwen/Qwen3-VL-235B-A22B-Instruct | image evidence extraction | 16 | 0 | 25,167 | 3,106 | 28,273 | 0.0113 |
| huggingface-routed | zai-org/GLM-5.3 | message evidence extraction | 215 | 1 | 143,369 | 48,013 | 191,382 | 0.1916 |

## Overall

- Model calls for the final evidence: **231** (215 messages + 16 images)
- Successful calls: **231** (every cached fact validated against its schema)
- Retries: **1**
- Cache hits during the final prediction run: **all 231** (0 misses, 0 new calls)
- Input tokens: **168,536**
- Output tokens: **51,119**
- Total tokens: **219,655**
- Evaluation requests: **250**
- Average tokens per request: **878.6** (amortised: extraction is per-message/per-image, not per-request)
- Estimated total cost: **USD 0.2029**
- Estimated cost per request: **USD 0.000812**

## Cost basis

The Hugging Face router reports its own `estimated_cost` for some providers and not
others. Where reported, that figure is used. Otherwise cost is computed from published
rates recorded in `bow/config.py` (`DEFAULT_INPUT_PRICES` / `DEFAULT_OUTPUT_PRICES`).

Development iterations over superseded prompt versions are not included above; the
cumulative spend across the whole project, recorded in `evaluation/usage.jsonl`, was
USD 0.68 against an available budget of about USD 4.20.

No API keys or credentials appear in this report or anywhere in the submission.
