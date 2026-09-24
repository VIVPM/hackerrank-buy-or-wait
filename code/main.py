"""Buy or Wait? - generate predictions for every request in dataset/requests.csv.

    python code/main.py                 # writes <repo root>/output.csv
    python code/main.py --samples       # writes predictions for the 25 solved samples instead
    python code/main.py --manifest      # print the frozen-run manifest and exit
    python code/main.py --agent --samples   # bounded tool-using agent -> output_agent*.csv

The default run is the submitted, fixed-order pipeline. `--agent` runs the bounded agent
(`bow/agent.py`) instead and writes to a separate file, so it can never overwrite output.csv.

Prediction is fully deterministic and makes no model calls: the AI-derived facts were extracted
once by `code/evaluation/extract.py` and are read from `code/ai_cache`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bow.ai.facts import load_facts                                   # noqa: E402
from bow.ai.schemas import (                                          # noqa: E402
    EXTRACTOR_VERSION, IMAGE_PROMPT_VERSION, IMAGE_SCHEMA_VERSION,
    MESSAGE_PROMPT_VERSION, MESSAGE_SCHEMA_VERSION,
)
from bow.config import RunConfig                                      # noqa: E402
from bow.dataset import Dataset                                       # noqa: E402
from bow.errors import BowError                                       # noqa: E402
from bow.outputs import write_output                                  # noqa: E402
from bow.predict import predict                                       # noqa: E402


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=str(Path(__file__).resolve().parents[1]), timeout=10)
        return out.stdout.strip() or "unavailable"
    except Exception:
        return "unavailable"


def manifest(cfg: RunConfig) -> dict:
    """Everything needed to reproduce this run."""
    fc = cfg.forecast
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "forecast_config": {f.name: str(getattr(fc, f.name))
                            for f in fc.__dataclass_fields__.values()},
        "models": {
            "text": cfg.models.text_model,
            "vision": cfg.models.vision_model,
            "provider": cfg.models.provider,
            "routing": "huggingface routed inference (router.huggingface.co/v1), no dedicated "
                       "endpoints",
            "temperature": cfg.models.temperature,
        },
        "extraction_versions": {
            "message_prompt": MESSAGE_PROMPT_VERSION,
            "message_schema": MESSAGE_SCHEMA_VERSION,
            "image_prompt": IMAGE_PROMPT_VERSION,
            "image_schema": IMAGE_SCHEMA_VERSION,
            "extractor": EXTRACTOR_VERSION,
        },
        "paths": {
            "dataset": str(cfg.dataset_dir),
            "cache": str(cfg.cache_dir),
            "output": str(cfg.output_path),
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Buy or Wait? prediction run")
    ap.add_argument("--samples", action="store_true",
                    help="predict the 25 solved samples instead of the evaluation set")
    ap.add_argument("--manifest", action="store_true", help="print the frozen manifest and exit")
    ap.add_argument("--out", type=Path, default=None, help="override the output path")
    ap.add_argument("--agent", action="store_true",
                    help="use the bounded tool-using agent (makes model calls unless cached)")
    args = ap.parse_args(argv)

    cfg = RunConfig()
    if args.manifest:
        print(json.dumps(manifest(cfg), indent=2))
        return 0

    ds = Dataset.load(cfg)
    store = load_facts(ds, cfg, strict=True)      # fails closed if any fact is uncached
    ids = ds.sample_request_ids if args.samples else ds.eval_request_ids
    target = args.out or (cfg.output_path if not args.samples
                          else cfg.output_path.with_name("output_samples.csv"))
    if args.agent and not args.out:
        target = target.with_name(target.stem.replace("output", "output_agent") + ".csv")

    results, failures = [], []
    for request_id in ids:
        ctx = ds.context_for(request_id)
        try:
            if args.agent:
                from bow.agent import run_agent
                results.append(run_agent(ctx, store, run_cfg=cfg,
                                         trace_dir=cfg.debug_dir / "agent" / "guided").result)
            else:
                results.append(predict(ctx, store, cfg.forecast).result)
        except BowError as exc:
            failures.append((request_id, f"{type(exc).__name__}: {exc}"))

    if failures:
        print(f"ERROR: {len(failures)} request(s) failed to predict:", file=sys.stderr)
        for request_id, why in failures:
            print(f"  {request_id}: {why}", file=sys.stderr)
        return 1

    write_output(results, target)
    print(f"wrote {len(results)} predictions to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
