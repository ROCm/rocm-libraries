#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Calibrate the existing legal knobs of the gfx950 dense prefill kernel.

Sweeps only knobs the production kernel already exposes (block_n, waves_per_eu,
persistent/num_persistent) at one exact shape, so the strongest *unchanged*
configuration can be used as the experiment baseline.

Usage::

    python calibrate_dense_prefill_knobs.py \
        --shape-json llama3_8b_dense_prefill_baseline_shape.json \
        --output-json /path/calib.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(__file__)
_RK = os.path.abspath(os.path.join(_HERE, "../../../../.."))
sys.path.insert(0, _RK + "/platform/python")
sys.path.insert(0, _RK + "/library")

import torch  # noqa: E402

from builders.gfx950.attention.prefill.attention_dense_prefill import (  # noqa: E402
    make_spec_from_shape,
    run_benchmark,
)

_DEFAULT_SHAPE = Path(_HERE) / "llama3_8b_dense_prefill_baseline_shape.json"


def _device_info() -> dict:
    p = torch.cuda.get_device_properties(0)
    return {
        "name": p.name,
        "gcnArchName": p.gcnArchName,
        "multi_processor_count": p.multi_processor_count,
        "torch": torch.__version__,
        "hip": torch.version.hip,
    }


def _grid_ctas(grid) -> int:
    n = 1
    for d in grid:
        n *= int(d)
    return n


def _configs(shape: dict) -> list[dict]:
    """Legal knob combinations only; NP values stay at or below 256 CUs."""
    out: list[dict] = []
    for bn in (64, 128):
        for wpe in (1, 2):
            out.append({"block_n": bn, "waves_per_eu": wpe, "persistent": False})
            for np_ in (128, 256):
                out.append(
                    {
                        "block_n": bn,
                        "waves_per_eu": wpe,
                        "persistent": True,
                        "num_persistent": np_,
                    }
                )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shape-json", type=Path, default=_DEFAULT_SHAPE)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=3, help="samples per config")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no HIP device", file=sys.stderr)
        return 1

    base = json.loads(args.shape_json.read_text())
    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_shape": base,
        "device": _device_info(),
        "warmup": args.warmup,
        "iters": args.iters,
        "repeats": args.repeats,
        "configs": [],
    }

    for knobs in _configs(base):
        shape = dict(base)
        shape.update(knobs)
        # q_reload stays off: this sweep documents the unchanged kernel.
        shape["q_reload"] = False
        label = (
            f"bn{knobs['block_n']}_wpe{knobs['waves_per_eu']}"
            + (f"_np{knobs['num_persistent']}" if knobs["persistent"] else "_nopers")
        )
        print(f"\n=== {label} ===", flush=True)

        samples: list[dict] = []
        failed: str | None = None
        for r in range(args.repeats):
            try:
                res = run_benchmark(
                    make_spec_from_shape(shape),
                    warmup=args.warmup,
                    iters=args.iters,
                    seed=args.seed,
                    # parity is shape/knob independent: check the first sample only
                    check=(not args.no_check) and r == 0,
                )
            except Exception as exc:  # unsupported knob combination
                failed = f"{type(exc).__name__}: {exc}"
                print(f"  SKIP {failed}", flush=True)
                break
            samples.append(res)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        entry: dict = {"label": label, "knobs": knobs, "error": failed}
        if samples:
            tflops = [s["tflops"] for s in samples]
            entry.update(
                {
                    "kernel_name": samples[0]["kernel_name"],
                    "persist_decode": samples[0]["persist_decode"],
                    "grid": samples[0]["grid"],
                    "block": samples[0]["block"],
                    "ctas": _grid_ctas(samples[0]["grid"]),
                    "ms_median": statistics.median(s["ms"] for s in samples),
                    "tflops_median": statistics.median(tflops),
                    "tflops_samples": tflops,
                    "max_abs": samples[0]["max_abs"],
                    "ok": all(s["ok"] for s in samples),
                }
            )
            print(
                f"  {entry['ms_median']:.4f} ms  {entry['tflops_median']:.1f} TFLOPS "
                f"ctas={entry['ctas']}  ok={entry['ok']}",
                flush=True,
            )
        report["configs"].append(entry)

    ranked = [c for c in report["configs"] if c.get("ok")]
    ranked.sort(key=lambda c: c["tflops_median"], reverse=True)
    report["ranking"] = [
        {"label": c["label"], "tflops_median": c["tflops_median"]} for c in ranked
    ]
    report["best"] = ranked[0] if ranked else None

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.output_json}")
    if report["best"]:
        print("best:", json.dumps(report["best"]["knobs"]), end=" ")
        print(f"{report['best']['tflops_median']:.1f} TFLOPS")
    for c in report["ranking"]:
        print(f"  {c['label']:<20} {c['tflops_median']:8.1f}")
    return 0 if report["best"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
