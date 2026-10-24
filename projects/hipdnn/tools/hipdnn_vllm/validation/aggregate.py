# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Aggregate e2e_matrix.py results into coverage, accuracy and perf tables.

usage: python aggregate.py RESULT_DIR [RESULT_DIR ...] --out DIR

Finds every ``summary.json`` below the given directories and writes:

* ``coverage.csv`` / ``coverage.md``: model x arch x run x op x route x reason,
  with eager and capture-time call counts;
* ``accuracy.md``: per run, teacher-forced top-1 agreement with base (min
  cosine for embedding models), the plugin-off control's agreement, the
  threshold, near ties, the mean |delta logprob| of the forced token, NaN and
  the verdict (``ERROR`` rows for failed runs, base included), then a
  Failures section with each summary's ``failures`` and every failed run;
* ``perf.md``: tokens/s, TTFT and start-up time relative to base and to
  vLLM's default backend, plus plan-build, norm pre-build and handle-creation
  stall time;
* ``rows.json``: the same rows, machine-readable.

When the same (model, arch) appears more than once, the newest summary wins.
Exits 1 when any failure is listed or no summary was found.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os


def _fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _md(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(_fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def load(dirs: list[str]) -> list[dict]:
    found: dict[tuple, dict] = {}
    for d in dirs:
        for path in glob.glob(os.path.join(d, "**", "summary.json"), recursive=True):
            s = json.load(open(path))
            s["_path"] = path
            key = (s["model"], s["arch"])
            if key not in found or s.get("time", "") > found[key].get("time", ""):
                found[key] = s
    return [found[k] for k in sorted(found)]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    summaries = load(args.dirs)

    cov, acc, perf, skipped, failures = [], [], [], [], []
    for s in summaries:
        model, arch = s["model"], s["arch"]
        if s.get("skipped"):
            skipped.append([model, arch, s["skipped"]])
            continue
        listed = set(s.get("failures", []))
        failures += [[model, arch, f] for f in s.get("failures", [])]
        ctrl = (s["runs"].get("control", {}).get("accuracy") or {}).get("agreement")
        swap = (s["runs"].get("cand-vllm_c", {}).get("accuracy") or {}).get("agreement")
        if "base" not in s["runs"]:
            failures.append([model, arch, "summary has no base run"])
        for name, run in s["runs"].items():
            if "error" in run and not any(f.startswith(f"{name}:") or f.startswith(f"{name} ") for f in listed):
                failures.append([model, arch, f"{name}: {run['error']} (log {run.get('log', '-')})"])
        for name, run in s["runs"].items():
            per: dict[tuple, dict] = {}
            for c in run.get("counters", []):
                d = per.setdefault((c["op"], c["route"], c["reason"] or "-"), {"eager": 0, "capture": 0})
                d[c["phase"]] = d.get(c["phase"], 0) + c["count"]
            for (op, route, reason), n in sorted(per.items()):
                cov.append([model, arch, name, op, route, reason, n["eager"], n["capture"]])
            a = run.get("accuracy") or {}
            if name != "base" or "error" in run:
                acc.append([model, arch, name, "ERROR" if "error" in run else run.get("verdict", "-"),
                            a.get("agreement"), ctrl, swap, s.get("threshold"),
                            a.get("positions"), f"{a.get('near_ties_lt_0.1_nats', '-')}/{a.get('disagreements', '-')}",
                            a.get("forced_logprob_abs_delta_mean"), a.get("nan"),
                            run.get("error", "")])
            st = run.get("stalls") or {}
            perf.append([model, arch, name, run.get("rel_tokens_per_s_vs_base"),
                         run.get("rel_tokens_per_s_vs_default"), run.get("rel_ttft_vs_base"),
                         run.get("rel_ttft_vs_default"), run.get("rel_init_vs_default"), st.get("plan_builds"),
                         st.get("plan_build_s"), st.get("norm_prebuild_s"), st.get("handle_create_s")])

    cov_header = ["model", "arch", "run", "op", "route", "reason", "eager calls", "capture calls"]
    with open(os.path.join(args.out, "coverage.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cov_header)
        w.writerows(cov)
    with open(os.path.join(args.out, "coverage.md"), "w") as f:
        f.write("# Coverage: model x arch x run x op x route x reason\n\n")
        f.write("Counts are host-side calls; capture calls are recorded once per captured graph, "
                "replays are not counted.\n\n")
        f.write(_md(cov, cov_header))
    with open(os.path.join(args.out, "accuracy.md"), "w") as f:
        f.write("# Accuracy vs plugin-off base (teacher-forced top-1; embeddings: min cosine)\n\n")
        f.write("PASS: the run routed at least one call of every enabled op family it exercised through hipDNN, "
                "no NaN, and agreement >= threshold = min(0.99, max(control, floor)); the floor keeps a noisy "
                "control from lowering the bar without limit, and a control below it fails the model. "
                "control: a second plugin-off run (run-to-run noise); its verdict is '-'. "
                "vllm_c: plugin off with vLLM's vllm_c norm kernels instead of native, "
                "a non-hipDNN kernel swap of the same ops, for scale. "
                "near ties: disagreements where base's top-1/top-2 margin < 0.1 nats. "
                "ERROR on base: the model has no accuracy result at all.\n\n")
        f.write(_md(acc, ["model", "arch", "run", "verdict", "agreement", "control", "vllm_c", "threshold",
                          "positions", "near ties/disagree", "mean abs dlogprob", "NaN", "error"]))
        if skipped:
            f.write("\n## Skipped\n\n" + _md(skipped, ["model", "arch", "reason"]))
        if failures:
            f.write("\n## Failures\n\n" + _md(failures, ["model", "arch", "failure"]))
    with open(os.path.join(args.out, "perf.md"), "w") as f:
        f.write("# Relative perf and stall time\n\n")
        f.write("Ratios are run/reference (tokens/s > 1 is faster; TTFT and start-up < 1 is faster). "
                "base = plugin off, TRITON_ATTN; default = plugin off, vLLM's default backend. "
                "TTFT is one request, median of warm calls. Start-up is compared with default rather than base: "
                "base runs first and fills the shared compile caches. "
                "Stall columns are seconds summed over the run's processes.\n\n")
        f.write(_md(perf, ["model", "arch", "run", "tok/s vs base", "tok/s vs default", "TTFT vs base",
                           "TTFT vs default", "start-up vs default", "plan builds", "plan build s",
                           "norm prebuild s", "handle s"]))
    json.dump({"coverage": cov, "accuracy": acc, "perf": perf, "skipped": skipped, "failures": failures,
               "sources": [s["_path"] for s in summaries]},
              open(os.path.join(args.out, "rows.json"), "w"), indent=1)
    print(f"{len(summaries)} summaries -> {args.out}")
    for name in ("accuracy.md", "perf.md"):
        print(open(os.path.join(args.out, name)).read())
    if not summaries:
        print(f"FAIL: no summary.json under {', '.join(args.dirs)}")
        return 1
    for model, arch, f in failures:
        print(f"FAIL: {model} on {arch}: {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
