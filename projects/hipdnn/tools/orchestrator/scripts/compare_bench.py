#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Decide whether one engine hit its speed target against another.

The orchestrator's condition grammar can compare two numbers but cannot divide them,
which is deliberate -- a ratio computed inside a config file is a ratio nobody can
test. So the arithmetic lives here, in a step, and the loop's exit condition reads the
single integer this writes.

    compare_bench.py --new bench_test.json --ref bench_ref.json \\
        --max-ratio 0.80 --out compare.json

Both inputs are `hipdnn_graph_bench --out` files. `--max-ratio` is the largest
acceptable `new_median / ref_median`, and it reads in either direction:

    0.80  the new engine must be at least 20% FASTER than the reference
    1.00  it must at least match it
    1.15  it may be up to 15% slower

The comparison is on the median. The mean of a 50-sample latency series on a shared
desktop GPU is dragged around by whichever sample caught a scheduler hiccup, and the
max is that hiccup.

The two files must describe the same graph, the same iteration count and the same
timing method, and must name different engines. Any of those being false makes the
ratio meaningless, so it is an error (exit 1) rather than a verdict -- comparing a
50-iteration run against a 5-iteration one produces a number, and that number is the
bug.

Exit codes: 0 comparison written (pass or fail -- the verdict is in the JSON), 1 the
two files cannot be compared.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Every key `hipdnn_graph_bench` writes that this tool reads. Checked up front so a
#: renamed field in the benchmark fails here with the field name rather than as a
#: KeyError three lines into the arithmetic.
REQUIRED_KEYS = (
    "engine",
    "graph",
    "iterations",
    "timing_method",
    "median_ms",
    "samples_ms",
)


def _load(path: str, role: str) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: could not read {role} benchmark {path}: {error}")
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise SystemExit(
            f"error: {role} benchmark {path} is missing {', '.join(missing)}"
        )
    return data


def _goal_text(max_ratio: float) -> str:
    """The target as a human sentence fragment, in whichever direction it points."""
    if max_ratio < 1.0:
        return f"at least {(1.0 - max_ratio) * 100:.0f}% faster than"
    if max_ratio > 1.0:
        return f"no more than {(max_ratio - 1.0) * 100:.0f}% slower than"
    return "at least as fast as"


def _standing_text(ratio: float) -> str:
    """Where the new engine actually landed, phrased the way a reader expects."""
    if ratio < 1.0:
        return f"{(1.0 - ratio) * 100:.1f}% faster"
    if ratio > 1.0:
        return f"{(ratio - 1.0) * 100:.1f}% slower"
    return "exactly level"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--new", required=True, help="benchmark JSON for the engine under test"
    )
    parser.add_argument(
        "--ref", required=True, help="benchmark JSON for the reference engine"
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=0.80,
        help="largest acceptable new/ref median ratio; 0.80 = must be 20%% faster (default)",
    )
    parser.add_argument(
        "--out", required=True, help="where to write the comparison JSON"
    )
    args = parser.parse_args()

    if args.max_ratio <= 0:
        print("error: --max-ratio must be > 0", file=sys.stderr)
        return 1

    new = _load(args.new, "new")
    ref = _load(args.ref, "ref")

    for field, label in (
        ("graph", "graph"),
        ("iterations", "iteration count"),
        ("timing_method", "timing method"),
    ):
        if new[field] != ref[field]:
            print(
                f"error: the two runs disagree on {label}: "
                f"{new['engine']}={new[field]!r} vs {ref['engine']}={ref[field]!r}",
                file=sys.stderr,
            )
            return 1
    if new["engine"] == ref["engine"]:
        print(
            f"error: both benchmarks are {new['engine']}; there is nothing to compare",
            file=sys.stderr,
        )
        return 1
    if (
        len(new["samples_ms"]) != new["iterations"]
        or len(ref["samples_ms"]) != ref["iterations"]
    ):
        print(
            "error: a benchmark reported fewer samples than iterations", file=sys.stderr
        )
        return 1
    if ref["median_ms"] <= 0.0:
        print(
            f"error: reference engine {ref['engine']} reported a non-positive median "
            f"({ref['median_ms']}); nothing was measured",
            file=sys.stderr,
        )
        return 1

    ratio = new["median_ms"] / ref["median_ms"]
    meets = ratio <= args.max_ratio
    budget_ms = args.max_ratio * ref["median_ms"]
    # How much wall clock still has to come off. Reported because "you are at ratio
    # 1.005 and need 0.800" is a number an optimiser cannot act on, while "cut
    # 1.07 ms off a 5.36 ms kernel" is.
    gap_ms = new["median_ms"] - budget_ms

    if meets:
        feedback = (
            f"Speed target met: {new['engine']} median {new['median_ms']:.4f} ms vs "
            f"{ref['engine']} median {ref['median_ms']:.4f} ms -- "
            f"{_standing_text(ratio)} (ratio {ratio:.3f} <= {args.max_ratio:.3f})."
        )
    else:
        feedback = (
            f"## Performance gate failed\n\n"
            f"`{new['engine']}` must be **{_goal_text(args.max_ratio)}** `{ref['engine']}` on "
            f"`{new['graph']}` over {new['iterations']} iterations. It is currently "
            f"**{_standing_text(ratio)}**.\n\n"
            f"| statistic | {new['engine']} | {ref['engine']} |\n"
            f"|---|---|---|\n"
            f"| median ms | {new['median_ms']:.4f} | {ref['median_ms']:.4f} |\n"
            f"| min ms | {new.get('min_ms', float('nan')):.4f} | "
            f"{ref.get('min_ms', float('nan')):.4f} |\n"
            f"| p90 ms | {new.get('p90_ms', float('nan')):.4f} | "
            f"{ref.get('p90_ms', float('nan')):.4f} |\n"
            f"| stddev ms | {new.get('stddev_ms', float('nan')):.4f} | "
            f"{ref.get('stddev_ms', float('nan')):.4f} |\n\n"
            f"Budget: the median must be at or under **{budget_ms:.4f} ms** "
            f"(ratio {args.max_ratio:.3f}). You are **{gap_ms:.4f} ms** over it.\n\n"
            f"Every integration case passed, so this round is a tuning problem and not a "
            f"correctness one: do not change what the kernel computes.\n\n"
            f"Note what this target rules out. A kernel that is a copy of "
            f"`{ref['engine']}`'s, or a port of the same algorithm with the same access "
            f"pattern, lands at ratio ~1.0 by construction and can never pass. Beating the "
            f"reference means doing something it does not do -- a wider vectorised access, "
            f"a different workgroup shape or tiling, fewer passes over the data, keeping "
            f"per-channel parameters in registers or LDS across the spatial loop, avoiding "
            f"a separate normalisation pass, or exploiting a layout the generic path "
            f"handles conservatively. Read the reference to learn what the work *is*, then "
            f"find the thing it leaves on the table on {new.get('graph', 'this shape')}."
        )

    report = {
        "meets_target": 1 if meets else 0,
        "ratio": ratio,
        "max_ratio": args.max_ratio,
        "budget_ms": budget_ms,
        "gap_ms": gap_ms,
        "speedup_percent": (1.0 - ratio) * 100.0,
        "goal": _goal_text(args.max_ratio),
        "standing": _standing_text(ratio),
        "metric": "median_ms",
        "graph": new["graph"],
        "iterations": new["iterations"],
        "new_engine": new["engine"],
        "ref_engine": ref["engine"],
        "new_median_ms": new["median_ms"],
        "ref_median_ms": ref["median_ms"],
        "new": new,
        "ref": ref,
        "feedback": feedback,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"{new['engine']} median {new['median_ms']:.4f} ms vs "
        f"{ref['engine']} median {ref['median_ms']:.4f} ms -> ratio {ratio:.3f} "
        f"({_standing_text(ratio)}); target {args.max_ratio:.3f} "
        f"({_goal_text(args.max_ratio)} the reference): {'PASS' if meets else 'FAIL'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
