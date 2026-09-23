#!/usr/bin/env python3
"""
probe_actor_critic.py - Run DEFAULT_PATIENT brute-force selection for every
actor-critic type/rank/layout combination and save the winning kernel names
to a JSON file.

Run this once per GPU ASIC that you want to cover.  Different ASICs support
different data-type combinations (e.g. F64 only on gfx90a/gfx942/gfx950), so
you will typically need at least two probe runs to fill all entries.

The output JSON is consumed by patch_actor_critic.py.

Usage
=====
    python3 scripts/update_actor_critic/probe_actor_critic.py \
        --build-dir build \
        --output    results_gfx90a.json

Options
=======
  --build-dir DIR   Directory containing compiled test binaries
                    (default: <repo_root>/build)
  --output FILE     Where to write results (default: actor_critic_probe.json
                    in the current directory)
  --ranks LIST      Comma-separated ranks to probe (default: 1,2,3,4,5,6)
  --timeout S       Per-binary timeout in seconds (default: 300)

JSON format
===========
{
  "<combo_key>": {
    "col_major": {
      "1": {"winner": "<kernel name>", "time_ms": 1.23},
      "2": {"winner": "<kernel name>", "time_ms": 0.98},
      ...
    },
    "row_major": { ... }
  },
  ...
}

A null "winner" means the binary was not found or no kernel succeeded for
that combination (type/rank pair not supported on this ASIC).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# The shared combo definitions live next to this script
sys.path.insert(0, str(Path(__file__).parent))
from _actor_critic_combos import (  # noqa: E402
    TYPE_COMBINATIONS,
    LAYOUTS,
    ALL_RANKS,
    make_probe_yaml,
    parse_best_kernel,
)


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------

def run_binary(binary: Path, yaml_path: str,
               extra_env: dict[str, str], timeout: int) -> str:
    """Run a test binary with the given YAML config and return combined output."""
    env = {**os.environ, **extra_env}
    cmd = [str(binary), "-y", yaml_path]
    print(f"    {' '.join(cmd)}", flush=True)
    try:
        r = subprocess.run(cmd, env=env, capture_output=True,
                           text=True, timeout=timeout)
        return r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        print(f"    WARNING: timed out after {timeout}s", file=sys.stderr)
        return ""
    except FileNotFoundError:
        # Binary simply doesn't exist in this build
        return ""


def probe_all(build_dir: Path, ranks: list[int],
              timeout: int) -> dict[str, dict]:
    """
    Returns a nested dict:
      results[combo_key][layout_name][rank] = {
          "winner": str | None,
          "time_ms": float | None,
      }
    """
    results: dict[str, dict] = {}

    with tempfile.TemporaryDirectory(prefix="hiptensor_probe_") as tmpdir:
        for combo in TYPE_COMBINATIONS:
            key = combo["key"]
            results[key] = {}
            print(f"\n=== {key} ===", flush=True)

            for lname, memory_layout in LAYOUTS:
                results[key][lname] = {}
                print(f"  [{lname}]", flush=True)

                for rank in ranks:
                    if combo["ranked"]:
                        binary = build_dir / "bin" / \
                            f"{combo['test_binary']}_m{rank}n{rank}k{rank}"
                    else:
                        binary = build_dir / "bin" / combo["test_binary"]

                    if not binary.exists():
                        print(f"    rank {rank}: binary not found — skipped",
                              flush=True)
                        results[key][lname][rank] = {
                            "winner": None, "time_ms": None}
                        continue

                    yaml_text = make_probe_yaml(combo, rank, memory_layout)
                    yaml_path = os.path.join(
                        tmpdir, f"{key}_{lname}_r{rank}.yaml")
                    Path(yaml_path).write_text(yaml_text)

                    output = run_binary(binary, yaml_path, {}, timeout)
                    winner, time_ms = parse_best_kernel(output)

                    if winner is None:
                        print(f"    rank {rank}: no winner — type/rank may "
                              f"not be supported on this ASIC", flush=True)
                    else:
                        print(f"    rank {rank}: {winner}  ({time_ms:.3f} ms)",
                              flush=True)

                    results[key][lname][rank] = {
                        "winner": winner, "time_ms": time_ms}

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Probe DEFAULT_PATIENT winners and save to JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--build-dir",
        default=str(repo_root / "build"),
        metavar="DIR",
        help="Directory containing compiled test binaries (default: %(default)s)",
    )
    parser.add_argument(
        "--output", "-o",
        default="actor_critic_probe.json",
        metavar="FILE",
        help="Output JSON file (default: %(default)s)",
    )
    parser.add_argument(
        "--ranks",
        default=",".join(str(r) for r in ALL_RANKS),
        metavar="LIST",
        help="Comma-separated ranks to probe (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        metavar="S",
        help="Per-binary timeout in seconds (default: %(default)s)",
    )
    args = parser.parse_args()

    build_dir = Path(args.build_dir)
    if not build_dir.exists():
        print(f"ERROR: build directory not found: {build_dir}", file=sys.stderr)
        sys.exit(1)

    ranks = [int(r.strip()) for r in args.ranks.split(",") if r.strip()]

    results = probe_all(build_dir, ranks, args.timeout)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")

    # Summary
    total = found = 0
    for combo_data in results.values():
        for layout_data in combo_data.values():
            for entry in layout_data.values():
                total += 1
                if entry["winner"] is not None:
                    found += 1
    print(f"Coverage: {found}/{total} (combo × layout × rank) slots filled")


if __name__ == "__main__":
    main()
