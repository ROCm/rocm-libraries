#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# check_batch_parity.py -- guard for the differential harness's fast path.
#
# run_diff.py emits the Python reference once per family in batch mode instead of
# spawning a fresh interpreter per config. This guard proves that optimization is
# a pure speedup: it runs run_diff twice against the SAME engine archive --
# once with --isolated (the original per-config interpreter path) and once in the
# default batch path -- and asserts the two dashboards are identical at every
# (family, idx) for verdict, shas, ref_sha, return codes, and family status.
#
# Any divergence means the batch protocol is not byte-faithful to the isolated
# reference and the byte-identity gate can no longer be trusted; this exits
# nonzero so CI fails loudly.
#
# Usage:
#   python check_batch_parity.py [--mode ll|ir|verify] [--only SUBSTR] [--all]
#       [--archive A] [--build-root DIR]
#
# By default it checks a small, representative subset of families (fast enough
# for a PR gate); --all checks every family (slower, for nightly/full audits).

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROCKE = HERE.parents[2]  # differential -> instances -> tests -> rocKE (platform)
RUN_DIFF = HERE / "run_diff.py"

# A representative default subset: elementwise exercises arch-tuple configs and a
# mid-range end-of-range; gemm and layernorm2d cover the heavier lowering paths;
# reduce is a small distinct family. Enough surface to catch a framing/protocol
# regression without paying for all ~65 families on every PR.
DEFAULT_SUBSET = "elementwise,gemm,layernorm2d,reduce"


def _ensure_archive(archive: Path, build_root: Path) -> Path:
    if archive.exists():
        return archive
    print(f"== building engine archive (none at {archive}) ==")
    subprocess.run(
        ["cmake", "-S", str(ROCKE), "-B", str(build_root), "-DCMAKE_BUILD_TYPE=Release"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        ["cmake", "--build", str(build_root), "--target", "rocke_core", "-j"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    if not archive.exists():
        sys.exit(f"FATAL: archive not produced at {archive}")
    return archive


def _run(mode: str, only: str, archive: Path, isolated: bool, out_json: Path) -> None:
    cmd = [
        sys.executable,
        str(RUN_DIFF),
        "--mode",
        mode,
        "--archive",
        str(archive),
        "--json",
        str(out_json),
    ]
    if only:
        cmd += ["--only", only]
    if isolated:
        cmd.append("--isolated")
    # run_diff exits nonzero on drift; that is orthogonal to batch/isolated
    # agreement (both would see the same drift), so we do not check its rc here --
    # we compare the two dashboards it produced.
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _normalize(dashboard_path: Path) -> dict:
    """Reduce a dashboard to the comparable per-(family, idx) facts."""
    results = json.loads(dashboard_path.read_text())
    out = {}
    for r in results:
        out[r["family"]] = {
            "status": r["status"],
            "n": r.get("n"),
            "range_drift": r.get("range_drift"),
            "configs": {
                c["idx"]: (
                    c["verdict"],
                    tuple(c["shas"]) if c["shas"] else None,
                    c["ref_sha"],
                    c["c_rc"],
                    c["p_rc"],
                )
                for c in r["configs"]
            },
        }
    return out


def _diff(iso: dict, bat: dict) -> list[str]:
    problems = []
    if set(iso) != set(bat):
        problems.append(f"family set differs: {sorted(set(iso) ^ set(bat))}")
        return problems
    for fam in sorted(iso):
        if iso[fam] == bat[fam]:
            continue
        if iso[fam]["status"] != bat[fam]["status"]:
            problems.append(
                f"{fam}: status {iso[fam]['status']} -> {bat[fam]['status']}"
            )
        ca, cb = iso[fam]["configs"], bat[fam]["configs"]
        if set(ca) != set(cb):
            problems.append(f"{fam}: config idx set differs: {sorted(set(ca) ^ set(cb))}")
        for i in sorted(set(ca) & set(cb)):
            if ca[i] != cb[i]:
                problems.append(f"{fam}[{i}]: {ca[i]} -> {cb[i]}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="batch-vs-isolated parity guard")
    ap.add_argument("--mode", default="ll", choices=["ll", "ir", "verify"])
    ap.add_argument("--only", default="", help="family substrings (overrides subset)")
    ap.add_argument("--all", action="store_true", help="check every family")
    ap.add_argument(
        "--archive",
        default=str(Path(tempfile.gettempdir()) / "rocke_verify" / "librocke_core.a"),
    )
    ap.add_argument(
        "--build-root", default=str(Path(tempfile.gettempdir()) / "rocke_verify")
    )
    args = ap.parse_args()

    only = "" if args.all else (args.only or DEFAULT_SUBSET)
    archive = _ensure_archive(Path(args.archive), Path(args.build_root))

    with tempfile.TemporaryDirectory() as td:
        iso_json = Path(td) / "isolated.json"
        bat_json = Path(td) / "batch.json"
        scope = "all families" if args.all else f"families~[{only}]"
        print(f"== batch-parity guard: mode={args.mode} {scope} ==")
        _run(args.mode, only, archive, isolated=True, out_json=iso_json)
        _run(args.mode, only, archive, isolated=False, out_json=bat_json)
        iso, bat = _normalize(iso_json), _normalize(bat_json)

    if not iso:
        sys.exit("no families matched -- nothing checked")

    problems = _diff(iso, bat)
    if problems:
        print(f"\nFAIL: batch path diverges from isolated in {len(problems)} place(s):")
        for p in problems[:40]:
            print(f"  {p}")
        return 1
    print(f"\nPASS: batch == isolated across {len(iso)} families (byte-identical).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
