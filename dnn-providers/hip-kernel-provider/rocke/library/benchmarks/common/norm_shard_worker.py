#!/usr/bin/env python
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""One shard's worker process for ``run_norm_shards.py``.

Runs sequentially, pinned to one GPU by the parent's ``HIP_VISIBLE_DEVICES``
(same convention as ``run_conv_shards.py``'s shards). Each line of ``--cases``
is one JSON object (a serialized
:class:`~benchmarks.common.parse_norm_miopen.ParsedCase`): this process shells
out to ``benchmark_batchnorm2d_aot.py``/``benchmark_layernorm2d_aot.py`` once
per case (a fresh process per shape, not an in-process loop over hundreds of
shapes -- see ``run_norm_shards.py``'s module docstring for why), then merges
that case's rows into this shard's own ``--results`` file.

Numeric validation (``--validate``, forwarded to the AOT script) only ever
checks rocke's own output against a torch reference -- see
``_validate_batchnorm``/``_validate_norm2d`` in the two AOT scripts. MIOpen is
not re-validated here: it is the trusted baseline, so this worker only times
it, in find mode (``MIOPEN_FIND_MODE=1 MIOPEN_FIND_ENFORCE=4``, i.e. a forced
database re-search rather than ``-S 0``'s fixed solver) -- the same env
``compare_norm_vs_miopen.py`` already uses, generalized here from that
script's three hardcoded cases to every case this worker runs. Best-effort:
skipped with a one-line notice if no ``MIOpenDriver`` is on ``PATH``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from benchmarks.common.norm_bench_common import case_key

_HERE = Path(__file__).resolve().parent

_SCRIPTS = {
    "bnorm": _HERE / "benchmark_batchnorm2d_aot.py",
    "lnorm": _HERE / "benchmark_layernorm2d_aot.py",
}

_MIOPEN_ENV_EXTRA = {"MIOPEN_FIND_MODE": "1", "MIOPEN_FIND_ENFORCE": "4"}
_MIOPEN_BN_RE = re.compile(
    r"GPU Kernel Min Time Forward Batch Normalization Elapsed:\s*([0-9.]+)\s*ms"
)
_MIOPEN_LN_RE = re.compile(
    r"GPU Kernel Time Forward LayerNorm Elapsed:\s*([0-9.]+)\s*ms"
)
_MIOPEN_BIN: Optional[str] = None
_MIOPEN_WARNED = False


def _miopen_binary() -> Optional[str]:
    global _MIOPEN_BIN, _MIOPEN_WARNED
    if _MIOPEN_BIN is None:
        _MIOPEN_BIN = shutil.which("MIOpenDriver") or ""
        if not _MIOPEN_BIN and not _MIOPEN_WARNED:
            _MIOPEN_WARNED = True
            print("[info] no MIOpenDriver on PATH; skipping MIOpen timing", flush=True)
    return _MIOPEN_BIN or None


def rocke_case_key(case: dict) -> str:
    """The exact key the matching AOT script computes for *case*.

    Must match ``benchmark_batchnorm2d_aot.py``'s/``benchmark_layernorm2d_aot.py``'s
    own ``case_key(...)`` call so this worker's MIOpen summary row lines up
    with that case's per-candidate rocke rows under the same ``key``.
    """
    dims = dict(case["dims"])
    if case["family"] == "bnorm":
        return case_key(dims, dtype=case["dtype"], family="batchnorm2d",
                         extra=f"training={case['mode'] == 'training'}")
    return case_key(dims, dtype=case["dtype"], family="norm2d", extra="kind=layernorm")


def run_miopen_case(case: dict, *, timeout: int = 120) -> Optional[float]:
    """MIOpenDriver find-mode GPU kernel time for *case*, in ms; ``None`` if unavailable."""
    binary = _miopen_binary()
    if binary is None:
        return None
    dims = dict(case["dims"])
    dtype_token = "fp16" if case["dtype"] == "fp16" else "bfp16"
    if case["family"] == "bnorm":
        forw = "2" if case["mode"] == "inference" else "1"
        save = "1" if case["mode"] == "training" else "0"
        argv = [
            binary, f"bnorm{dtype_token}", "-F", forw,
            "-n", str(dims["n"]), "-c", str(dims["c"]),
            "-H", str(dims["h"]), "-W", str(dims["w"]),
            "-m", "1", "-r", "1", "-s", save,
            "-t", "1", "-w", "1", "-V", "0", "-i", "20",
        ]
        pattern = _MIOPEN_BN_RE
    else:
        argv = [
            binary, f"layernorm{dtype_token}", "-F", "1",
            "-X", f"{dims['rows']}x{dims['cols']}",
            "-o", "1", "-t", "1", "-w", "1", "-V", "0", "-i", "20",
        ]
        pattern = _MIOPEN_LN_RE

    import os

    env = dict(os.environ)
    env.update(_MIOPEN_ENV_EXTRA)
    try:
        proc = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None
    m = pattern.search(proc.stdout + "\n" + proc.stderr)
    return float(m.group(1)) if m else None


def _read_cases(path) -> list:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def case_argv(case: dict, *, arch: str, dtype: str) -> list:
    """The AOT script's own flags for one case -- everything but ``--results``."""
    if case["family"] == "bnorm":
        dims = dict(case["dims"])
        return [
            "--n", str(dims["n"]), "--c", str(dims["c"]),
            "--h", str(dims["h"]), "--w", str(dims["w"]),
            "--training" if case["mode"] == "training" else "--inference",
            "--dtype", dtype, "--arch", arch,
        ]
    dims = dict(case["dims"])
    return [
        "--rows", str(dims["rows"]), "--cols", str(dims["cols"]),
        "--kind", "layernorm", "--dtype", dtype, "--arch", arch,
    ]


def case_label(case: dict) -> str:
    dims = ",".join(f"{k}={v}" for k, v in sorted(dict(case["dims"]).items()))
    return f"{case['family']}/{case['dtype']}/{case['mode']}/{dims}"


def run_one_case(case: dict, *, script: Path, arch: str, forwarded: Sequence[str]) -> dict:
    """Run one case in its own subprocess; return ``{rows, rc, stderr_tail}``."""
    with tempfile.TemporaryDirectory(prefix="rocke-norm-case-") as tmp:
        tmp_results = Path(tmp) / "case.jsonl"
        argv = [
            sys.executable, str(script),
            *case_argv(case, arch=arch, dtype=case["dtype"]),
            "--results", str(tmp_results),
            *forwarded,
        ]
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=600)
        rows = []
        if tmp_results.is_file():
            for line in tmp_results.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return {
            "rows": rows,
            "rc": proc.returncode,
            "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
        }


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", choices=sorted(_SCRIPTS), required=True)
    ap.add_argument("--cases", required=True, metavar="FILE",
                    help="JSON Lines of ParsedCase dicts, this shard's share")
    ap.add_argument("--results", required=True, metavar="FILE",
                    help="JSON Lines this shard appends one row per candidate to")
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("forwarded", nargs=argparse.REMAINDER,
                    help="forwarded verbatim to the AOT script, e.g. --fast-rank")
    return ap


def main(argv=None) -> int:
    args = make_parser().parse_args(argv if argv is not None else sys.argv[1:])
    forwarded = list(args.forwarded)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]

    cases = _read_cases(args.cases)
    script = _SCRIPTS[args.family]

    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    rc_all = 0
    with results_path.open("w", encoding="utf-8") as out_fh:
        for i, case in enumerate(cases):
            label = case_label(case)
            print(f"[{i + 1}/{len(cases)}] {label}", flush=True)
            outcome = run_one_case(case, script=script, arch=args.arch, forwarded=forwarded)
            if outcome["rc"] != 0:
                rc_all = rc_all or outcome["rc"]
                print(f"  FAIL rc={outcome['rc']}\n{outcome['stderr_tail']}", flush=True)
                continue
            best = None
            for row in outcome["rows"]:
                out_fh.write(json.dumps(row, sort_keys=True) + "\n")
                ms = row.get("ms")
                if ms is not None and (best is None or ms < best):
                    best = ms

            miopen_ms = run_miopen_case(case)
            if miopen_ms is not None:
                summary = {
                    "key": rocke_case_key(case),
                    "miopen_ms": miopen_ms,
                    "rocke_best_ms": best,
                    "speedup": (miopen_ms / best) if best else None,
                }
                out_fh.write(json.dumps(summary, sort_keys=True) + "\n")

            out_fh.flush()
            print(
                f"  {len(outcome['rows'])} row(s), best={best}, miopen={miopen_ms}",
                flush=True,
            )
    return rc_all


if __name__ == "__main__":
    raise SystemExit(main())
