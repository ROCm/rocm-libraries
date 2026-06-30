# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# run_all.py -- example-repeatability driver for the rocke examples tree.
#
# The examples/ tree (~40k LOC of perf / parity drivers) is the place CK DSL
# proves its kernels build, launch, and verify on real hardware. Historically
# each driver was a one-off `python3 -m ...` you ran by hand and eyeballed.
# This driver makes a curated subset *repeatable* and *single-command*:
#
#   * a REGISTRY of example modules, each pinned to a fixed-seed / determi-
#     nistic argv so two runs produce the same observable result;
#   * a `--check` mode that runs each registered example, distills its output
#     to a stable DIGEST (return code + the numeric pass/fail markers it
#     prints -- max_abs_diff, bad=, PASS/FAIL -- which are seed-deterministic),
#     and asserts that digest against a golden captured by `--bless`;
#   * graceful SKIP-with-reason for examples that need a GPU we don't have,
#     missing data, or a non-matching arch -- the run stays green, the skip is
#     reported.
#
# This is deliberately a FRAMEWORK + a representative wired subset, not an
# attempt to make all 40k LOC repeatable in one shot. Registering another
# example is one `Example(...)` row (see REGISTRY.md).
#
# Determinism contract for a registered example:
#   1. It must accept a fixed argv (no wall-clock / hostname / pid in output
#      that leaks into the digest -- the digest filter drops timing lines).
#   2. Its inputs must be seeded (the rocke examples use np.random
#      .default_rng(0xC0FFEE) / torch.manual_seed -- already deterministic).
#   3. It must return 0 on success, nonzero on failure, and print its
#      numeric verdict (max_abs_diff / bad= / PASS|FAIL).
#
# Usage (GPU steps need sudo -E on this box; see numeric.py header):
#   # capture goldens (first time / after an intended change):
#   python3 -m rocke.examples.run_all --bless
#   # assert against goldens (CI / repeat):
#   python3 -m rocke.examples.run_all --check
#   # list what's registered:
#   python3 -m rocke.examples.run_all --list
#
# Goldens live next to this file under examples/_goldens/<name>.json so they
# travel with the tree; build artifacts / dashboards go to /tmp.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

HERE = Path(__file__).resolve().parent
GOLDEN_DIR = HERE / "_goldens"
PYROOT = HERE.parent.parent  # python/ (holds rocke)


# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Example:
    """One repeatable example invocation.

    ``module``  -- dotted module run via ``python -m`` (must have a ``main``
                   returning an int exit code, the rocke example convention).
    ``script``  -- optional absolute path run via ``python <script>`` instead of
                   ``-m module`` (for lanes that live outside the package, e.g.
                   the numeric differential lane under tests/). ``module`` is
                   then just the human label shown in the report.
    ``argv``    -- fixed deterministic arguments appended after the module.
    ``family``  -- coarse grouping for the report (gemm / elementwise / ...).
    ``needs_gpu`` -- if True, SKIP (not FAIL) when no GPU is visible.
    ``arch``    -- if set, SKIP when the live device's gfx arch != this.
    ``timeout`` -- per-example wall-clock guard (s).
    ``digest_keep`` -- extra regexes whose matching lines are kept in the
                   digest (on top of the default numeric-verdict patterns).
    """

    name: str
    module: str
    script: Optional[str] = None
    argv: Tuple[str, ...] = ()
    family: str = "misc"
    needs_gpu: bool = True
    arch: Optional[str] = None
    timeout: int = 300
    digest_keep: Tuple[str, ...] = ()


# The wired representative subset (proof the framework works across families).
# Each is a real rocke example with a deterministic, seeded verify path.
REGISTRY: List[Example] = [
    # elementwise (HIP path, numpy-seeded add) -- fast, no torch dependency
    Example(
        name="elementwise_add_hip",
        module="rocke.examples.common.elementwise_verify_hip",
        argv=("--arch", "gfx950", "--n", "4096"),
        family="elementwise",
        arch="gfx950",
        timeout=240,
    ),
    # universal GEMM (HIP path -> run_manifest verify), seeded manifest runner
    Example(
        name="universal_gemm_hip_512",
        module="rocke.examples.common.universal_gemm_verify_hip",
        argv=("--arch", "gfx950", "--m", "512", "--n", "512", "--k", "512"),
        family="gemm",
        arch="gfx950",
        timeout=360,
    ),
    # FMHA forward (HIP path), seeded numpy reference, MHA non-causal
    Example(
        name="fmha_fwd_hip_mha",
        module="rocke.examples.common.fmha_fwd_verify_hip",
        argv=(
            "--arch",
            "gfx950",
            "--seqlen-q",
            "64",
            "--seqlen-k",
            "64",
            "--head-size",
            "64",
            "--heads",
            "4",
            "--batch",
            "2",
        ),
        family="attention",
        arch="gfx950",
        timeout=300,
    ),
    # row reduction demo (torch, seeded), exact-match verdict
    Example(
        name="distribution_reduce_demo",
        module="rocke.examples.common.distribution_reduce_demo",
        argv=("--M", "128", "--N", "4096", "--block-size", "256", "--vec", "8"),
        family="reduce",
        arch="gfx950",
        timeout=240,
    ),
    # L6 NUMERIC differential lane itself, as a repeatable example: it already
    # prints a per-config PASS/DRIFT table that is fully seed-deterministic.
    Example(
        name="numeric_differential_lane",
        module="tests/instances/differential/numeric.py",
        script=str(
            PYROOT.parent / "tests" / "instances" / "differential" / "numeric.py"
        ),
        argv=("--arch", "gfx950"),
        family="differential",
        arch="gfx950",
        timeout=590,
        digest_keep=(
            r"^\s*(GREEN|DRIFT|XFAIL|REJECTED)\s",
            r"L6 NUMERIC SUMMARY",
            r"PASS=\d+",
        ),
    ),
]


# ---------------------------------------------------------------------
# Digest: distill an example's stdout to a stable, seed-deterministic blob
# ---------------------------------------------------------------------
# We do NOT digest the whole stdout (it carries hipcc/total timings, byte
# counts that vary with the compiler, device-name banners, tmp paths). We
# keep only the lines that encode the *numeric verdict*, which is what
# repeatability is actually about.
_DEFAULT_KEEP = (
    r"max_abs_diff",
    r"\bbad=\d+",
    r"\bmax_abs=",  # numeric.py style
    r"->\s*(PASS|FAIL)",
    r"\b(PASS|FAIL)\b",
    r"margin=",
    r"diff\s*=",
)
# Lines to always drop even if a keep-pattern matches (volatile fields).
_DROP = (
    r"hipcc=\d",
    r"total=\d",
    r"\d+\s*ms",
    r"\bB,\s*isa=",
    r"/tmp/",
)
# Volatile substrings to scrub *within* a kept line (timings embedded in an
# otherwise-stable verdict line).
_SCRUB = [
    (re.compile(r"\(\d+ B[^)]*\)"), "(<bytes>)"),
    (re.compile(r"hipcc=[0-9.]+ms"), "hipcc=<t>"),
    (re.compile(r"total=[0-9.]+ms"), "total=<t>"),
    (re.compile(r"[0-9]+ms"), "<t>ms"),
    (re.compile(r"device=[^\n]+"), "device=<dev>"),
    # PerfJSON / perf lines carry wall-clock fields (ms/tflops/gbps/us/
    # bandwidth) that vary run-to-run; scrub the value so only the numeric
    # *correctness* fields (max_abs_diff / bad_count) drive the digest.
    (
        re.compile(
            r'("(?:ms|tflops|gbps|us|gflops|bw|bandwidth|sec|seconds|'
            r'time|elapsed)"\s*:\s*)[-0-9.eE+]+'
        ),
        r"\1<t>",
    ),
    (re.compile(r"\b(ms|tflops|gbps|us|gflops)\s*=\s*[-0-9.eE+]+"), r"\1=<t>"),
]


def _distill(stdout: str, keep: Tuple[str, ...]) -> List[str]:
    keeps = [re.compile(p) for p in (_DEFAULT_KEEP + keep)]
    drops = [re.compile(p) for p in _DROP]
    out: List[str] = []
    for raw in stdout.splitlines():
        line = raw.rstrip()
        if not any(k.search(line) for k in keeps):
            continue
        if any(d.search(line) for d in drops):
            # Don't drop a verdict line just because it has a byte/ms count;
            # scrub instead so the verdict survives.
            if not re.search(r"(PASS|FAIL|max_abs|margin|bad=)", line):
                continue
        for rx, repl in _SCRUB:
            line = rx.sub(repl, line)
        out.append(line)
    return out


def _digest_of(lines: List[str]) -> str:
    h = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    return h


# ---------------------------------------------------------------------
# GPU / arch probe
# ---------------------------------------------------------------------
def _live_arch() -> Optional[str]:
    """Return the live device's gfx arch, or None if no GPU is visible."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    if not torch.cuda.is_available():
        return None
    try:
        name = torch.cuda.get_device_properties(0).gcnArchName  # e.g. "gfx950:..."
        return name.split(":")[0]
    except Exception:  # noqa: BLE001
        # Fall back to "some GPU present" -- arch-gated examples will still skip
        # if they can't confirm, but un-gated ones can run.
        return "unknown"


# ---------------------------------------------------------------------
# Run one example
# ---------------------------------------------------------------------
@dataclass
class RunResult:
    name: str
    family: str
    status: str  # PASS | FAIL | DRIFT | SKIPPED | ERROR
    rc: int = 0
    digest: str = ""
    golden: str = ""
    reason: str = ""
    lines: List[str] = field(default_factory=list)


def _golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.json"


def run_one(ex: Example, *, bless: bool, live_arch: Optional[str]) -> RunResult:
    r = RunResult(name=ex.name, family=ex.family, status="PASS")

    # Skip gates --------------------------------------------------------
    if ex.needs_gpu and live_arch is None:
        r.status = "SKIPPED"
        r.reason = "no GPU visible (run under sudo -E, see numeric.py header)"
        return r
    if ex.arch and live_arch not in (None, "unknown", ex.arch):
        r.status = "SKIPPED"
        r.reason = f"example pinned to {ex.arch}, live device is {live_arch}"
        return r

    # Run --------------------------------------------------------------
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(PYROOT))
    env.setdefault("TMPDIR", "/tmp")
    # Force determinism knobs that some torch kernels honor.
    env.setdefault("PYTHONHASHSEED", "0")
    if ex.script:
        cmd = [sys.executable, ex.script, *ex.argv]
    else:
        cmd = [sys.executable, "-m", ex.module, *ex.argv]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=ex.timeout, env=env
        )
    except subprocess.TimeoutExpired:
        r.status = "ERROR"
        r.reason = f"timeout after {ex.timeout}s"
        return r
    except Exception as e:  # noqa: BLE001
        r.status = "ERROR"
        r.reason = f"spawn failed: {e}"
        return r

    r.rc = proc.returncode
    combined = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    r.lines = _distill(combined, ex.digest_keep)
    r.digest = _digest_of(r.lines)

    # A nonzero rc means the example's own verify failed -> a real FAIL,
    # independent of golden comparison.
    if proc.returncode != 0:
        # rc==2 in the rocke examples means "validate rejected (unsupported
        # spec on this arch)" -- treat as SKIP, not FAIL.
        if proc.returncode == 2:
            r.status = "SKIPPED"
            r.reason = "example validate rejected spec on this arch (rc=2)"
            return r
        r.status = "FAIL"
        r.reason = f"example exited rc={proc.returncode}"
        if not r.lines:
            r.reason += f"; tail: {combined.strip().splitlines()[-3:]}"
        return r

    # Golden comparison ------------------------------------------------
    gpath = _golden_path(ex.name)
    if bless:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        gpath.write_text(
            json.dumps(
                {
                    "name": ex.name,
                    "module": ex.module,
                    "argv": list(ex.argv),
                    "digest": r.digest,
                    "lines": r.lines,
                },
                indent=2,
            )
        )
        r.status = "PASS"
        r.reason = "blessed"
        return r

    if not gpath.exists():
        r.status = "ERROR"
        r.reason = "no golden -- run with --bless first"
        return r
    golden = json.loads(gpath.read_text())
    r.golden = golden.get("digest", "")
    if r.digest == r.golden:
        r.status = "PASS"
    else:
        r.status = "DRIFT"
        # show a short diff of the verdict lines for the report
        gl = golden.get("lines", [])
        diff_lines = [f"- {x}" for x in gl if x not in r.lines]
        diff_lines += [f"+ {x}" for x in r.lines if x not in gl]
        r.reason = "digest != golden:\n      " + "\n      ".join(diff_lines[:12])
    return r


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument(
        "--check",
        action="store_true",
        help="run examples and assert digests vs goldens (default)",
    )
    g.add_argument(
        "--bless", action="store_true", help="run examples and (re)capture goldens"
    )
    g.add_argument("--list", action="store_true", help="list the registry and exit")
    ap.add_argument(
        "--only", default="", help="comma-separated name/family substrings to filter"
    )
    ap.add_argument(
        "--json",
        default=os.path.join(
            os.environ.get("TMPDIR", "/tmp"),
            f"ck_examples_run_{os.getuid()}.json",
        ),
    )
    args = ap.parse_args(argv)

    if args.list:
        print(f"{'name':32s} {'family':12s} {'arch':8s} module")
        for ex in REGISTRY:
            print(
                f"{ex.name:32s} {ex.family:12s} "
                f"{(ex.arch or '-'):8s} {ex.module} {' '.join(ex.argv)}"
            )
        return 0

    subs = [s for s in args.only.split(",") if s]

    def want(ex: Example) -> bool:
        return not subs or any(s in ex.name or s in ex.family for s in subs)

    live = _live_arch()
    mode = "BLESS" if args.bless else "CHECK"
    print(f"rocke examples {mode}  live_arch={live}  registry={len(REGISTRY)}")

    results: List[RunResult] = []
    for ex in REGISTRY:
        if not want(ex):
            continue
        print(f"  running {ex.name} ...", flush=True)
        results.append(run_one(ex, bless=args.bless, live_arch=live))

    npass = nfail = ndrift = nskip = nerr = 0
    rows = []
    for r in results:
        rows.append(
            {
                "name": r.name,
                "family": r.family,
                "status": r.status,
                "rc": r.rc,
                "digest": r.digest,
                "golden": r.golden,
                "reason": r.reason,
            }
        )
        if r.status == "PASS":
            npass += 1
        elif r.status == "FAIL":
            nfail += 1
        elif r.status == "DRIFT":
            ndrift += 1
        elif r.status == "SKIPPED":
            nskip += 1
        else:
            nerr += 1
        line = f"  {r.status:8s} {r.family}/{r.name}"
        if r.digest:
            line += f"  digest={r.digest[:12]}"
        if r.reason:
            line += f"  -- {r.reason}"
        print(line)

    try:
        Path(args.json).write_text(json.dumps(rows, indent=2))
    except OSError as e:  # noqa: BLE001 - dashboard is best-effort
        print(f"  (could not write dashboard {args.json}: {e})")
    print(f"\n=== EXAMPLES {mode} SUMMARY ===")
    print(f"  PASS={npass} FAIL={nfail} DRIFT={ndrift} SKIPPED={nskip} ERROR={nerr}")
    print(f"  goldens: {GOLDEN_DIR}")
    print(f"  dashboard: {args.json}")
    # Nonzero only on a real failure/drift/error; skips keep the run green.
    return 1 if (nfail or ndrift or nerr) else 0


if __name__ == "__main__":
    raise SystemExit(main())
