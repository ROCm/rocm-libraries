# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Forward-conv performance comparison: rocKE vs external references.

Runs a corpus of convolution problems (grouped or not) through the rocKE
**dispatcher-selected** kernel and one or more external references, and reports
per-row throughput deltas. This is the AICK-1752 acceptance evidence step:
measure pinned MIOpen / CK-Tile / old-CK deltas for the assigned rows and hand
the selected configurations + numbers to the paired performance owner (CF5P).

Corpus format
-------------
The corpus is a list of ``MIOpenDriver`` command strings (one per line via
``--miopen-file``, or a single ``--miopen-cmd``). Using MIOpenDriver commands as
the shape source lets the *same* line drive both rocKE (parsed via
:func:`rocke.benchmark.benchmark_implicit_gemm_conv.parse_miopen_cmd`) and the
MIOpen reference (executed directly), so a row is defined once.

References
----------
* **MIOpen** — executed via the ``MIOpenDriver`` binary (``--miopen-bin``,
  default ``MIOpenDriver`` on ``PATH``); timed with ``-t 1``. Always attempted
  when the binary is found.
* **CK-Tile / old-CK** — executed via a user-supplied command *template*
  (``--ck-cmd-template`` / ``--oldck-cmd-template``) because this repo ships no
  prebuilt grouped-conv reference binary. The template may reference
  ``{n} {c} {hi} {wi} {k} {y} {x} {ph} {pw} {sh} {sw} {dh} {dw} {groups}
  {dtype}``; the runner formats it, runs it, and parses the elapsed time via
  ``--ext-time-regex`` (default matches ``... <float> ms``). Left as N/A when no
  template is given.

Throughput uses a single consistent FLOP count (``ConvProblem.flops``, which
matches MIOpen's reported ``flopCnt``), so ``TFLOP/s = flops / (time_ms * 1e9)``
for every engine and the speedup is simply ``ref_ms / rocke_ms``.

Example
-------
    python -m rocke.benchmark.conv_perf_compare --arch gfx950 \
        --miopen-file corpus_grouped_bf16.txt --out deltas.json

where ``corpus_grouped_bf16.txt`` contains lines such as::

    MIOpenDriver convbfp16 -n 2 -c 256 -H 8 -W 8 -k 256 -y 3 -x 3 \
        -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 32 -F 1
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional

_DTYPE_TO_MIOPEN = {"fp16": "convfp16", "bf16": "convbfp16", "fp32": "conv"}


@dataclass
class RowResult:
    short: str
    dtype: str
    groups: int
    flops: int
    rocke_spec_id: str = "n/a"
    rocke_ms: Optional[float] = None
    miopen_ms: Optional[float] = None
    ck_ms: Optional[float] = None
    oldck_ms: Optional[float] = None
    note: str = ""

    def _tflops(self, ms: Optional[float]) -> Optional[float]:
        if ms is None or ms <= 0.0:
            return None
        return self.flops / (ms * 1e9)

    def as_dict(self) -> dict:
        d = {
            "short": self.short,
            "dtype": self.dtype,
            "groups": self.groups,
            "flops": self.flops,
            "rocke_spec_id": self.rocke_spec_id,
            "rocke_ms": self.rocke_ms,
            "rocke_tflops": self._tflops(self.rocke_ms),
            "miopen_ms": self.miopen_ms,
            "miopen_tflops": self._tflops(self.miopen_ms),
            "ck_ms": self.ck_ms,
            "ck_tflops": self._tflops(self.ck_ms),
            "oldck_ms": self.oldck_ms,
            "oldck_tflops": self._tflops(self.oldck_ms),
            "note": self.note,
        }
        for ref in ("miopen", "ck", "oldck"):
            ref_ms = getattr(self, f"{ref}_ms")
            d[f"speedup_vs_{ref}"] = (
                ref_ms / self.rocke_ms
                if (ref_ms and self.rocke_ms and self.rocke_ms > 0.0)
                else None
            )
        return d


# ---------------------------------------------------------------------
# rocKE (dispatcher-selected config), timed on the live GPU
# ---------------------------------------------------------------------


def _time_rocke(problem, dtype: str, arch: str, warmup: int, iters: int):
    """Return ``(spec_id, avg_ms)`` for the dispatcher-selected rocKE kernel."""
    from ..dispatch.families.conv import ConvRequest, dispatch_conv
    from ..helpers.compile import compile_kernel
    from ..instances.common.conv_implicit_gemm import build_implicit_gemm_conv
    from ..instances.common.manifest_runner.conv import run_conv_manifest_problem
    from ..runtime.hip_module import Runtime

    try:
        from ..runtime.launcher import time_launches
    except Exception:  # pragma: no cover - torch-optional
        time_launches = None

    p = problem
    req = ConvRequest(
        N=p.N,
        C=p.C,
        K=p.K,
        Hi=p.Hi,
        Wi=p.Wi,
        Y=p.Y,
        X=p.X,
        G=p.groups,
        stride_h=p.sH,
        stride_w=p.sW,
        pad_h=p.pH,
        pad_w=p.pW,
        dilation_h=p.dH,
        dilation_w=p.dW,
        arch=arch,
        dtype=dtype,
    )
    res = dispatch_conv(req)
    spec = res.spec
    kernel = build_implicit_gemm_conv(spec, arch=arch)
    art = compile_kernel(kernel, arch=arch)

    manifest = {
        "conv": [
            p.N,
            p.Hi,
            p.Wi,
            p.C,
            p.K,
            p.Y,
            p.X,
            p.sH,
            p.sW,
            p.pH,
            p.pW,
            p.dH,
            p.dW,
        ],
        "groups": p.groups,
        "cpg": p.cpg,
        "kpg": p.kpg,
        "dtype": "bf16" if dtype == "bf16" else ("fp32" if dtype == "fp32" else "fp16"),
        "grid_explicit": list(res.grid),
        "threads_per_block": res.block[0],
        "sig_has_bytes": 1,
        "kernel_name": art.kernel_name,
    }
    make_args, _grid, block, _flop, _bytes, _check = run_conv_manifest_problem(
        manifest, None, False
    )
    rt = Runtime()
    module = rt.load_module(art.hsaco)
    fn = module.get_function(art.kernel_name)
    args, _ptrs = make_args(rt)

    def _one():
        rt.launch(fn, res.grid, block, args)

    if time_launches is not None:
        ms = time_launches(_one, warmup=warmup, iters=iters)
    else:  # simple fallback timer
        import time as _t

        for _ in range(warmup):
            _one()
        rt.synchronize()
        t0 = _t.perf_counter()
        for _ in range(iters):
            _one()
        rt.synchronize()
        ms = (_t.perf_counter() - t0) * 1e3 / iters
    module.unload()
    return res.candidate.spec_id, float(ms)


# ---------------------------------------------------------------------
# MIOpen reference (MIOpenDriver binary)
# ---------------------------------------------------------------------

_MIOPEN_TIME_RE = re.compile(r"Elapsed:\s*([0-9.]+)\s*ms")


def _time_miopen(problem, dtype: str, miopen_bin: str) -> Optional[float]:
    """Run MIOpenDriver forward conv for ``problem`` and return avg kernel ms."""
    base = _DTYPE_TO_MIOPEN.get(dtype, "convfp16")
    p = problem
    cmd = [
        miopen_bin,
        base,
        "-n",
        str(p.N),
        "-c",
        str(p.C),
        "-H",
        str(p.Hi),
        "-W",
        str(p.Wi),
        "-k",
        str(p.K),
        "-y",
        str(p.Y),
        "-x",
        str(p.X),
        "-p",
        str(p.pH),
        "-q",
        str(p.pW),
        "-u",
        str(p.sH),
        "-v",
        str(p.sW),
        "-l",
        str(p.dH),
        "-j",
        str(p.dW),
        "-g",
        str(p.groups),
        "-F",
        "1",
        "-t",
        "1",
        "-V",
        "0",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"[miopen] run failed: {e}", file=sys.stderr)
        return None
    text = (out.stdout or "") + (out.stderr or "")
    matches = _MIOPEN_TIME_RE.findall(text)
    if not matches:
        print(f"[miopen] no timing parsed (rc={out.returncode})", file=sys.stderr)
        return None
    # The forward-conv line is the relevant one; take the last elapsed match.
    return float(matches[-1])


# ---------------------------------------------------------------------
# Generic external reference (CK-Tile / old-CK) via a command template
# ---------------------------------------------------------------------


def _time_external(problem, dtype: str, template: str, time_re: str) -> Optional[float]:
    p = problem
    cmd = template.format(
        n=p.N,
        c=p.C,
        hi=p.Hi,
        wi=p.Wi,
        k=p.K,
        y=p.Y,
        x=p.X,
        ph=p.pH,
        pw=p.pW,
        sh=p.sH,
        sw=p.sW,
        dh=p.dH,
        dw=p.dW,
        groups=p.groups,
        dtype=dtype,
    )
    try:
        out = subprocess.run(
            shlex.split(cmd), capture_output=True, text=True, timeout=300
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"[ext] run failed: {e}", file=sys.stderr)
        return None
    text = (out.stdout or "") + (out.stderr or "")
    m = re.findall(time_re, text)
    return float(m[-1]) if m else None


# ---------------------------------------------------------------------
# Corpus + driver
# ---------------------------------------------------------------------


def _load_corpus(args) -> List[str]:
    if args.miopen_cmd:
        return [args.miopen_cmd]
    if args.miopen_file:
        with open(args.miopen_file, "r") as f:
            return [
                ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")
            ]
    # Default placeholder corpus (NOT the pinned CF5P corpus — supply
    # --miopen-file with the assigned rows for real evidence).
    print(
        "[warn] no --miopen-cmd/--miopen-file; using a small built-in demo "
        "corpus (NOT the pinned CF5P rows).",
        file=sys.stderr,
    )
    return [
        "MIOpenDriver convbfp16 -n 2 -c 256 -H 8 -W 8 -k 256 -y 3 -x 3 "
        "-p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 32 -F 1",
        "MIOpenDriver convbfp16 -n 8 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 "
        "-p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 4 -F 1",
    ]


def _print_table(rows: List[RowResult]) -> None:
    hdr = (
        f"{'shape':<34} {'dt':<4} {'G':>4} "
        f"{'rocKE TF':>9} {'MIOpen TF':>10} {'CK TF':>8} {'oldCK TF':>9} "
        f"{'vsMIOpen':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        d = r.as_dict()

        def _f(x, w, p=1):
            return f"{x:>{w}.{p}f}" if isinstance(x, (int, float)) else f"{'n/a':>{w}}"

        sp = d["speedup_vs_miopen"]
        print(
            f"{r.short:<34} {r.dtype:<4} {r.groups:>4} "
            f"{_f(d['rocke_tflops'],9)} {_f(d['miopen_tflops'],10)} "
            f"{_f(d['ck_tflops'],8)} {_f(d['oldck_tflops'],9)} "
            f"{(f'{sp:.2f}x' if sp else 'n/a'):>9}"
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default=None, help="gfx target (default: live device)")
    ap.add_argument("--miopen-cmd", default=None, help="single MIOpenDriver command")
    ap.add_argument("--miopen-file", default=None, help="file of MIOpenDriver commands")
    ap.add_argument("--miopen-bin", default="MIOpenDriver", help="MIOpenDriver binary")
    ap.add_argument(
        "--no-miopen", action="store_true", help="skip the MIOpen reference"
    )
    ap.add_argument("--ck-cmd-template", default=None, help="CK-Tile command template")
    ap.add_argument(
        "--oldck-cmd-template", default=None, help="old-CK command template"
    )
    ap.add_argument(
        "--ext-time-regex",
        default=r"([0-9.]+)\s*ms",
        help="regex with one group capturing elapsed ms from CK/old-CK output",
    )
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", default=None, help="write the per-row report as JSON")
    args = ap.parse_args(argv)

    from ..benchmark.benchmark_implicit_gemm_conv import parse_miopen_cmd
    from ..runtime.hip_module import get_device_arch

    arch = args.arch or get_device_arch(0) or "gfx950"
    corpus = _load_corpus(args)
    rows: List[RowResult] = []

    for cmd in corpus:
        try:
            problem, dtype = parse_miopen_cmd(cmd)
        except ValueError as e:
            print(f"[skip] {e}", file=sys.stderr)
            continue
        row = RowResult(
            short=problem.short(),
            dtype=dtype,
            groups=problem.groups,
            flops=problem.flops,
        )
        try:
            row.rocke_spec_id, row.rocke_ms = _time_rocke(
                problem, dtype, arch, args.warmup, args.iters
            )
        except Exception as e:  # noqa: BLE001 - record and continue the corpus
            row.note = f"rocke: {e}"
            print(f"[rocke] {problem.short()}: {e}", file=sys.stderr)
        if not args.no_miopen:
            row.miopen_ms = _time_miopen(problem, dtype, args.miopen_bin)
        if args.ck_cmd_template:
            row.ck_ms = _time_external(
                problem, dtype, args.ck_cmd_template, args.ext_time_regex
            )
        if args.oldck_cmd_template:
            row.oldck_ms = _time_external(
                problem, dtype, args.oldck_cmd_template, args.ext_time_regex
            )
        rows.append(row)

    print(f"\n# rocKE forward-conv perf comparison on {arch}\n")
    _print_table(rows)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"arch": arch, "rows": [r.as_dict() for r in rows]}, f, indent=2)
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
