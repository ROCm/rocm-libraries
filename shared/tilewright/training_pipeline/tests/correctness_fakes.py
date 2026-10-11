# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A CPU-only stand-in for the correctness gate's exact harness
(tools/errmap.cpp) and small gfx950 libraries for it.

`write_harness(path)` writes an executable that takes the harness's
arguments and prints its JSON lines. FAKE_ERRMAP_SPEC names a JSON file:

  names        {"<solution index>": name} the harness reports (required)
  fail         {"<solution index>": rule}: "always"; "first" (the first GEMM
               of a process only); "first_in:<n>" (the first GEMM of the n-th
               process that launches it first, counted in `state_dir`);
               "k<K>" (K equals that); "shape:<M>x<N>" (M and N equal
               those); "rate:<r>" (every r-th repetition of the process),
               "rate:<r>@<M>x<N>" (only on that shape)
  nbad         {"<solution index>": n}: the wrong elements a failing
               repetition reports (default 5)
  ignore_scale {"<solution index>": "A", "B" or "AB"}: the solution leaves
               out the scales of those operands, so it is wrong wherever the
               harness gives that operand scales (--scale-a, --scale-b not
               0), whose every scale is HARNESS_SCALES' power of 2, not 1
  unwritten    [solution index, ...]: every repetition leaves an element
               with its NaN pattern and writes an infinity into another; its
               line holds them as the harness's printf writes them (-nan,
               inf), which is not JSON
  depth_u      {"<solution index>": d}: unsupported unless K % d == 0
  unsupported  [solution index, ...]: never supported
  missing      [solution index, ...]: not found, as getAlgosFromIndex skips
               an index its library does not hold
  missing_alone
               [solution index, ...]: not found by a process of no other
               kernel
  crash        [solution index, ...]: the process dies at that solution
  crash_once   [solution index, ...]: the same, in the first process to run
               it only (counted in `state_dir`)
  killed       [solution index, ...]: the process is killed by SIGKILL there
  hang         [solution index, ...]: the process stops making progress there
  hang_shape   "<M>x<N>": it hangs on that shape only
  delay        seconds before each repetition's line
  delay_shape  "<M>x<N>": it waits on that shape only
  broken       true: every process fails before it reports ready
  broken_ids   [--ids value, ...]: a process of those --ids does
  log          a file each process appends its arguments to, one JSON line

Like the harness, it refuses a leading dimension below the rows its matrix
is stored with in the layout of --trans-a and --trans-b (a block-scaled
operand stores K rounded up to a multiple of 256), 8- and 4-bit A and B
without scales, scales of other types (one f32 scale, --scale-a 1 and
--scale-b 1, takes f8 and bf8; block-32 scales, 3, take f8, bf8 and f4),
other scaling formats, A and B scaled in two formats, block-32 scales with K
not a multiple of 32, an odd leading dimension of an f4 operand, and C, D or
bias types other than bf16 and f32.
"""
from __future__ import annotations

import json
import os
import signal
import stat
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import msgpack

TESTS_DIR = Path(__file__).resolve().parent
PREFIX = (
    "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk"
)
STEM = f"{PREFIX}_ID75a0_gfx950"
BANK = f"{PREFIX}_CU256_ID75a0_gfx950"
# gfx950's F8BS problem type: f8 A and B with one f32 scale per tensor (SAB),
# bf16 C and D, a bias epilogue (Bias).
F8BS_STEM = (
    "TensileLibrary_F8F8_BF8_HA_Bias_SAB_SAV_UA_Type_F8B_HPA_Contraction_l_Alik_"
    "Bljk_Cijk_Dijk_ID75a0_gfx950"
)
F8BS_KIND = "F8BS_BH_Bias_HA_S_SAB_SAV_UserArgs"
GFX950 = {"type": "AMDGPU", "value": {"type": "Processor", "value": "gfx950"}}
# The scale of every element of A and of B the harness gives a scaled operand
# (tools/errmap.cpp kScaleExpA and kScaleExpB).
HARNESS_SCALES = {"a": 2.0, "b": 4.0}


def _counted(spec: Dict[str, Any], what: str) -> int:
    """How often `what` happened before, this time included."""
    path = Path(spec["state_dir"]) / what
    n = int(path.read_text()) + 1 if path.is_file() else 1
    path.write_text(str(n))
    return n


NARROW_TYPES = ("f8", "bf8", "f4")
# The A and B types each scaling format takes: one f32 scale per tensor (1)
# and block-32 E8M0 scales (3).
SCALED_TYPES = {"1": ("f8", "bf8"), "3": NARROW_TYPES}


def _type_error(k: int, opts: Dict[str, str]) -> str:
    for op in ("a", "b"):
        kind = opts.get(f"--{op}-type", "bf16")
        scale = opts.get(f"--scale-{op}", "0")
        if scale not in ("0", *SCALED_TYPES):
            return f"scaling format {scale} of {op.upper()} is not run"
        if kind not in SCALED_TYPES.get(scale, ()) and (
            scale != "0" or kind in NARROW_TYPES
        ):
            return (
                f"{op.upper()} takes scales exactly when it is 8- or 4-bit, one f32 "
                f"scale only when it is 8-bit"
            )
        if scale == "3" and k % 32:
            return "block-32 scales need K to be a multiple of 32"
    scales = {opts.get(f"--scale-{op}", "0") for op in ("a", "b")} - {"0"}
    if len(scales) > 1:
        return "A and B take scales of one format"
    for flag in ("--c-type", "--d-type", "--bias-type"):
        if opts.get(flag, "bf16") not in ("bf16", "f32"):
            return "C, D and the bias are bf16 or f32"
    return ""


def _stored_k(k: int, opts: Dict[str, str], op: str) -> int:
    """K as the harness stores operand `op`: rounded up to a multiple of 256
    when it is block-scaled."""
    return -(-k // 256) * 256 if opts.get(f"--scale-{op}", "0") == "3" else k


def _leading_dimension_error(m: int, n: int, k: int, opts: Dict[str, str]) -> str:
    rows_a = _stored_k(k, opts, "a") if opts.get("--trans-a", "T") == "T" else m
    rows_b = n if opts.get("--trans-b", "N") == "T" else _stored_k(k, opts, "b")
    f4_a = opts.get("--a-type") == "f4"
    f4_b = opts.get("--b-type") == "f4"
    lda = int(opts.get("--lda", rows_a + (rows_a % 2 if f4_a else 0)))
    ldb = int(opts.get("--ldb", rows_b + (rows_b % 2 if f4_b else 0)))
    ldc = int(opts.get("--ldc", m))
    sep = opts.get("--sep", "0") != "0"
    ldd = int(opts.get("--ldd", m if sep else ldc))
    if lda < rows_a or ldb < rows_b or ldc < m or ldd < m:
        return "a leading dimension is below its matrix's rows"
    if (f4_a and lda % 2) or (f4_b and ldb % 2):
        return "an f4 operand needs an even leading dimension"
    if not sep and ldd != ldc:
        return "C == D needs equal leading dimensions and types (or --sep 1)"
    return ""


def harness_main(argv: Sequence[str]) -> int:
    spec = json.loads(Path(os.environ["FAKE_ERRMAP_SPEC"]).read_text())
    m, n, k, b = (int(x) for x in argv[:4])
    opts = dict(zip(argv[4::2], argv[5::2]))
    ids = [int(x) for x in opts["--ids"].split(",")]
    reps = int(opts.get("--reps", "1"))
    if spec.get("log"):
        with open(spec["log"], "a") as f:
            f.write(json.dumps({"args": list(argv), "pid": os.getpid()}) + "\n")
    problem = _type_error(k, opts) or _leading_dimension_error(m, n, k, opts)
    if problem:
        print(problem, file=sys.stderr)
        return 1
    if spec.get("broken") or opts["--ids"] in spec.get("broken_ids", []):
        print("hipBLASLt status 3 at line 1", file=sys.stderr)
        return 3
    missing = set(spec.get("missing", []))
    if len(ids) == 1:
        missing |= set(spec.get("missing_alone", []))
    found = [sid for sid in ids if sid not in missing]
    print(json.dumps({"ready": True, "algos": len(found), "sids": found}), flush=True)
    here = f"{m}x{n}"
    delay = float(spec.get("delay", 0)) * (spec.get("delay_shape", here) == here)
    scaled = {op for op in "ab" if opts.get(f"--scale-{op}", "0") != "0"}
    first = True
    count: Dict[int, int] = {}
    for sid in found:
        name = spec["names"][str(sid)]
        du = spec.get("depth_u", {}).get(str(sid))
        if sid in spec.get("unsupported", []) or (du and k % int(du)):
            print(
                json.dumps({"sid": sid, "supported": False, "ws_need": 0, "name": name})
            )
            continue
        if sid in spec.get("crash", []) or (
            sid in spec.get("crash_once", []) and _counted(spec, f"crash_{sid}") == 1
        ):
            sys.stdout.flush()
            os._exit(139)
        if sid in spec.get("killed", []):
            sys.stdout.flush()
            os.kill(os.getpid(), signal.SIGKILL)
        if sid in spec.get("hang", []) and spec.get("hang_shape", here) == here:
            sys.stdout.flush()
            time.sleep(3600)
        ignored = str(spec.get("ignore_scale", {}).get(str(sid), "")).lower()
        drops_scale = any(op in scaled and HARNESS_SCALES[op] != 1.0 for op in ignored)
        wrong = int(spec.get("nbad", {}).get(str(sid), 5))
        for rep in range(reps):
            time.sleep(delay)
            rule = str(spec.get("fail", {}).get(str(sid), ""))
            count[sid] = count.get(sid, 0) + 1
            rate, _, where = rule.partition("@")
            nth_first = (
                _counted(spec, f"first_{sid}")
                if first and rule.startswith("first_in:")
                else 0
            )
            bad = (
                drops_scale
                or rule == "always"
                or (rule == "first" and first)
                or rule == f"first_in:{nth_first}"
                or rule == f"k{k}"
                or rule == f"shape:{m}x{n}"
                or (
                    rate.startswith("rate:")
                    and where in ("", f"{m}x{n}")
                    and count[sid] % int(rate[5:]) == 0
                )
            )
            first = False
            if sid in spec.get("unwritten", []):
                print(
                    f'{{"sid": {sid}, "rep": {rep}, "supported": true, "ws_need": 0, '
                    f'"nbad": 2, "nnan": 1, "maxabs": inf, "bad": [[0,0,0,-nan,4,0,""], '
                    f'[1,0,0,-inf,2,0,""]], "name": "{name}"}}',
                    flush=True,
                )
                continue
            print(
                json.dumps(
                    {
                        "sid": sid,
                        "rep": rep,
                        "supported": True,
                        "ws_need": 0,
                        "nbad": wrong if bad else 0,
                        "nnan": 0,
                        "maxabs": 3.0 if bad else 0.0,
                        "bad": [],
                        "name": name,
                    }
                ),
                flush=True,
            )
    return 0


def write_harness(path: Path) -> Path:
    path.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        f"sys.path.insert(0, {str(TESTS_DIR)!r})\n"
        "from correctness_fakes import harness_main\n"
        "sys.exit(harness_main(sys.argv[1:]))\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def solution(
    index: int, mt: Sequence[int], *, custom: bool = False, kind: str = "BBS"
) -> Dict[str, Any]:
    from bench_fakes import solution as base

    sol = base(index, index % 7, tuple(mt))
    sol["name"] = f"Cijk_Alik_Bljk_{kind}_K{index}_MT{mt[0]}x{mt[1]}x{mt[2]}"
    sol["kernelName"] = sol["name"]
    sol["hardwarePredicate"] = GFX950
    if custom:
        sol["customKernel"] = {"name": sol["name"], "generated": False}
    return sol


# Table kernels (index: macro tile); 103 is hand-written.
TABLE = {100: (64, 64, 64), 101: (128, 64, 32), 102: (128, 64, 32), 103: (256, 256, 64)}
# Bank kernels of the CU256 file.
BANK_KERNELS = {200: (64, 64, 64), 201: (128, 128, 32)}
# Table kernels of the F8BS library; 302 and 303 are of one family.
F8BS_TABLE = {
    300: (128, 128, 128),
    301: (64, 256, 256),
    302: (256, 128, 128),
    303: (256, 128, 128),
}


def _write_table(
    path: Path, table: Dict[int, Sequence[int]], kind: str, custom: Sequence[int] = ()
) -> None:
    sols = [solution(i, mt, custom=i in custom, kind=kind) for i, mt in table.items()]
    doc = {
        "solutions": sols,
        "library": {
            "type": "Problem",
            "rows": [
                {
                    "predicate": {"type": "PredictionMatching"},
                    "library": {"type": "Prediction", "table": list(table)},
                }
            ],
        },
    }
    path.write_bytes(msgpack.packb(doc))


def write_library(libdir: Path) -> Path:
    """`STEM` with a Prediction table of TABLE and `BANK` with Equality rows
    of BANK_KERNELS, both for every gfx950 device."""
    libdir.mkdir(parents=True, exist_ok=True)
    _write_table(libdir / f"{STEM}.dat", TABLE, "BBS", custom=(103,))
    bank = [solution(i, mt) for i, mt in BANK_KERNELS.items()]
    doc = {
        "solutions": bank,
        "library": {
            "type": "Problem",
            "rows": [
                {
                    "predicate": {"type": "EqualityMatching"},
                    "library": {
                        "type": "Matching",
                        "table": [
                            {"key": [1, i, 1, 1], "index": i} for i in BANK_KERNELS
                        ],
                    },
                }
            ],
        },
    }
    (libdir / f"{BANK}.dat").write_bytes(msgpack.packb(doc))
    return libdir


def write_f8bs_library(libdir: Path) -> Path:
    """`F8BS_STEM` with a Prediction table of F8BS_TABLE, for every gfx950
    device."""
    libdir.mkdir(parents=True, exist_ok=True)
    _write_table(libdir / f"{F8BS_STEM}.dat", F8BS_TABLE, F8BS_KIND)
    return libdir


def names() -> Dict[str, str]:
    return {
        str(i): solution(i, mt)["name"] for i, mt in {**TABLE, **BANK_KERNELS}.items()
    }


def f8bs_names() -> Dict[str, str]:
    return {
        str(i): solution(i, mt, kind=F8BS_KIND)["name"] for i, mt in F8BS_TABLE.items()
    }


def config(**correctness: Any) -> Dict[str, Any]:
    return {
        "config_id": "gate_test",
        "arch": "gfx950",
        "hipblaslt": {
            "a_type": "bf16_r",
            "b_type": "bf16_r",
            "c_type": "bf16_r",
            "d_type": "bf16_r",
            "scale_type": "f32_r",
            "compute_type": "c_f32_r",
            "transA": "T",
            "transB": "N",
            "library_stem": STEM,
        },
        "hardware": {
            "n_cu": 256,
            "lds_bytes": 163840,
            "l2_bytes": 4194304,
            "pci_chip_id": 0x75A0,
        },
        "runtime": {"devices": [0], "rocm_libraries_root": "/nonexistent"},
        "planner": {"enabled": True, "bank_library_stems": [BANK]},
        "correctness": {"enabled": True, **correctness},
    }


def mx_config(
    a_type: str = "f8_r", b_type: str = "", **correctness: Any
) -> Dict[str, Any]:
    """`config` with A and B of `a_type` and `b_type` (default: A's) and
    block-32 E8M0 scales on both."""
    cfg = config(**correctness)
    cfg["hipblaslt"].update(a_type=a_type, b_type=b_type or a_type, scaleA=3, scaleB=3)
    return cfg


def scalar_config(
    a_type: str = "f8_r", b_type: str = "", **correctness: Any
) -> Dict[str, Any]:
    """`config` with A and B of `a_type` and `b_type` (default: A's) and one
    f32 scale per tensor on both."""
    cfg = config(**correctness)
    cfg["hipblaslt"].update(a_type=a_type, b_type=b_type or a_type, scaleA=1, scaleB=1)
    return cfg


def f8bs_config(**correctness: Any) -> Dict[str, Any]:
    """`config` of gfx950's F8BS problem type: f8 A and B with one f32 scale
    per tensor (scaleA and scaleB 1), bf16 C and D, f32 compute, TN, the
    library's table alone."""
    cfg = config(**correctness)
    cfg["hipblaslt"].update(
        a_type="f8_r", b_type="f8_r", scaleA=1, scaleB=1, library_stem=F8BS_STEM
    )
    cfg["library"] = {"kind": "origami"}
    cfg["planner"] = {"enabled": True}
    return cfg


def harness_calls(log: Path) -> List[Dict[str, Any]]:
    if not log.is_file():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]
