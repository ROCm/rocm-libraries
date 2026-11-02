# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Register-resident WMMA roofline probe for gfx1250 -- one instruction or a set.

For every selected instruction and every (chains, block, passes) config it
generates a HIP kernel that runs ``chains`` INDEPENDENT accumulators back to
back with no memory traffic in the loop, so throughput is bounded by the WMMA
issue rate. All configs compile into one binary, which is run once; the report
is TFLOP/s per instruction at its best config.

Method:

* Big grid. The total wave count is held fixed (default 131072 waves) and the
  block count is derived from it. A small grid's partially filled last wave
  round understates the ceiling by several percent.
* Peak protocol. Idle gap (default 25 ms) before every sample; each sample is
  the mean over ``--iters`` launches; the reported figure is the MEDIAN over
  ``--repeats`` samples, and best/median > 1.05 is flagged. The sustained
  counterpart is ``--gap-ms 0 --warmup 20 --iters 50``.
* Chains are swept because each class peaks at a different count: more chains
  hide latency until accumulator VGPRs cross the occupancy cliff.
* Operand data (``--data``). The rate depends on the operand bits: more
  toggling in the datapath draws more power and lowers the clock. ``const`` is
  one bit pattern everywhere; ``int`` and ``rand`` draw per-lane e4m3 bytes from
  the GEMM's two input sets (``fp8_data``), alternating between two operand
  sets so consecutive WMMAs see different bits, as GEMM fragments do. Compare a
  GEMM only against the ceiling measured on matching data.

This is a RATE probe: results are not checked. Needs ``hipcc`` (``$HIPCC``,
``PATH`` or ``$ROCM_PATH/bin``) and a gfx1250 device.

Examples: ``--list``, or see EXAMPLES at the bottom of this file / ``--help``.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ARCH = "gfx1250"
WAVE = 32

# --------------------------------------------------------------------------
# Instruction catalog
# --------------------------------------------------------------------------
# a/b/acc are HIP vector typedef names declared in _PRELUDE. ``call`` is the
# builtin invocation with {a}, {b}, {c} placeholders for the operands.
SC32 = "0x7F7F7F7F"  # four E8M0 bytes = 2^0 (scale, block 32)
SC64 = "0x7F7F7F7F7F7F7F7FL"  # eight E8M0 bytes = 2^0 (scale16, block 16)
FMTS = {"fp8": 0, "bf8": 1, "fp6": 2, "bf6": 3, "fp4": 4}


@dataclass(frozen=True)
class Inst:
    name: str
    m: int
    n: int
    k: int
    a: str
    b: str
    acc: str
    call: str
    chains: tuple[int, ...]  # default chain sweep for this class
    note: str = ""

    @property
    def flops(self) -> int:  # per instruction, per wave
        return 2 * self.m * self.n * self.k


def _catalog() -> dict[str, Inst]:
    cat: dict[str, Inst] = {}

    def add(i: Inst) -> None:
        cat[i.name] = i

    fp8_ch = (12, 16, 20, 24)
    b16_ch = (24, 40, 56)

    # 16-bit, K=32: (negA, A, negB, B, modC, C, reuseA, reuseB)
    for nm, a, acc in (
        ("f32_16x16x32_f16", "v16h", "v8f"),
        ("f32_16x16x32_bf16", "v16b", "v8f"),
        ("f16_16x16x32_f16", "v16h", "v8h"),
        ("bf16_16x16x32_bf16", "v16b", "v8b"),
    ):
        add(
            Inst(
                nm,
                16,
                16,
                32,
                a,
                a,
                acc,
                f"__builtin_amdgcn_wmma_{nm}(false,{{a}},false,{{b}},(short)0,{{c}},false,false)",
                b16_ch,
            )
        )

    # Dense fp8/bf8 pairings, K=64 and K=128, f32 and f16 accumulate:
    # (A, B, modC, C, reuseA, reuseB)
    for k, frag in ((64, "v8i"), (128, "v16i")):
        for accn, acc in (("f32", "v8f"), ("f16", "v8h")):
            for pair in ("fp8_fp8", "fp8_bf8", "bf8_fp8", "bf8_bf8"):
                nm = f"{accn}_16x16x{k}_{pair}"
                note = "correct but ~5.6x slow" if (k == 128 and accn == "f16") else ""
                add(
                    Inst(
                        nm,
                        16,
                        16,
                        k,
                        frag,
                        frag,
                        acc,
                        f"__builtin_amdgcn_wmma_{nm}({{a}},{{b}},(short)0,{{c}},false,false)",
                        fp8_ch,
                        note,
                    )
                )

    # f8f6f4 family: one opcode, the format immediate selects the encoding.
    for fmt, code in FMTS.items():
        add(
            Inst(
                f"f32_16x16x128_f8f6f4.{fmt}",
                16,
                16,
                128,
                "v16i",
                "v16i",
                "v8f",
                f"__builtin_amdgcn_wmma_f32_16x16x128_f8f6f4({code},{{a}},{code},{{b}},(short)0,{{c}})",
                fp8_ch,
            )
        )
        add(
            Inst(
                f"scale_f32_16x16x128_f8f6f4.{fmt}",
                16,
                16,
                128,
                "v16i",
                "v16i",
                "v8f",
                f"__builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4({code},{{a}},{code},{{b}},(short)0,{{c}},"
                f"0,0,{SC32},0,0,{SC32},false,false)",
                fp8_ch,
            )
        )
        add(
            Inst(
                f"scale16_f32_16x16x128_f8f6f4.{fmt}",
                16,
                16,
                128,
                "v16i",
                "v16i",
                "v8f",
                f"__builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4({code},{{a}},{code},{{b}},(short)0,{{c}},"
                f"0,0,{SC64},0,0,{SC64},false,false)",
                fp8_ch,
            )
        )

    # 32x16x128 FP4: A <16 x i32>, B <8 x i32>, C <16 x float>.
    f4_ch = (8, 12, 16)
    add(
        Inst(
            "f32_32x16x128_f4",
            32,
            16,
            128,
            "v16i",
            "v8i",
            "v16f",
            "__builtin_amdgcn_wmma_f32_32x16x128_f4({a},{b},(short)0,{c})",
            f4_ch,
        )
    )
    add(
        Inst(
            "scale_f32_32x16x128_f4",
            32,
            16,
            128,
            "v16i",
            "v8i",
            "v16f",
            f"__builtin_amdgcn_wmma_scale_f32_32x16x128_f4({{a}},{{b}},(short)0,{{c}},"
            f"0,0,{SC32},0,0,{SC32},false,false)",
            f4_ch,
        )
    )
    add(
        Inst(
            "scale16_f32_32x16x128_f4",
            32,
            16,
            128,
            "v16i",
            "v8i",
            "v16f",
            f"__builtin_amdgcn_wmma_scale16_f32_32x16x128_f4({{a}},{{b}},(short)0,{{c}},"
            f"0,0,{SC64},0,0,{SC64},false,false)",
            f4_ch,
        )
    )
    return cat


CATALOG = _catalog()

# Named sets, selected with @name. Values are glob patterns over CATALOG.
SETS: dict[str, list[str]] = {
    "ceiling": [
        "f32_16x16x128_f8f6f4.fp8",
        "f32_16x16x64_fp8_fp8",
        "f32_16x16x32_bf16",
    ],
    "fp8-k64": ["f32_16x16x64_*", "f16_16x16x64_*"],
    "fp8-k128": [
        "f32_16x16x128_??8_??8",
        "f16_16x16x128_*",
        "f32_16x16x128_f8f6f4.fp8",
    ],
    "fp8-dense": [
        "f32_16x16x64_*",
        "f16_16x16x64_*",
        "f32_16x16x128_??8_??8",
        "f16_16x16x128_*",
    ],
    "f8f6f4": [
        "f32_16x16x128_f8f6f4.*",
        "scale_f32_16x16x128_f8f6f4.*",
        "scale16_f32_16x16x128_f8f6f4.*",
    ],
    "mx": ["scale_*", "scale16_*"],
    "16bit": ["*_16x16x32_*"],
    "f4": ["*_32x16x128_f4", "*f8f6f4.fp4"],
    "all": ["*"],
}


def resolve(spec: str) -> list[str]:
    """Comma-separated names, globs (``f32_16x16x64_*``) and sets (``@f8f6f4``)."""
    out: list[str] = []
    for tok in (t.strip() for t in spec.split(",")):
        if not tok:
            continue
        pats = SETS[tok[1:]] if tok.startswith("@") and tok[1:] in SETS else [tok]
        if tok.startswith("@") and tok[1:] not in SETS:
            raise SystemExit(
                f"unknown set {tok}; sets: {', '.join('@' + s for s in SETS)}"
            )
        hit = False
        for pat in pats:
            for nm in CATALOG:
                if fnmatch.fnmatchcase(nm, pat) and nm not in out:
                    out.append(nm)
                    hit = True
            if pat in CATALOG:
                hit = True
        if not hit:
            raise SystemExit(f"no instruction matches {tok!r}; try --list")
    return out


# --------------------------------------------------------------------------
# HIP source generation
# --------------------------------------------------------------------------
_PRELUDE = r"""
#include <hip/hip_runtime.h>
#include <cstdio>
#include <unistd.h>
typedef int      v8i  __attribute__((ext_vector_type(8)));
typedef int      v16i __attribute__((ext_vector_type(16)));
typedef float    v8f  __attribute__((ext_vector_type(8)));
typedef float    v16f __attribute__((ext_vector_type(16)));
typedef _Float16 v8h  __attribute__((ext_vector_type(8)));
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef __bf16   v8b  __attribute__((ext_vector_type(8)));
typedef __bf16   v16b __attribute__((ext_vector_type(16)));
"""

_LANES = {
    "v8i": 8,
    "v16i": 16,
    "v8f": 8,
    "v16f": 16,
    "v8h": 8,
    "v16h": 16,
    "v8b": 8,
    "v16b": 16,
}
_ELEM = {
    "v16h": "(_Float16)1.0f",
    "v16b": "(__bf16)1.0f",
    "v8i": "0x3C3C3C3C",
    "v16i": "0x3C3C3C3C",
}  # constant, non-identity operands (non-zero e4m3 bytes / 1.0)


def _data_table(mode: str) -> list[int]:
    """64 operand bytes the kernel samples per lane/slot for --data int|rand."""
    import random

    import numpy as np

    from .fp8_data import e4m3_round

    rng = random.Random(42)
    if mode == "int":
        xs = [rng.randint(-5, 5) for _ in range(64)]
    else:
        xs = [rng.gauss(0.0, 0.1) for _ in range(64)]
    return [int(c) for c in e4m3_round(np, np.array(xs, dtype=np.float32))]


DATA_MODE = "const"


def _ident(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z]", "_", s)


def _kernel(inst: Inst, ch: int, blk: int, passes: int, kname: str) -> str:
    na, nb, nc = _LANES[inst.a], _LANES[inst.b], _LANES[inst.acc]
    if DATA_MODE == "const" or inst.a not in ("v8i", "v16i"):
        call = inst.call.format(a="a", b="b", c="acc[c]")
        init = (
            f"  {inst.a} a; for (int i = 0; i < {na}; ++i) a[i] = {_ELEM[inst.a]};\n"
            f"  {inst.b} b; for (int i = 0; i < {nb}; ++i) b[i] = {_ELEM[inst.b]};"
        )
    else:
        # Two operand sets, alternating chain to chain, so consecutive WMMAs see
        # different per-lane bytes drawn from the --data table (as GEMM fragments do).
        call = inst.call.format(
            a="((c & 1) ? a1 : a0)", b="((c & 1) ? b1 : b0)", c="acc[c]"
        )
        tbl = ", ".join(str(v) for v in _data_table(DATA_MODE))
        init = f"""  const unsigned char T[64] = {{{tbl}}};
  auto pk = [&](unsigned s) {{ unsigned w = 0; for (int k = 0; k < 4; ++k) {{
      s = s * 1664525u + 1013904223u; w |= (unsigned)T[(s >> 24) & 63] << (8 * k); }} return (int)w; }};
  const unsigned seed = threadIdx.x * 2654435761u ^ blockIdx.x * 40503u;
  {inst.a} a0, a1; {inst.b} b0, b1;
  for (int i = 0; i < {na}; ++i) {{ a0[i] = pk(seed + 17u * i + 1u); a1[i] = pk(seed + 17u * i + 5u); }}
  for (int i = 0; i < {nb}; ++i) {{ b0[i] = pk(seed + 31u * i + 9u); b1[i] = pk(seed + 31u * i + 13u); }}"""
    return f"""
extern "C" __global__ __launch_bounds__({blk}) void {kname}(float* dst, int n) {{
{init}
  {inst.acc} acc[{ch}];
  for (int c = 0; c < {ch}; ++c) for (int i = 0; i < {nc}; ++i) acc[c][i] = 0;
  for (int it = 0; it < n; ++it) {{
#pragma unroll
    for (int p = 0; p < {passes}; ++p) {{
#pragma unroll
      for (int c = 0; c < {ch}; ++c) acc[c] = {call};
    }}
  }}
  float t = 0;
  for (int c = 0; c < {ch}; ++c) for (int i = 0; i < {nc}; ++i) t += (float)acc[c][i];
  if (t == 1234.5f) dst[blockIdx.x * {blk} + threadIdx.x] = t;   // defeat DCE
}}
"""


def gen_source(
    cases: list[dict],
    loop: int,
    warmup: int,
    iters: int,
    repeats: int,
    gap_us: int,
    lds: int = 0,
) -> str:
    parts = [_PRELUDE]
    for cs in cases:
        parts.append(
            _kernel(
                CATALOG[cs["inst"]], cs["chains"], cs["blk"], cs["passes"], cs["kname"]
            )
        )
    max_threads = max(cs["blocks"] * cs["blk"] for cs in cases)
    rows = ",\n".join(
        f'  {{"{cs["id"]}", (void*){cs["kname"]}, {cs["blocks"]}, {cs["blk"]}}}'
        for cs in cases
    )
    parts.append(
        f"""
struct Case {{ const char* id; void* fn; int blocks; int blk; }};
const size_t LDS = {lds};
int main(int argc, char** argv) {{
  Case cs[] = {{
{rows}
  }};
  float* dst; hipMalloc(&dst, (size_t){max_threads} * 4);
  if (LDS) for (auto& c : cs) hipFuncSetAttribute(c.fn, hipFuncAttributeMaxDynamicSharedMemorySize, (int)LDS);
  for (auto& c : cs) {{
    void (*f)(float*, int) = (void (*)(float*, int))c.fn;
    for (int i = 0; i < {warmup}; ++i) hipLaunchKernelGGL(f, dim3(c.blocks), dim3(c.blk), LDS, 0, dst, {loop});
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) {{ printf("FAIL %s %s\\n", c.id, hipGetErrorString(e)); fflush(stdout); continue; }}
    printf("RES %s", c.id);
    for (int r = 0; r < {repeats}; ++r) {{
      usleep({gap_us});
      hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
      hipEventRecord(e0);
      for (int i = 0; i < {iters}; ++i) hipLaunchKernelGGL(f, dim3(c.blocks), dim3(c.blk), LDS, 0, dst, {loop});
      hipEventRecord(e1); hipEventSynchronize(e1);
      float ms; hipEventElapsedTime(&ms, e0, e1);
      printf(" %.6f", ms / {iters});
      hipEventDestroy(e0); hipEventDestroy(e1);
    }}
    printf("\\n"); fflush(stdout);
  }}
  hipFree(dst);
  return 0;
}}
"""
    )
    return "".join(parts)


# --------------------------------------------------------------------------
# Build / run / report
# --------------------------------------------------------------------------
def _hipcc() -> str:
    rocm = os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME")
    for cand in (
        os.environ.get("HIPCC"),
        shutil.which("hipcc"),
        str(Path(rocm) / "bin" / "hipcc") if rocm else None,
    ):
        if cand and Path(cand).exists():
            return cand
    raise SystemExit("hipcc not found; set HIPCC, put hipcc on PATH, or set ROCM_PATH")


def build(src: str, work: Path, check_asm: bool) -> tuple[Path, str | None]:
    work.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha1(src.encode()).hexdigest()[:12]
    hip, exe, asm = (
        work / f"probe_{tag}.hip",
        work / f"probe_{tag}",
        work / f"probe_{tag}.s",
    )
    hip.write_text(src)
    cc = _hipcc()
    if not exe.exists():
        subprocess.run(
            [cc, "-O3", "-w", f"--offload-arch={ARCH}", str(hip), "-o", str(exe)],
            check=True,
        )
    text = None
    if check_asm:
        if not asm.exists():
            subprocess.run(
                [
                    cc,
                    "-O3",
                    "-w",
                    f"--offload-arch={ARCH}",
                    "--offload-device-only",
                    "-S",
                    str(hip),
                    "-o",
                    str(asm),
                ],
                check=True,
            )
        text = asm.read_text()
    return exe, text


def asm_summary(text: str, kname: str) -> str:
    """WMMA mnemonics and spill/VGPR facts for one kernel, from the device .s."""
    m = re.search(
        rf"^{re.escape(kname)}:.*?$(.*?)^\s*\.size\s+{re.escape(kname)},",
        text,
        re.DOTALL | re.MULTILINE,
    )
    if not m:
        return "?"
    body = m.group(1)
    ops: dict[str, int] = {}
    for op in re.findall(r"^\s*(v_wmma\w*)", body, re.MULTILINE):
        ops[op] = ops.get(op, 0) + 1
    # amdhsa.kernels metadata: one "  - " YAML entry per kernel, keys alphabetical,
    # so .vgpr_count may precede .name within the entry.
    vg = sp = None
    for entry in re.split(r"\n  - ", text):
        if re.search(rf"\.name:\s+{re.escape(kname)}\s*$", entry, re.MULTILINE):
            g = re.search(r"\.vgpr_count:\s+(\d+)", entry)
            s = re.search(r"\.vgpr_spill_count:\s+(\d+)", entry)
            vg, sp = (g and g.group(1)), (s and s.group(1))
            break
    opstr = " ".join(f"{k}x{v}" for k, v in ops.items()) or "NO-WMMA"
    return f"{opstr} vgpr={vg} spill={sp}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )
    p.add_argument(
        "-i",
        "--inst",
        default="@ceiling",
        help="names, globs and @sets, comma-separated (default @ceiling)",
    )
    p.add_argument(
        "--list", action="store_true", help="list instructions and sets, then exit"
    )
    p.add_argument(
        "-c",
        "--chains",
        default="",
        help="chain counts, e.g. 16,24 (default: per-class sweep)",
    )
    p.add_argument(
        "--blk", default="128", help="threads per block, list allowed (default 128)"
    )
    p.add_argument(
        "--passes", default="2", help="unrolled passes per loop iteration (default 2)"
    )
    p.add_argument(
        "--waves",
        type=int,
        default=131072,
        help="total waves per launch; blocks = waves*32/blk (default 131072)",
    )
    p.add_argument("--loop", type=int, default=1024, help="loop iterations per kernel")
    p.add_argument(
        "--lds-kb",
        type=int,
        default=0,
        help="dynamic LDS per block (KiB), unused; pins occupancy like a GEMM's LDS",
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=5, help="launches per timed sample")
    p.add_argument("--repeats", type=int, default=7, help="timed samples per config")
    p.add_argument("--gap-ms", type=float, default=25.0, help="idle before each sample")
    p.add_argument("--stat", choices=("median", "best"), default="median")
    p.add_argument(
        "--ref", default="", help="instruction to normalise against (default: first)"
    )
    p.add_argument(
        "--check-asm",
        action="store_true",
        help="also emit device asm: WMMA opcodes, VGPRs, spills per kernel",
    )
    p.add_argument(
        "--all-configs",
        action="store_true",
        help="print every config, not only each instruction's peak",
    )
    p.add_argument(
        "--work",
        type=Path,
        default=Path(tempfile.gettempdir()) / "roofline_probe",
        help="build directory (binaries are cached by source hash)",
    )
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print the cases and exit")
    p.add_argument(
        "--json", type=Path, help="write all samples and summaries to this file"
    )
    p.add_argument(
        "--data",
        choices=("const", "int", "rand"),
        default="const",
        help="fp8 operand contents: const (0x3C everywhere, the original), "
        "int (per-lane -5..5) or rand (per-lane randn*0.1); int/rand alternate "
        "two operand sets between chains",
    )
    args = p.parse_args(argv)
    global DATA_MODE
    DATA_MODE = args.data

    if args.list:
        w = max(map(len, CATALOG))
        print(f"{'instruction':<{w}}  MxNxK      A     B     acc   chains")
        for i in CATALOG.values():
            print(
                f"{i.name:<{w}}  {i.m}x{i.n}x{i.k:<4} {i.a:<5} {i.b:<5} {i.acc:<5} "
                f"{','.join(map(str, i.chains))}  {i.note}"
            )
        print("\nsets:")
        for s, pats in SETS.items():
            print(f"  @{s:<10} {len(resolve('@' + s)):3d}  {' '.join(pats)}")
        return 0

    names = resolve(args.inst)
    blks = [int(x) for x in args.blk.split(",")]
    passes = [int(x) for x in args.passes.split(",")]
    user_ch = [int(x) for x in args.chains.split(",")] if args.chains else None

    cases = []
    for nm in names:
        inst = CATALOG[nm]
        for ch in user_ch or inst.chains:
            for blk in blks:
                if blk % WAVE or args.waves * WAVE % blk:
                    raise SystemExit(
                        f"--blk {blk} must divide waves*32 and be a multiple of 32"
                    )
                for ps in passes:
                    cid = f"{nm}|c{ch}|b{blk}|p{ps}"
                    cases.append(
                        {
                            "id": cid,
                            "inst": nm,
                            "chains": ch,
                            "blk": blk,
                            "passes": ps,
                            "blocks": args.waves * WAVE // blk,
                            "kname": "k_" + _ident(cid),
                        }
                    )
    if args.dry_run:
        for cs in cases:
            print(cs["id"], "blocks", cs["blocks"])
        print(f"{len(cases)} configs")
        return 0

    src = gen_source(
        cases,
        args.loop,
        args.warmup,
        args.iters,
        args.repeats,
        int(args.gap_ms * 1000),
        args.lds_kb * 1024,
    )
    print(
        f"[roofline] {len(names)} instruction(s), {len(cases)} config(s); building ...",
        file=sys.stderr,
    )
    exe, asm = build(src, args.work, args.check_asm)
    if args.build_only:
        print(exe)
        return 0

    print(
        f"[roofline] running {exe.name} (gap {args.gap_ms:g} ms, {args.repeats}x{args.iters} "
        f"launches, {args.stat})",
        file=sys.stderr,
    )
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=False)
    if out.returncode:
        sys.stderr.write(out.stdout + out.stderr)
        return out.returncode

    by_id = {cs["id"]: cs for cs in cases}
    for line in out.stdout.splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "FAIL":
            by_id[tok[1]]["error"] = " ".join(tok[2:])
            continue
        if tok[0] != "RES":
            continue
        cs = by_id[tok[1]]
        inst = CATALOG[cs["inst"]]
        flops = (
            cs["blocks"]
            * (cs["blk"] // WAVE)
            * args.loop
            * cs["passes"]
            * cs["chains"]
            * inst.flops
        )
        tfs = [flops / (float(ms) * 1e-3) / 1e12 for ms in tok[2:]]
        cs["tf_samples"] = tfs
        cs["tf_median"] = statistics.median(tfs)
        cs["tf_best"] = max(tfs)
        cs["tf"] = cs["tf_median"] if args.stat == "median" else cs["tf_best"]
        if asm is not None:
            cs["asm"] = asm_summary(asm, cs["kname"])

    # Per-instruction peak over configs.
    peaks: dict[str, dict] = {}
    for cs in cases:
        if "tf" in cs and (
            cs["inst"] not in peaks or cs["tf"] > peaks[cs["inst"]]["tf"]
        ):
            peaks[cs["inst"]] = cs
    ref_name = args.ref or (names[0] if names else "")
    ref = peaks.get(ref_name, {}).get("tf")

    w = max(len(n) for n in names)
    if args.all_configs:
        print(
            f"\n{'instruction':<{w}} {'ch':>3} {'blk':>4} {'ps':>2} {'TF':>8} {'best/med':>8}"
        )
        for cs in cases:
            if "error" in cs:
                print(
                    f"{cs['inst']:<{w}} {cs['chains']:>3} {cs['blk']:>4} {cs['passes']:>2}  FAILED {cs['error']}"
                )
            elif "tf" in cs:
                print(
                    f"{cs['inst']:<{w}} {cs['chains']:>3} {cs['blk']:>4} {cs['passes']:>2} "
                    f"{cs['tf']:8.0f} {cs['tf_best'] / cs['tf_median']:8.3f}"
                    + (f"  {cs['asm']}" if "asm" in cs else "")
                )

    print(
        f"\npeak per instruction ({args.stat} of {args.repeats}; ratio vs {ref_name}):"
    )
    print(f"{'instruction':<{w}} {'TF':>8} {'ratio':>6}  {'best cfg':<16} note")
    for nm in names:
        cs = peaks.get(nm)
        if cs is None:
            errs = [c.get("error") for c in cases if c["inst"] == nm and "error" in c]
            print(f"{nm:<{w}} {'FAILED':>8}  {errs[0] if errs else ''}")
            continue
        flag = (
            "  BOOST? best/med>1.05" if cs["tf_best"] / cs["tf_median"] > 1.05 else ""
        )
        ratio = f"{cs['tf'] / ref:6.3f}" if ref else "     -"
        cfg = f"c{cs['chains']} b{cs['blk']} p{cs['passes']}"
        extra = f"  {cs['asm']}" if "asm" in cs and not args.all_configs else ""
        print(
            f"{nm:<{w}} {cs['tf']:8.0f} {ratio}  {cfg:<16} {CATALOG[nm].note}{flag}{extra}"
        )

    if args.json:
        args.json.write_text(
            json.dumps(
                {"args": {k: str(v) for k, v in vars(args).items()}, "cases": cases},
                indent=2,
            )
        )
        print(f"\n[roofline] wrote {args.json}", file=sys.stderr)
    return 0


EXAMPLES = r"""
examples:
  P="-m rocke.examples.gfx1250.gemm_perf_fp8.wmma_roofline_probe"

  python3 $P --list                                   # catalog + named sets
  python3 $P                                          # @ceiling: f8f6f4.fp8, k64 fp8, bf16
  python3 $P -i f32_16x16x128_f8f6f4.fp8              # one instruction, default chain sweep
  python3 $P -i f32_16x16x128_f8f6f4.fp8 -c 16 --blk 128 --passes 2   # one exact config
  python3 $P -i f32_16x16x64_fp8_fp8,f32_16x16x128_f8f6f4.fp8 --ref f32_16x16x64_fp8_fp8
  python3 $P -i @f8f6f4 -c 24 --ref f32_16x16x128_f8f6f4.fp8   # 3 opcodes x 5 formats
  python3 $P -i 'f32_16x16x64_*' -c 24                # glob: the four K=64 pairings
  python3 $P -i @fp8-dense -c 16,24                   # all 16 dense fp8/bf8 forms
  python3 $P -i @16bit                                # f16/bf16 K=32, chains 24,40,56
  python3 $P -i @all -c 16 --json roofline_all.json
  python3 $P -i @ceiling --blk 64,128,256 --passes 1,2 --all-configs   # loop-shape scan
  python3 $P -i @ceiling --check-asm --all-configs    # confirm opcodes, VGPRs, no spills
  python3 $P -i @ceiling --stat best --repeats 15     # best-of (watch best/med flag)
  python3 $P -i @ceiling --blk 128 --lds-kb 300 -c 64   # pin 1 block/CU: 1 wave/SIMD at blk 128
  python3 $P -i @mx --dry-run                         # show configs without building

  # the fp8 GEMM case study's ceiling, on each input set, peak and sustained
  python3 $P -i f32_16x16x128_f8f6f4.fp8 --data rand
  python3 $P -i f32_16x16x128_f8f6f4.fp8 --data rand --gap-ms 0 --warmup 20 --iters 50
"""

if __name__ == "__main__":
    raise SystemExit(main())
