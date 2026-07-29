#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Inner-loop ISA analysis + issue-cycle model for the swapqk kernel.

Finds the K-loop (the backward branch whose body holds the WMMAs), histograms
that body by opcode, and prices it with the issue costs measured by peaks.py.
On RDNA3 WMMA executes on the VALU pipe, so WMMA and VALU cycles ADD rather
than overlap -- meaning the loop body has a hard issue-cycle floor that can be
compared directly against the measured cycles per wave-iteration.

Usage:
    python3 isa_deep.py --block-n 64 --vt 1
"""

from __future__ import annotations

import argparse
import collections
import re
import subprocess
import tempfile

from rocke.helpers import compile_kernel

from kernels.gfx1151.wmma_fmha_swapqk import SwapQKCfg, build_wmma_fmha_swapqk

# measured on gfx1151 by peaks.py: cycles per instruction per SIMD
COST = {"wmma": 36.15, "trans": 4.00, "valu": 1.31, "other": 0.0}

TRANS = re.compile(r"^v_(exp|log|rcp|sqrt|rsq|sin|cos)_")


def classify(op: str) -> str:
    if op.startswith("v_wmma"):
        return "wmma"
    if TRANS.match(op):
        return "trans"
    if op.startswith("v_"):
        return "valu"
    return "other"  # scalar / memory / branch / waitcnt issue on other ports


def group(op: str) -> str:
    """Coarse functional bucket, for spotting where the VALU work actually is."""
    if op.startswith("v_wmma"):
        return "WMMA"
    if TRANS.match(op):
        return "transcendental"
    if op.startswith(("v_permlane", "v_mov_b32_dpp", "ds_permute")):
        return "lane shuffle"
    if op.startswith(("v_cvt", "v_pack", "v_mov_b16", "v_perm_b32")):
        return "pack/convert"
    if op.startswith(("v_max", "v_min", "v_cmp", "v_cndmask")):
        return "max/select"
    if op.startswith(
        (
            "v_add",
            "v_sub",
            "v_mul",
            "v_fma",
            "v_fmac",
            "v_mad",
            "v_dot",
            "v_lshl",
            "v_and",
            "v_or",
            "v_xor",
            "v_bfe",
            "v_ashr",
            "v_lshr",
            "v_not",
            "v_bfi",
            "v_alignbit",
        )
    ):
        return "valu arith"
    if op.startswith("v_mov"):
        return "v_mov"
    if op.startswith("v_"):
        return "valu other"
    if op.startswith(
        (
            "buffer_load",
            "global_load",
            "scratch_load",
            "ds_read",
            "ds_load",
            "flat_load",
        )
    ):
        return "MEM load"
    if op.startswith(
        ("buffer_store", "global_store", "scratch_store", "ds_write", "flat_store")
    ):
        return "MEM store"
    if op.startswith("s_waitcnt") or op.startswith("s_wait"):
        return "waitcnt"
    if op.startswith(("s_branch", "s_cbranch", "s_setpc", "s_endpgm")):
        return "branch"
    if op.startswith("s_"):
        return "scalar"
    return "misc"


def disasm(hsaco: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=False) as f:
        f.write(hsaco)
        path = f.name
    for exe in ("/opt/rocm/llvm/bin/llvm-objdump", "llvm-objdump"):
        try:
            r = subprocess.run(
                [exe, "-d", "--mcpu=gfx1151", path], capture_output=True, text=True
            )
            if r.returncode == 0:
                return r.stdout
        except FileNotFoundError:
            continue
    raise RuntimeError("llvm-objdump not found")


# llvm-objdump prints: "\t<opcode> <operands>   // <ADDR>: <encoding>"
LINE = re.compile(r"^\t(\S+)(.*?)//\s*([0-9A-Fa-f]+):")


def parse(text: str):
    """-> list of (addr, opcode, full_line)"""
    out = []
    for ln in text.splitlines():
        m = LINE.match(ln)
        if m:
            out.append((int(m.group(3), 16), m.group(1), ln))
    return out


def find_kloop(insts):
    """The K-loop = body of the backward branch containing the most WMMAs.

    llvm-objdump prints the branch operand as a relative word count but also
    annotates the absolute destination as "<kernel+0xOFFSET>", which is what we
    key off. Offsets are relative to the kernel symbol = first instruction.
    """
    base = insts[0][0]
    idx = {a: i for i, a in enumerate(a for a, _, _ in insts)}
    best = None
    for i, (a, op, ln) in enumerate(insts):
        if not op.startswith(("s_cbranch", "s_branch")):
            continue
        m = re.search(r"\+0x([0-9a-fA-F]+)>", ln)
        if not m:
            continue
        tgt = base + int(m.group(1), 16)
        if tgt >= a or tgt not in idx:
            continue  # forward branch / unknown target
        j = idx[tgt]
        n_wmma = sum(1 for _, o, _ in insts[j : i + 1] if o.startswith("v_wmma"))
        if n_wmma and (best is None or n_wmma > best[0]):
            best = (n_wmma, j, i)
    return best


def resource(hsaco: bytes):
    raw = bytes(hsaco)

    def after(key):
        i = raw.find(key.encode())
        if i < 0:
            return None
        j = i + len(key)
        b0 = raw[j]
        if b0 < 0x80:
            return b0
        if b0 == 0xCC:
            return raw[j + 1]
        if b0 == 0xCD:
            return int.from_bytes(raw[j + 1 : j + 3], "big")
        if b0 == 0xCE:
            return int.from_bytes(raw[j + 1 : j + 5], "big")
        return None

    return {
        k: after("." + k)
        for k in (
            "vgpr_count",
            "sgpr_count",
            "vgpr_spill_count",
            "private_segment_fixed_size",
            "group_segment_fixed_size",
        )
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seqlen", type=int, default=16384)
    ap.add_argument("--head-size", type=int, default=128)
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--block-n", type=int, default=64)
    ap.add_argument("--waves", type=int, default=2)
    ap.add_argument("--mq", type=int, default=1)
    ap.add_argument("--of16", type=int, default=0)
    ap.add_argument("--ilp", type=int, default=2)
    ap.add_argument("--vt", type=int, default=1)
    ap.add_argument("--dual", type=int, default=1)
    ap.add_argument("--qkdo", type=int, default=0)
    ap.add_argument("--qh", type=int, default=0, help="q_hoist (Q out of K-loop)")
    ap.add_argument("--wpe", type=int, default=0, help="waves_per_eu hint (0=none)")
    ap.add_argument(
        "--measured-mcyc",
        type=float,
        default=0.0,
        help="measured GRBM_GUI_ACTIVE (millions) to compare against",
    )
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    cfg = SwapQKCfg(
        head_size=args.head_size,
        num_query_heads=args.heads,
        num_kv_heads=0,
        mask_mode="none",
        n_waves=args.waves,
        q_block=args.mq,
        o_f16=bool(args.of16),
        block_n=args.block_n,
        qk_ilp=args.ilp,
        sched_mode="pingpong",
        buffer_gather=True,
        dual_gather=bool(args.dual),
        fast_exp2=True,
        v_transposed=bool(args.vt),
        qk_douter=bool(args.qkdo),
        q_hoist=bool(args.qh),
        waves_per_eu=(args.wpe or None),
    )
    art = compile_kernel(build_wmma_fmha_swapqk(cfg, arch="gfx1151"), arch="gfx1151")
    insts = parse(disasm(art.hsaco))
    res = resource(art.hsaco)

    print(f"kernel: {art.kernel_name}")
    print(
        f"  vgpr={res['vgpr_count']} sgpr={res['sgpr_count']} "
        f"spill={res['vgpr_spill_count']} "
        f"scratch={res['private_segment_fixed_size']}B "
        f"lds={res['group_segment_fixed_size']}B"
    )
    # gfx1151 wave32 allocates VGPRs in granules of 16 out of a 1536-entry file,
    # so occupancy is a step function: only crossing a granule boundary buys a wave.
    gran = -(-res["vgpr_count"] // 16) * 16
    print(
        f"  vgpr granule={gran} -> {1536 // gran} waves/SIMD "
        f"(next wave at <={1536 // (1536 // gran + 1)} vgpr)"
    )
    print(f"  whole-kernel instructions: {len(insts)}")

    loop = find_kloop(insts)
    if loop is None:
        print("  !! K-loop not found")
        return 1
    n_wmma, j, i = loop
    body = insts[j : i + 1]
    print(
        f"  K-loop body: {len(body)} instructions, {n_wmma} WMMA "
        f"(0x{body[0][0]:x}..0x{body[-1][0]:x})\n"
    )

    # ---- opcode histogram of the loop body ----
    ops = collections.Counter(o for _, o, _ in body)
    groups = collections.Counter()
    cycles = collections.Counter()
    for _, o, _ in body:
        groups[group(o)] += 1
        cycles[group(o)] += COST[classify(o)]

    print(f"{'functional group':<18}{'count':>7}{'issue cyc':>11}{'% cyc':>8}")
    print("-" * 44)
    tot_cyc = sum(cycles.values())
    for g, n in groups.most_common():
        print(
            f"{g:<18}{n:>7}{cycles[g]:>11.0f}"
            f"{100 * cycles[g] / max(tot_cyc, 1):>7.1f}%"
        )
    print("-" * 44)
    print(f"{'TOTAL':<18}{len(body):>7}{tot_cyc:>11.0f}")

    loads = collections.Counter(
        o
        for _, o, _ in body
        if o.startswith(("global_load", "buffer_load", "scratch_load"))
    )
    ndual = sum(1 for _, o, _ in body if o.startswith("v_dual"))
    print(f"  in-loop v_dual_* (VOPD): {ndual}")
    print(f"\n  in-loop loads: " + ", ".join(f"{o}={n}" for o, n in loads.items()))

    print(f"\ntop {args.top} opcodes in the K-loop:")
    for o, n in ops.most_common(args.top):
        print(f"  {n:>5}  {o:<34} {group(o)}")

    # ---- issue-cycle model vs measurement ----
    L, D = args.seqlen, args.head_size
    iters = L // args.block_n
    rows_per_cta = 16 * args.mq * args.waves
    ctas = L // rows_per_cta
    waves = ctas * args.waves
    simds = 80
    wps = waves / simds
    print(
        f"\nmodel at L={L} bn={args.block_n}: {iters} iterations, "
        f"{ctas} CTAs, {waves} waves ({wps:.1f}/SIMD)"
    )
    per_simd = wps * iters * tot_cyc
    print(
        f"  issue-cycle floor  = {wps:.1f} waves x {iters} iters x "
        f"{tot_cyc:.0f} cyc = {per_simd / 1e6:.2f} Mcyc"
    )
    wmma_only = wps * iters * cycles["WMMA"]
    print(
        f"  of which WMMA      = {wmma_only / 1e6:.2f} Mcyc "
        f"({100 * cycles['WMMA'] / tot_cyc:.0f}%)"
    )
    flops = 4.0 * L * L * D
    for label, c in (("WMMA-only roof", wmma_only), ("issue floor", per_simd)):
        print(f"  {label:<18} -> {flops / c * 2.4e9 / 1e12:>6.2f} TF at 2.4 GHz")
    if args.measured_mcyc:
        m = args.measured_mcyc * 1e6
        print(
            f"  measured           = {args.measured_mcyc:.2f} Mcyc "
            f"-> {flops / m * 2.4e9 / 1e12:.2f} TF at 2.4 GHz"
        )
        print(
            f"  issue-bound frac   = {100 * per_simd / m:.0f}%  "
            f"(stall/other = {100 * (1 - per_simd / m):.0f}%)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
