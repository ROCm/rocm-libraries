#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-only ISA check of comp_tdm_scheduled's K tail.

Compiles representative comp_tdm_scheduled kernels for gfx1250, GPU code only (-S, nothing runs),
and checks their ISA:

- post_loop: after the K loop's back-edge (the tile after the loop: the K tail, or TailNumber::Full's
  odd tile) no tensor_load_to_lds and no global_prefetch_b8. No tile follows the last one, so it
  refills no buffer and prefetches nothing.
- dscnt: at every tensor_load_to_lds, every ds_load still in flight was issued after the barrier
  before the refill's own one. A refill overwrites a buffer once its reads have drained, while
  the next tile's reads, issued after the publish barrier, may stay in flight.
- gl2_clamp: every GL2 prefetch address in the K loop goes through a min: a partial tile's data
  lines are clamped to the end of the row, as its scale lines are. Any instance may prefetch one:
  its K tail, or a partial last tile of every K step that the loop runs.
- vgpr: with --baseline REV, each kernel's VGPR count and scratch next to REV's pipeline headers;
  more scratch than the baseline fails.

Usage: tail_isa_check.py [--baseline HEAD] [--jobs N] [--only SUBSTRING] [--keep DIR]
"""

import argparse
import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

CK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
PIPELINE_DIR = "include/ck_tile/ops/gemm/pipeline"
PIPELINE_HEADERS = [
    f"{PIPELINE_DIR}/mx_gemm_pipeline_ag_bg_cr_comp_tdm_scheduled.hpp",
    f"{PIPELINE_DIR}/mx_gemm_pipeline_ag_bg_cr_comp_tdm_scheduled_policy.hpp",
]

# The compile options of test/ck_tile/gemm_mx/tdm_scheduled (TEST_MX_GEMM_COMPILE_OPTIONS and
# MX_TDM_SCHEDULED_SCHED_OPTIONS) with the gfx1250 definitions of the CK build.
FLAGS = [
    "-x", "hip", "--offload-arch=gfx1250", "--offload-device-only", "-S", "-O3", "-std=c++20",
    "-mllvm", "-amdgpu-early-inline-all=true", "-mllvm", "-amdgpu-function-calls=false",
    "-mllvm", "--lsr-drop-solution=1", "-fno-offload-uniform-block",
    "-mllvm", "-amdgpu-expert-scheduling-mode", "-mllvm", "-enable-post-misched=false",
    "-mllvm", "-greedy-regclass-priority-trumps-globalness",
    "-D__HIP_PLATFORM_AMD__", "-DCK_CMAKE_GPU_TARGET_IDS=0x1250", "-DCK_USE_OCP_FP8",
    "-DCK_TILE_USE_OCP_FP8", "-DCK_TILE_USE_WMMA=1", "-DCK_GFX12_SUPPORT", "-DCK_USE_GFX1250",
    "-DCK_USE_NATIVE_MX_SUPPORT", "-DCK_GFX1250_SUPPORT", "-DUSE_NEW_UNIFIED_FRAMEWORK=0",
    "-Wno-undefined-func-template",
]  # fmt: skip
SINGLE_TAIL_FLAGS = ["-DCK_TILE_FORCE_SINGLE_TAIL_HANDLER"]


@dataclass(frozen=True)
class Instance:
    """A comp_tdm_scheduled kernel: types, tile, wave group, atom, cluster (CX N tiles by CY M
    tiles), policy knobs, shape traits, and build variant."""

    name: str
    a: str = "fp8_t"
    b: str = "bf8_t"
    o: str = "fp8_t"
    tile: tuple = (128, 128, 256)
    warps: tuple = (2, 2)
    atom_m: int = 16
    cluster: tuple = (1, 1)
    gl2: int = 4
    lds_prefetch: bool = True
    split_b: bool = True
    edge: tuple = (True, True)
    tail_steps: int = 1
    single_tail: bool = False

    def checks(self):
        """The check() options of the instance's kernels."""
        return dict(post_loop_refill_free=not (self.single_tail and self.tail_steps))

    def tail_nums(self):
        return ["Full"] if self.single_tail else ["One", "Two"]

    def flags(self):
        return SINGLE_TAIL_FLAGS if self.single_tail else []

    def source(self, tail_num):
        b = lambda v: "true" if v else "false"  # noqa: E731
        cx, cy = self.cluster
        return f"""
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
using namespace ck_tile;
using P = MxGemmPipelineAgBgCrCompTDMScheduled<
    typename MxGemmTdmScheduledProblem<
        {self.a}, {self.b}, {self.o},
        ClusterTileGemmShape<sequence<{cy}, {cx}, 1>,
                             sequence<{self.tile[0]}, {self.tile[1]}, {self.tile[2]}>,
                             sequence<{self.warps[0]}, {self.warps[1]}, 1>,
                             sequence<{self.atom_m}, 16, 128>>,
        MxTdmScheduledGemmTraits<{b(self.edge[0])}, {b(self.edge[1])}, false, false, false,
                                 {self.tail_steps}>>::template WithTailNum<TailNumber::{tail_num}>,
    MxGemmPipelineAgBgCrCompTDMScheduledPolicy<{self.gl2}, {b(self.lds_prefetch)},
                                               {b(self.split_b)}, true>>;
using K = MxGemmKernel<GemmTile2DNMajorPartitioner<typename P::BlockGemmShape>, P,
                       MxTdmStoreEpilogue<P>>;
void touch() {{ (void)&kentry<1, K, typename K::KernelArgs>; }}
"""


FP4_256 = dict(a="pk_fp4_t", b="pk_fp4_t", tile=(256, 256, 512), atom_m=32)
FP4_1024 = dict(a="pk_fp4_t", b="pk_fp4_t", tile=(128, 128, 1024), atom_m=32)
INSTANCES = [
    # The EightBitTypes K-tail instance: SplitB without a cluster (fill_stage<kPendingHalfB>).
    Instance("half_b"),
    Instance("half_b_no_tail", tail_steps=0),
    Instance("full", split_b=False),
    Instance("direct", lds_prefetch=False, split_b=False),
    Instance("half_b_single_tail", single_tail=True),
    Instance("half_b_single_tail_no_tail", tail_steps=0, single_tail=True),
    # No edge and no K tail: the loop may still run a partial last tile.
    Instance("half_b_whole", edge=(False, False), tail_steps=0),
    # kSplitCluster: the tuned FP4 4x4 cluster, tail 0, 1 and 3.
    Instance(
        "fp4_cluster_4x4_no_tail",
        cluster=(4, 4),
        edge=(False, False),
        tail_steps=0,
        **FP4_256,
    ),
    Instance("fp4_cluster_4x4", cluster=(4, 4), edge=(False, False), **FP4_256),
    Instance(
        "fp4_cluster_4x4_tail_3",
        cluster=(4, 4),
        edge=(False, False),
        tail_steps=3,
        **FP4_256,
    ),
    # FP4 tile_k 1024: two GL2 lines per row.
    Instance("fp4_k1024", edge=(False, False), **FP4_1024),
    Instance("fp4_k1024_cluster_2x2", cluster=(2, 2), edge=(False, False), **FP4_1024),
]

BRANCH = re.compile(r"^\s*s_(?:c)?branch\w*\s+(\.LBB\w+)")
LABEL = re.compile(r"^(\.LBB\w+):")
DS_OP = re.compile(r"^\s*ds_(\w+)")
DSCNT = re.compile(r"^\s*s_wait_dscnt\s+(\S+)")


def kernel_body(asm):
    """The lines of the kernel with TDM loads, from its symbol to s_endpgm."""
    lines = asm.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^_Z\w*:", line):
            start = i
        elif start is not None and "s_endpgm" in line:
            body = lines[start : i + 1]
            if any("tensor_load_to_lds" in x for x in body):
                return body, start
            start = None
    raise RuntimeError("no kernel with tensor_load_to_lds")


def kernel_stats(asm):
    """NumVgprs and ScratchSize of the kernel with TDM loads (the largest NumVgprs)."""
    vgprs = [int(v) for v in re.findall(r"; NumVgprs: (\d+)", asm)]
    scratch = [int(v) for v in re.findall(r"; ScratchSize: (\d+)", asm)]
    i = max(range(len(vgprs)), key=lambda j: vgprs[j])
    return vgprs[i], scratch[i]


def k_loop(body):
    """(header, back-edge) line indices of the K loop: the last back-edge whose body has a TDM
    load, or None."""
    labels = {}
    loop = None
    for i, line in enumerate(body):
        m = LABEL.match(line)
        if m:
            labels[m.group(1)] = i
        m = BRANCH.match(line)
        if m and m.group(1) in labels:
            head = labels[m.group(1)]
            if any("tensor_load_to_lds" in x for x in body[head:i]):
                loop = (head, i)
    return loop


def check_dscnt(path):
    """Failures of one straight-line path of (line number, text): a tensor_load_to_lds with a
    ds_load in flight that was issued before the barrier preceding the refill's barrier.
    """
    failures = []
    in_flight = []  # (line number, is_load, barrier epoch at issue), oldest first
    epoch = 0
    # Adjacent barriers (a workgroup and a cluster barrier, only waits between) count once.
    adjacent = False
    for num, line in path:
        text = line.strip()
        if not text or text.startswith((";", ".")) or LABEL.match(text):
            continue
        m = DS_OP.match(line)
        if m:
            in_flight.append((num, m.group(1).startswith("load"), epoch))
        elif m := DSCNT.match(line):
            keep = int(m.group(1), 0)
            in_flight = in_flight[max(0, len(in_flight) - keep) :] if keep else []
            continue
        elif text.startswith("s_barrier_wait"):
            epoch += not adjacent
            adjacent = True
            continue
        elif text.startswith(("s_barrier_signal", "s_wait_", "s_nop")):
            continue
        elif "tensor_load_to_lds" in line:
            stale = [n for n, is_load, e in in_flight if is_load and e < epoch - 1]
            if stale:
                failures.append(
                    f"line {num}: tensor_load_to_lds with ds_loads in flight from before the "
                    f"previous barrier (lines {', '.join(map(str, stale[:8]))}"
                    f"{', ...' if len(stale) > 8 else ''})"
                )
        adjacent = False
    return failures


REG = re.compile(
    r"\b([vs])(\d+)(?:\s*/\*v(\d+)\*/)?|\b([vs])\[(\d+):(\d+)\](?:\s*/\*v\[(\d+):(\d+)\]\*/)?"
)
# Instructions whose first operand is a register they write.
WRITES = re.compile(r"^\s*(v_|s_|ds_load|global_load|buffer_load|scratch_load)")
NO_DEST = re.compile(
    r"^\s*s_(cmp|bitcmp|cbranch|branch|wait|barrier|nop|set|sleep|endpgm|delay|setprio|"
    r"cluster|prefetch|dcache|icache|ttrace|sendmsg)"
)
CLAMP = re.compile(r"^\s*[vs]_(min|med3|min3)\w*\s")


def regs(operands):
    """The VGPRs and SGPRs ("v12", "s4") of an operand string; with VGPR MSBs the real VGPR is in
    the comment."""
    found = set()
    for m in REG.finditer(operands):
        if m.group(1):
            found.add(f"{m.group(1)}{m.group(3) or m.group(2)}")
        else:
            lo, hi = (
                (m.group(7), m.group(8)) if m.group(7) else (m.group(5), m.group(6))
            )
            found.update(f"{m.group(4)}{r}" for r in range(int(lo), int(hi) + 1))
    return found


def split_operands(line):
    """(mnemonic, first operand, the other operands) of an instruction line."""
    text = line.split(";")[0].strip()
    mnemonic, _, rest = text.partition(" ")
    depth, i = 0, 0
    for i, c in enumerate(rest):
        depth += c == "["
        depth -= c == "]"
        if c == "," and depth == 0:
            return mnemonic, rest[:i], rest[i + 1 :]
    return mnemonic, rest, ""


def address_clamped(path, at):
    """Whether a min (v_ or s_min, v_med3, v_min3) computes part of the address of the
    instruction at path[at]: a backward slice over the registers it reads."""
    _, first, rest = split_operands(path[at][1])
    wanted = regs(first + "," + rest)
    for i in range(at - 1, -1, -1):
        line = path[i][1]
        if not wanted:
            return False
        if not WRITES.match(line) or NO_DEST.match(line):
            continue
        _, dst, src = split_operands(line)
        written = regs(dst) & wanted
        if not written:
            continue
        if CLAMP.match(line):
            return True
        wanted = (wanted - written) | regs(src)
    return False


def check_gl2_clamp(pre, loop_body):
    """Failures of a kernel with a K tail: a GL2 prefetch in the K loop whose address no v_min
    clamps (the data lines' offset to the end of the row). Edge instances clamp the row with a
    v_min too, so only instances without edges tell the clamps apart."""
    path = pre + loop_body + loop_body
    failures = []
    for at in range(len(pre) + len(loop_body), len(path)):
        num, line = path[at]
        if "global_prefetch" in line and not address_clamped(path, at):
            failures.append(
                f"line {num}: GL2 prefetch address not clamped: {line.strip()}"
            )
    return failures


def check(asm, post_loop_refill_free=True, gl2_clamp=True):
    """The failures of a kernel's ISA. post_loop_refill_free is false when a tile that refills
    follows the loop: TailNumber::Full's odd tile, which feeds the K tail. gl2_clamp: every GL2
    prefetch address in the K loop goes through a v_min."""
    body, start = kernel_body(asm)
    numbered = [(start + i + 1, line) for i, line in enumerate(body)]
    loop = k_loop(body)
    failures = []
    if loop is None:
        pre, loop_body, post = numbered, [], []
    else:
        head, edge = loop
        pre, loop_body, post = (
            numbered[:head],
            numbered[head : edge + 1],
            numbered[edge + 1 :],
        )
    for num, line in post if post_loop_refill_free else []:
        if "tensor_load_to_lds" in line or "global_prefetch" in line:
            failures.append(f"line {num}: after the K loop: {line.strip()}")
    # The paths through the loop: skipped, and run twice (its steady state) before the tail.
    paths = [pre + post] + ([pre + loop_body + loop_body + post] if loop_body else [])
    for path in paths:
        for f in check_dscnt(path):
            if f not in failures:
                failures.append(f)
    if gl2_clamp and loop_body:
        failures += check_gl2_clamp(pre, loop_body)
    return failures


def compile_asm(clang, inst, tail_num, overlay, workdir):
    tag = f"{inst.name}.{tail_num}" + (".base" if overlay else "")
    src = os.path.join(workdir, f"{tag}.hip")
    out = os.path.join(workdir, f"{tag}.s")
    with open(src, "w") as f:
        f.write(inst.source(tail_num))
    includes = (["-I", overlay] if overlay else []) + [
        "-I",
        os.path.join(CK_ROOT, "include"),
    ]
    cmd = [clang, src, "-o", out] + FLAGS + inst.flags() + includes
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{tag}: compile failed:\n{res.stderr[-4000:]}")
    with open(out) as f:
        return f.read()


def make_overlay(rev, workdir):
    """A directory with the pipeline headers at git revision rev, ahead of include/ on -I."""
    overlay = os.path.join(workdir, "baseline_include")
    for header in PIPELINE_HEADERS:
        text = subprocess.run(
            ["git", "show", f"{rev}:./{header}"],
            cwd=CK_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        dst = os.path.join(overlay, os.path.relpath(header, "include"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w") as f:
            f.write(text)
    return overlay


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clang", default="/opt/rocm/llvm/bin/clang++")
    parser.add_argument(
        "--baseline", metavar="REV", help="compare VGPRs with REV's headers"
    )
    parser.add_argument("--jobs", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--only", help="instances whose name contains this")
    parser.add_argument("--keep", metavar="DIR", help="keep the sources and ISA in DIR")
    parser.add_argument(
        "--verbose", action="store_true", help="print the baseline's failures"
    )
    args = parser.parse_args()

    workdir = args.keep or tempfile.mkdtemp(prefix="tdm_tail_isa_")
    os.makedirs(workdir, exist_ok=True)
    overlay = make_overlay(args.baseline, workdir) if args.baseline else None
    jobs = [
        (inst, tail_num)
        for inst in INSTANCES
        if not args.only or args.only in inst.name
        for tail_num in inst.tail_nums()
    ]
    try:
        with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
            new = {
                job: pool.submit(compile_asm, args.clang, *job, None, workdir)
                for job in jobs
            }
            base = {
                job: pool.submit(compile_asm, args.clang, *job, overlay, workdir)
                for job in (jobs if overlay else [])
            }
            failed = 0
            for job in jobs:
                inst, tail_num = job
                asm = new[job].result()
                failures = check(asm, **inst.checks())
                vgprs, scratch = kernel_stats(asm)
                line = (
                    f"{inst.name:32} {tail_num:4} vgprs {vgprs:3} scratch {scratch:4}"
                )
                base_failures = []
                if overlay:
                    base_asm = base[job].result()
                    base_failures = check(base_asm, **inst.checks())
                    base_vgprs, base_scratch = kernel_stats(base_asm)
                    line += (
                        f"  baseline vgprs {base_vgprs:3} scratch {base_scratch:4}"
                        f"  baseline failures {len(base_failures)}"
                    )
                    if scratch > base_scratch:
                        failures.append(f"scratch {scratch} > baseline {base_scratch}")
                print(("FAIL " if failures else "ok   ") + line)
                for f in failures:
                    print(f"       {f}")
                for f in base_failures if args.verbose else []:
                    print(f"       baseline: {f}")
                failed += bool(failures)
    finally:
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
    print(f"{len(jobs) - failed} of {len(jobs)} kernels pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
