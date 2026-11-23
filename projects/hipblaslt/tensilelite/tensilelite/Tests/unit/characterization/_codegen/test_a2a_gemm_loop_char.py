# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 A2A+GEMM (FusedA2A: A2AGemm) codegen characterization.

Checks that a FusedA2A: A2AGemm solution emits the enqueuer election, the SDMA
ring instructions, the exit tally and the drain-send wait, one segment switch per
prefetch stage after the first, one after the unrolled loop and one at the first
loop copy's exit under ExpandPointerSwap or UnrollLoopSwapGlobalReadOrder, their
step-backs and branch targets, the A2ASubRounds-dependent round count, flag-slot
and sub-round instructions, no tail loop and no StaggerU wrap iteration, that the
segment switches touch no VGPR but lane 0 of Serial under EXEC=1 and restore it,
no LDS or barrier, that the drain-send wait runs right before s_endpgm, and that
the result assembles for gfx950. UseSubtileImpl kernels switch before the PGR=2
batch-1 skip and after the loop exit label, never inside the loop body, leave
the fixed Srd+2 window alone, and count the work-group out after the loop. The
enqueuer's lanes stop at the group start, and a2aBatchSpan, run on a grid of
shapes, leaves the batch, the group's first block and the first gathered segment
in the A2A registers. The enqueuer's group seed and group loop head, run on a grid
of batches, visit each queue's peer groups in order.
"""

import functools
import itertools
import math
import os
import re

import pytest

from config_harness import assert_assembles, emit_kernels_from_config
from tensilelite.Components.Signature import fusedA2AKernArgLayout

pytestmark = pytest.mark.unit

_ARCH = "gfx950"

_COMM = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,  # characterization/
        os.pardir,  # unit/
        os.pardir,  # tensilelite/Tests/
        "common", "comm", "gfx950",
    )
)
_CONFIG = os.path.join(_COMM, "a2a_gemm_loop.yaml")
_SUB_ROUNDS = re.compile(r"\.amdhsa_kernel \S*_AASR(\d+)_")
_PGR = re.compile(r"\.amdhsa_kernel \S*_PGR(\d+)_")
_EPS = re.compile(r"\.amdhsa_kernel \S*_EPS(\d+)_")
_ULSGRO = re.compile(r"\.amdhsa_kernel \S*_ULSGRO(\d+)_")
_USI = re.compile(r"\.amdhsa_kernel \S*_USI(\d+)_")
_DEPTHU = re.compile(r"\.amdhsa_kernel \S*_MT\d+x\d+x(\d+)_")
_PEER_GROUPS = re.compile(r"\.amdhsa_kernel \S*_AAPG(\d+)_")
_GROUPS = [0, 1, 2, 3, 4, 5, 6, 7]
_LIMIT = 24

_A2A_GEMM_MARKERS = (
    "label_A2AWaitFlag:",
    "label_A2ALastSegment:",
    "label_A2ASkipEnqueue:",
    "s_atomic_inc",
    "s_atomic_umax_x2",
    "label_a2a_exit_tally_end:",
    "label_a2a_drain_send_end:",
)


def _sub_round_lines(subRounds, switches):
    """Full-line patterns of the A2ASubRounds-dependent instructions -> expected count."""
    split = subRounds > 1
    lines = {
        r"s_mov_b32 s\[sgprA2AShardCounter\], s\[sgprA2AW\] +// rounds = W": int(not split),
        rf"s_lshl_b32 s\[sgprA2AShardCounter\], s\[sgprA2AShardCounter\], {int(math.log2(subRounds))} +// \(W-1\) \* S": int(split),
        r"s_add_u32 s\[sgprA2AShardCounter\], s\[sgprA2AShardCounter\], 1 +// rounds = 1 \+ \(W-1\) \* S": int(split),
        r"s_lshl_b32 (s\d+), \1, 2 +// flag slot byte offset": subRounds + switches,
        rf"s_lshl_b32 (s\d+), \1, {int(math.log2(subRounds))} +// \(gLo\*\(W-1\) \+ i\) \* S": subRounds if split else 0,
        rf"s_lshl_b32 (s\d+), \1, {int(math.log2(subRounds))} +// slot \* S": switches if split else 0,
        r"s_add_u32 (s\d+), \1, s\d+ +// \+ sub-round": switches if split else 0,
        rf"s_and_b32 (s\d+), \1, {subRounds - 1} +// sub-round = \(1 - rounds left\) mod S": switches if split else 0,
    }
    for h in range(1, subRounds):
        lines[rf"s_add_u32 (s\d+), \1, {h} +// \+ sub-round"] = 1
    return lines


def _switches(src):
    """Bodies of the segment switches, from the flag slot to the EXEC restore."""
    return re.findall(r"^s_add_u32 \S+, s\[sgprA2AFlagRow\], s\[sgprA2AShardIdx\].*?"
                      r"^label_A2AFlagReady\w*:\n[^\n]*\n^s_mov_b64 exec, -1\b[^\n]*",
                      src, re.M | re.S)


# The only VGPR accesses a segment switch may make: lane 0 of Serial carries the flag poll.
_SERIAL_LANE0 = re.compile(
    r"^(?:v_readfirstlane_b32 s\d+, v\[vgprSerial\]"
    r"|v_mov_b32 v\[vgprSerial\], s\d+"
    r"|global_load_dword v\[vgprSerial\], v\[vgprSerial\], s\[sgprFusedCounterPtr:sgprFusedCounterPtr\+1\] .*\bsc0 sc1\b)")

# SALU op -> (result, SCC) from (operands, SCC).
_SALU = {
    "s_mul_i32": lambda a, b, c: (a * b, c),
    "s_add_u32": lambda a, b, c: (a + b, a + b > 0xFFFFFFFF),
    "s_sub_u32": lambda a, b, c: (a - b, b > a),
    "s_subb_u32": lambda a, b, c: (a - b - c, b + c > a),
    "s_min_u32": lambda a, b, c: (min(a, b), a < b),
    "s_max_u32": lambda a, b, c: (max(a, b), a > b),
    "s_max_i32": lambda a, b, c: (max(_i32(a), _i32(b)), _i32(a) > _i32(b)),
    "s_cselect_b32": lambda a, b, c: (a if c else b, c),
}
_SCMPS = {
    "s_cmp_eq_u32": lambda a, b: a == b,
    "s_cmp_gt_u32": lambda a, b: a > b,
    "s_cmp_ge_u32": lambda a, b: a >= b,
}
_OPS = {
    "v_cvt_f64_u32": float,
    "v_rcp_f64": lambda a: 1.0 / a,
    "v_mul_f64": lambda a, b: a * b,
    "v_cvt_u32_f64": int,
    "v_mul_lo_u32": lambda a, b: a * b,
    "v_sub_u32": lambda a, b: a - b,
    "v_add_u32": lambda a, b: a + b,
    "v_readfirstlane_b32": lambda a: a,
    "v_mov_b32": lambda a: a,
    "s_mov_b32": lambda a: a,
}


def _i32(x):
    return x - ((x & 0x80000000) << 1)


def _lane(v, L):
    return v if L == "0" else f"{v}@{L}"


def _insts(text):
    """(op, operands) of the instructions in text."""
    lines = [line.split("//")[0].strip() for line in text.splitlines()]
    return [(op, [re.sub(r"v\[(\d+)(?::\d+)?\]", r"v\1", a) for a in args.split(", ")])
            for op, _, args in (line.partition(" ") for line in lines if line)]


def _batch_span(src):
    """The FusedNumCu register and the (op, operands) of a2aBatchSpan from b * F through
    the first gathered segment."""
    layout = fusedA2AKernArgLayout()
    wOff = int(re.search(r"^s_load_dword s\[sgprA2AW\], \S+, (\w+)", src, re.M).group(1), 0)
    cuOff = wOff + layout["FusedNumCu"] - layout["FusedW"]
    [numCu] = re.findall(rf"^s_load_dword (s\d+), \S+, {cuOff:#x}\b", src, re.M)
    [first] = re.finditer(r"^.*// b \* F *$", src, re.M)
    start = first.start()
    end = src.index("\n", src.index("// first gathered segment = G + 1", start))
    return numCu, _insts(src[start:end])


def _run(insts, regs):
    """regs after running insts on one lane; key f"{v}@{L}" is lane L > 0 of VGPR v."""
    lane = True
    scc = None
    for op, args in insts:
        vals = [regs[a] if a in regs else int(a, 0) for a in args[op not in _SCMPS:]]
        if op == "v_cmpx_ge_u32":
            lane = regs[args[1]] >= regs[args[2]]
        elif op == "s_mov_b64":
            assert args == ["exec", "-1"], args
            lane = True
        elif op == "v_readlane_b32":
            regs[args[0]] = regs[_lane(args[1], args[2])]
        elif op == "v_writelane_b32":
            regs[_lane(args[0], args[2])] = vals[0]
        elif op in _SCMPS:
            scc = _SCMPS[op](*vals)
        elif op in _SALU:
            assert scc is not None or op not in ("s_subb_u32", "s_cselect_b32"), f"{op} before SCC is set"
            v, scc = _SALU[op](*vals, scc)
            regs[args[0]] = v & 0xFFFFFFFF
        elif op != "s_nop" and (op != "v_add_u32" or lane):
            v = _OPS[op](*vals)
            regs[args[0]] = v & 0xFFFFFFFF if isinstance(v, int) else v
            for key in [k for k in regs if k.startswith(args[0] + "@")]:
                del regs[key]
    return regs


def _group_sizes(n, W, cap=0):
    """Block counts of the peer groups of a batch of n blocks, in order; cap is A2APeerGroups."""
    if cap:
        ng = min(max(W - 1, 1), n, cap)
        gk = -(-n // ng)
    else:
        gk = -(-n // max(W - 1, 1))
        ng = -(-n // gk)
    s = ng * gk - n
    return [gk] * (ng - s) + [gk - 1] * s


@functools.cache
def _peer_group_kernels():
    """(A2APeerGroups, basename, source) of one A2ASubRounds 1 kernel per A2APeerGroups value."""
    [(base, src, _err)] = emit_kernels_from_config(_CONFIG, limit=1, arch=_ARCH,
                                                   size_group_index=_GROUPS[-2])
    kernels = [(0, base, src)]
    for base, src, _err in emit_kernels_from_config(_CONFIG, limit=_LIMIT, arch=_ARCH,
                                                    size_group_index=_GROUPS[-1]):
        if _SUB_ROUNDS.search(src).group(1) == "1":
            kernels.append((int(_PEER_GROUPS.search(src).group(1)), base, src))
    kernels.sort(key=lambda kernel: kernel[0])
    assert [k for k, _b, _s in kernels] == [0, 1, 2, 7], [b for _k, b, _s in kernels]
    return kernels


def _group_of(i, sizes):
    """(G, the group's first block offset) of block offset i."""
    lo = 0
    for g, d in enumerate(sizes):
        if i < lo + d:
            return g, lo
        lo += d
    raise AssertionError((i, sizes))


@pytest.mark.parametrize("group", _GROUPS)
def test_a2a_gemm_gfx950_emits_assembly(group):
    results = emit_kernels_from_config(_CONFIG, limit=_LIMIT, arch=_ARCH, size_group_index=group)
    assert len(results) >= 1, f"Expected >=1 kernel, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), [
        (b, e) for (b, _s, e) in results if e != 0
    ]
    for base, src, _err in results:
        assert_assembles(src, base)
        missing = [m for m in _A2A_GEMM_MARKERS if m not in src]
        assert not missing, f"Kernel {base!r}: missing A2AGemm markers {missing}"
        subRounds = _SUB_ROUNDS.search(src)
        assert subRounds, f"Kernel {base!r}: no AASR<n> in the kernel name"
        pgr = int(_PGR.search(src).group(1))
        eps = int(_EPS.search(src).group(1))
        ulsgro = int(_ULSGRO.search(src).group(1))
        subtile = int(_USI.search(src).group(1))
        if subtile:
            eps = ulsgro = 0
        switches = _switches(src)
        assert len(switches) == max(pgr - 1, 0) + 1 + eps + ulsgro, \
            f"Kernel {base!r}: {len(switches)} segment switches, PGR={pgr} EPS={eps} ULSGRO={ulsgro}"
        for switch in switches:
            touched = [line for line in switch.splitlines()
                       if re.match(r"(v_|ds_|buffer_|global_|s_barrier)", line) and not _SERIAL_LANE0.match(line)]
            assert not touched, f"Kernel {base!r}: segment switch emits {touched}"
            save = re.search(r"^s_mov_b64 exec, 1\b.*\n^v_readfirstlane_b32 (s\d+), v\[vgprSerial\]", switch, re.M)
            assert save and switch.index(save.group(0)) < switch.index("v_mov_b32 v[vgprSerial]"), \
                f"Kernel {base!r}: segment switch writes Serial before narrowing EXEC to lane 0"
            assert re.search(rf"^label_A2AFlagReady\w*:\n^v_mov_b32 v\[vgprSerial\], {save.group(1)}\b.*\n"
                             r"^s_mov_b64 exec, -1\b", switch, re.M), \
                f"Kernel {base!r}: segment switch does not restore Serial lane 0 and EXEC"
        if subtile:
            stepBack = rf"{2 * int(_DEPTHU.search(src).group(1))}"
            backs = [len(re.findall(rf"^s_sub_u32 s\[sgprSrd[AB]\+0\], s\[sgprSrd[AB]\+0\], {stepBack} +"
                                    r"// one DepthU back$", switch, re.M)) for switch in switches]
            assert backs == [0] * (pgr == 2) + [2 * (pgr >= 1)], \
                f"Kernel {base!r}: DepthU step-backs per switch {backs}, PGR={pgr}"
            assert not any(re.search(r"s\[sgpr(?:Srd[AB]\+2|ShadowLimit[AB]|GlobalReadIncs[AB])", switch)
                           for switch in switches), f"Kernel {base!r}: segment switch restores a read limit"
            exitLabel = "SkipToNGLL" if pgr == 2 else "SkipToNLL"
            assert re.search(rf"^label_{exitLabel}:\n^s_cmp_eq_u32 s\[sgprA2AShardCounter\], 1 +// last segment\?$",
                             src, re.M), f"Kernel {base!r}: no segment switch right after label_{exitLabel}"
            if pgr == 2:
                assert re.search(r"^label_A2APrefetch1:\n^s_cmp_le_u32 s\[sgprLoopCounterL\], 1\b", src, re.M), \
                    f"Kernel {base!r}: no segment switch right before the batch-1 skip"
            loop = re.search(r"^label_LoopBeginL:$.*?^s_cbranch_scc0 label_LoopBeginL\b", src, re.M | re.S)
            assert loop and "A2A" not in loop.group(0), f"Kernel {base!r}: A2A code inside the loop body"
            assert "sgprStaggerUIter" not in src, f"Kernel {base!r}: StaggerU code emitted"
            assert re.search(r"^label_SkipToEnd:$.*^label_a2a_exit_tally_end:$.*^label_Summation_End\w*:",
                             src, re.M | re.S), f"Kernel {base!r}: exit tally not between the loop exit and the store"
        else:
            stepBacks = len(re.findall(r"^s_sub_u32 s\[sgprSrd[AB]\+0\], s\[sgprSrd[AB]\+0\], s\[sgprGlobalReadIncs[AB]\+0\] +// one GlobalReadIncs back$",
                                       src, re.M))
            assert stepBacks == 2 * (pgr >= 2) * (1 + ulsgro), \
                f"Kernel {base!r}: {stepBacks} GlobalReadIncs step-backs, PGR={pgr} ULSGRO={ulsgro}"
            assert re.search(r"^s_mov_b32 s\[sgprStaggerUIter\], 0xffffffff\b", src, re.M), \
                f"Kernel {base!r}: StaggerUIter can still select a wrap iteration"
        targets = sorted(re.findall(r"^s_branch label_(\w+) +// read the next segment$", src, re.M))
        assert targets == sorted(["LoopBeginL"] + ["LoopCopy2L"] * (eps + ulsgro)), \
            f"Kernel {base!r}: segment switches branch to {targets}"
        expected = _sub_round_lines(int(subRounds.group(1)), len(switches))
        lines = src.splitlines()
        counts = {p: sum(1 for line in lines if re.fullmatch(p, line)) for p in expected}
        assert counts == expected, f"Kernel {base!r}: A2ASubRounds lines {counts} != {expected}"
        assert "label_SkipTailLoopL:" not in src, f"Kernel {base!r}: tail loop emitted"
        row = re.search(r"^s_add_u32 (\S+), \1, \S+ +// first block of group g$", src, re.M)
        bounds = set(re.findall(r"^v_cmp_le_i32 vcc, (\S+), v\d+ +// inside the group\?$", src, re.M))
        assert row and bounds == {row.group(1)}, \
            f"Kernel {base!r}: lanes bounded by {bounds}, not the group start"
        assert src.count("label_a2a_drain_send_end:") == 1 and \
            re.search(r"^label_a2a_drain_send_end:\n^s_endpgm\b", src, re.M), \
            f"Kernel {base!r}: drain-send wait does not end the kernel"


def test_a2a_gemm_gfx950_batch_span():
    assert _group_sizes(16, 8) == [3, 3, 3, 3, 2, 2] and _group_sizes(22, 4) == [8, 7, 7]
    assert _group_sizes(8, 8) == [2, 2, 2, 2] and _group_sizes(8, 4) == [3, 3, 2]
    assert _group_sizes(16, 8, 7) == [3, 3, 2, 2, 2, 2, 2] and _group_sizes(8, 8, 7) == [2, 1, 1, 1, 1, 1, 1]
    assert _group_sizes(16, 8, 1) == [16] and _group_sizes(5, 8, 2) == [3, 2]
    shapes = list(itertools.product((32, 256), range(2, 9), (2, 11, 12, 32, 44), (2, 7, 8, 44)))
    # F = 1, numCu = n: batches of exactly n blocks, then a short last batch.
    shapes += [(n, W, 1, 2 * n + max(n // 3, 1)) for W in range(1, 9) for n in range(1, 41)]
    for cap, base, src in _peer_group_kernels():
        numCuReg, insts = _batch_span(src)
        for numCu, W, F, T in shapes:
            for b in range(T):
                regs = _run(insts, {numCuReg: numCu, "s[sgprA2AW]": W,
                                    "s[sgprWorkGroup1]": b, "s[sgprNumWorkGroups0]": F,
                                    "s[sgprNumWorkGroups1]": T})
                iB = b * F // numCu
                bLo = -(-iB * numCu // F)
                bHi = min(-(-(iB + 1) * numCu // F), T)
                g, gLo = _group_of(b - bLo, _group_sizes(bHi - bLo, W, cap))
                got = [regs[f"s[sgpr{r}]"] for r in ("A2ABlockLo", "A2ABlockCount", "A2AShardIdx",
                                                     "A2AShardCounter")]
                assert got == [bLo, bHi - bLo, g + 1, bLo + gLo], \
                    f"Kernel {base!r}: numCu={numCu} W={W} F={F} T={T} b={b} -> {got}"


def test_a2a_gemm_gfx950_group_walk():
    for cap, base, src in _peer_group_kernels():
        [seed] = re.finditer(r"// wrap s once at W *\n(.*?// NG -> lane 1 *)$", src, re.M | re.S)
        [walk] = re.finditer(r"^[^\n]*// blocks left on this queue *$.*?// first block of this group *$",
                             src, re.M | re.S)
        pre, _, head = walk.group(0).partition("\nlabel_a2a_group_loop")
        head = head.partition("\n")[2]
        vGK = int(re.search(r"^v_mov_b32 v(\d+), \S+ +// GK -> packet state$", seed.group(1), re.M).group(1))
        for lo, hi in re.findall(r"^v_\w+ v\[?(\d+)(?::(\d+)\])?,", src[seed.end():walk.start()], re.M):
            assert not int(lo) <= vGK <= int(hi or lo), f"Kernel {base!r}: v{vGK} written before the group loop"
        qReg = re.search(r"^s_add_u32 \S+, (s\d+), 1 +// i \+ 1$", pre, re.M).group(1)
        loReg = re.search(r"^s_add_u32 (\S+), \1, (s\d+) +// first block of group g$", pre, re.M).group(2)
        rowReg, nBlkReg = re.search(r"^s_sub_u32 (\S+), \1, (\S+) +// first block of this group$",
                                    head, re.M).groups()
        setup, head = _insts(seed.group(1)) + _insts(pre), _insts(head)
        bLo = 1000
        for W in range(2, 9):
            for n in range(1, 71):
                sizes = _group_sizes(n, W, cap)
                starts = [sum(sizes[:g]) for g in range(len(sizes))]
                for i in range(W - 1):
                    regs = _run(setup, {"s[sgprA2AW]": W, "s[sgprA2ABlockCount]": n, qReg: i, loReg: bLo})
                    got, left = [], n
                    while left > 0 and len(got) <= len(sizes):
                        regs = _run(head, regs)
                        got.append((regs[rowReg] - bLo, regs[nBlkReg]))
                        left -= regs[nBlkReg]
                    order = [g for g in itertools.chain(range(i, -1, -1), range(W - 2, i, -1)) if g < len(sizes)]
                    assert got == [(starts[g], sizes[g]) for g in order], \
                        f"Kernel {base!r}: W={W} n={n} queue {i} places (first, count) {got}"


@pytest.mark.parametrize("group", _GROUPS)
def test_a2a_gemm_gfx950_golden(snapshot, group):
    """Order-invariant golden: pin {basename, err} for every emitted A2AGemm kernel."""
    digest = sorted(
        (
            {"basename": b, "err": e}
            for (b, _s, e) in emit_kernels_from_config(_CONFIG, limit=_LIMIT, arch=_ARCH,
                                                       size_group_index=group)
        ),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot

