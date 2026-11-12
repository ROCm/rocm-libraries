# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Split-K / stream-K work units with a fused, deterministic fix-up for the MXFP6 tile library (``splitk="fused"``).

Every field below is default-off: a spec with ``splitk="none"`` and ``m_mask=False`` builds the IR it built before.

Work units (one CTA each, grid ``(units, 1, batch)``):

* the first ``D`` tiles of an entry (grouped tile order) are data-parallel units: the whole K, C straight from the
  accumulators;
* each of the other ``T - D`` tiles is split into ``s`` units along K: unit ``j`` runs K tiles ``[j c, (j + 1) c)``
  (``c = K / 128 / s``). ``D = T`` (``sk_split = 1``) is the plain data-parallel grid; ``D = 0`` is pure split-K;
  ``sk_hybrid`` keeps the full waves of tiles data-parallel and splits only the last partial wave (stream-K's DP + SK
  hybrid, quantized to whole K chunks).

K range of a unit: the operand / scale base pointers move by ``k0 = j c`` K tiles (every K-tile address of the
builder is linear in the tile index; the c0c1 row-block stride keeps the whole K) and the loop runs ``c`` tiles (the
runtime-K loop count of ``rt_tail="fixed"``: ``c`` must be in the build's K residue class).

Fix-up (no waiting anywhere, so no co-residency assumption and no deadlock): every wave of a split unit stores its
fp32 accumulators to its slot of the workspace (``ws``: [entry][split tile][split][wave][atom][lane] x 16 bytes),
releases them (agent-scope release fence, or write-through stores + ``vmcnt(0)`` under ``sk_release="wt"``), then adds
1 to its own counter (``cnt``: [entry][split tile][wave] int32, zero before the first launch). The wave that sees the
counter at ``s - 1`` is the last of its tile's ``s`` waves: it acquires, sums the ``s`` partials **in split order
0, 1, .., s - 1** (its own from registers) and stores C, then resets the counter to 0 for the next launch. The
result is the same bits whichever unit arrives last: repeat launches are bitwise identical.

Must-fail mutants (``SK_MUTANTS``): ``sk_drop`` (the reducer skips the partial of split (j + 1) mod s),
``sk_order`` (the partials summed in reverse split order: caught by the order set for s >= 3), ``sk_noflag``
(split 0 never signals: no unit reduces), ``sk_noreset`` (the counter is not reset: the next launch never reduces).
``sk_arrival`` (``SK_SHOULD_FAIL``: the reducer's own partial first, i.e. an arrival-dependent order) differs from the
split order only when the reducer is split >= 2 (fp32 addition commutes, so the first two may swap): it fails
whenever such a unit arrives last somewhere, which the GPU runs report but do not require.

``m_mask`` (decode, any M >= 1): the canonical A buffer descriptor ends at row M (rows past M read as zeros), tiles
along M are ceil(M / tile_m) without the edge shift, and C stores of rows >= M are predicated off. A then needs only
M rows (the per-call A scale image is built per CTA tile as usual).

Knobs (all default-off / host-only where noted): ``sk_static`` (the split count baked in: the reducer issues every
partial load of a group of atoms before its first add), ``sk_release`` ("fence" | "wt" write-through sc0 sc1 |
"wt1" sc1 only), ``sk_fix`` ("last" | "owner": split s - 1 polls its peers' flags; co-residency-gated by
``problem_ok``), ``sk_hybrid`` / ``sk_wave`` / ``sk_split`` (host plan). An **exact-K short build** (reference k =
128 c, ``short_spec`` / ``unit_spec``) serves units shorter than the library's two-iteration minimum. K tails
(``k_tail``) run on data-parallel units only (s = 1). ``select`` / ``select_config`` / ``predict_us``: the per-problem
choice ``library.dispatch`` uses for M < MT1 and short grids.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from rocke.core.ir import F32, I32, I64, PtrType, Value, VectorType

SPLITK_MODES = ("none", "fused")
SK_RELEASES = ("fence", "wt", "wt1")
SK_FIXES = ("last", "owner")
SK_MUTANTS = ("sk_drop", "sk_order", "sk_noflag", "sk_noreset", "sk_arrival")
MASK_MUTANTS = ("mask_over", "mask_under")  # m_mask must-fails: C rows < M + 1 stored (row M of the last row tile:
#   past C -- required when M % tile_m != 0, caught by the harness's guard after C) / rows < M - 1 (row M - 1 unwritten)
SK_SHOULD_FAIL = ("sk_arrival",)  # arrival-dependent: fails in practice, not by construction
#: workspace stores of the "wt" release: sc0 sc1 (write-through at system scope, as MXFP4 TensileLite's partials)
CP_WT = 1 | 16
CP_WT1 = 16  # "wt1": sc1 only (agent scope, as LLVM's gfx940 agent-scope atomic store)
OOB = 0x7FFFFFF0  # a voffset past every workspace descriptor: the load reads 0 and moves no data
N_CU = 256
SC_SHIFT = 16  # kernel argument sk_sc = s | c << SC_SHIFT (s <= 64 splits, c < 2^16 K tiles per split)


def _lcm(a: int, b: int) -> int:
    from math import gcd

    return a * b // gcd(a, b)


def validate(spec) -> Tuple[bool, str]:
    """Structural checks of the split-K / m_mask fields (called from gemm._validate_library)."""
    from .gemm import line_operands, operand_layout_of

    if spec.splitk not in SPLITK_MODES:
        return False, f"splitk must be one of {SPLITK_MODES}"
    if spec.sk_release not in SK_RELEASES:
        return False, f"sk_release must be one of {SK_RELEASES}"
    if spec.sk_kargs not in ("packed", "wide"):
        return False, "sk_kargs must be packed or wide"
    if not isinstance(spec.sk_split, int) or not 1 <= spec.sk_split <= 64:
        return False, "sk_split must be 1..64"
    if spec.mutant in SK_MUTANTS and spec.splitk == "none":
        return False, f"mutant {spec.mutant} needs splitk fused"
    if spec.mutant in MASK_MUTANTS and not spec.m_mask:
        return False, f"mutant {spec.mutant} needs m_mask"
    if spec.m_mask and spec.splitk == "none":
        return False, "m_mask: splitk fused (the masked C stores are the fix-up's)"
    if spec.splitk == "none":
        return True, ""
    if not spec.runtime_shape or spec.rt_tail != "fixed":
        return False, "splitk: runtime_shape, rt_tail fixed (a unit's K range is a runtime K of the build's residue)"
    if spec.schedule != "dp" or spec.persist_overlap or getattr(spec, "group_table", False):
        return False, "splitk: schedule dp, no persist_overlap / group_table"
    if getattr(spec, "k_tail", False) and (spec.sk_split != 1 or spec.sk_hybrid):
        return False, "splitk with k_tail: data-parallel units only (sk_split 1; split units are whole 128-K tiles)"
    if spec.atom != "16x16x128" or spec.scale_layout != "wtile" or spec.scale_raw:
        return False, "splitk: the 16x16x128 atom and wtile scales (not scale_raw)"
    if spec.probe != "none":
        return False, "splitk: no probe (the probe of a split build is its tile's dp build)"
    if spec.mutant in ("batch_swap", "persist_drop"):
        return False, f"splitk: mutant {spec.mutant} does not apply"
    if spec.m_mask:
        la = operand_layout_of(spec, "A")
        if la == "canonical" and (not spec.canon_rsrc_once or "A" in line_operands(spec)):
            return False, "m_mask: canonical A with canon_rsrc_once in the stage ring (one exact-range descriptor)"
        if la not in ("canonical", "mcontig"):
            return False, "m_mask: canonical or M-contiguous A"
    if line_operands(spec) and (spec.k // 128) % 8:
        return False, "splitk with a line ring: the reference K a multiple of 8 K tiles (the line pattern)"
    if spec.sk_fix not in SK_FIXES:
        return False, f"sk_fix must be one of {SK_FIXES}"
    if spec.sk_fix == "owner" and not spec.sk_static:
        return False, "sk_fix owner: sk_static (the owner's reduction is built for the split count)"
    if spec.sk_fix == "owner" and spec.mutant in ("sk_noflag", "sk_noreset", "sk_arrival"):
        return False, f"sk_fix owner: mutant {spec.mutant} does not apply (a missing flag hangs the owner)"
    # (sk_static with sk_split 1: data-parallel units only -- no unit map, no fix-up code; the masked / exact-K decode
    # units of select)
    if spec.mutant in ("sk_order", "sk_arrival") and spec.sk_split < 3:
        return False, f"mutant {spec.mutant} needs sk_split >= 3 (two partials commute: any order is the same sum)"
    if spec.mutant in SK_MUTANTS and spec.sk_split < 2:
        return False, f"mutant {spec.mutant} needs sk_split >= 2"
    return True, ""


def short_spec(spec, c_tiles: int):
    """``spec`` built for units of exactly ``c_tiles`` K tiles (reference k = 128 c): split units shorter than the
    library's two-iteration minimum (the runtime K must then equal k: ``problem_ok``)."""
    from dataclasses import replace

    return replace(spec, k=128 * c_tiles)


def granule(spec) -> int:
    """K tiles a split's start must be a multiple of: the wtile scale period, and the line rings' 8-tile address
    pattern (their 128-byte lines keep their phase)."""
    from .gemm import line_operands

    g = max(1, spec.wtile_period)
    if line_operands(spec):
        g = _lcm(g, 8)
    return g


def occupancy_wgs(spec) -> int:
    from .gemm import lds_bytes

    return max(1, min(spec.occupancy if spec.occupancy > 1 else 16, (160 * 1024) // max(1, lds_bytes(spec))))


def plan(spec, m: int, n: int, k: int, batch: int = 1, n_cu: int = N_CU) -> Dict[str, int]:
    """Host plan of a ``splitk="fused"`` launch: tiles T, data-parallel tiles D, splits s, K tiles per split c, units
    per entry, workspace / counter bytes."""
    from .gemm import n_tiles

    T = n_tiles(spec, "A", m) * n_tiles(spec, "B", n)
    s = spec.sk_split
    KT = -(-k // 128) if getattr(spec, "k_tail", False) else k // 128
    if s <= 1:
        D = T
    elif spec.sk_hybrid:
        W = spec.sk_wave or n_cu * occupancy_wgs(spec)
        D = T - T % W if T % W else T
        if T < W:
            D = 0
    else:
        D = 0
    if D == T:
        s = 1
    c = KT // s if s > 1 else KT
    nw = spec.waves_m * spec.waves_n
    tb = spec.tile_m * spec.tile_n * 4
    return {"T": T, "D": D, "s": s, "c": c, "KT": KT, "units": D + (T - D) * s, "nsk": T - D,
            "ws_bytes": batch * (T - D) * s * tb, "tile_bytes": tb,
            "cnt_bytes": batch * (T - D) * nw * 4 * (s if spec.sk_fix == "owner" else 1)}


def problem_ok(spec, m: int, n: int, k: int, batch: int = 1) -> Tuple[bool, str]:
    """Runtime predicates of a ``splitk="fused"`` launch (replaces gemm.runtime_problem_ok's K rule)."""
    from .gemm import loop_plan, operand_layout_of

    if spec.mutant == "batch_swap":
        return False, "batch_swap does not apply"
    if spec.m_mask:
        if m < 1:
            return False, "M >= 1"
        # (M-contiguous A, M % 16 != 0: the M-padded producer layout, ``gemm.mcontig_rows``: K rows of ceil16(M)
        # codes; the padding rows feed only the masked C rows)
    elif spec.edge_tiles:
        if m < spec.tile_m:
            return False, f"edge_tiles: M >= tile_m ({spec.tile_m}; m_mask serves smaller M)"
        if operand_layout_of(spec, "A") == "mcontig" and m % 16:
            return False, "edge_tiles with M-contiguous A: M a multiple of 16"
    elif m < 1 or m % spec.tile_m:
        return False, f"M must be a positive multiple of tile_m ({spec.tile_m})"
    if spec.edge_tiles:
        if n < spec.tile_n:
            return False, f"edge_tiles: N >= tile_n ({spec.tile_n})"
    elif n < 1 or n % spec.tile_n:
        return False, f"N must be a positive multiple of tile_n ({spec.tile_n})"
    if not 1 <= batch <= 65535:
        return False, "batch must be 1..65535 (grid z)"
    if k % (32 if getattr(spec, "k_tail", False) else 128):
        return False, "K must be a multiple of 128 (k_tail: of 32)"
    U, H, L, T0 = loop_plan(spec)
    p = plan(spec, m, n, k, batch)
    if spec.sk_kargs == "packed" and p["c"] >= 1 << (32 - SC_SHIFT):
        return False, f"K tiles per split {p['c']} must be < {1 << (32 - SC_SHIFT)} (kernel argument sk_sc)"

    def k_ok(kt: int, what: str) -> Tuple[bool, str]:
        if L < 2 and kt != T0:  # a short build: its vmcnt model holds for exactly its own K (no extra iterations)
            return False, f"{what}: K tiles {kt} must be exactly {T0} (short split build, k = {spec.k})"
        if kt < T0 or (kt - T0) % U:
            return False, f"{what}: K tiles {kt} must be {T0} + j {U} (j >= 0; build k = {spec.k})"
        return True, ""

    if p["D"] > 0:
        ok, why = k_ok(p["KT"], "data-parallel units")
        if not ok:
            return ok, why
    if p["D"] < p["T"]:
        s, c = p["s"], p["c"]
        if p["KT"] % s:
            return False, f"K / 128 = {p['KT']} must split into {s} equal parts"
        if c % granule(spec):
            return False, f"K tiles per split ({c}) must be a multiple of {granule(spec)}"
        ok, why = k_ok(c, "split units")
        if not ok:
            return ok, why
    if p["units"] > 2 ** 31 - 1:
        return False, "too many work units"
    if spec.sk_fix == "owner" and p["D"] < p["T"] and batch * p["units"] > N_CU * occupancy_wgs(spec):
        return False, (f"sk_fix owner: {batch * p['units']} units exceed the {N_CU} x {occupancy_wgs(spec)} co-resident "
                       "slots (the owner waits for its peers)")
    return True, ""


def grid(spec, m: int, n: int, k: int, batch: int = 1) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    p = plan(spec, m, n, k, batch)
    return (p["units"], 1, batch), (64 * spec.waves_m * spec.waves_n, 1, 1)


def kargs_tail(spec, m: int, n: int, k: int, batch: int, ws: int, cnt: int) -> bytes:
    """The split-K kernel arguments after the runtime ones (and the magic numbers): ws, cnt, D, s, c."""
    import struct

    p = plan(spec, m, n, k, batch)
    if p["D"] < p["T"] and (not ws or not cnt):
        raise ValueError("splitk: pass the workspace (ws) and counter (cnt, zeroed once) device pointers "
                         f"({p['ws_bytes']} / {p['cnt_bytes']} bytes: splitk.plan)")
    # (s, c) share one dword: the kernel arguments end at 128 bytes, so the prologue fetches them with two
    # s_load_dwordx16 behind ONE wait (a 129th+ byte cost a second, serialized kernarg round trip)
    if spec.sk_kargs == "wide":
        return struct.pack("<QQiii", ws, cnt, p["D"], p["s"], p["c"])
    return struct.pack("<QQii", ws, cnt, p["D"], p["s"] | p["c"] << SC_SHIFT)


def buffers(spec, m: int, n: int, k: int, batch: int = 1) -> Tuple[int, int]:
    """(workspace bytes, counter bytes) of a launch (the counters must be zero before the first launch; every launch
    leaves them zero)."""
    p = plan(spec, m, n, k, batch)
    return max(16, p["ws_bytes"]), max(16, p["cnt_bytes"])


# ===================================================================================================== dispatch
#: split-K selection model (``select`` / ``predict_us``), calibrated on MI355X decode / short-grid A/Bs (notes/
#: tile-library.md "Stream-K / split-K lane"): a unit streams its K range at a per-CU rate set by the bytes its ring
#: keeps in flight (latency-bound below ~175 busy CUs, HBM-bound above), computes at the per-CU MFMA rate, and pays a
#: fixed cost per unit run in sequence on a CU plus the fix-up once per launch when it splits.
SEL_LAT_US = 1.8  # load latency the ring hides: per-CU rate = (stages - 1) x stage bytes / latency
SEL_RCU_MIN, SEL_RCU_MAX = 30e9, 60e9  # per-CU streaming rate clamp (bytes / s)
SEL_HBM = 5.4e12  # sustained HBM rate of the GEMM streams (bytes / s; 85 % of the measured 6.33 TB/s copy roof)
SEL_CU_FLOPS = 16e12  # sustained MXFP6 MFMA rate per CU (FLOP / s; ~4.1 PF chip-wide at the power cap)
SEL_UNIT_US = 3.0  # per-unit fixed cost (setup + prologue fill + epilogue), per unit a CU runs in sequence
SEL_FIX_US = 2.0  # the split fix-up's critical path (partial store ack + counter + reduction)
SEL_MAX_SPLIT = 32
SEL_MARGIN = 0.9  # a split / fused build replaces a library kernel that runs the problem only below this time ratio
SEL_DP_STATIC = False  # s = 1 units built static (sk_static: no unit map / fix-up code); on once its A/B wins
#: compute-bound short grids (M > SEL_BW_M): measured (final round, 47 ship / plain pairs) a split only pays while the
#: grid is far below the GPU and every unit keeps a long K range; HBM-bound ones (decode / small M) split freely
SEL_BW_M = 160
SEL_CB_MAX_TILES = 64
SEL_CB_MIN_UNIT_TILES = 16
def min_unit_tiles(spec) -> int:
    """The shortest split unit a build of ``spec``'s structure runs (an exact-K short build: stages + 2 K tiles, a
    multiple of the granule)."""
    g = granule(spec)
    t = spec.stages + 2
    return -(-t // g) * g


def unit_spec(spec, c_tiles: int):
    """The build serving split units of ``c_tiles`` K tiles: the residue-class build when the library minimum fits
    in c, else the exact-K short build."""
    from .gemm import library_k_ref, loop_plan
    from dataclasses import replace

    try:
        k = library_k_ref(spec, k_target=128 * c_tiles)
        sp = replace(spec, k=k)
        U, H, L, T = loop_plan(sp)
        if (c_tiles - T) % U == 0 and c_tiles >= T:
            return sp
    except ValueError:
        pass
    return short_spec(spec, c_tiles)


def predict_us(spec, m: int, n: int, k: int, batch: int = 1, s: int = 1, n_cu: int = N_CU, **cal) -> float:
    """Model time of ``spec`` on (m, n, k, batch) as s-way split units (s = 1: one unit per tile): waves of units over
    the CU x occupancy slots; a unit's time is the max of its streaming at the per-CU rate, its share of the HBM
    stream and its MFMAs -- the CU's rate and MFMA pipe shared by the units resident on it -- plus a fixed cost; plus
    the fix-up when it splits. ``cal``: constant overrides (calibration)."""
    g = {**{k_: globals()[k_] for k_ in ("SEL_LAT_US", "SEL_RCU_MIN", "SEL_RCU_MAX", "SEL_HBM", "SEL_CU_FLOPS",
                                          "SEL_UNIT_US", "SEL_FIX_US")}, **cal}
    tm = spec.tile_m
    T = batch * -(-m // tm) * -(-n // spec.tile_n)  # tiles (edge / masked tiles included)
    KT = -(-k // 128)
    c = KT // s
    units = T * s
    occ = max(1, spec.occupancy)
    slots = n_cu * occ
    waves = -(-units // slots)
    q = min(occ, -(-min(units, slots) // n_cu))  # units resident per CU
    stage = (spec.tile_m + spec.tile_n) * 97
    rcu = min(g["SEL_RCU_MAX"], max(g["SEL_RCU_MIN"], (spec.stages - 1) * stage * q / (g["SEL_LAT_US"] * 1e-6)))
    per_unit_stream = c * stage * q / rcu
    total_bytes = batch * (n * k * 0.78125 + min(m, T * tm) * k * 0.78125) + units * (s > 1) * spec.tile_m * spec.tile_n * 8
    hbm = total_bytes / g["SEL_HBM"] / waves
    mfma = c * 2.0 * spec.tile_m * spec.tile_n * 128 * q / g["SEL_CU_FLOPS"]
    t = waves * (max(per_unit_stream, hbm, mfma) + g["SEL_UNIT_US"] * 1e-6)
    if s > 1:
        t += g["SEL_FIX_US"] * 1e-6
    return t * 1e6


def mask_needed(spec, m: int) -> bool:
    """Whether problem rows ``m`` need the fused path's ``m_mask`` on ``spec``'s config: fewer rows than the tile, or
    M-contiguous A with M % 16 != 0 (the edge shift would put a tile origin off the 16-row blocks)."""
    from .gemm import operand_layout_of

    return m < spec.tile_m or (operand_layout_of(spec, "A") == "mcontig" and m % 16 != 0)


def select(spec, m: int, n: int, k: int, batch: int = 1, n_cu: int = N_CU, splits=None):
    """The split-K plan for ``spec`` (a library build of one config, ``runtime_shape``) on problem (m, n, k, batch):
    ``spec`` itself when its grid fills the GPU (tiles >= 2 x the CUs) and it runs the problem; else the fastest by
    ``predict_us`` over the even splits s of K -- s = 1 is ``spec`` when it runs the problem, else the fused path's
    data-parallel units (``m_mask`` for M < tile_m, or a K the library build's residue class does not hold); ties ->
    fewer splits. Every candidate is validated and its problem checked (``problem_ok``); a split unit is the
    residue-class build when it fits, else the exact-K short build (shallower ring when the units are short)."""
    from dataclasses import replace

    from .gemm import n_tiles, runtime_problem_ok, validate_spec

    orig = spec
    mask = mask_needed(spec, m)
    if getattr(spec, "k_tail", False):  # K tails (K % 128 != 0): data-parallel units only (splits are whole K tiles)
        if not mask and runtime_problem_ok(orig, m, n, k, batch)[0]:
            return orig
        sp = replace(spec, schedule="dp", persist_overlap=False, splitk="fused", sk_split=1, m_mask=mask)
        for cand in (sp, unit_spec(sp, -(-k // 128))):  # (the exact-K build when the residue class misses K)
            if validate_spec(cand)[0] and runtime_problem_ok(cand, m, n, k, batch)[0]:
                return cand
        return orig
    orig_ok = not mask and runtime_problem_ok(orig, m, n, k, batch)[0]
    spec = replace(spec, schedule="dp", persist_overlap=False)  # the fused path's units form a data-parallel grid
    T = n_tiles(replace(spec, m_mask=mask), "A", m) * n_tiles(spec, "B", n) * batch
    KT = k // 128
    if T >= 2 * n_cu and orig_ok:
        return orig
    best = None
    cands = splits or [s for s in range(1, SEL_MAX_SPLIT + 1) if KT % s == 0]

    def unit(s: int, c: int):
        bases = [spec] + ([replace(spec, stages=2)] if spec.stages > 2 else [])  # a shallower ring: shorter units
        if m <= 64 and (spec.stages, spec.occupancy) != (4, 1):
            # decode units: 4 stages at occupancy 1 (r2c: +3-4 % over the family's shallower, occupied rings; deeper
            # rings lose)
            bases.insert(0, replace(spec, stages=4, occupancy=1))
        for base in bases:
            if c < min_unit_tiles(base) or c % granule(base):
                continue
            try:
                # the shipped fix-up: write-through partials (no L2 write-back per wave; 1.6x the fence on decode)
                # and the static reducer (all partial loads in flight at once)
                sp = unit_spec(replace(base, splitk="fused", sk_split=s, m_mask=mask, sk_release="wt",
                                       sk_static=s >= 2 or SEL_DP_STATIC), c)
            except ValueError:
                continue
            if validate_spec(sp)[0] and runtime_problem_ok(sp, m, n, k, batch)[0]:
                return sp
        return None

    t_orig = predict_us(orig, m, n, k, batch, 1, n_cu) if orig_ok else None
    for s in cands:
        c = KT // s
        if s > 1 and m > SEL_BW_M and (T > SEL_CB_MAX_TILES or c < SEL_CB_MIN_UNIT_TILES):
            continue  # compute-bound: only few-tile grids with long units split
        if s == 1 and orig_ok:
            sp = orig
        else:
            sp = unit(s, c)
            if sp is None:
                continue
        t = predict_us(sp, m, n, k, batch, s, n_cu)
        if sp is not orig and t_orig is not None and t > SEL_MARGIN * t_orig:
            continue  # not clearly better than the library kernel: keep it
        key = (round(t, 2), s)
        if best is None or key < best[0]:
            best = (key, sp)
    if best is None:
        return orig
    return best[1]


def select_config(mt0: int, mt1: int, du: int, family: str, fmt: str, m: int, n: int, k: int, batch: int = 1, **over):
    """``library.dispatch`` + ``select``: the shipped kernel of config ``MT0 x MT1 x DepthU`` / ``family`` on problem
    (m, n, k, batch), split along K when its grid is short of the GPU. A work-group tile larger than the problem
    (WG_MULT) falls back to the native tile (edge tiles along N, ``m_mask`` along M)."""
    from . import library as LIB

    sp = None
    if m >= mt1:  # (library.dispatch sends M < MT1 here: no recursion)
        try:
            sp = LIB.dispatch(mt0, mt1, du, family, fmt, m, n, batch, k_target=k, _sk_base=True, **over)
            if m < sp.tile_m or n < sp.tile_n:
                sp = None
        except ValueError:
            sp = None
    if sp is None:
        try:
            sp = LIB.family_spec(mt0, mt1, du, family, fmt, k_target=k, **{"edge_tiles": True, **over})
        except ValueError:
            sp = LIB.family_spec(mt0, mt1, du, family, fmt, k_target=None, **{"edge_tiles": True, **over})
    from .gemm import runtime_problem_ok

    sel = select(sp, m, n, k, batch)
    if not runtime_problem_ok(sel, m, n, k, batch)[0] and sp.canon_ring != "stage":
        # a line ring's 8-tile K pattern / long loop body leaves no unit length: the stage ring of the same tile
        try:
            alt = LIB.family_spec(mt0, mt1, du, family, fmt, k_target=None, ring="stage", **{"edge_tiles": True, **over})
            sel2 = select(alt, m, n, k, batch)
            if runtime_problem_ok(sel2, m, n, k, batch)[0]:
                return sel2
        except ValueError:
            pass
    return sel


# ===================================================================================================== device side
def emit(B) -> None:
    """Build the split-K kernel body into ``B`` (a ``gemm._Builder`` whose params a .. c, the runtime ones and the
    magic numbers are declared)."""
    b, spec = B.b, B.spec
    c32 = B.c32
    # the reducer's acquire needs no L2 write-back (it only reads what the others released): the direct compiler's
    # acquire-only fence (lever acqfence: wait + buffer_inv sc1, as LLVM's gfx940 cache control) via the kernel's own
    # levers (dlevers: a direct compiler that reads them, rustgemm crate-up)
    lv = [x for x in (spec.dlevers or "").split(",") if x]
    b.kernel.attrs["dlevers"] = ",".join(lv + ["acqfence"])
    p_ws = b.param("ws", PtrType(F32, "global"))
    p_cnt = b.param("cnt", PtrType(I32, "global"))
    p_dp = b.param("sk_dp", I32)
    if spec.sk_kargs == "wide":
        p_ns, p_sc = b.param("sk_ns", I32), b.param("sk_kc", I32)
    else:
        p_sc = b.param("sk_sc", I32)  # s | c << SC_SHIFT (kargs_tail)
    # entry z of a strided batch: every pointer moves by z x its stride
    z32 = b.block_id_z()
    z = b.zext(z32, I64)
    B.pa, B.pb, B.psa, B.psb, B.pc = (b.global_ptr_add(p, b.mul(z, s_))
                                      for p, s_ in zip((B.pa, B.pb, B.psa, B.psb, B.pc), B.p_bs))
    u = b.block_id_x()
    gm, gn = B._ntile_rows("A"), B._ntile_rows("B")
    T = b.to_sgpr_u32(b.mul(gm, gn))
    if spec.sk_kargs == "wide":
        s, D, c = (c32(spec.sk_split) if spec.sk_static else p_ns), p_dp, p_sc
    else:
        s = c32(spec.sk_split) if spec.sk_static else b.to_sgpr_u32(b.land(p_sc, c32((1 << SC_SHIFT) - 1)))
        D, c = p_dp, b.to_sgpr_u32(b.lshr(p_sc, c32(SC_SHIFT)))
    dp_only = spec.sk_static and spec.sk_split == 1  # every unit a whole tile: unit = tile, no fix-up
    KT = B.k_tiles  # the whole K (strides)
    if dp_only:
        is_sk, j, t, k0, klen = None, c32(0), u, c32(0), KT
    else:
        is_sk = b.cmp_ge(u, D)
        v = b.select(is_sk, b.sub(u, D), c32(0))
        q = b.div(v, s)
        j = b.to_sgpr_u32(b.select(is_sk, b.sub(v, b.mul(q, s)), c32(0)))
        t = b.to_sgpr_u32(b.select(is_sk, b.add(D, q), u))
        k0 = b.to_sgpr_u32(b.mul(j, c))
        klen = b.to_sgpr_u32(b.select(is_sk, c, KT))
    # the unit's K range: operand / scale bases move by k0 K tiles
    from .gemm import ROW_BYTES, operand_layout_of, scale_img_bytes

    k064 = b.zext(k0, I64)
    la, lb = operand_layout_of(spec, "A"), operand_layout_of(spec, "B")
    if la == "canonical":
        sha = b.mul(k064, b.const_i64(ROW_BYTES))
    elif la == "mcontig":
        lda = B.p_ld["A"]
        B.sk_a_shift32 = b.to_sgpr_u32(b.mul(k0, b.mul(lda, c32(128))))
        sha = b.zext(B.sk_a_shift32, I64)
    else:
        sha = b.mul(k064, b.const_i64(spec.tile_m * ROW_BYTES))
    shb = b.mul(k064, b.const_i64(ROW_BYTES if lb == "canonical" else spec.tile_n * ROW_BYTES))
    B.pa = b.global_ptr_add(B.pa, sha)
    B.pb = b.global_ptr_add(B.pb, shb)
    for X, gt in (("A", gm), ("B", gn)):  # wtile images: K tile t at t (tiles along X) img + tile img
        sh = b.mul(k064, b.zext(b.mul(gt, c32(scale_img_bytes(spec, X))), I64))
        if X == "A":
            B.psa = b.global_ptr_add(B.psa, sh)
        else:
            B.psb = b.global_ptr_add(B.psb, sh)
    B.k_tiles_full = KT
    B.k_tiles = klen
    B.persist_t = t  # _cta_map: the unit's tile (grouped order, no XCD chunking)
    if spec.m_mask:
        B.sk_m = B.p_dim["A"]
    B.epilogue_done = False
    B._setup()
    B._mainloop()
    B.persist_t = None
    # the fix-up's / epilogue's lane-only address math must not be hoisted above the K loop (LICM would keep it
    # live through the loop's register peak): an opaque copy of the lane id pins it after the loop
    B.lane = b.inline_asm("v_mov_b32 $0, $1", "=v,v", [B.lane], I32, sideeffect=True)
    n_acc = B.fm * B.fn
    zero = b.zero_vec_f32(4)
    acc = [zero if a_ is None else a_ for a_ in B.acc]
    if dp_only:
        store_c(B, acc)
        return
    with b.scf_if(b.cmp_lt(u, D)):  # data-parallel unit: C straight from the accumulators
        store_c(B, acc)
    nw = B.nw
    tb = spec.tile_m * spec.tile_n * 4
    wb = n_acc * 1024
    with b.scf_if(is_sk):
        nsk = b.sub(T, D)
        tid = b.to_sgpr_u32(b.add(b.mul(z32, nsk), b.sub(t, D)))  # split tile id over the batch
        base = b.add(b.mul(b.mul(b.zext(tid, I64), b.zext(s, I64)), b.const_i64(tb)),
                     b.zext(b.mul(B.wave, c32(wb)), I64))
        rs = b.buffer_rsrc(b.global_ptr_add(p_ws, base), b.to_sgpr_u32(b.mul(s, c32(tb))))
        voff = b.mul(B.lane, c32(16))
        jsoff = b.to_sgpr_u32(b.mul(j, c32(tb)))
        attrs = {"cachepolicy": {"wt": CP_WT, "wt1": CP_WT1}[spec.sk_release]} if spec.sk_release != "fence" else None

        def store_partial():
            for mi in range(B.fm):  # this unit's partial: 1 KiB per atom and wave, one dwordx4 per lane
                with _rows_live(B, mi):  # m_mask: an atom row block entirely past M is neither stored nor read
                    for a in range(mi * B.fn, (mi + 1) * B.fn):
                        soff = b.to_sgpr_u32(b.add(jsoff, c32(a * 1024)))
                        b._op("tile.buffer_store_vN", [rs, voff, soff, acc[a]], attrs=attrs)

        if spec.sk_fix == "owner":
            _owner_fixup(B, p_cnt, tid, j, s, rs, voff, tb, acc, store_partial)
            return
        store_partial()
        if spec.sk_release != "fence":
            b.s_waitcnt(vmcnt=0)  # write-through stores acknowledged before the signal
        else:
            b.fence(scope="agent", ordering="release")
        cidx = b.to_sgpr_u32(b.add(b.mul(tid, c32(nw)), B.wave))
        inc = c32(1)
        if spec.mutant == "sk_noflag":  # must-fail: split 0 never signals
            inc = b.to_sgpr_u32(b.select(b.cmp_eq(j, c32(0)), c32(0), c32(1)))
        old = b.global_atomic_add(p_cnt, cidx, inc, ordering="monotonic")
        old0 = b.readfirstlane(old)  # lane 0's old value: the wave's one atomic added 64 (one per lane)
        last = b.cmp_eq(old0, b.mul(b.sub(s, c32(1)), c32(64)))
        if spec.mutant == "sk_noflag":
            last = b.land(last, b.cmp_ne(j, c32(0)))
        with b.scf_if(last):
            b.fence(scope="agent", ordering="acquire")
            if spec.sk_static:
                _reduce_static(B, rs, voff, j, spec.sk_split, tb, acc)
            else:
                _reduce(B, rs, voff, j, s, tb, acc)
            if spec.mutant != "sk_noreset":
                b.global_store(p_cnt, cidx, c32(0), align=4)


def _owner_fixup(B, p_flag, tid: Value, j: Value, s: Value, rs, voff, tb: int, acc: List[Value], store_partial) -> None:
    """``sk_fix="owner"``: split s - 1 owns the tile. The other splits store their partial (write-through), wait
    for it to land and raise their flag (``flags``: [entry][split tile][split][wave]); the owner polls its peers'
    flags (lane l < s - 1 polls split l's: one acquire poll loop for all of them), sums the partials in split order
    with its own from registers, stores C and lowers the flags for the next launch. The owner waits, so every unit
    of the launch must be co-resident: ``problem_ok`` admits the mode only when the units fit the GPU's CU x
    occupancy slots in one wave (each XCD's round-robin share fits its slots), and the owner is the highest unit
    index of its tile (its peers are dispatched first)."""
    b, spec = B.b, B.spec
    c32 = B.c32
    nw = B.nw
    S = spec.sk_split
    owner = b.cmp_eq(j, b.sub(s, c32(1)))
    fbase = b.to_sgpr_u32(b.mul(tid, s))
    with b.scf_if(b.cmp_ne(j, b.sub(s, c32(1)))):  # a peer: partial, then the flag
        store_partial()
        b.s_waitcnt(vmcnt=0)  # (write-through stores acknowledged; sk_release="fence" adds the L2 write-back)
        if spec.sk_release == "fence":
            b.fence(scope="agent", ordering="release")
        fidx = b.to_sgpr_u32(b.add(b.mul(b.add(fbase, j), c32(nw)), B.wave))
        b.global_flag_store(p_flag, fidx, c32(1), scope="agent", ordering="monotonic")
    with b.scf_if(owner):
        if len(acc) > 32:  # the reducer reads every slot (no register room for its own partial): store it first
            store_partial()
            b.s_waitcnt(vmcnt=0)
        peer = b.smin(B.lane, b.sub(s, c32(2)))  # lane l < s - 1 polls split l's flag (the rest repeat s - 2's)
        pidx = b.add(b.mul(b.add(fbase, peer), c32(nw)), B.wave)
        b.global_flag_wait_eq(p_flag, pidx, c32(1), scope="agent")
        _reduce_static(B, rs, voff, j, S, tb, acc)
        with b.scf_if(b.cmp_lt(B.lane, b.sub(s, c32(1)))):  # lower the peers' flags for the next launch
            b.global_store(p_flag, pidx, c32(0), align=4)


def _reduce_static(B, rs, voff, j: Value, S: int, tb: int, own: List[Value]) -> None:
    """``sk_static``: the reduction with the split count S known at build time -- per group of atoms every one of
    the S partials' loads issues before the first add (one memory latency per group, not S), then the adds in split
    order. Groups hold at most 128 VGPRs of loads."""
    b, spec = B.b, B.spec
    c32 = B.c32
    n_acc = len(own)
    budget = (128 if spec.occupancy <= 1 else 48) // (4 * S)  # atoms whose S loads fit in flight (VGPR budget)
    own_regs = n_acc <= 32
    zero = b.zero_vec_f32(4)
    oob = c32(OOB)

    def add4(x: Value, y: Value) -> Value:
        return b.vec_pack([b.fadd(b.vec_extract(x, e), b.vec_extract(y, e)) for e in range(4)], F32)

    if spec.mutant == "sk_drop":
        j1 = b.add(j, c32(1))
        d = b.sub(j1, b.mul(b.div(j1, c32(S)), c32(S)))
    if spec.m_mask:  # groups within one atom row block (skipped whole when its rows are past M)
        G = max(1, min(B.fn, budget))
        groups = [(mi, list(range(mi * B.fn + n0, mi * B.fn + min(B.fn, n0 + G)))) for mi in range(B.fm)
                  for n0 in range(0, B.fn, G)]
    else:  # groups across row blocks: as many atoms as the budget allows (one memory latency per group)
        G = max(1, budget)
        groups = [(None, list(range(a0, min(n_acc, a0 + G)))) for a0 in range(0, n_acc, G)]
    for mi, grp in groups:
        with (_rows_live(B, mi) if mi is not None else _Null()):
            xs = []
            for jj in range(S):
                mine = b.cmp_eq(c32(jj), j)
                vo = b.select(mine, oob, voff) if own_regs else voff
                row = [b.buffer_load_vN(rs, vo, b.to_sgpr_u32(c32(jj * tb + a * 1024)), F32, 4) for a in grp]
                xs.append((mine, row))
            parts = []
            for jj, (mine, row) in enumerate(xs):
                r = [b.select(mine, own[a], x) for a, x in zip(grp, row)] if own_regs else row
                if spec.mutant == "sk_drop":  # must-fail: the partial of split (j + 1) mod S is skipped
                    skip = b.cmp_eq(c32(jj), d)
                    r = [b.select(skip, zero, x) for x in r]
                parts.append(r)
            if spec.mutant == "sk_arrival":  # should-fail: own partial first (arrival-dependent order)
                sums = [own[a] for a in grp]
                for jj in range(S):
                    keep = b.cmp_eq(c32(jj), j)
                    sums = [b.select(keep, x, add4(x, y)) for x, y in zip(sums, parts[jj])]
            else:
                order = list(range(S))[::-1] if spec.mutant == "sk_order" else list(range(S))
                sums = parts[order[0]]
                for jj in order[1:]:
                    sums = [add4(x, y) for x, y in zip(sums, parts[jj])]
            store_atoms(B, grp, sums)


def _reduce(B, rs, voff, j: Value, s: Value, tb: int, own: List[Value]) -> None:
    """Sum the tile's s partials of this wave in split order 0 .. s - 1 and store C, G atoms at a time: one runtime
    loop over the splits per group (the group's G loads issue before any is used), then the group's C stores.
    Small accumulator sets take this unit's own partial from registers (its slot is not read back: an OOB voffset
    moves no data); larger ones (> 32 atoms: no VGPR room next to 256 AGPRs) read every slot."""
    b, spec = B.b, B.spec
    c32 = B.c32
    n_acc = len(own)
    G = min(n_acc, 8)
    own_regs = n_acc <= 32
    zero = b.zero_vec_f32(4)
    oob = c32(OOB)

    def parts(jj: Value, grp: List[int]) -> List[Value]:
        sj = b.mul(jj, c32(tb))
        if own_regs:
            mine = b.cmp_eq(jj, j)
            vo = b.select(mine, oob, voff)
        else:
            vo = voff
        xs = [b.buffer_load_vN(rs, vo, b.to_sgpr_u32(b.add(sj, c32(a * 1024))), F32, 4) for a in grp]
        if own_regs:
            xs = [b.select(mine, own[a], x) for a, x in zip(grp, xs)]
        if spec.mutant == "sk_drop":  # must-fail: the partial of split (j + 1) mod s is skipped
            j1 = b.add(j, c32(1))
            d = b.sub(j1, b.mul(b.div(j1, s), s))
            skip = b.cmp_eq(jj, d)
            xs = [b.select(skip, zero, x) for x in xs]
        return xs

    def add4(x: Value, y: Value) -> Value:
        return b.vec_pack([b.fadd(b.vec_extract(x, e), b.vec_extract(y, e)) for e in range(4)], F32)

    groups = [(mi, list(range(mi * B.fn + n0, mi * B.fn + min(B.fn, n0 + G)))) for mi in range(B.fm)
              for n0 in range(0, B.fn, G)]
    for gi, (mi, grp) in enumerate(groups):
        g0 = grp[0]
        rl = _rows_live(B, mi)
        rl.__enter__()
        if spec.mutant == "sk_arrival":  # should-fail: own partial first, then the others (arrival-dependent order)
            init, lo = parts(j, grp), c32(0)
        elif spec.mutant == "sk_order":  # must-fail: reverse split order s - 1 .. 0
            init, lo = parts(b.sub(s, c32(1)), grp), c32(1)
        else:
            init, lo = parts(c32(0), grp), c32(1)
        loop = b.scf_for_iter(lo, s, c32(1), [(f"sk{g0}_{i}", x) for i, x in enumerate(init)], iv_name=f"sj{g0}",
                              elide_trailing_barrier=False)
        with loop as (jj, vs):
            src = b.sub(b.sub(s, c32(1)), jj) if spec.mutant == "sk_order" else jj
            xs = parts(src, grp)
            nxt = [add4(vs[i], x) for i, x in enumerate(xs)]
            if spec.mutant == "sk_arrival":
                keep = b.cmp_eq(jj, j)
                nxt = [b.select(keep, vs[i], y) for i, y in enumerate(nxt)]
            b.scf_yield(*nxt)
        store_atoms(B, grp, list(loop.results))
        rl.__exit__(None, None, None)


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _rows_live(B, mi: int):
    """m_mask: a uniform branch on atom row block ``mi`` of this wave holding rows < M (else a no-op context)."""
    if not B.spec.m_mask:
        return _Null()
    b = B.b
    r0 = b.add(b.add(B.row0["A"], B.wrow0["A"]), B.c32(mi * B.T))
    return b.scf_if(b.cmp_lt(r0, _mask_m(B)))


def _mask_m(B):
    """m_mask: the row bound of the C stores (M; the must-fail mutants move it by one row)."""
    mu = B.spec.mutant
    if mu == "mask_over":
        return B.b.add(B.sk_m, B.c32(1))
    if mu == "mask_under":
        return B.b.sub(B.sk_m, B.c32(1))
    return B.sk_m


def store_c(B, vals: List[Value]) -> None:
    """bf16 C of the CTA tile from ``vals`` (one f32x4 per atom): the builder's own epilogue, or with ``m_mask`` the
    row-predicated stores."""
    if not B.spec.m_mask:
        B.acc = list(vals)
        B._epilogue()
        return
    B.b.s_waitcnt(vmcnt=0)
    store_atoms(B, list(range(len(vals))), list(vals))


def _store_setup(B) -> dict:
    """Loop-invariant C addressing of ``store_atoms`` (computed once per call site)."""
    b = B.b
    c32 = B.c32
    lane, T = B.lane, B.T
    rr = b.mod(lane, c32(T))
    grp = b.div(lane, c32(T))
    d = {"rr": rr, "grp": grp, "rsrc": b.buffer_rsrc(B.pc, c32(0x7FFFFFF0)), "cnb": B._cnb(),
         "m0": b.add(B.row0["A"], B.wrow0["A"]), "n0": b.add(B.row0["B"], B.wrow0["B"])}
    if not B.spec.swap_ab:
        par = b.land(rr, c32(1))
        d.update(par=par, odd=b.cmp_eq(par, c32(1)), row0=b.add(b.mul(grp, c32(4)), b.mul(par, c32(2))),
                 col_b=b.mul(b.sub(rr, par), c32(2)))
    else:
        d["lane_off"] = b.add(b.mul(rr, d["cnb"]), b.mul(grp, c32(8)))
    return d


def store_atoms(B, atoms: List[int], vals: List[Value]) -> None:
    """bf16 C stores of the given atoms (the builder epilogue's lane maps; ``out_exp2`` applied; ``m_mask``: rows >= M
    predicated off). Atom a = mi fn + ni."""
    b, spec = B.b, B.spec
    g = B.geo
    T = B.T
    c32 = B.c32
    d = _store_setup(B)
    M = _mask_m(B) if spec.m_mask else None
    cnb, rsrc, m0, n0 = d["cnb"], d["rsrc"], d["m0"], d["n0"]
    if spec.out_exp2:
        f = b.const_f32(2.0 ** spec.out_exp2)
        vals = [b.vec_pack([b.fmul(b.vec_extract(v, e), f) for e in range(g.c_len)], F32) for v in vals]
    soffs: Dict[int, Value] = {}

    def soff_of(mi: int) -> Value:
        if mi not in soffs:
            soffs[mi] = b.to_sgpr_u32(b.add(b.mul(b.add(m0, c32(mi * T)), cnb), b.mul(n0, c32(2))))
        return soffs[mi]

    for a, v in zip(atoms, vals):
        mi, ni = divmod(a, B.fn)
        mrow = b.add(m0, c32(mi * T))
        if not spec.swap_ab:  # a lane holds C[m0 + mi T + 4 grp + (0..3)][n0 + ni T + rr] (DPP-paired dwords)
            h = b.vec_bitcast(b.vec_trunc_f32_to_bf16(v), VectorType(I32, 2))
            d0, d1 = b.vec_extract(h, 0), b.vec_extract(h, 1)
            x0, x1 = b.dpp_xor(d0, 1), b.dpp_xor(d1, 1)
            lo = b.select(d["odd"], x1, d0)
            hi = b.select(d["odd"], d1, x0)
            words = (b.lor(b.land(lo, c32(0xFFFF)), b.shl(hi, c32(16))),
                     b.lor(b.lshr(lo, c32(16)), b.land(hi, c32(-65536))))
            so = soff_of(mi)  # (defined outside the predicated regions)
            for jw, w in enumerate(words):
                r = b.add(d["row0"], c32(jw))
                off = b.add(b.add(b.mul(r, cnb), d["col_b"]), c32(ni * T * 2))
                if M is None:
                    b.buffer_store_vN(rsrc, off, so, b.vec_pack([w], I32))
                else:
                    with b.scf_if(b.cmp_lt(b.add(mrow, r), M)):
                        b.buffer_store_vN(rsrc, off, so, b.vec_pack([w], I32))
            continue
        h = b.vec_trunc_f32_to_bf16(v)  # a lane holds C[m0 + mi T + rr][n0 + ni T + 4 grp + (0..3)]
        off = b.add(d["lane_off"], c32(ni * T * 2))
        so = soff_of(mi)

        def st(h=h, off=off, so=so):
            if spec.store_nt:
                b.buffer_store_vN_bf16(rsrc, off, so, h, 2, cachepolicy=2)
            else:
                b.buffer_store_vN(rsrc, off, so, b.vec_bitcast(h, VectorType(I32, 2)))

        if M is None:
            st()
        else:
            with b.scf_if(b.cmp_lt(b.add(mrow, d["rr"]), M)):
                st()
