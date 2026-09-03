# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Reusable LDS bank-conflict tooling: validated write-port model + simulator, bit-exact
address map, isolation micro-probes, rocprof harness, and the 3-panel register->LDS
dataflow renderer.

WHY THIS MODULE EXISTS
----------------------
The /bank-conflict skill repeatedly rebuilt these same pieces as throwaway scripts (one per
investigation), which burned tokens and let the model drift between systems. The MECHANISM (the
CDNA2 LDS write-port rule) and the TOOLING (simulator, probes, validator, renderer) are stable and
arch-parameterized, so they live here once, committed, cross-system consistent. Only the per-CASE
*numbers* (a new kernel/tile/dtype's measured conflicts/access) must be regenerated on the real GPU
each time -- this module stores the METHOD, never a kernel's answer. The `_MECHANISM_*` tables below
are the model's VALIDATION CORPUS (the reference measurements that PROVE the write-port mechanism
reproduces hardware to the integer); they are not a substitute for measuring a new case.

THE CARDINAL RULE (unchanged): never state a conflict number until (1) you have rocprof hardware
counters for it AND (2) this simulator predicts those exact counters from the address map. If the
sim does not reproduce the measurement, the MODEL is wrong -- fix it, do not "meet in the middle".

THE MECHANISM (validated bit-exact, gfx90a; SOT: helpers/tiling/docs/lds_banks.md §1.4)
--------------------------------------------------------------------------------------------
The CDNA2 LDS *write* datapath is NOT the naive per-address replay counter. The naive rule
`sum_bank (distinct_addresses - 1)` OVER-COUNTS K-aliased stores ~8-9x. The write port has two
hardware constants, both confirmed integer-exact on every measured config AND by independently
reproducing the measured `productive` floor they were NOT fit against:
  1. WRITE-PORT WIDTH = 8 banks/cycle  (`min(banks_used, 8)`): the store port covers an 8-bank
     stripe per cycle, not all 32 banks. (Write-side rule; the read port differs.)
  2. WRITE-COMBINE DEPTH = 4            (`depth / 4`):          same-bank stacked stores drain 4
     per cycle (write-combine of a 4-deep column).

  served_cycles(half-wave, phase) = min(banks_used, PORT_BANKS) * max_bank_depth / COMBINE
  productive (floor)              = ceil(distinct_dwords_in_instruction / NB)   (per instruction)
  SQ_LDS_IDX_ACTIVE   (IDX)       = served
  SQ_LDS_BANK_CONFLICT (BC)       = served - productive
  conflicts/access                = BC / (IDX - BC) = BC / productive

PHASE COMBINE: one instruction's dword-phases PIPELINE through the write port -- a half-wave's
served = MAX over its phases (busiest phase sets the rate). The counter is PER INSTRUCTION; a store
forced to a narrow width issues several instructions, each measured separately with its own
footprint (productive = footprint_instr / NB).

THE PAD-SWEEP STRIPE-ALIGNMENT MODEL (the conflict-FREE fix)
-----------------------------------------------------------
The write-port histogram model reproduces the CONFLICTED configs, but the naive `bank = dword mod
NB` map it is built on CANNOT tell pad8 (HW 1.0) from pad16 (HW 0.0): both give byte-identical
served-group histograms, and the naive verdict is in fact INVERTED vs hardware for padded K-aliased
f16 stores. The physical distinguisher is the K-row's HALF-STRIPE PARITY, not its address:

  Let s = (lds_row_stride_in_dwords) mod NB, and W = dwords/lane (b64->2, b128->4). The half-stripe
  unit is 4*W dwords. The store is CONFLICT-FREE (BC=0) iff the K-row shift is an ODD multiple of
  that unit:            s % (4*W) == 0  AND  (s // (4*W)) is ODD.
  Otherwise it sits at the throughput floor (conflicts/access = 1.0), except the fully-aliased
  s == 0 b64 case which piles the whole 8-wide K column onto one stripe (conflicts/access = 3.0).

`conflict_free_bank_of()` gives the one-lane-per-bank permutation that BC=0 physically means (SOT
lds_banks.md §2) -- what the FIXED-panel bank grid must draw; the naive map must NOT be drawn there.

PUBLIC API & CONTRACTS
----------------------
Preferred entry point is `analyze_store`; the rest are the layers it composes (usable directly for
custom access patterns). Every function that could emit a mislabeled artifact GATES internally.

  analyze_store(descs, *, tile_free, wtag, measure=None, render_to=None, verify_fix=False, ...)
      -> ConflictReport. Chains: store_datum -> simulate -> (measure on GPU + HARD gate sim==HW) ->
      recommend_pad -> (optional verify_fix on GPU) -> render. CONTRACT: with `measure` it returns a
      VALIDATED report; without it the report is UNVALIDATED and `render_to` is REFUSED (cardinal
      rule). RAISES ConflictModelError if the probe is not bit-exact or sim != HW.
  measure callable (INJECTED by the caller): `measure(pad:int, mode='store') -> dict` with keys
      BC, IDX, conflicts_per_access (+ optional ADDR, max_abs_diff). Encapsulates the container
      rocprof run; keeps this module container-agnostic. It is the ONLY host-specific glue.
  ConflictReport: dataclass. `.verdict`, `.conflicts_per_access`, `.fix_pad`, `.located`, `.png`,
      `.facts_table()` -> the skill's markdown rows.

  selftest(arch) -> bool           GATE: model reproduces THAT arch's measured corpus; refuses an
                                   arch with no corpus. Run before trusting any number.
  simulate(accesses, arch)         write-port model over an address map -> {IDX,BC,productive,c/a}.
  simulate_hist(hists, footprint)  the authoritative per-(half-wave,phase) predictor.
  addr_map(desc, strides)          bit-exact (lane,reg)->element-address map from the REAL emit.
  store_datum(desc, tile_free)     shared (acc, vw, datum{(lane,ph)->(K,free,dword,bank)}) builder.
  collision_lanes(datum)           lanes piling on a bank in a served group (the located conflict).
  predict_pad_sweep / is_conflict_free / recommend_pad / conflict_free_bank_of  the stripe rule.
  gate(sim, measured)              HARD assert sim==HW; RAISES ConflictModelError on mismatch.
  ProbeDescs(.from_coop) / build_probe / run_probe   bit-exact isolation micro-kernels.
  COUNTER_PMC / ROCPROF_RECIPE / parse_counter_csv   the rocprof harness.
  render_conflict_3panel(...)      the figure; GATES sim==measured, fix conflict-free, fixed panel
                                   collision-free before drawing (refuses a mislabeled figure).
  register_arch(arch, hists, pad_sweep)   add + store a new arch's model and its own corpus.

EXTENDING TO A NEW ARCH
-----------------------
Call `register_arch(ArchLDS(name, NB, HALF, PORT_BANKS, COMBINE), hists, pad_sweep)` where the
constants come from that arch's ISA + a probe sweep and `hists`/`pad_sweep` are FRESHLY MEASURED on
that arch (same format as gfx90a's `_VALIDATION_CORPUS` entry). Then run `selftest(name)` until it
PASSES. `selftest` REFUSES an arch that has no corpus of its own -- gfx90a's numbers must never be
used to "validate" another arch. Do NOT assume gfx90a constants carry over -- gfx942 / RDNA differ.
"""
from __future__ import annotations

import csv
import glob
import math
import os
from collections import defaultdict
from dataclasses import dataclass


# ==================================================================================================
# Arch model
# ==================================================================================================
@dataclass(frozen=True)
class ArchLDS:
    """Per-arch LDS write-port constants. gfx90a is validated bit-exact; other arches must be
    validated with a fresh probe sweep before use (see module docstring)."""

    name: str
    NB: int           # number of LDS banks
    HALF: int         # served-group size (half-wave lanes arbitrated together)
    PORT_BANKS: int   # write-port width: distinct banks served per cycle
    COMBINE: int      # write-combine depth: same-bank stores drained per cycle


GFX90A = ArchLDS("gfx90a", NB=32, HALF=32, PORT_BANKS=8, COMBINE=4)
ARCHS = {"gfx90a": GFX90A}


def arch_lds(arch) -> ArchLDS:
    """Resolve an arch name (or an ArchLDS) to its validated LDS model."""
    if isinstance(arch, ArchLDS):
        return arch
    if arch not in ARCHS:
        raise ValueError(
            f"no validated LDS model for {arch!r}; validated arches: {sorted(ARCHS)}. "
            f"Add an ArchLDS entry + validate with a fresh probe sweep before use.")
    return ARCHS[arch]


# ==================================================================================================
# Write-port model (the authoritative conflict predictor)
# ==================================================================================================
def served_phase(banks_used, max_depth, arch=GFX90A):
    """Served cycles for one (half-wave, phase): the write-port rule. Write-COMBINE folds accesses to
    distinct banks together, but it can only combine across banks that are ACTUALLY active -- a deep pile
    on fewer than COMBINE banks cannot combine across idle banks, so it drains at ~depth. Capping COMBINE
    at ``min(banks_used, COMBINE)`` is what makes a narrow-deep K-alias pile (e.g. b128 with k_lanes=32 ->
    2 banks x depth 16) cost the measured 3x, while every wider store (banks_used >= COMBINE) is unchanged
    -- so the validated corpus (all rows have banks_used >= 4 = COMBINE) is untouched."""
    a = arch_lds(arch)
    return min(banks_used, a.PORT_BANKS) * max_depth / min(banks_used, a.COMBINE)


def simulate_hist(hists, footprint_dwords, arch=GFX90A):
    """AUTHORITATIVE. `hists` = {(half_wave, phase): {bank: depth, ...}} for ONE instruction.
    `footprint_dwords` = distinct dwords THIS instruction writes (for the productive floor).

    The instruction's dword-phases pipeline (MAX per half-wave); served sums over half-waves. NOTE: this
    per-phase MAX does not model cross-phase bank SPREAD, so it is CONSERVATIVE on the conflict-free pad
    (it will not falsely report a spread as conflict-free); the deep/narrow K-alias pile magnitude is
    captured via the COMBINE cap in :func:`served_phase`. Returns per-instruction {IDX, BC, productive}."""
    a = arch_lds(arch)
    by_hw = defaultdict(dict)
    for (hw, ph), hist in hists.items():
        by_hw[hw][ph] = served_phase(len(hist), max(hist.values()), a) if hist else 0.0
    served = round(sum(max(pc.values() or [0.0]) for pc in by_hw.values()))
    productive = -(-footprint_dwords // a.NB)  # ceil
    return {"IDX": served, "BC": served - productive, "productive": productive}


def _dwords(access, dtype_bytes=2):
    """Dword indices this lane's op touches (vw*dtype_bytes/4 dwords, consecutive from base)."""
    per_dword = 4 // dtype_bytes  # f16 -> 2 elems per dword
    d0 = access["base"] // per_dword
    ndw = max(1, access["vw"] // per_dword)
    return [d0 + i for i in range(ndw)]


def _lane_dwords(accesses, dtype_bytes=2):
    """Aggregate all run-entries per lane into the ordered list of dwords the lane writes."""
    per_lane = defaultdict(list)
    for a in accesses:
        per_lane[a["lane"]].append((a["reg0"], _dwords(a, dtype_bytes)))
    out = {}
    for lane, runs in per_lane.items():
        runs.sort()
        dws = []
        for _, ds in runs:
            dws.extend(ds)
        out[lane] = dws
    return out


def simulate(accesses, arch=GFX90A, dtype_bytes=2):
    """Address-map driver: build per-(half-wave, phase) histograms from the exact `accesses`
    ({lane, reg0, base, vw}) and apply the port rule via `simulate_hist`. Returns the per-instruction
    result plus `conflicts_per_access` and the `detail` histograms.

    KNOWN LIMITATION: the emit-derived address map under-scales the multi-run footprint for
    forced-narrow stores; drive full validation from `simulate_hist` with the measured histograms.
    The single-contiguous-run configs (natural A b64 / B b128 stores) reproduce end-to-end."""
    a = arch_lds(arch)
    lane_dw = _lane_dwords(accesses, dtype_bytes)
    ndw = max(len(v) for v in lane_dw.values())

    hists = {}
    footprint = set()
    for hw in range(0, 64, a.HALF):
        for ph in range(ndw):
            seen = defaultdict(set)
            for lane in range(hw, hw + a.HALF):
                dws = lane_dw.get(lane)
                if not dws or ph >= len(dws):
                    continue
                d = dws[ph]
                footprint.add(d)
                seen[d % a.NB].add(d)
            hists[(hw, ph)] = {b: len(s) for b, s in seen.items()}
    r = simulate_hist(hists, len(footprint), a)
    r["conflicts_per_access"] = r["BC"] / r["productive"] if r["productive"] else 0.0
    r["detail"] = hists
    return r


# ==================================================================================================
# Pad-sweep stripe-alignment model (the conflict-free fix)
# ==================================================================================================
def dwords_per_lane(wtag):
    """W = dwords a single lane writes per store op. b32 -> 1, b64 -> 2, b128 -> 4."""
    return {"b32": 1, "b64": 2, "b128": 4}[wtag]


def predict_pad_sweep(stride_dwords, wtag, arch=GFX90A, *, pad0_depth=None):
    """Validated conflicts/access for a K-aliased coop store at LDS row stride `stride_dwords`
    (in dwords) and store width `wtag`. Reproduces the measured rocprof pad sweep to the number.

    The CONFLICT-FREE stripe unit is DERIVED from the K-alias DEPTH at pad0 -- the max number of lanes
    stacked on one bank when the LDS row stride is a whole number of banks (``s=0``). That depth is read
    straight off the address map (``max`` bank depth of the pad0 histogram), NOT a per-config constant:

        unit = NB * W / pad0_depth

    A deeper alias column needs a smaller K-shift to spread across the banks -> a nearer conflict-free pad.
    ``pad0_depth=None`` reproduces the legacy ``4*W`` (the ``depth = NB/4`` geometry the corpus was measured
    at). This predicts the conflict-FREE pad only; the conflict MAGNITUDE (3.0 pile vs 1.0 floor) comes from
    ``simulate`` / ``simulate_hist`` (the write-port model), which carries the footprint the closed form lacks."""
    a = arch_lds(arch)
    W = dwords_per_lane(wtag)
    s = stride_dwords % a.NB
    unit = (a.NB * W // pad0_depth) if pad0_depth else 4 * W
    if unit and s % unit == 0 and (s // unit) % 2 == 1:
        return 0.0
    if wtag == "b64" and s == 0:
        return 3.0  # whole 8-wide K column stacked on one 8-bank stripe
    return 1.0      # throughput floor (64 lanes > 32 banks): unavoidable, not fixable


def is_conflict_free(stride_dwords, wtag, arch=GFX90A, *, pad0_depth=None):
    return predict_pad_sweep(stride_dwords, wtag, arch, pad0_depth=pad0_depth) == 0.0


def conflict_free_bank_of(lane, arch=GFX90A):
    """The served group's bank for `lane` in a CONFLICT-FREE store: a full permutation, one lane per
    bank (bank = lane mod NB). This is the physical meaning of BC=0 (SOT lds_banks.md §2) and is what
    the FIXED-panel bank grid must draw -- NOT the naive `dword mod NB` map, which is inverted vs HW
    for the padded stores."""
    return lane % arch_lds(arch).NB


def recommend_pad(tile_free, wtag, arch=GFX90A, *, pad0_depth=None, max_extra_pad=None, align=8,
                  dtype_bytes=2):
    """Smallest trailing row pad (in elems) that makes a K-aliased store CONFLICT-FREE by the
    validated stripe-alignment rule -- closed-form, no GPU. `align` keeps the pad a multiple of the
    store's alignment (8 f16 = b128) so it does not narrow the access width. `pad0_depth` (the pad0 K-alias
    depth read off the address map) sets the stripe unit, so the fix pad is correct at any geometry -- e.g.
    a depth-16 alias fixes at +16 while the depth-8 default fixes at +32. Returns the pad, or None if no
    conflict-free pad exists within `max_extra_pad` (default: one full NB stripe)."""
    a = arch_lds(arch)
    per_dword = 4 // dtype_bytes
    limit = max_extra_pad if max_extra_pad is not None else a.NB * per_dword
    for pad in range(0, limit + 1, align):
        if is_conflict_free((tile_free + pad) // per_dword, wtag, a, pad0_depth=pad0_depth):
            return pad
    return None


class ConflictModelError(AssertionError):
    """Raised when the validated simulator does NOT reproduce the measured counters. The cardinal
    rule: if the model does not match hardware, the MODEL is wrong -- fix it, never 'meet in the
    middle'. This is a hard stop, not a warning."""


def gate(sim, measured, *, tol=1e-6, rtol=2e-2, label="", absolute=False):
    """HARD gate: assert the simulator reproduces the measured HARDWARE. The authoritative quantity is
    `conflicts_per_access = BC/(IDX-BC)` -- a RATIO with an IDENTICAL definition on both sides
    (`simulate` and `parse_counter_csv`), so it is comparable REGARDLESS of scale: per-served-group sim
    vs whole-run counters. That ratio IS the scale-invariant form of "the model matches the GPU" (the
    cardinal-rule number), and is always checked.

    Absolute BC / IDX are only meaningful when BOTH sides are at the SAME scale -- e.g. the per-
    instruction validation corpus, or a deliberate per-group cross-check. Pass `absolute=True` for those;
    do NOT for a live whole-run measurement, where per-group sim BC vs whole-run measured BC is a SCALE
    error, not a model error (comparing them would guarantee a spurious failure). Raises
    ConflictModelError on any mismatch -- there is no soft path (enforcement of the cardinal rule). A NaN
    on either side is itself a failure: a degenerate counter (e.g. IDX==BC) is never a silent pass."""
    tag = f"[{label}] " if label else ""
    # conflicts_per_access is the scale-invariant HW-match quantity, but it is a LIVE whole-run RATIO with
    # sub-percent counter noise (steady-state variance + the ~0 ADDR-broadcast events) -- an absolute 1e-6
    # tol is a corpus/exact-arithmetic tolerance, unreachable on hardware. Gate it with a small RELATIVE
    # tolerance instead: a WRONG model is off by whole conflict factors (3 vs 7, 3 vs 1), never by 0.2%.
    # BC/IDX (absolute=True, the per-instruction corpus) stay strict -- those are exact by construction.
    keys = [("conflicts_per_access", max(tol, rtol * abs(float(measured.get("conflicts_per_access", 0.0)))))]
    if absolute:
        keys = [("BC", 0.5), ("IDX", 0.5)] + keys
    checked = False
    for key, itol in keys:
        if key in sim and key in measured:
            checked = True
            sv, mv = float(sim[key]), float(measured[key])
            if math.isnan(sv) or math.isnan(mv) or abs(sv - mv) > itol:
                raise ConflictModelError(
                    f"{tag}model does NOT reproduce hardware: {key} sim={sim[key]} "
                    f"measured={measured[key]}. The MODEL is wrong -- fix lds_conflict.py, do not "
                    f"proceed.")
    if not checked:
        raise ConflictModelError(
            f"{tag}gate has nothing to compare -- neither conflicts_per_access nor (with absolute=True) "
            f"BC/IDX is present in BOTH sim and measured. Refusing to pass a vacuous gate.")
    return True


# ==================================================================================================
# Bit-exact (lane, register) -> LDS address map (drives the REAL emit)
# ==================================================================================================
class NumBuilder:
    """Numeric evaluator implementing exactly the IRBuilder ops emit_tensor_coordinates uses, over
    plain python ints for ONE concrete thread id. No SSA -- values ARE ints."""

    def __init__(self, thread: int):
        self._thread = thread

    def const_i32(self, v):
        return int(v)

    def thread_id_x(self):
        return self._thread

    def div(self, a, b):
        return int(a) // int(b)

    def mod(self, a, b):
        return int(a) % int(b)

    def mul(self, a, b):
        return int(a) * int(b)

    def add(self, a, b):
        return int(a) + int(b)

    def xor(self, a, b):
        return int(a) ^ int(b)

    def shl(self, a, b):
        return int(a) << int(b)


def access_width(tile_desc, strides, dtype_name="f16", lds_swizzle=False):
    """The vw the emit would choose for this LDS access (drives the per-access dword count)."""
    from rocke.helpers.tiling.emit import _contiguous_run, _swizzle_vw

    _ALIGN = 2 if dtype_name == "f16" else 4

    class _Win:
        bounds = None

        class tensor:
            pass

    _Win.tensor.strides = strides
    _Win.tensor.dtype = type("dt", (), {"name": dtype_name})
    vw = _contiguous_run(tile_desc.layout, _Win, _Win.tensor.dtype)
    if lds_swizzle:
        vw = _swizzle_vw(lds_swizzle, vw, _ALIGN)
    return vw


def addr_map(tile_desc, strides, origin=(0, 0), n_lanes=64, dtype_name="f16", lds_swizzle=False):
    """Return (accesses, vw). Each access = one (lane, register-run) wide op:
        {lane, reg0, vw, base}   where base is the element address of the run start.

    The emit issues one wide op per `vw` registers from reg0 (store_fragment loop
    `for register in range(0, regcount, vw)`), writing `vw` CONSECUTIVE elems from the base
    position. Bit-identical to what the kernel emits -- no re-derivation of the encoding math."""
    from rocke.helpers.tiling.emit import (
        _swizzle_lds_positions, emit_tensor_coordinates,
    )

    vw = access_width(tile_desc, strides, dtype_name, lds_swizzle)
    regcount = tile_desc.register_count
    accesses = []
    swz = _swizzle_lds_positions if lds_swizzle is True else lds_swizzle
    for lane in range(n_lanes):
        nb = NumBuilder(lane)
        for reg0 in range(0, regcount, vw):
            coords = emit_tensor_coordinates(nb, tile_desc.layout, lane, reg0)
            positions = [origin[ax] + coords[ax] for ax in range(len(coords))]
            if lds_swizzle:
                positions = swz(nb, positions)
            base = sum(positions[ax] * strides[ax] for ax in range(len(positions)))
            accesses.append({"lane": lane, "reg0": reg0, "vw": vw, "base": base})
    return accesses, vw


# ==================================================================================================
# Isolation micro-probes (generic -- caller supplies the exact kernel descriptors)
# ==================================================================================================
@dataclass
class ProbeDescs:
    """The exact descriptors an LDS store/read probe needs, supplied by the caller so the probe
    stays bit-identical to the kernel under study (no re-derivation).

    coop_native : load layout in the coop band's NATIVE (free, K) order (global load).
    coop_store  : the wide LDS store layout, (K, free) memref order (transpose of coop_native).
    wave_read   : the wave-tile read layout, (K, free) order (the MMA-operand read).
    """

    coop_native: object
    coop_store: object
    wave_read: object

    @classmethod
    def from_coop(cls, coop_native, wave_native, *, transpose):
        """Build ProbeDescs from the NATIVE (free, K) coop and wave descriptors, applying `transpose`
        (e.g. `_transpose_desc`) to BOTH to get the (K, free) store/read memref order the kernel
        emits. Removes the recurring 'which one do I transpose, and which direction' mistake -- both
        the store and the wave read are transposes of their natives, so this does it once, correctly.
        """
        return cls(coop_native=coop_native, coop_store=transpose(coop_native),
                   wave_read=transpose(wave_native))


def build_probe(descs: ProbeDescs, mode: str, *, name=None, tile_free=128, tile_k=16, n_waves=8,
                warp_free=64, lds_pad=0, n_iter=64, force_vw=0, lds_swizzle=False, dtype=None):
    """Build a store-mirror or read-only probe KernelDef using the caller's exact descriptors.

    mode="store": loop{ store(coop); sync; read(store-layout); sync } -- read keeps the store live,
                  measures the store pattern (write + read of it).
    mode="read" : store once; loop{ read(wave); sync } -- isolates the read pattern.

    force_vw (elems) forces a narrower access width via an identity swizzle; lds_swizzle installs a
    real position swizzle. Every probe is a round-trip identity (verified max_abs_diff==0.0)."""
    from rocke.core.ir import F16, I32, IRBuilder, PtrType
    from rocke.helpers.tiling import (
        load_fragment, make_fragment, make_tensor_desc, make_window, store_fragment,
    )

    dt = dtype or F16
    coop_free = tile_free // n_waves

    if lds_swizzle:
        swz = lds_swizzle
    elif force_vw:
        def _identity(_b, positions):
            return positions
        _identity.vw_elems = force_vw
        swz = _identity
    else:
        swz = False

    vwtag = ("_swz" if lds_swizzle else f"_vw{force_vw}" if force_vw else "")
    kname = name or f"lds_probe_{mode}_pad{lds_pad}{vwtag}_{tile_free}x{tile_k}"
    b = IRBuilder(kname)
    b.kernel.attrs["max_workgroup_size"] = 64

    in_ptr = b.param("IN", PtrType(dt, "global"), noalias=True, readonly=True, align=16)
    out_ptr = b.param("OUT", PtrType(dt, "global"), noalias=True, writeonly=True, align=16)
    b.param("N", I32)

    tid = b.thread_id_x()
    zero = b.const_i32(0)

    band_td = make_tensor_desc((coop_free, tile_k), (tile_k, 1), dt)
    stride = tile_free + lds_pad
    lds = b.smem_alloc(dt, [tile_k, stride], name_hint="lds_probe")
    lds_td = make_tensor_desc((tile_k, tile_free), (stride, 1), dt)

    def _load_band():
        return load_fragment(b, in_ptr, make_window(band_td, (zero, zero)), descs.coop_native, tid)

    def _store(frag):
        f = make_fragment(descs.coop_store, dt, frag.value)
        store_fragment(b, lds, make_window(lds_td, (zero, zero)), f, tid, lds_swizzle=swz)

    def _read_store_layout():
        return load_fragment(b, lds, make_window(lds_td, (zero, zero)), descs.coop_store, tid,
                             lds_swizzle=swz)

    def _read_wave():
        return load_fragment(b, lds, make_window(lds_td, (zero, zero)), descs.wave_read, tid,
                             lds_swizzle=False)

    n_c = b.const_i32(n_iter)
    if mode == "store":
        band = _load_band()
        loop = b.scf_for_iter(zero, n_c, b.const_i32(1), [], iv_name="i")
        with loop:
            _store(band)
            b.sync_lds_only()
            _read_store_layout()
            b.sync_lds_only()
            b.scf_yield()
        _store(band)
        b.sync_lds_only()
        rd = _read_store_layout()
        store_fragment(b, out_ptr, make_window(band_td, (zero, zero)),
                       make_fragment(descs.coop_native, dt, rd.value), tid)
    elif mode == "read":
        band = _load_band()
        _store(band)
        b.sync_lds_only()
        loop = b.scf_for_iter(zero, n_c, b.const_i32(1), [], iv_name="i")
        with loop:
            _read_wave()
            b.sync_lds_only()
            b.scf_yield()
        rd = _read_wave()
        out2 = make_tensor_desc((tile_k, warp_free), (warp_free, 1), dt)
        store_fragment(b, out_ptr, make_window(out2, (zero, zero)),
                       make_fragment(descs.wave_read, dt, rd.value), tid)
    else:
        raise ValueError(mode)

    b.ret()
    return b.kernel


def run_probe(descs: ProbeDescs, mode, *, arch="gfx90a", tile_free=128, tile_k=16, n_waves=8,
              warp_free=64, lds_pad=0, n_iter=64, grid_ctas=512, verify=True, force_vw=0,
              block_lanes=64, lds_swizzle=False):
    """Compile, launch and verify a probe on the real GPU. Returns a dict incl. max_abs_diff
    (None when the config masks lanes so a full-band compare would false-flag untouched cells)."""
    import numpy as np

    from rocke.helpers.compile import compile_kernel
    from rocke.helpers.spec import SignatureBuilder
    from rocke.runtime.hip_module import Runtime, get_device_arch
    from rocke.runtime.host_buffers import as_u8_buffer
    from rocke.runtime.launcher import (
        DeviceMem, KernelLauncher, LaunchConfig, synchronize_and_release,
    )

    if get_device_arch(0) != arch:
        raise RuntimeError(f"need {arch}")
    kernel = build_probe(descs, mode, tile_free=tile_free, tile_k=tile_k, n_waves=n_waves,
                         warp_free=warp_free, lds_pad=lds_pad, n_iter=n_iter, force_vw=force_vw,
                         lds_swizzle=lds_swizzle)
    art = compile_kernel(kernel, arch=arch)
    sig = SignatureBuilder().ptr("IN", "f16").ptr("OUT", "f16").scalar("N", "i32").build()
    launcher = KernelLauncher(hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig)

    coop_free = tile_free // n_waves
    rng = np.random.default_rng(0)
    in_h = rng.integers(-5, 6, size=(coop_free, tile_k)).astype(np.float16)
    out_h = (np.zeros((coop_free, tile_k), dtype=np.float16) if mode == "store"
             else np.zeros((tile_k, warp_free), dtype=np.float16))

    rt = Runtime()
    in_d, out_d = DeviceMem(in_h.nbytes), DeviceMem(out_h.nbytes)
    rt.memcpy_h2d(in_d.ptr(), as_u8_buffer(in_h), in_h.nbytes)
    rt.memcpy_h2d(out_d.ptr(), as_u8_buffer(out_h), out_h.nbytes)
    launcher({"IN": in_d, "OUT": out_d, "N": n_iter},
             config=LaunchConfig(grid=(grid_ctas, 1, 1), block=(block_lanes, 1, 1)))
    synchronize_and_release()
    rt.memcpy_d2h(as_u8_buffer(out_h), out_d.ptr(), out_h.nbytes)

    diff = None
    if verify and mode == "store" and block_lanes == 64:
        diff = float(np.abs(out_h.astype(np.float32) - in_h.astype(np.float32)).max())
    return {"mode": mode, "pad": lds_pad, "force_vw": force_vw, "lds_swizzle": bool(lds_swizzle),
            "block_lanes": block_lanes, "kernel": kernel.name, "max_abs_diff": diff}


# ==================================================================================================
# rocprof harness
# ==================================================================================================
COUNTER_PMC = ("pmc: SQ_LDS_BANK_CONFLICT SQ_LDS_ADDR_CONFLICT SQ_LDS_IDX_ACTIVE "
               "SQ_INSTS_LDS SQ_WAVES")

# The validated recipe: bare-metal rocprofv3 SIGABRTs on this host (HSA 8.19); profile inside the
# ROCm-7.14 container. Do NOT wrap in env/bash -c re-exec chains (double-exec re-registers the tool
# and SIGABRTs). See the /bank-conflict skill for the full container bring-up.
ROCPROF_RECIPE = r"""
# 1) write COUNTER_PMC to lds_counters.txt, then, inside the ROCm-7.14 container:
export LD_LIBRARY_PATH=/opt/venv/lib/python3.14/site-packages/_rocm_sdk_devel/lib:\
/opt/venv/lib/python3.14/site-packages/_rocm_sdk_core/lib:$LD_LIBRARY_PATH
export PYTHONPATH=python ROCKE_CPP_QUIET_FALLBACK=1
rocprofv3 -i lds_counters.txt --kernel-include-regex '<kernel-name>' --truncate-kernels \
  --output-format csv -d <outdir> -- python3 <probe-runner-script>
# 2) CSV lands at <outdir>/pmc_1/*/*_counter_collection.csv (root-owned; rm from inside container).
"""


def parse_counter_csv(outdir):
    """Parse a rocprofv3 counter_collection CSV: sum each counter across SE rows for the steady-state
    dispatches (drop dispatch 0 = JIT warm-up). Returns {kernel, BC, IDX, ADDR, INSTS_LDS, WAVES,
    conflicts_per_access}."""
    files = glob.glob(f"{outdir}/**/*_counter_collection.csv", recursive=True)
    if not files:
        raise FileNotFoundError(f"no counter_collection CSV under {outdir}")
    rows = []
    with open(files[0]) as f:
        rows = list(csv.DictReader(f))
    disp = sorted({int(r["Dispatch_Id"]) for r in rows})
    steady = [d for d in disp if d != disp[0]] or disp
    agg = defaultdict(float)
    for r in rows:
        if int(r["Dispatch_Id"]) in steady:
            agg[r["Counter_Name"]] += float(r["Counter_Value"])
    bc = agg.get("SQ_LDS_BANK_CONFLICT", 0.0)
    idx = agg.get("SQ_LDS_IDX_ACTIVE", 0.0)
    cpa = bc / (idx - bc) if (idx - bc) > 0 else float("nan")
    return {"kernel": rows[0]["Kernel_Name"], "BC": bc, "IDX": idx,
            "ADDR": agg.get("SQ_LDS_ADDR_CONFLICT", 0.0),
            "INSTS_LDS": agg.get("SQ_INSTS_LDS", 0.0), "WAVES": agg.get("SQ_WAVES", 0.0),
            "dispatches": len(steady), "conflicts_per_access": cpa}


# ==================================================================================================
# 3-panel register->LDS dataflow renderer
# ==================================================================================================
def store_datum(store_desc, tile_free, arch=GFX90A, strides=None, dtype_name="f16"):
    """The single source of the per-slot store picture, shared by the simulator driver, the renderer,
    and the orchestrator so they can never diverge. Returns (acc, vw, datum) where
    `datum[(lane, phase)] = (K, free, dword, bank)` for each dword-phase the lane writes."""
    a = arch_lds(arch)
    strides = strides or (tile_free, 1)
    per_dword = 4 // (2 if dtype_name == "f16" else 4)
    acc, vw = addr_map(store_desc, strides, n_lanes=64, dtype_name=dtype_name)
    datum = {}
    for ac in acc:
        d0 = ac["base"] // per_dword
        for ph in range(vw // per_dword):
            elem = ac["base"] + per_dword * ph
            datum[(ac["lane"], ph)] = (elem // tile_free, elem % tile_free, d0 + ph, (d0 + ph) % a.NB)
    return acc, vw, datum


def collision_lanes(datum, arch=GFX90A, bank=0, phase=0):
    """The lanes of the representative served group (half-wave 0, `phase`) that pile on `bank` -- the
    collision the counters/sim prove. Single definition, used by the renderer and `_locate_collision`."""
    a = arch_lds(arch)
    return [l for l in range(a.HALF) if datum[(l, phase)][3] == bank]


def render_conflict_3panel(out_path, *, store_desc, tile_free, wtag, measured_cpa, measured_bc=None,
                           fix_pad, fix_label, arch="gfx90a", operand_label="A", dims_label="M",
                           macro_label="macro 128x256, waves 2x4, tile_k=16", strides=None,
                           dtype_name="f16", subject_pad=0, max_banks=1, max_lanes=16, full=False):
    """Render a two-row, 3-panel register->LDS dataflow figure for one operand's store conflict.

    ROW 1 (CONFLICTED, at `subject_pad`): (1) register file tid x reg, the shown threads highlighted;
      (2) funnel arrows for the same threads piling onto the shown bank(s);
      (3) LDS bank grid, RED BOX + N-way on the piled columns.
    ROW 2 (FIXED): the SAME threads with the validated de-aliasing `fix_pad` -- arrows fan out to
      distinct banks, boxes gone.

    By DEFAULT the funnel shows ONE representative piled bank (the cleanest single-conflict story).
    Pass `full=True` to draw EVERY conflicted bank (ignores `max_banks`/`max_lanes`); or set
    `max_banks`/`max_lanes` for a bounded handful. The bank grid (panel 3) always shows all banks.

    `subject_pad` selects WHICH pad state is the conflicted subject (0 = the raw pad0 pile; a partial
    pad shows its residual piles). The representative bank + multiplicity are DERIVED from the data,
    not hardcoded. EVERY number is gated: the subject panel's BC/c-a must equal the supplied MEASURED
    values via the validated simulator, and the fix pad must be conflict-free by the stripe rule. ALL
    DRAWING is delegated to `layout_render.render_conflict_dataflow` -- this module never touches
    matplotlib, so the figure is machine/model-independent. Returns out_path."""
    from rocke.helpers.tiling.visualization.layout_render import render_conflict_dataflow

    a = arch_lds(arch)
    per_dword = 4 // (2 if dtype_name == "f16" else 4)
    strides = strides or (tile_free + subject_pad, 1)  # subject pad sets the LDS row stride

    acc, vw, datum = store_datum(store_desc, tile_free, a, strides, dtype_name)

    # --- GATE 1: reproduce the SUBJECT collision via the validated write-port sim. The gate is on
    # conflicts/access (scale-invariant: per-served-group sim vs whatever scale the caller measured);
    # `measured_bc`, when given, is an OPTIONAL per-served-group BC cross-check (do NOT pass a whole-run
    # counter here -- it is a different scale). ---
    r = simulate(acc, arch=a, dtype_bytes=(4 // per_dword))
    cpa0 = r["BC"] / (r["IDX"] - r["BC"]) if (r["IDX"] - r["BC"]) else 0.0
    gate({"conflicts_per_access": cpa0}, {"conflicts_per_access": measured_cpa},
         label=f"{operand_label} render subject")
    if measured_bc is not None:
        assert r["BC"] == measured_bc, (
            f"{operand_label} sim per-group BC {r['BC']} != supplied {measured_bc} (per-group scale?)")

    # representative colliding banks: up to `max_banks` most-piled banks in the served group. The
    # SAME threads are drawn conflicted (piled) on top and conflict-free (fanned) on the bottom.
    occ = defaultdict(list)
    for lane in range(a.HALF):
        occ[datum[(lane, 0)][3]].append(lane)
    # most-piled bank first, so the DEFAULT single-bank view shows the worst pile
    piled = sorted((b for b in occ if len(occ[b]) > 1), key=lambda b: (-len(occ[b]), b))
    subject_bank = piled[0] if piled else max(sorted(occ), key=lambda b: len(occ[b]))
    bank_budget = len(piled) if full else (max_banks if max_banks is not None else len(piled))
    lane_budget = 10**9 if full or max_lanes is None else max_lanes
    show_banks, shown_lanes = [], []
    for b in piled:
        if len(show_banks) >= bank_budget:
            break
        if shown_lanes and len(shown_lanes) + len(occ[b]) > lane_budget:
            break
        show_banks.append(b)
        shown_lanes += occ[b]
    assert shown_lanes, (
        f"{operand_label} at pad{subject_pad}: no >=2-way pile to show (already conflict-free?)")

    # --- GATE 2: the fix pad is conflict-free per the validated stripe-alignment rule (== HW). The rule is
    # DEPTH-AWARE: a wide store (b128) into a wide tile K-aliases at a deeper stripe unit (e.g. depth 16, not
    # the depth-8 default), so the check MUST pass the pad0 K-alias depth read off THIS store's map -- exactly
    # as `recommend_pad` does. Omitting it uses the depth-8 default and spuriously rejects a pad the GPU
    # confirms conflict-free (e.g. A b128 into a 256-wide tile: pad16 measures BC=0 but depth-8 predicts 1). ---
    fix_stride_dw = (tile_free + fix_pad) // per_dword
    pad0_depth = max((d for h in r.get("detail", {}).values() for d in h.values()), default=None)
    assert predict_pad_sweep(fix_stride_dw, wtag, a, pad0_depth=pad0_depth) == 0.0, (
        f"{operand_label} fix pad{fix_pad} not conflict-free by the validated (depth-aware) rule")

    def fix_bank(lane):
        return conflict_free_bank_of(lane, a)

    # --- GATE 3: the FIXED panel is drawn collision-free (one lane per bank in the served group) ---
    fixed_occ = defaultdict(list)
    for lane in range(a.HALF):
        fixed_occ[fix_bank(lane)].append(lane)
    assert max(len(v) for v in fixed_occ.values()) == 1, (
        f"{operand_label} FIXED panel would draw a collision -- refuse to label BC=0 over it")

    nway = len(occ[subject_bank])
    subj = f"pad{subject_pad}" if subject_pad else "pad0"
    suptitle = (
        f"CRC {operand_label}-store LDS bank conflict  ({wtag}, {macro_label}, {a.name} NB={a.NB})\n"
        f"K-alias: LDS row stride = {per_dword * (tile_free + subject_pad)} {dtype_name} = "
        f"{(tile_free + subject_pad) // per_dword} dwords -> {nway}-way {dims_label} piles "
        f"(showing {len(show_banks)} of {len(piled)} conflicted banks)\nMEASURED (rocprof) "
        f"conflicts/access = {measured_cpa:.2f}  (sim reproduces to the integer: BC={r['BC']}, "
        f"IDX={r['IDX']}, productive={r['productive']})   |   TOP=conflicted ({subj})  "
        f"BOTTOM={fix_label}")

    # All matplotlib drawing lives in the viz module (one visual language, model-independent).
    return render_conflict_dataflow(out_path, datum=datum, shown_lanes=shown_lanes, half=a.HALF,
                                    nreg=vw // per_dword, nbanks=a.NB, fix_bank_fn=fix_bank,
                                    wtag=wtag, suptitle=suptitle, subject_bank=subject_bank,
                                    cpa=measured_cpa)


# ==================================================================================================
# High-level orchestrator (chains address-map -> sim -> HW gate -> fix -> render into one call)
# ==================================================================================================
@dataclass
class ConflictReport:
    """Everything one operand's store analysis produces, gated and packaged so the skill formats
    tables instead of assembling loose values (removes the hand-built-table error surface)."""

    operand_label: str
    arch: str
    wtag: str
    tile_free: int
    vw: int
    sim: dict                      # {IDX, BC, productive, conflicts_per_access}
    measured: dict | None          # parse_counter_csv output, or None if HW not yet gathered
    gate_passed: bool              # sim reproduced HW to the number (False if no HW yet)
    conflicts_per_access: float    # authoritative value (HW when present, else sim)
    fix_pad: int | None            # smallest conflict-free pad (elems), closed-form
    fix_verified_hw: bool          # True only if the fix pad was ALSO measured conflict-free on GPU
    located: dict                  # {half_wave, phase, bank, cells:[T{l}R{r}...], nway}
    bit_exact: float | None        # probe max_abs_diff (must be 0.0), or None if not run here
    png: str | None                # rendered 3-panel path, or None

    @property
    def verdict(self):
        if self.measured is None:
            return "UNVALIDATED (no hardware counters yet -- do NOT ship this number)"
        return "VALIDATED (sim == hardware)" if self.gate_passed else "MODEL MISMATCH"

    def facts_table(self):
        """Markdown rows for the skill's 'Hard facts' + 'Model validation' tables."""
        m = self.measured or {}
        hard = (f"| {self.operand_label} store pad0 | {m.get('BC', '?')} | {m.get('IDX', '?')} | "
                f"{self.conflicts_per_access:.4f} | {m.get('ADDR', '?')} |")
        val = (f"| {self.operand_label} store pad0 | {self.sim['conflicts_per_access']:.4f} | "
               f"{m.get('conflicts_per_access', float('nan')):.4f} | "
               f"{'PASS' if self.gate_passed else 'FAIL'} |")
        return {"hard_facts_row": hard, "model_validation_row": val}


def _locate_collision(datum, arch):
    """The representative served group the counters/sim proved: half-wave 0, phase 0, the lanes
    piling on bank 0."""
    cells = collision_lanes(datum, arch)
    return {"half_wave": 0, "phase": 0, "bank": 0, "nway": len(cells),
            "cells": [f"T{l}R0" for l in cells]}


def analyze_store(descs: ProbeDescs, *, tile_free, wtag, arch="gfx90a", operand_label="A",
                  dims_label="M", measure=None, verify_fix=False, render_to=None,
                  macro_label="macro 128x256, waves 2x4, tile_k=16", strides=None, dtype_name="f16",
                  **probe_kwargs) -> ConflictReport:
    """One call that runs the whole store analysis and returns a gated ConflictReport:

      address map (bit-exact from emit) -> simulate -> [measure on GPU + HARD gate sim==HW] ->
      recommend the conflict-free pad (closed form) -> [optionally verify the fix on GPU] ->
      render the 3-panel figure.

    `measure` is an INJECTED callable `measure(pad:int, mode:str='store') -> dict` (with BC / IDX /
    conflicts_per_access, optionally ADDR / max_abs_diff). It encapsulates the container rocprof run
    so THIS module stays container-agnostic; the /bank-conflict skill supplies it. If `measure` is
    None the report is returned UNVALIDATED (sim only) and rendering is refused -- a number without
    hardware must never be presented (the cardinal rule).

    `probe_kwargs` (tile_k, n_waves, warp_free, ...) are forwarded to the measure callable's probe.
    """
    a = arch_lds(arch)
    strides = strides or (tile_free, 1)
    per_dword = 4 // (2 if dtype_name == "f16" else 4)

    # 1) bit-exact address map + simulated prediction (shared builder -> renderer sees the same datum)
    acc, vw, datum = store_datum(descs.coop_store, tile_free, a, strides, dtype_name)
    sim = simulate(acc, arch=a, dtype_bytes=(4 // per_dword))
    located = _locate_collision(datum, a)

    # 2) measure on the GPU + HARD gate (skipped only if no measure callable was supplied)
    measured = None
    gate_passed = False
    bit_exact = None
    if measure is not None:
        measured = measure(0, mode="store")
        bit_exact = measured.get("max_abs_diff")
        if bit_exact is not None and bit_exact != 0.0:
            raise ConflictModelError(
                f"{operand_label} store probe not bit-exact (max_abs_diff={bit_exact}); the "
                f"addressing is wrong, counters are meaningless.")
        gate(sim, measured, label=f"{operand_label} store pad0")
        gate_passed = True

    cpa = (measured["conflicts_per_access"] if measured is not None
           else sim["conflicts_per_access"])

    # 3) closed-form fix pad, optionally HW-verified. The conflict-free stripe unit is set by the pad0
    #    K-alias depth read off THIS store's address map (max bank depth) -- no per-config constant, so the
    #    fix pad is correct at any geometry (deep alias -> nearer pad). (Assumes the analysis strides are
    #    pad0, the default; `sim` is then the pad0 histogram.)
    pad0_depth = max((d for h in sim.get("detail", {}).values() for d in h.values()), default=None)
    fix_pad = recommend_pad(tile_free, wtag, a, pad0_depth=pad0_depth, dtype_bytes=per_dword)
    # MODEL-SIDE fix gate (== render GATE 2). The conflict-FREE verdict is a half-stripe PARITY property, which
    # the address-map `simulate` (naive bank=dword mod NB histogram) is structurally blind to -- it reproduces
    # the magnitude of CONFLICTED pads but can never reach 0 at the parity-resolved pads (e.g. a depth-16 b128
    # store: sim keeps a spurious residual at pad16/48 where HW = 0). So the fix is validated by the DEPTH-AWARE
    # stripe rule (`is_conflict_free`), never by `simulate` on the padded strides. This brings the correct model
    # predictor into the analysis path so a report is model-gated even without a GPU (the GPU stays the arbiter).
    if fix_pad is not None:
        assert is_conflict_free((tile_free + fix_pad) // per_dword, wtag, a, pad0_depth=pad0_depth), (
            f"{operand_label} recommended pad {fix_pad} is not conflict-free by the stripe rule -- "
            f"recommend_pad and is_conflict_free disagree (model bug).")
    fix_verified_hw = False
    if verify_fix and measure is not None and fix_pad is not None:
        fm = measure(fix_pad, mode="store")
        if fm.get("conflicts_per_access", 1.0) != 0.0:
            raise ConflictModelError(
                f"{operand_label} recommended pad {fix_pad} measured "
                f"{fm['conflicts_per_access']} conflicts/access on GPU, not 0 -- the stripe rule and "
                f"hardware disagree; fix the model.")
        fix_verified_hw = True

    # 4) render (only with HW-gated numbers -- never present a figure over an ungated number)
    png = None
    if render_to is not None:
        if measured is None:
            raise ConflictModelError(
                f"refusing to render {operand_label}: no hardware counters. Supply `measure` so the "
                f"figure carries a GPU-gated number, not a simulated one.")
        fix_label = (f"pad +{fix_pad} {dtype_name} -> 0-way / BC=0 (closed-form; "
                     f"{'HW-verified' if fix_verified_hw else 'stripe-rule validated'})")
        # NOTE: pass only the scale-invariant measured conflicts/access -- NOT measured["BC"], which is a
        # whole-run counter and does not share scale with the sim's per-served-group BC. The render gate
        # reconciles on conflicts/access; the figure annotates the sim's own per-group BC/IDX.
        png = render_conflict_3panel(
            render_to, store_desc=descs.coop_store, tile_free=tile_free, wtag=wtag,
            measured_cpa=measured["conflicts_per_access"],
            fix_pad=fix_pad, fix_label=fix_label, arch=a, operand_label=operand_label,
            dims_label=dims_label, macro_label=macro_label, strides=strides, dtype_name=dtype_name)

    return ConflictReport(
        operand_label=operand_label, arch=a.name, wtag=wtag, tile_free=tile_free, vw=vw, sim=sim,
        measured=measured, gate_passed=gate_passed, conflicts_per_access=cpa, fix_pad=fix_pad,
        fix_verified_hw=fix_verified_hw, located=located, bit_exact=bit_exact, png=png)


# ==================================================================================================
# Per-arch model validation corpus + self-test
# (proves the mechanism reproduces THAT arch's hardware; NOT per-case answers)
# ==================================================================================================
# The corpus is keyed PER ARCH. Each arch's model is validated ONLY against measurements taken on
# that arch -- gfx90a's numbers must never be used to "validate" gfx942 (different NB / port / combine).
# To add a new arch: (1) add its ArchLDS to ARCHS, (2) measure its own `hists` + `pad_sweep` on the
# real GPU and add a `_VALIDATION_CORPUS[<name>]` entry, (3) run `selftest(<name>)` until it PASSES.
# Until an arch has its own corpus, `selftest` REFUSES it (no silent cross-arch validation).
#
# A bank conflict is a property of the PHYSICAL store geometry ONLY -- the store WIDTH, the LDS row
# stride, the K-alias depth -- NEVER of which operand (A/B) or tensor it came from. So the corpus is keyed
# by PHYSICAL descriptors, context-agnostic: two stores with the same (wtag, tile_free) but a different
# K-alias depth are DIFFERENT rows (e.g. a b128 store into a 256-wide tile is depth-8 for one coop layout,
# depth-16 for another) -- the depth is what the model reads, not the operand.
#
# hists   : per-INSTRUCTION measured histograms. Each store is K-aliased so every used bank has the same
#           depth. (name, banks_used, depth, n_phases, footprint_dwords, HW_IDX, HW_BC). `name` is a
#           physical descriptor (wtag / footprint / banks), not a tensor.
# pad_sweep: measured store-mirror pad sweep. (wtag, tile_free, pad, HW conflicts/access, pad0_depth);
#           row stride in dwords = (tile_free + pad) / 2. `pad0_depth` = the pad0 K-alias depth of THAT
#           geometry (read off its address map), which sets the stripe unit NB*W/depth. The legacy depth-8
#           rows keep pad0_depth=8 (== the old 4*W default); a wider/deeper alias needs a nearer pad (the
#           b128 depth-16 store into a 256-wide tile is conflict-free at +16, not +32).
_VALIDATION_CORPUS = {
    "gfx90a": {  # rocprofv3, bit-exact isolation probes
        "hists": [
            ("b64  fp128 b4  pad0", 4, 8, 2, 128, 16, 12),
            ("b64  fp128 b16 pad8", 16, 2, 2, 128, 8, 4),
            ("b32  fp128 b8  pad0", 8, 8, 2, 128, 32, 28),
            ("b32  fp128 b32 pad8", 32, 2, 2, 128, 8, 4),
            ("b128 fp256 b4  pad0", 4, 8, 4, 256, 16, 8),
            ("b128 fp256 b8  pad8", 8, 4, 4, 256, 16, 8),
            ("b64  fp256 b8  pad0", 8, 8, 2, 256, 32, 24),
            ("b32  fp128 b16 pad0", 16, 8, 2, 128, 32, 28),
        ],
        "pad_sweep": [
            # (wtag, tile_free, pad, HW conflicts/access, pad0_depth) -- purely physical, no operand.
            ("b64", 128, 0, 3.0, 8), ("b64", 128, 8, 1.0, 8), ("b64", 128, 16, 0.0, 8),
            ("b64", 128, 24, 1.0, 8), ("b64", 128, 32, 1.0, 8), ("b64", 128, 40, 1.0, 8),
            ("b64", 128, 48, 0.0, 8),
            ("b128", 256, 0, 1.0, 8), ("b128", 256, 8, 1.0, 8), ("b128", 256, 16, 1.0, 8),
            ("b128", 256, 24, 1.0, 8), ("b128", 256, 32, 0.0, 8), ("b128", 256, 40, 1.0, 8),
            ("b128", 256, 48, 1.0, 8), ("b128", 256, 56, 1.0, 8), ("b128", 256, 64, 1.0, 8),
            # b128 into a 256-wide tile at a DEPTH-16 K-alias (unit NB*W/16 = 8) -- rocprof-measured on
            # gfx90a. pad0's 3.0 MAGNITUDE is a `simulate`/hists fact, not a stripe-rule one
            # (predict_pad_sweep is the conflict-free VERDICT: 0 at the odd half-stripe pads 16, 48).
            ("b128", 256, 8, 1.0, 16), ("b128", 256, 16, 0.0, 16), ("b128", 256, 24, 1.0, 16),
            ("b128", 256, 32, 1.0, 16), ("b128", 256, 48, 0.0, 16),
        ],
    },
}


def register_arch(arch: ArchLDS, hists, pad_sweep):
    """Store a NEW arch's model + its own validation corpus, the same way gfx90a is stored. `hists`
    and `pad_sweep` must be freshly MEASURED on that arch (see `_VALIDATION_CORPUS` format). After
    registering, `selftest(arch.name)` must PASS before the model is trusted for any number."""
    ARCHS[arch.name] = arch
    _VALIDATION_CORPUS[arch.name] = {"hists": list(hists), "pad_sweep": list(pad_sweep)}


def _uniform(banks, depth, n_half, n_phase):
    return {(hw, ph): {b: depth for b in range(banks)}
            for hw in range(n_half) for ph in range(n_phase)}


def selftest(arch=GFX90A, verbose=True):
    """Gate: the write-port model + stripe-alignment rule must reproduce THIS arch's OWN measured
    corpus to the integer / to the number. Returns True iff the model is valid for `arch`. Refuses
    (raises) an arch that has no measured corpus -- no cross-arch validation."""
    a = arch_lds(arch)
    if a.name not in _VALIDATION_CORPUS:
        raise ConflictModelError(
            f"no validation corpus for {a.name}: cannot self-test its LDS model. Measure {a.name}'s "
            f"own hists + pad_sweep on the real GPU and register_arch(...) them first -- gfx90a's "
            f"corpus must NOT be used to validate another arch.")
    corpus = _VALIDATION_CORPUS[a.name]
    ok = True
    if verbose:
        print(f"== write-port model vs measured histograms ({a.name}) ==")
        print(f"{'config':13s} | {'IDX':>3} {'hw':>3} | {'BC':>3} {'hw':>3} | ok")
    for name, banks, depth, nph, fp, hidx, hbc in corpus["hists"]:
        r = simulate_hist(_uniform(banks, depth, 2, nph), fp, a)
        row_ok = (r["IDX"] == hidx and r["BC"] == hbc)
        ok &= row_ok
        if verbose:
            print(f"{name:13s} | {r['IDX']:>3} {hidx:>3} | {r['BC']:>3} {hbc:>3} | "
                  f"{'OK' if row_ok else 'FAIL'}")
    if verbose:
        print(f"\n== pad-sweep stripe-alignment rule vs measured conflicts/access ({a.name}) ==")
        print(f"{'config':16s} | stride_dw s | {'sim c/a':>7} {'HW c/a':>7} | ok")
    for wtag, tf, pad, hw, pad0_depth in corpus["pad_sweep"]:
        stride = (tf + pad) // 2
        sim = predict_pad_sweep(stride, wtag, a, pad0_depth=pad0_depth)
        row_ok = abs(sim - hw) < 1e-9
        ok &= row_ok
        if verbose:
            print(f"{wtag + ' tf' + str(tf) + ' d' + str(pad0_depth) + ' pad' + str(pad):22s} | "
                  f"{stride:8d} {stride % a.NB:2d} | {sim:7.2f} {hw:7.2f} | {'OK' if row_ok else 'FAIL'}")
    if verbose:
        print("\nGATE:", "PASS" if ok else "FAIL - model wrong, do not trust any number it produces")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
