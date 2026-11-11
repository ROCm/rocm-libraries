# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tuning registry of the attention backward (layout-general family).

Every performance knob of the backward is one :class:`KnobAxis`: the
compile-time fields of :class:`~kernels.common.attention_bwd.AttnBwdSpec`,
the runtime plan values of :mod:`kernels.common.attention_bwd_plan`
(``RUNTIME_KNOBS``) and the prep / convert tuning values (``AUX_KNOBS``).
``choices[0]`` is the default (the spec field's declared default, or the
plan's value when the hook passes nothing), so a configuration that leaves an
axis alone emits today's code.

Each axis also records its lever family (the release order of the sweep
driver), the funnel stage that owns it, whether it recompiles (``compile``)
or is a plan value (``runtime``), the sweep level that enumerates it, and the
values that are registered but not emitted:

* ``not_built``: values the kernel's emission check (``_not_built``) or the
  plan refuses. They stay registered (levers are retained) and the sweep
  records them ``not_built`` without compiling them.
* ``full_only``: values only the ``full`` level (the per-shape dispatcher
  sweep) enumerates.
* ``gated``: values a device probe has not cleared yet; no level emits them.

:data:`BWD_KNOWN_WRONG_KNOBS` holds, per target, the knob values whose
kernels computed wrong outputs (a failed reference check). They are held out
of every level (:meth:`KnobAxis.values`), refused by the legality verdict,
refused in the tuned tables (checked at import) and declined by the
dispatcher's candidates.

Legality is the kernel's own: :func:`backward_spec_legality` resolves a spec
with ``validate_attn_bwd_spec`` and then applies ``_not_built`` (and the
known-wrong values); there is no second model here. This module holds the
registry only; the request, candidates and ranker of the backward dispatcher
are :mod:`.registry`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from kernels.common._attention_bwd_caps import AGPR_ASM_TAIL_NOP, asm_tail_nop_min
from kernels.common._attention_bwd_tiles import g02_grids, g13_grids
from kernels.common.attention_bwd import (
    AGPR_ALLOCS,
    AGPR_ROUTES,
    ASM_CLUSTERS,
    ASM_TAIL_NOP_MAX,
    ATOMS,
    BARRIER_MERGE_ARCHS,
    BARRIER_MERGE_STAGE_ORDERS,
    BARRIER_MERGES,
    BF16_CASTS,
    BLOCK_K4_VALUES,
    BLOCK_M_VALUES,
    BLOCK_N_VALUES,
    COUNTED_ISSUES,
    BWD_EXP2S,
    DKV_ATOMIC_GUARDS,
    DKV_EPILOGUES,
    DKV_CELL_VECS,
    DKV_SINK_NOT_BUILT,
    DKV_SINKS,
    DKV_STORE_VECS,
    NARROW_VECS,
    WS_BOUNDS,
    DQ_ISSUES,
    DQ_MODES,
    FENCED_SCHEDS,
    G4_SLICINGS,
    GLOBAL_PATHS,
    LDS_ORDERS,
    LOOP_SYNCS,
    NOT_YET_EFFECTIVE_KNOBS,
    PT_ROUTES,
    PTR_NOALIAS,
    QDO_STAGES,
    RESIDENCIES,
    RING_ISSUES,
    SCHEDS,
    SPLIT_DQ_AVAILABLE,
    STAGE_ORDERS,
    STAGE_TABLE_SCHEDS,
    STATIC_SCALES,
    STATS_LOADS,
    STATS_VECS,
    STEP_ORDERS,
    SWIZZLES,
    TRANSPOSE_SOURCES,
    WAVES_PER_EU_VALUES,
    WS_LAYOUTS,
    AttnBwdSpec,
    _not_built,
    attn_bwd_arch_facts,
    validate_attn_bwd_spec,
)
from kernels.common._attention_bwd_valu import NATIVE_BF16_CVT_ARCHS
from kernels.common.attention_bwd_aux import CONVERT_VECS
from kernels.common.attention_bwd_dq import (
    DQ_AGPR_ALLOCS,
    DQ_BLOCK_M_VALUES,
    DQ_BLOCK_N_VALUES,
    DQ_HEAD_LOOPS,
    DQ_KV_PADS,
    DQ_SCHEDS,
    DQ_TRANSPOSE_SOURCES,
)
from kernels.common.attention_bwd_plan import (
    AUX_KNOBS,
    AUX_ROW_MULTIPLES,
    DECLARED_ARCHS,
    DQ_KNOBS,
    LB_ORDERS,
    MIRROR_PAIRING_BUILT,
    RUNTIME_KNOBS,
    SCALE_PLACEMENTS,
)

__all__ = [
    "ASM_TAIL_NOP_FLOOR",
    "BWD_KNOB_AXES",
    "BWD_KNOWN_WRONG_KNOBS",
    "CODEGEN_STRATEGIES",
    "FUNNEL_STAGES",
    "KIND_COMPILE",
    "KIND_DQ",
    "KIND_RUNTIME",
    "KnobAxis",
    "LEVER_FAMILIES",
    "NUMERICS",
    "PROBLEM_FIELDS",
    "SWEEP_LEVELS",
    "BwdSpecLegality",
    "axes_of_stage",
    "backward_axis",
    "backward_spec_legality",
    "backward_tuning_axes",
    "check_known_wrong",
    "knob_family",
    "known_wrong_findings",
    "known_wrong_values",
    "sweepable",
]

# The eight lever families, in release order (the sweep driver releases them
# one at a time; an unreleased family contributes only its default).
LEVER_FAMILIES: Mapping[int, str] = {
    1: "geometry",
    2: "residency_registers",
    3: "transpose_global_path",
    4: "atoms_operand_routes",
    5: "schedule",
    6: "codegen_minor_swizzle",
    7: "runtime",
    8: "split_dq",
}
# Funnel stage words that own an axis: geometry G, structure T, schedule H,
# codegen K, swizzle S, runtime R.
FUNNEL_STAGES = ("geometry", "structure", "schedule", "codegen", "swizzle", "runtime")
SWEEP_LEVELS = ("production", "full")
KIND_COMPILE = "compile"  # a spec field: recompiles, salted into the kernel name
KIND_RUNTIME = "runtime"  # a plan value (a kernarg, or an aux-kernel recompile)
# a split dQ kernel value (the plan's DQ_KNOBS, dq_mode = "split" only): an
# AttnBwdDqSpec field (recompiles the dQ kernel, never main) or its XCD chunk
KIND_DQ = "dq_kernel"
# Numerics class of a non-default value against the default:
#   bitwise      the same fp32 operations in the same order: dK / dV (and the
#                dQ partials) bit for bit those of the default;
#   reorders_fp  the same algorithm with another fp32 summation order or
#                rounding point: within the backward tolerance, not bit-equal;
#   tile         another tile, atom, operand route or data placement: a kernel
#                of its own, held to the backward tolerance against the
#                oracle; no bitwise claim against another tile;
#   n/a          no non-default value is built.
NUMERICS = ("bitwise", "reorders_fp", "tile", "n/a")

# AttnBwdSpec fields that describe the problem class, not a tuning decision.
PROBLEM_FIELDS = frozenset(
    {"head_size", "dtype", "seq_mode", "dkv_mode", "stage_vec", "mask_class", "name"}
)
# Knob values whose kernels computed wrong outputs on a target (a failed
# reference check): {arch: {knob: values}}. Held out of every sweep level,
# refused by the legality verdicts (general, dense, unified), by the tuned
# tables' import check and by the dispatcher's candidates. Not the same as
# ``not_built`` (refused by the emission check), ``gated`` (not yet cleared by
# a device probe) or the exact resource refusals (``RESOURCE_REFUSED``: one
# failing spec, not a knob value). A knob name names its lever in every
# backward family that registers it (the dense body knobs and the shared
# runtime knobs are the general axes; the unified knobs are the general
# tile-step knobs). A default value is never known wrong. Empty on every
# target: each backward kernel found computing wrong values was fixed (the
# barrier fence) or is refused exactly by its validator.
BWD_KNOWN_WRONG_KNOBS: Mapping[str, Mapping[str, frozenset]] = MappingProxyType(
    {arch: MappingProxyType({}) for arch in DECLARED_ARCHS}
)


def known_wrong_values(arch: str, knob: str) -> frozenset:
    """The known-wrong values of ``knob`` on ``arch`` (empty if none)."""
    return frozenset(BWD_KNOWN_WRONG_KNOBS.get(arch, {}).get(knob, ()))


def known_wrong_findings(arch: str, knobs: Mapping[str, Any]) -> Tuple[str, ...]:
    """``"knob=value"`` for every known-wrong value of ``arch`` in ``knobs``
    (a knob mapping or a spec's fields), sorted."""
    out = []
    for knob, values in BWD_KNOWN_WRONG_KNOBS.get(arch, {}).items():
        if knob in knobs and knobs[knob] in values:
            out.append(f"{knob}={knobs[knob]!r}")
    return tuple(sorted(out))


# Codegen machine-scheduler strategies (None: the backend default).
CODEGEN_STRATEGIES: Tuple[Optional[str], ...] = (
    None,
    "max-ilp",
    "max-memory-clause",
    "iterative-ilp",
    "iterative-minreg",
    "iterative-maxocc",
)


# Smallest legal asm_g13 tail s_nop per MFMA target, over every atom the
# target has: the ISA's MFMA result wait states minus one
# (``asm_tail_nop_min``; the validator refuses a smaller count). The device
# probe (``test_attention_bwd_agpr_probe``: a negative control that fires
# below it, every count from the floor bitwise, and an ISA audit of whole
# asm_g13 kernels at the floor) found the hardware needs no more. Values
# below the floor are a hazard: they are not registered at all.
def _nop_floor(arch: str) -> int:
    floors = []
    for k in (16, 32):
        try:
            floors.append(asm_tail_nop_min(arch, "bf16", k))
        except ValueError:
            continue  # the atom is not on this target
    return max(floors)


ASM_TAIL_NOP_FLOOR: Mapping[str, int] = MappingProxyType(
    {a: _nop_floor(a) for a in ("gfx942", "gfx950")}
)
# Tail counts the production level sweeps (with each target's floor); the
# full level adds every other legal count. 15 is the shipped AGPR_ASM_TAIL_NOP
# (resolved to the default, None).
_PROD_NOPS = (7, 8, 11)


@dataclass(frozen=True)
class KnobAxis:
    """One backward tuning decision; ``choices[0]`` is the default.

    ``choices`` lists every registered value on any arch. ``arch_values``
    (when set) gives the values the sweep enumerates on one arch, in sweep
    order (it may drop values the arch cannot take; the policy ``None`` of a
    resolved field is not a sweep value there). ``requires`` is a cheap
    coupling test on a knob mapping (spec field names plus the slice's
    problem fields): False where a non-default value cannot be legal or
    changes nothing, so the sweep does not enumerate it; the kernel validator
    still decides every candidate.
    """

    name: str
    choices: Tuple[Any, ...]
    family: int
    stage: str
    kind: str = KIND_COMPILE
    level: str = "production"
    numerics: str = "bitwise"
    not_built: Tuple[Any, ...] = ()
    full_only: Tuple[Any, ...] = ()
    gated: Tuple[Any, ...] = ()
    derived: str = ""  # runtime values computed per request (description)
    plan_owned: bool = False  # set by the plan / the slice, not enumerated
    arch_values: Optional[Callable[[str], Tuple[Any, ...]]] = None
    requires: Optional[Callable[[Any, Mapping[str, Any]], bool]] = None
    note: str = ""

    @property
    def default(self) -> Any:
        return self.choices[0]

    def values(self, arch: str, level: str = "full") -> Tuple[Any, ...]:
        """Values the sweep enumerates on ``arch`` at ``level`` (sweep order).

        Not-built values stay in (they are recorded, never compiled); gated
        and known-wrong values (:data:`BWD_KNOWN_WRONG_KNOBS`) are dropped at
        every level, ``full_only`` ones below ``full``.
        """
        if level not in SWEEP_LEVELS:
            raise ValueError(f"sweep level {level!r} not in {SWEEP_LEVELS}")
        vals = self.arch_values(arch) if self.arch_values else self.choices
        wrong = known_wrong_values(arch, self.name)
        out = tuple(v for v in vals if v not in self.gated and v not in wrong)
        if level != "full":
            out = tuple(v for v in out if v not in self.full_only)
        return out

    def applies(self, value: Any, knobs: Mapping[str, Any]) -> bool:
        """Whether ``value`` is worth enumerating on top of ``knobs``."""
        if value == self.default or self.requires is None:
            return True
        return bool(self.requires(value, knobs))


# --------------------------------------------------------------------------- arch values
def _facts(arch: str):
    return attn_bwd_arch_facts(arch)


def _mfma(arch: str) -> bool:
    return _facts(arch).matrix_path == "mfma"


def _waves(arch):
    return _facts(arch).legal_waves


def _atoms(arch):
    return _facts(arch).legal_atoms


def _pt_routes(arch):
    return PT_ROUTES if _mfma(arch) else ("lds",)


def _residencies(arch):
    return RESIDENCIES if _mfma(arch) else ("lds",)


def _transpose(arch):
    return _facts(arch).legal_transpose_sources


def _global_paths(arch):
    return _facts(arch).legal_global_paths


def _rings(arch):
    return _facts(arch).legal_ring_depths


def _mfma_only(values, wmma_values):
    return lambda arch: values if _mfma(arch) else wmma_values


def _waves_per_eu(arch):
    if not _mfma(arch):
        return (None, 1, 2)
    # 1..4 on every CDNA target (the validator's 512 / n register budget
    # decides each hint)
    return (None,) + WAVES_PER_EU_VALUES


def _asm_tail_nops(arch):
    if not _mfma(arch):
        return (None,)
    top = min(AGPR_ASM_TAIL_NOP.values())  # the shipped count: the default
    return (None,) + tuple(range(ASM_TAIL_NOP_FLOOR[arch], top))


_NOP_CHOICES = (None,) + tuple(
    range(min(ASM_TAIL_NOP_FLOOR.values()), ASM_TAIL_NOP_MAX + 1)
)
_FULL_NOPS = tuple(
    n
    for n in _NOP_CHOICES[1:]
    if n not in _PROD_NOPS and n not in ASM_TAIL_NOP_FLOOR.values()
)


def _grids(rows_first: bool) -> Tuple[tuple, ...]:
    natural = tuple((1, w) if rows_first else (w, 1) for w in (1, 2, 4, 8))
    alt = tuple((2, w // 2) for w in (2, 4, 8))
    return tuple(dict.fromkeys(natural + alt))


# Every G0 / G2 and G1 / G3 grid of every wave count (the small-tile grids of
# _attention_bwd_tiles: transposed and, for G0 / G2, replicated grids).
# Every grid of the small-tile set that some registered tile can take: a G0 /
# G2 grid splits kM0 x kN0 (rows x cols into 16 x 16 atoms), a G1 / G3 grid
# kN0 x D (the largest D is 128). (8, 1) for G0 / G2, for example, would need
# kM0 = 128, above every registered block_m.
_G02_GRIDS = tuple(
    dict.fromkeys(
        g
        for w in (1, 2, 4, 8)
        for g in g02_grids(w)
        if 16 * g[0] <= max(BLOCK_M_VALUES) and 16 * g[1] <= max(BLOCK_N_VALUES)
    )
)
_G13_GRIDS = tuple(
    dict.fromkeys(
        g
        for w in (1, 2, 4, 8)
        for g in g13_grids(w)
        if 16 * g[0] <= max(BLOCK_N_VALUES) and 16 * g[1] <= 128
    )
)


# --------------------------------------------------------------------------- couplings
def _vgpr_staging(k: Mapping[str, Any]) -> bool:
    return k.get("global_path") in (None, "vgpr") and k.get("ring_depth") in (None, 1)


def _req_asm(_v, k) -> bool:
    return k.get("agpr_route") == "asm_g13"


def _req_route(_v, k) -> bool:
    a = k.get("agpr_alloc")
    return isinstance(a, tuple) and a[0] > 0


def _req_stage_order(_v, k) -> bool:
    return k.get("sched") not in STAGE_TABLE_SCHEDS


def _req_loop_sync(v, k) -> bool:
    if k.get("barrier_merge") == "on":
        # the merged ring keeps loop_sync's meaning for its merged and
        # prologue barriers; counted owns the same loads (excluded)
        return v != "counted"
    # the prefetch ring owns its (LDS-only) barriers
    if k.get("global_path") not in (None, "vgpr") or k.get("ring_depth") not in (
        None,
        1,
    ):
        return False
    if v != "counted":
        return True
    if _xt(k) or not _g4_step(k):
        return False  # the XT images and the dS key slices: synchronous body
    # counted owns the next-step loads: register staging without a ring or a
    # merged barrier, the default staging / stats order, not the grouped
    # head-outer form (a schedule value is legal; stage_table_atomics only
    # with the early issue, see _req_counted)
    return (
        _vgpr_staging(k)
        and k.get("qdo_stage", "split") == "split"
        and k.get("stats_load", "at_use") == "at_use"
        and not (k.get("step_order") == "head_outer" and k.get("edge_tiles") is True)
        and k.get("dq_mode", "atomic") == "atomic"
    )


def _req_counted(v, k) -> bool:
    # the late issue loads the next step after the deferred atomics of
    # stage_table_atomics (COUNTED_LATE_DEFERRED)
    if v == "late" and k.get("sched") == "stage_table_atomics":
        return False
    return k.get("loop_sync") == "counted"


def _req_dma(v, k) -> bool:
    # 16-byte DMA rows; head packing is refused with the DMA ring; the DMA
    # ring is no synchronous body (the XT images, the G4 key slices; its
    # stage table is built)
    if v == "dma" and (_xt(k) or not _g4_step(k)):
        return False
    return k.get("stage_vec", 8) == 8 and not k.get("head_pack")


def _xt(k) -> bool:
    return k.get("transpose_source") == "xt_lds"


def _req_xt(v, k) -> bool:
    # the XT images run on the synchronous body, and the fenced read-ahead of
    # dO^T would precede their publishing barrier
    if v != "xt_lds":
        return True
    return (
        _vgpr_staging(k)
        and k.get("barrier_merge") in (None, "off")
        and k.get("loop_sync") != "counted"
        and k.get("sched") not in FENCED_SCHEDS
    )


def _sync_sched(k) -> bool:
    """No schedule value: ``sched`` none and no ``setprio``."""
    return k.get("sched") in (None, "none") and not k.get("setprio")


def _req_sched(v, k) -> bool:
    # a schedule value runs on the bodies its stage table models (the
    # synchronous step, its counted waits, the prefetch ring, the
    # merged-barrier ring; with G4 in the step, or the split-dQ main kernel's
    # stages A..C); stage_table_atomics defers dQ atomics, so not with split
    # dQ (none to defer) nor with the counted waits' late issue
    if v in (None, "none"):
        return True
    late_counted = k.get("loop_sync") == "counted" and k.get("counted_issue") == "late"
    split = k.get("dq_mode") == "split"
    return (
        not (v == "stage_table_atomics" and (late_counted or split))
        and not (v in FENCED_SCHEDS and _xt(k))
        and _g4_step(k)
    )


def _req_setprio(_v, k) -> bool:
    # the priority raise sits around the stages of a stage table
    return k.get("sched") in STAGE_TABLE_SCHEDS and _req_sched(k["sched"], k)


def _req_ring(v, k) -> bool:
    # the prefetch / merged ring is no synchronous body (the XT images, the
    # G4 key slices; its stage table is built), and the grouped head-outer
    # form (head_outer + edge tiles) cannot carry it
    return v in (None, 1) or (
        not _xt(k)
        and _g4_step(k)
        and not (k.get("step_order") == "head_outer" and k.get("edge_tiles") is True)
    )


def _req_dq_mode(v, k) -> bool:
    # the split-dQ main kernel issues no G4: no merged barrier (it sits before
    # G4), no counted atomics, no deferred dQ atomics (stage_table_atomics);
    # the other schedule values run on its stages A..C
    if v != "split":
        return True
    return (
        k.get("barrier_merge") in (None, "off")
        and k.get("loop_sync") != "counted"
        and k.get("sched") != "stage_table_atomics"
        and _g4_step(k)
    )


def _req_vgpr_staging(_v, k) -> bool:
    # loop_sync = "counted" owns the staging and stats issue order
    return _vgpr_staging(k) and k.get("loop_sync") != "counted"


def _req_dkv_epilogue(_v, k) -> bool:
    # acc_in_lds reads the fp32 images: only the staged default there
    return not (k.get("acc_in_lds") and k.get("dkv_mode", "direct") == "direct")


def _req_store_vec(_v, k) -> bool:
    # the row width of the staged epilogue: direct dK / dV, or the slice sink's
    # fp32 rows (inert with the atomic sink); the slice sink reads LDS
    # accumulator images in rows whatever the epilogue
    slices = k.get("dkv_mode", "direct") == "atomic" and k.get("dkv_sink") == "slices"
    if slices and k.get("acc_in_lds"):
        return True
    if k.get("dkv_epilogue") == "direct":
        return False
    if k.get("dkv_mode", "direct") == "direct":
        return True
    return slices


def _req_atomic(_v, k) -> bool:
    return k.get("dkv_mode", "direct") == "atomic"


def _req_atomic_sink(_v, k) -> bool:
    # the atomic epilogue's row guard (inert with the slice sink)
    return _req_atomic(_v, k) and k.get("dkv_sink", "atomic") == "atomic"


def _req_dq_issue(_v, k) -> bool:
    # LDS accumulators always issue G4 per fragment (inert there)
    return not k.get("acc_in_lds")


def _req_noalias(v, k) -> bool:
    # every pointer only where main writes nothing but the workspace
    return v != "all" or k.get("dkv_mode", "direct") == "atomic"


def _req_thd_direct(_v, k) -> bool:
    return k.get("seq_mode") == "thd" and k.get("dkv_mode", "direct") == "direct"


def _req_edge_tiles(_v, k) -> bool:
    return k.get("edge_tiles") is True


def _req_barrier_merge(_v, k) -> bool:
    # the route the merged barrier's hazard proof covers (the validator's
    # couplings): register staging into the 2-slot ring, relabel P / dS,
    # register dK / dV, G4 last, the default staging order, atomic dQ, not
    # the grouped head-outer form (no ring there: its range loops spill SGPRs)
    return (
        k.get("global_path") in (None, "vgpr")
        and k.get("ring_depth") in (None, 2)
        and k.get("pt_route") in (None, "relabel")
        and not k.get("acc_in_lds")
        and k.get("stage_order", STAGE_ORDERS[0]) in BARRIER_MERGE_STAGE_ORDERS
        and k.get("qdo_stage", QDO_STAGES[0]) == QDO_STAGES[0]
        and k.get("stats_load", STATS_LOADS[0]) == STATS_LOADS[0]
        and k.get("dq_mode", "atomic") == "atomic"
        and k.get("loop_sync") != "counted"
        and not _xt(k)
        and _g4_step(k)
        and not (k.get("step_order") == "head_outer" and k.get("edge_tiles") is True)
    )


def _req_ring_issue(_v, k) -> bool:
    return k.get("barrier_merge") == "on"


def _req_bf16(_v, k) -> bool:
    return k.get("dtype") == "bf16"


def _req_head_major(_v, k) -> bool:
    return k.get("ws_layout", "head_major") == "head_major"


def _bf16_casts(arch):
    # where the generic cast is one packed conversion the guard-free rounding
    # can only add VALU: legal there, not enumerated; the softmax-gradient
    # levers are enumerated on the MFMA targets they were run on
    if arch in NATIVE_BF16_CVT_ARCHS or not _mfma(arch):
        return BF16_CASTS[:1]
    return BF16_CASTS


def _req_thd_natural(_v, k) -> bool:
    # the work list serves THD and fixes the CTA order (natural lb_order)
    return k.get("seq_mode") == "thd" and k.get("lb_order", "natural") == "natural"


def _req_g4_slicing(v, k) -> bool:
    # MFMA register accumulators with the block dQ issue; "slice_db_g3" reads
    # the first dS slice under G3, so G3 must issue right before G4
    if k.get("acc_in_lds") or k.get("dq_issue", "block") != "block":
        return False
    order = k.get("stage_order") or STAGE_ORDERS[0]
    if v == "slice_db_g3" and "g3g4" not in order:
        return False
    # the synchronous body with atomic dQ and no stage table; K^T from
    # registers under xt_lds
    return v == G4_SLICINGS[0] or (
        _vgpr_staging(k)
        and k.get("barrier_merge") in (None, "off")
        and k.get("loop_sync") != "counted"
        and k.get("dq_mode", "atomic") == "atomic"
        and _sync_sched(k)
        and not (_xt(k) and k.get("kt_source") == "lds")
    )


def _g4_step(k) -> bool:
    """No G4 dS key slicing (``g4_slicing`` step)."""
    return k.get("g4_slicing") in (None, G4_SLICINGS[0])


def _req_k4_sliced(_v, k) -> bool:
    # kK4 sizes the dS key slices; under "step" every value emits one kernel
    return k.get("g4_slicing", G4_SLICINGS[0]) != G4_SLICINGS[0]


def _cell_vecs(arch):
    return DKV_CELL_VECS if _mfma(arch) else (1,)


def _req_ws_bounds(_v, k) -> bool:
    # where it acts (INERT_KNOBS): the slice sink's stores, or the scalar
    # workspace stats loads (one row per load, unpacked steps, not the counted
    # waits' staged loads nor the DMA ring's copies); head-major workspace
    if k.get("ws_layout", "head_major") != "head_major":
        return False
    slices = k.get("dkv_mode") == "atomic" and k.get("dkv_sink") == "slices"
    scalar_stats = (
        k.get("stats_vec", 1) == 1
        and not k.get("head_pack")
        and k.get("loop_sync") != "counted"
        and k.get("global_path") in (None, "vgpr")
    )
    return slices or scalar_stats


def _req_cell_vec(_v, k) -> bool:
    if k.get("dkv_epilogue") != "direct" or k.get("acc_in_lds"):
        return False
    if k.get("dkv_mode", "direct") == "direct":
        return True
    return k.get("dkv_sink") == "slices"


def _req_narrow(_v, k) -> bool:
    return k.get("stage_vec", 8) == 1


def _req_slab_carry(_v, k) -> bool:
    # the synchronous step (no prefetch ring, no counted waits), not q_rotate;
    # inert in head-packed steps and the grouped head_outer + edge_tiles loops
    return (
        not k.get("q_rotate")
        and not k.get("head_pack")
        and k.get("global_path") in (None, "vgpr")
        and k.get("ring_depth") in (None, 1)
        and k.get("loop_sync") != "counted"
        and not (k.get("step_order") == "head_outer" and k.get("edge_tiles") is True)
    )


# --------------------------------------------------------------------------- registry
# Numerics class per knob (see NUMERICS); every other knob is "bitwise".
_NUMERICS_CLASS: Mapping[str, str] = {
    # the tile decomposition, atoms, operand routes and data placement
    **dict.fromkeys(
        (
            "waves",
            "block_m",
            "block_n",
            "warp_grid_g4",
            "kv_residency",
            "kt_source",
            "transpose_source",
            "global_path",
            "ring_depth",
            "atom_g02",
            "atom_g13",
            "atom_g4",
            "pt_route",
            "warp_grid_g02",
            "warp_grid_g13",
        ),
        "tile",
    ),
    # fp32 accumulate through LDS images; packed heads regroup the K rows
    "acc_in_lds": "reorders_fp",
    "head_pack": "reorders_fp",
    # dK / dV summed over (head group, q tile) in another order
    "step_order": "reorders_fp",
    # each q-tile range starts at kt mod its length: another dK / dV order
    "q_rotate": "reorders_fp",
    # P below 2^-126 flushed to 0 / P's argument rounded once (fma)
    "bwd_exp2": "reorders_fp",
    "stats_fma": "reorders_fp",
    # g_split > 1: dK / dV through fp32 atomics; the scale multiply moves
    "g_split": "reorders_fp",
    # slice sink: the CTA partials summed in slice order, not atomic order
    "dkv_sink": "reorders_fp",
    "scale_placement": "reorders_fp",
    # split dQ: dQ is one fixed-order register sum (dK / dV bit for bit)
    "dq_mode": "reorders_fp",
    # the dQ kernel's tile, atoms and transpose source; its stats prescale
    **dict.fromkeys(
        (
            "dq_waves",
            "dq_block_m",
            "dq_block_n",
            "dq_atom_g02",
            "dq_atom_g4",
            "dq_transpose_source",
        ),
        "tile",
    ),
    "dq_stats_prescale": "reorders_fp",
}


def _axis(name, choices, family, stage, **kw) -> KnobAxis:
    kw.setdefault("numerics", _NUMERICS_CLASS.get(name, "bitwise"))
    return KnobAxis(name, tuple(choices), family, stage, **kw)


_SPEC_AXES: Tuple[KnobAxis, ...] = (
    # 1 geometry (G)
    _axis("waves", (None, 1, 2, 4, 8), 1, "geometry", arch_values=_waves),
    _axis("block_m", (None,) + BLOCK_M_VALUES, 1, "geometry",
          arch_values=lambda a: BLOCK_M_VALUES),  # fmt: skip
    _axis("block_n", (None,) + BLOCK_N_VALUES, 1, "geometry",
          arch_values=lambda a: BLOCK_N_VALUES),  # fmt: skip
    _axis(
        "block_k4", (None,) + BLOCK_K4_VALUES, 1, "geometry",
        arch_values=lambda a: BLOCK_K4_VALUES, requires=_req_k4_sliced,
        note="the G4 dS key-slice width: inert under g4_slicing='step' "
        "(INERT_KNOBS), so the geometry stage pins it and the schedule stage "
        "enumerates it with the slicing values",
    ),  # fmt: skip
    _axis("warp_grid_g4", (None,) + _grids(True), 1, "geometry",
          note="(1, W) or (2, W / 2), enumerated with the wave count"),  # fmt: skip
    # 2 residency / registers (T, H, K)
    _axis("kv_residency", (None,) + RESIDENCIES, 2, "structure",
          arch_values=_residencies),  # fmt: skip
    _axis("kt_source", (None,) + RESIDENCIES, 2, "structure",
          arch_values=lambda a: RESIDENCIES if _mfma(a) else ("lds",)),  # fmt: skip
    _axis("s_in_agpr", (None, False, True), 2, "structure",
          arch_values=_mfma_only((False, True), (False,)),
          note="attr route: a declaration (DECLARATION_ONLY); asm_g13: G0 / G2 "
          "as AGPR-form asm clusters (S / dP in AGPRs)"),  # fmt: skip
    _axis("waves_per_eu", (None,) + WAVES_PER_EU_VALUES, 2, "schedule",
          arch_values=_waves_per_eu),  # fmt: skip
    _axis("agpr_alloc", AGPR_ALLOCS, 2, "schedule",
          arch_values=_mfma_only(AGPR_ALLOCS, (None,))),  # fmt: skip
    _axis(
        "agpr_route", AGPR_ROUTES, 2, "schedule",
        arch_values=_mfma_only(AGPR_ROUTES, (None,)), requires=_req_route,
        note="needs an AGPR reservation agpr_alloc=(n, n), n > 0",
    ),  # fmt: skip
    _axis("acc_in_lds", (None, False, True), 2, "codegen",
          arch_values=_mfma_only((False, True), (True,))),  # fmt: skip
    # 3 transpose / global path (T)
    _axis("transpose_source", (None,) + TRANSPOSE_SOURCES, 3, "structure",
          arch_values=_transpose, requires=_req_xt,
          note="xt_lds: XT images of Q^T / dO^T / K^T (synchronous body only: "
          "no ring, merged or counted barriers, no fenced sched)"),  # fmt: skip
    _axis("global_path", (None,) + GLOBAL_PATHS, 3, "structure",
          arch_values=_global_paths, requires=_req_dma,
          note="dma: gfx950 LDS DMA ring (stage_vec 8, no head packing; with "
          "tr_read the DMA-filled q / do take the slab XOR)"),  # fmt: skip
    _axis("ring_depth", (None, 1, 2, 3), 3, "structure", arch_values=_rings,
          requires=_req_ring,
          note="the Q / dO / stats prefetch ring: 2 on gfx942, 2 / 3 on "
          "gfx950 (register or DMA ring); 2 also under barrier_merge"),  # fmt: skip
    # 4 atoms / operand routes (T)
    _axis("atom_g02", (None,) + ATOMS, 4, "structure", arch_values=_atoms),
    _axis("atom_g13", (None,) + ATOMS, 4, "structure", arch_values=_atoms),
    _axis("atom_g4", (None,) + ATOMS, 4, "structure", arch_values=_atoms),
    _axis("pt_route", (None,) + PT_ROUTES, 4, "structure", arch_values=_pt_routes),
    _axis("warp_grid_g02", (None,) + _G02_GRIDS, 4, "structure",
          note="(1, W); with pt_route='lds' every grid of the small-tile "
          "set (transposed, (2, W / 2), replicated k < W)"),  # fmt: skip
    _axis("warp_grid_g13", (None,) + _G13_GRIDS, 4, "structure",
          note="(W, 1); (1, W), (2, W / 2) with pt_route='lds'"),  # fmt: skip
    # 5 schedule (H)
    _axis("sched", (None,) + SCHEDS, 5, "schedule", arch_values=lambda a: SCHEDS,
          requires=_req_sched,
          note="the fenced values read operands one stage ahead; "
          "stage_table_atomics defers the dQ atomics one q step (atomic dQ "
          "only); split dQ runs stages A..C"),  # fmt: skip
    _axis("setprio", (None, False, True), 5, "schedule",
          arch_values=lambda a: (False, True), requires=_req_setprio,
          note="stage-table scheds only"),  # fmt: skip
    _axis(
        "asm_tail_nop", _NOP_CHOICES, 5, "schedule", arch_values=_asm_tail_nops,
        full_only=_FULL_NOPS, requires=_req_asm,
        note="asm_g13 only; ASM_TAIL_NOP_FLOOR (the ISA floor per target) to 15",
    ),  # fmt: skip
    _axis("asm_cluster", ASM_CLUSTERS, 5, "schedule",
          arch_values=_mfma_only(ASM_CLUSTERS, ASM_CLUSTERS[:1]),
          requires=_req_asm, note="asm_g13 only"),  # fmt: skip
    _axis("asm_load_hoist", (False, True), 5, "schedule",
          arch_values=_mfma_only((False, True), (False,)), requires=_req_asm,
          note="asm_g13 only: the second G1 / G3 GEMM's LDS operand reads "
          "before the first one's asm cluster"),  # fmt: skip
    _axis("stage_order", STAGE_ORDERS, 5, "schedule", full_only=STAGE_ORDERS[3:],
          requires=_req_stage_order,
          note="excludes the stage-table scheds"),  # fmt: skip
    _axis("loop_sync", LOOP_SYNCS, 5, "schedule", requires=_req_loop_sync,
          arch_values=_mfma_only(LOOP_SYNCS, LOOP_SYNCS[:2]),
          note="lds_only / counted need global_path='vgpr'; counted (CDNA) "
          "loads the next step before the dQ atomics and waits vmcnt(N)"),  # fmt: skip
    _axis("counted_issue", COUNTED_ISSUES, 5, "schedule", requires=_req_counted,
          arch_values=_mfma_only(COUNTED_ISSUES, COUNTED_ISSUES[:1]),
          note="loop_sync='counted' only: next-step loads after the staging "
          "barrier (early) or before G4 (late)"),  # fmt: skip
    _axis("qdo_stage", QDO_STAGES, 5, "schedule", requires=_req_vgpr_staging,
          note="joint needs block_m * D / 8 below the CTA threads"),  # fmt: skip
    _axis("stats_load", STATS_LOADS, 5, "schedule", requires=_req_vgpr_staging),
    _axis("dq_issue", DQ_ISSUES, 5, "schedule", requires=_req_dq_issue,
          note="per-fragment G4 then its dQ atomics; 'fragment' is the block "
          "IR over one G4 fragment per wave (INERT_KNOBS)"),  # fmt: skip
    _axis(
        "barrier_merge", BARRIER_MERGES, 5, "schedule",
        arch_values=lambda a: (
            BARRIER_MERGES if a in BARRIER_MERGE_ARCHS else BARRIER_MERGES[:1]
        ),
        requires=_req_barrier_merge,
        note="one barrier per q step on a 2-slot Q / dO / stats ring: "
        "ring_depth 2 (None resolves to it), vgpr staging, relabel P / dS, "
        "register dK / dV, G4 last, the split / at_use staging, atomic dQ",
    ),  # fmt: skip
    _axis("ring_issue", RING_ISSUES, 5, "schedule", requires=_req_ring_issue,
          note="barrier_merge='on' only: the next step's loads at the step "
          "top (early) or before the slot stores (late)"),  # fmt: skip
    _axis(
        "g4_slicing", G4_SLICINGS, 5, "schedule",
        arch_values=_mfma_only(G4_SLICINGS, G4_SLICINGS[:1]),
        requires=_req_g4_slicing,
        note="G4 dS read in block_k4 key slices (enumerated with block_k4); "
        "MFMA register accumulators, dq_issue='block'; slice_db_g3 needs G3 "
        "right before G4",
    ),  # fmt: skip
    # 6 codegen / minor / swizzle (K, S)
    _axis("scheduler_strategy", CODEGEN_STRATEGIES, 6, "codegen"),
    _axis("edge_tiles", (None, False, True), 6, "codegen",
          arch_values=lambda a: (False, True)),  # fmt: skip
    _axis("ws_layout", WS_LAYOUTS, 6, "codegen"),
    _axis("step_order", STEP_ORDERS, 6, "codegen",
          note="head_outer reorders dK / dV over head groups x q tiles"),  # fmt: skip
    _axis("dkv_epilogue", DKV_EPILOGUES, 6, "codegen", requires=_req_dkv_epilogue),
    _axis("dkv_store_vec", DKV_STORE_VECS, 6, "codegen", requires=_req_store_vec),
    _axis("dkv_atomic_guard", DKV_ATOMIC_GUARDS, 6, "codegen",
          requires=_req_atomic_sink),  # fmt: skip
    _axis("ptr_noalias", PTR_NOALIAS, 6, "codegen", requires=_req_noalias,
          note="'all' needs dkv_mode='atomic'"),  # fmt: skip
    _axis("q_rotate", (False, True), 6, "codegen",
          note="the kv-tile CTAs of a head walk its q tiles out of lockstep"),  # fmt: skip
    _axis("dkv_slab_hoist", (False, True), 6, "codegen", requires=_req_thd_direct,
          note="THD direct: dK / dV slabs formed before the q loops (SGPRs)"),  # fmt: skip
    _axis("interior_clamp", (False, True), 6, "codegen", requires=_req_edge_tiles,
          note="edge_tiles only: the interior q steps clamp their rows too "
          "(an identity; it changes the loop's register allocation)"),  # fmt: skip
    # softmax-gradient VALU levers (kernels.common._attention_bwd_valu)
    _axis("bf16_cast", BF16_CASTS, 6, "codegen", arch_values=_bf16_casts,
          requires=_req_bf16,
          note="bf16 only: P / dS rounded to nearest even without the NaN "
          "guard (bit for bit on non-NaN values; a NaN input gives an "
          "unspecified result); enumerated on gfx942 (not where the cast "
          "is native; WMMA targets: legal, not enumerated)"),  # fmt: skip
    _axis("bwd_exp2", BWD_EXP2S, 6, "codegen",
          arch_values=_mfma_only(BWD_EXP2S, BWD_EXP2S[:1]),
          note="fast: a bare v_exp_f32 for P (P below 2^-126 flushed to 0); "
          "WMMA targets: legal, not enumerated"),  # fmt: skip
    _axis("stats_fma", (False, True), 6, "codegen",
          arch_values=_mfma_only((False, True), (False,)),
          note="P's argument S * scale_log2 - lse2 as one fma; WMMA targets: "
          "legal, not enumerated"),  # fmt: skip
    _axis("stats_vec", STATS_VECS, 6, "codegen",
          arch_values=_mfma_only(STATS_VECS, STATS_VECS[:1]),
          requires=_req_head_major,
          note="lse2 / Dsum as <n x f32> loads of consecutive C rows "
          "(head-major workspace, MFMA C maps; same values)"),  # fmt: skip
    # memory-access width / bounds levers (bit for bit: the same bytes move)
    _axis("ws_bounds", WS_BOUNDS, 6, "codegen", requires=_req_ws_bounds,
          note="buffer-descriptor bounds for the scalar workspace stats loads "
          "and the slice stores (head-major)"),  # fmt: skip
    _axis("dkv_cell_vec", DKV_CELL_VECS, 6, "codegen", arch_values=_cell_vecs,
          requires=_req_cell_vec,
          note="dkv_epilogue='direct': 4 columns per store after an in-quad "
          "transpose (MFMA C layouts)"),  # fmt: skip
    _axis("narrow_vec", NARROW_VECS, 6, "codegen", requires=_req_narrow,
          note="stage_vec = 1: the access width cap; the plan narrows it to the "
          "request's strides and base alignment"),  # fmt: skip
    _axis("slab_carry", (False, True), 6, "codegen", requires=_req_slab_carry,
          note="Q / dO slab offsets carried across q steps (i64 deltas instead "
          "of per-step head and row products); the synchronous step, not "
          "q_rotate"),  # fmt: skip
    _axis(
        "head_pack", (None, False, True), 6, "codegen", plan_owned=True,
        arch_values=lambda a: (False, True),
        note="the plan packs short-query groups; the head_pack slice forces it",
    ),  # fmt: skip
    _axis(
        "lds_swizzle", (None,) + SWIZZLES, 6, "swizzle",
        arch_values=lambda a: ("none", "pad8", "pad16", "pad32", "xor"),
        note="per LDS buffer 'buf=value,...'; values are the per-buffer choices "
        "(xor: the 128-byte line XOR; an XT image left out takes the "
        "conflict predictor's choice)",
    ),  # fmt: skip
    _axis("lds_order", LDS_ORDERS, 6, "swizzle"),
    # 7 runtime-following compile fields (R): the plan sets each from a
    # runtime value
    _axis(
        "static_scale", STATIC_SCALES, 7, "runtime", plan_owned=True,
        note="the plan compiles main for its scale_placement (the multipliers "
        "that are 1.0 there are not applied); 'runtime' is the one-binary "
        "kernel applying all three; never enumerated on its own",
    ),  # fmt: skip
    _axis("lb_interleave", (False, True), 7, "runtime", plan_owned=True,
          note="set by the plan for lb_order='interleave'"),  # fmt: skip
    _axis("lb_mirror", (False, True), 7, "runtime", plan_owned=True,
          note="set by the plan for lb_order='mirror' (two kv tiles per CTA)"),  # fmt: skip
    _axis("xcd_remap", (False, True), 7, "runtime", plan_owned=True,
          note="set by the plan when num_xcds and xcd_chunk give xcd_n > 0; "
          "with xcd_n = 0 the remap is the identity"),  # fmt: skip
    _axis("worklist", (False, True), 7, "runtime", plan_owned=True,
          note="set by the plan for use_worklist (THD only); with "
          "use_worklist = 0 the decode is the grid one"),  # fmt: skip
    # the atomic-mode dK / dV sink (it changes code only where dK / dV are
    # atomic: an atomic slice here; stage R also crosses it with g_split > 1)
    _axis(
        "dkv_sink", DKV_SINKS, 6, "codegen", requires=_req_atomic,
        not_built=tuple(DKV_SINK_NOT_BUILT),
        note="'slices': per-CTA fp32 slices, plain stores, fixed-order reduce "
        "(no atomics, deterministic); 'last_cta' is registered, not built "
        "(DKV_SINK_NOT_BUILT)",
    ),  # fmt: skip
    # 8 split dQ (T)
    _axis(
        "dq_mode", DQ_MODES, 8, "structure", requires=_req_dq_mode,
        not_built=() if SPLIT_DQ_AVAILABLE else ("split",),
        note="'split' on the targets of SPLIT_DQ_ARCHS (the validator refuses "
        "it elsewhere); the dQ kernel's own values are the dq_* axes",
    ),  # fmt: skip
)


def _dq_atoms(arch):
    return _facts(arch).legal_atoms if _mfma(arch) else (None,)


def _dq_transpose(arch):
    if not _mfma(arch):
        return (None,)
    return (None,) + tuple(
        t for t in _facts(arch).legal_transpose_sources if t in DQ_TRANSPOSE_SOURCES
    )


def _req_split(_v, k) -> bool:
    return k.get("dq_mode") == "split"


# 8 split dQ: the dQ kernel's values (the plan's DQ_KNOBS; dq_mode = "split"
# only). Stage T varies each field on its split points (the sweep's
# dq_values, one at a time at both levels); the XCD chunk is a stage R value
# of split candidates.
_DQ_AXES: Tuple[KnobAxis, ...] = tuple(
    _axis(name, choices, 8, stage, kind=KIND_DQ, requires=_req_split, **kw)
    for name, choices, stage, kw in (
        ("dq_waves", (None, 1, 2, 4, 8), "structure", {}),
        ("dq_block_m", (None,) + DQ_BLOCK_M_VALUES, "structure", {}),
        ("dq_block_n", (None,) + DQ_BLOCK_N_VALUES, "structure", {}),
        ("dq_atom_g02", (None, "16x16x16", "16x16x32"), "structure",
         {"arch_values": lambda a: (None,) + _dq_atoms(a)}),
        ("dq_atom_g4", (None, "16x16x16", "16x16x32"), "structure",
         {"arch_values": lambda a: (None,) + _dq_atoms(a),
          "note": "16x16x32 is K-permuted: lds_plain only"}),
        ("dq_transpose_source", (None,) + DQ_TRANSPOSE_SOURCES, "structure",
         {"arch_values": _dq_transpose}),
        ("dq_head_loop", DQ_HEAD_LOOPS, "structure",
         {"note": "group packs the G query heads of a K / V head (G | kM0)"}),
        ("dq_stats_prescale", (True, False), "structure", {}),
        ("dq_edge_tiles", (False, True), "structure", {}),
        ("dq_kv_pad", DQ_KV_PADS, "structure", {}),
        ("dq_kv_prefetch", (False, True), "structure", {}),
        ("dq_waves_per_eu", (None, 1, 2), "structure", {}),
        ("dq_agpr_alloc", DQ_AGPR_ALLOCS, "structure", {}),
        ("dq_scheduler_strategy", CODEGEN_STRATEGIES, "structure", {}),
        ("dq_sched", DQ_SCHEDS, "structure",
         {"note": "the key step's stage hints (G0' | G2' + softmax | G4'): "
          "iglp, fences, the stage table, with the wave priority; dQ bit for "
          "bit; enumerated with the other dQ values on split points"}),
        ("dq_xcd_chunk", (0, "gx", "2gx", "4gx", 64), "runtime",
         {"note": "needs the request's num_xcds; multiples of the dQ grid x"}),
    )
)  # fmt: skip

_RUNTIME_AXES: Tuple[KnobAxis, ...] = (
    _axis("g_split", (1,), 7, "runtime", kind=KIND_RUNTIME,
          derived="divisors of the head group G (g_split > 1 makes dK / dV "
          "atomic, a main recompile)"),  # fmt: skip
    _axis(
        "lb_order", tuple(LB_ORDERS), 7, "runtime", kind=KIND_RUNTIME,
        not_built=() if MIRROR_PAIRING_BUILT else ("mirror",),
    ),  # fmt: skip
    _axis(
        "xcd_chunk", (0, "gx", "2gx", "4gx", 64), 7, "runtime", kind=KIND_RUNTIME,
        note="needs the request's num_xcds; 'gx' etc. are multiples of the "
        "main grid x (the plan resolves them)",
    ),  # fmt: skip
    _axis("scale_placement", SCALE_PLACEMENTS, 7, "runtime", kind=KIND_RUNTIME),
    _axis(
        "use_worklist", (False, True), 7, "runtime", kind=KIND_RUNTIME,
        requires=_req_thd_natural,
        note="THD with lb_order natural: a prep role lists the live (sequence, "
        "kv tile) items heaviest first; dK / dV bit for bit in direct mode",
    ),  # fmt: skip
    _axis("prep_rows_per_cta", tuple(AUX_ROW_MULTIPLES), 7, "runtime",
          kind=KIND_RUNTIME, note="prep recompile only"),  # fmt: skip
    _axis("convert_rows_per_cta", tuple(AUX_ROW_MULTIPLES), 7, "runtime",
          kind=KIND_RUNTIME, note="convert recompile only"),  # fmt: skip
    _axis("convert_vec", CONVERT_VECS, 7, "runtime", kind=KIND_RUNTIME,
          note="convert recompile only"),  # fmt: skip
)

BWD_KNOB_AXES: Tuple[KnobAxis, ...] = _SPEC_AXES + _RUNTIME_AXES + _DQ_AXES
_BY_NAME: Mapping[str, KnobAxis] = {a.name: a for a in BWD_KNOB_AXES}


def _check_registry() -> None:
    names = [a.name for a in BWD_KNOB_AXES]
    if len(names) != len(set(names)):
        raise AssertionError("duplicate backward knob axis")
    spec = {f.name for f in fields(AttnBwdSpec)} - PROBLEM_FIELDS
    compile_names = {a.name for a in _SPEC_AXES}
    if compile_names != spec:
        raise AssertionError(
            f"AttnBwdSpec tuning fields without an axis: {sorted(spec - compile_names)}"
            f"; axes without a field: {sorted(compile_names - spec)}"
        )
    runtime = set(RUNTIME_KNOBS) | set(AUX_KNOBS)
    if {a.name for a in _RUNTIME_AXES} != runtime:
        raise AssertionError("runtime axes must be RUNTIME_KNOBS + AUX_KNOBS")
    if {a.name for a in _DQ_AXES} != set(DQ_KNOBS):
        raise AssertionError("dQ kernel axes must be the plan's DQ_KNOBS")
    for a in BWD_KNOB_AXES:
        if a.family not in LEVER_FAMILIES or a.stage not in FUNNEL_STAGES:
            raise AssertionError(f"{a.name}: bad family / stage")
        if a.level not in SWEEP_LEVELS or a.numerics not in NUMERICS:
            raise AssertionError(f"{a.name}: bad level / numerics")
        extra = set(a.not_built) | set(a.full_only) | set(a.gated)
        if not extra <= set(a.choices) or a.default in extra:
            raise AssertionError(f"{a.name}: marked values must be non-default choices")
    check_known_wrong(_BY_NAME)


def check_known_wrong(axes: Mapping[str, KnobAxis]) -> None:
    """Raise unless :data:`BWD_KNOWN_WRONG_KNOBS` names declared targets and
    no axis of ``axes`` (``{name: axis}``) has a known-wrong default."""
    for arch, knobs in BWD_KNOWN_WRONG_KNOBS.items():
        if arch not in DECLARED_ARCHS:
            raise AssertionError(f"known-wrong knobs of undeclared target {arch!r}")
        for knob, values in knobs.items():
            if not isinstance(values, frozenset):
                raise AssertionError(
                    f"{arch} {knob}: known-wrong values not a frozenset"
                )
            if knob in axes and axes[knob].default in values:
                raise AssertionError(f"{arch} {knob}: the default is known wrong")


_check_registry()


def backward_axis(name: str) -> KnobAxis:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError(f"no backward knob axis {name!r}") from None


def knob_family() -> Mapping[str, int]:
    """``{knob: lever family}`` for every registered axis."""
    return {a.name: a.family for a in BWD_KNOB_AXES}


def backward_tuning_axes(arch: str, level: str = "full") -> Tuple[KnobAxis, ...]:
    """The registered axes for ``arch`` (every axis; values via
    :meth:`KnobAxis.values`). ``level`` filters the axes by sweep level."""
    attn_bwd_arch_facts(arch)  # raises for a target the family does not know
    if level not in SWEEP_LEVELS:
        raise ValueError(f"sweep level {level!r} not in {SWEEP_LEVELS}")
    if level == "full":
        return BWD_KNOB_AXES
    return tuple(a for a in BWD_KNOB_AXES if a.level == "production")


@dataclass(frozen=True)
class BwdSpecLegality:
    """Verdict of :func:`backward_spec_legality`."""

    ok: bool
    resolved: Optional[AttnBwdSpec] = None
    error: str = ""  # the validator's message (illegal)
    not_built: Tuple[str, ...] = ()  # the emission check's findings
    known_wrong: Tuple[str, ...] = ()  # BWD_KNOWN_WRONG_KNOBS findings (illegal)

    @property
    def status(self) -> str:
        if self.ok:
            return "legal"
        return "not_built" if self.not_built else "illegal"


def backward_spec_legality(
    spec_or_knobs: AttnBwdSpec | Mapping[str, Any],
    arch: str,
    *,
    head_size: Optional[int] = None,
    **problem: Any,
) -> BwdSpecLegality:
    """Legality of one backward configuration on ``arch``: the family
    validator (``validate_attn_bwd_spec``), the known-wrong values
    (:data:`BWD_KNOWN_WRONG_KNOBS`, illegal), then the emission check
    (``_not_built``). ``spec_or_knobs`` is a spec or a knob mapping (spec
    field names) completed by ``head_size`` and the problem fields."""
    if isinstance(spec_or_knobs, AttnBwdSpec):
        spec = spec_or_knobs
    else:
        if head_size is None:
            raise ValueError("a knob mapping needs head_size")
        try:
            spec = AttnBwdSpec(head_size=head_size, **problem, **dict(spec_or_knobs))
        except (TypeError, ValueError) as ex:
            return BwdSpecLegality(False, None, str(ex))
    try:
        resolved = validate_attn_bwd_spec(spec, arch)
    except ValueError as ex:
        return BwdSpecLegality(False, None, str(ex))
    wrong = known_wrong_findings(
        arch, {f.name: getattr(resolved, f.name) for f in fields(resolved)}
    )
    if wrong:
        return BwdSpecLegality(
            False, resolved, f"known wrong on {arch}: {', '.join(wrong)}", (), wrong
        )
    nb = tuple(_not_built(resolved))
    return BwdSpecLegality(not nb, resolved, "", nb)


def sweepable(axis: KnobAxis) -> bool:
    """Whether a funnel stage enumerates ``axis`` (plan-owned axes and the
    not-yet-effective knobs are pinned)."""
    return not axis.plan_owned and axis.name not in NOT_YET_EFFECTIVE_KNOBS


def axes_of_stage(stage: str, names: Sequence[str] = ()) -> Tuple[KnobAxis, ...]:
    """The sweepable axes a funnel stage owns, minus ``names``."""
    if stage not in FUNNEL_STAGES:
        raise ValueError(f"unknown funnel stage {stage!r}")
    return tuple(
        a
        for a in BWD_KNOB_AXES
        if a.stage == stage and sweepable(a) and a.name not in names
    )
