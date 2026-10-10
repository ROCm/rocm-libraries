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

Legality is the kernel's own: :func:`backward_spec_legality` resolves a spec
with ``validate_attn_bwd_spec`` and then applies ``_not_built``; there is no
second model here. This module holds the registry only; the request,
candidate and ranker parts of a backward dispatcher are not part of it.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from kernels.common._attention_bwd_caps import AGPR_ASM_TAIL_NOP, asm_tail_nop_min
from kernels.common.attention_bwd import (
    AGPR_ALLOCS,
    AGPR_ROUTES,
    ASM_CLUSTERS,
    ASM_TAIL_NOP_MAX,
    ATOMS,
    BLOCK_K4_VALUES,
    BLOCK_M_VALUES,
    BLOCK_N_VALUES,
    DKV_ATOMIC_GUARDS,
    DKV_EPILOGUES,
    DKV_STORE_VECS,
    DQ_ISSUES,
    DQ_MODES,
    GLOBAL_PATHS,
    LDS_ORDERS,
    LOOP_SYNCS,
    NOT_YET_EFFECTIVE_KNOBS,
    PT_ROUTES,
    PTR_NOALIAS,
    QDO_STAGES,
    RESIDENCIES,
    SCHEDS,
    SPLIT_DQ_AVAILABLE,
    STAGE_ORDERS,
    STAGE_TABLE_SCHEDS,
    STATIC_SCALES,
    STATS_LOADS,
    STEP_ORDERS,
    SWIZZLES,
    TRANSPOSE_SOURCES,
    WS_LAYOUTS,
    AttnBwdSpec,
    _not_built,
    attn_bwd_arch_facts,
    validate_attn_bwd_spec,
)
from kernels.common.attention_bwd_aux import CONVERT_VECS
from kernels.common.attention_bwd_plan import (
    AUX_KNOBS,
    AUX_ROW_MULTIPLES,
    LB_ORDERS,
    MIRROR_PAIRING_BUILT,
    RUNTIME_KNOBS,
    SCALE_PLACEMENTS,
)

__all__ = [
    "ASM_TAIL_NOP_FLOOR",
    "BWD_KNOB_AXES",
    "CODEGEN_STRATEGIES",
    "FUNNEL_STAGES",
    "KIND_COMPILE",
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
    "knob_family",
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
        values are dropped at every level, ``full_only`` ones below ``full``.
        """
        if level not in SWEEP_LEVELS:
            raise ValueError(f"sweep level {level!r} not in {SWEEP_LEVELS}")
        vals = self.arch_values(arch) if self.arch_values else self.choices
        out = tuple(v for v in vals if v not in self.gated)
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
    return (None, 1, 2) + ((3,) if arch == "gfx942" else ())


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


def _req_loop_sync(_v, k) -> bool:
    return k.get("global_path") in (None, "vgpr")


def _req_vgpr_staging(_v, k) -> bool:
    return _vgpr_staging(k)


def _req_dkv_epilogue(_v, k) -> bool:
    # acc_in_lds reads the fp32 images: only the staged default there
    return not (k.get("acc_in_lds") and k.get("dkv_mode", "direct") == "direct")


def _req_store_vec(_v, k) -> bool:
    # the row width of the staged direct epilogue (inert in atomic mode)
    return k.get("dkv_mode", "direct") == "direct" and k.get("dkv_epilogue") != "direct"


def _req_atomic(_v, k) -> bool:
    return k.get("dkv_mode", "direct") == "atomic"


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
    # g_split > 1: dK / dV through fp32 atomics; the scale multiply moves
    "g_split": "reorders_fp",
    "scale_placement": "reorders_fp",
    "dq_mode": "n/a",
    "use_worklist": "n/a",
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
        arch_values=lambda a: BLOCK_K4_VALUES,
        note="not yet effective (NOT_YET_EFFECTIVE_KNOBS): pinned per geometry",
    ),  # fmt: skip
    _axis("warp_grid_g4", (None,) + _grids(True), 1, "geometry",
          note="(1, W) or (2, W / 2), enumerated with the wave count"),  # fmt: skip
    # 2 residency / registers (T, H, K)
    _axis("kv_residency", (None,) + RESIDENCIES, 2, "structure",
          arch_values=_residencies),  # fmt: skip
    _axis("kt_source", (None,) + RESIDENCIES, 2, "structure",
          arch_values=lambda a: RESIDENCIES if _mfma(a) else ("lds",)),  # fmt: skip
    _axis("s_in_agpr", (None, False, True), 2, "structure",
          arch_values=_mfma_only((False, True), (False,))),  # fmt: skip
    _axis("waves_per_eu", (None, 1, 2, 3), 2, "schedule", arch_values=_waves_per_eu),
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
          arch_values=_transpose, not_built=("xt_lds",)),  # fmt: skip
    _axis("global_path", (None,) + GLOBAL_PATHS, 3, "structure",
          arch_values=_global_paths, not_built=("dma",)),  # fmt: skip
    _axis("ring_depth", (None, 1, 2, 3), 3, "structure", arch_values=_rings,
          not_built=(2, 3)),  # fmt: skip
    # 4 atoms / operand routes (T)
    _axis("atom_g02", (None,) + ATOMS, 4, "structure", arch_values=_atoms),
    _axis("atom_g13", (None,) + ATOMS, 4, "structure", arch_values=_atoms),
    _axis("atom_g4", (None,) + ATOMS, 4, "structure", arch_values=_atoms),
    _axis("pt_route", (None,) + PT_ROUTES, 4, "structure", arch_values=_pt_routes),
    _axis("warp_grid_g02", (None,) + _grids(True), 4, "structure",
          note="(1, W); (2, W / 2) with pt_route='lds'"),  # fmt: skip
    _axis("warp_grid_g13", (None,) + _grids(False), 4, "structure",
          note="(W, 1); (2, W / 2) with pt_route='lds'"),  # fmt: skip
    # 5 schedule (H)
    _axis("sched", (None,) + SCHEDS, 5, "schedule", arch_values=lambda a: SCHEDS,
          not_built=SCHEDS[1:]),  # fmt: skip
    _axis("setprio", (None, False, True), 5, "schedule",
          arch_values=lambda a: (False, True), not_built=(True,)),  # fmt: skip
    _axis(
        "asm_tail_nop", _NOP_CHOICES, 5, "schedule", arch_values=_asm_tail_nops,
        full_only=_FULL_NOPS, requires=_req_asm,
        note="asm_g13 only; ASM_TAIL_NOP_FLOOR (the ISA floor per target) to 15",
    ),  # fmt: skip
    _axis("asm_cluster", ASM_CLUSTERS, 5, "schedule",
          arch_values=_mfma_only(ASM_CLUSTERS, ASM_CLUSTERS[:1]),
          requires=_req_asm, note="asm_g13 only"),  # fmt: skip
    _axis("stage_order", STAGE_ORDERS, 5, "schedule", full_only=STAGE_ORDERS[3:],
          requires=_req_stage_order,
          note="excludes the stage-table scheds"),  # fmt: skip
    _axis("loop_sync", LOOP_SYNCS, 5, "schedule", requires=_req_loop_sync,
          note="lds_only needs global_path='vgpr'"),  # fmt: skip
    _axis("qdo_stage", QDO_STAGES, 5, "schedule", requires=_req_vgpr_staging,
          note="joint needs block_m * D / 8 below the CTA threads"),  # fmt: skip
    _axis("stats_load", STATS_LOADS, 5, "schedule", requires=_req_vgpr_staging),
    _axis("dq_issue", DQ_ISSUES, 5, "schedule", requires=_req_dq_issue,
          note="per-fragment G4 then its dQ atomics; 'fragment' is the block "
          "IR over one G4 fragment per wave (INERT_KNOBS)"),  # fmt: skip
    # 6 codegen / minor / swizzle (K, S)
    _axis("scheduler_strategy", CODEGEN_STRATEGIES, 6, "codegen"),
    _axis("edge_tiles", (None, False, True), 6, "codegen",
          arch_values=lambda a: (False, True)),  # fmt: skip
    _axis("ws_layout", WS_LAYOUTS, 6, "codegen"),
    _axis("step_order", STEP_ORDERS, 6, "codegen",
          note="head_outer reorders dK / dV over head groups x q tiles"),  # fmt: skip
    _axis("dkv_epilogue", DKV_EPILOGUES, 6, "codegen", requires=_req_dkv_epilogue),
    _axis("dkv_store_vec", DKV_STORE_VECS, 6, "codegen", requires=_req_store_vec),
    _axis("dkv_atomic_guard", DKV_ATOMIC_GUARDS, 6, "codegen", requires=_req_atomic),
    _axis("ptr_noalias", PTR_NOALIAS, 6, "codegen", requires=_req_noalias,
          note="'all' needs dkv_mode='atomic'"),  # fmt: skip
    _axis("q_rotate", (False, True), 6, "codegen",
          note="the kv-tile CTAs of a head walk its q tiles out of lockstep"),  # fmt: skip
    _axis("dkv_slab_hoist", (False, True), 6, "codegen", requires=_req_thd_direct,
          note="THD direct: dK / dV slabs formed before the q loops (SGPRs)"),  # fmt: skip
    _axis("interior_clamp", (False, True), 6, "codegen", requires=_req_edge_tiles,
          note="edge_tiles only: the interior q steps clamp their rows too "
          "(an identity; it changes the loop's register allocation)"),  # fmt: skip
    _axis(
        "head_pack", (None, False, True), 6, "codegen", plan_owned=True,
        arch_values=lambda a: (False, True),
        note="the plan packs short-query groups; the head_pack slice forces it",
    ),  # fmt: skip
    _axis(
        "lds_swizzle", (None,) + SWIZZLES, 6, "swizzle", not_built=("xor",),
        arch_values=lambda a: ("none", "pad8", "pad16", "pad32"),
        note="per LDS buffer 'buf=value,...'; values are the per-buffer choices",
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
    _axis("xcd_remap", (False, True), 7, "runtime", plan_owned=True,
          note="set by the plan when num_xcds and xcd_chunk give xcd_n > 0; "
          "with xcd_n = 0 the remap is the identity"),  # fmt: skip
    # 8 split dQ (T)
    _axis(
        "dq_mode", DQ_MODES, 8, "structure",
        not_built=() if SPLIT_DQ_AVAILABLE else ("split",),
    ),  # fmt: skip
)

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
    _axis("use_worklist", (False, True), 7, "runtime", kind=KIND_RUNTIME,
          not_built=(True,)),  # fmt: skip
    _axis("prep_rows_per_cta", tuple(AUX_ROW_MULTIPLES), 7, "runtime",
          kind=KIND_RUNTIME, note="prep recompile only"),  # fmt: skip
    _axis("convert_rows_per_cta", tuple(AUX_ROW_MULTIPLES), 7, "runtime",
          kind=KIND_RUNTIME, note="convert recompile only"),  # fmt: skip
    _axis("convert_vec", CONVERT_VECS, 7, "runtime", kind=KIND_RUNTIME,
          note="convert recompile only"),  # fmt: skip
)

BWD_KNOB_AXES: Tuple[KnobAxis, ...] = _SPEC_AXES + _RUNTIME_AXES
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
    for a in BWD_KNOB_AXES:
        if a.family not in LEVER_FAMILIES or a.stage not in FUNNEL_STAGES:
            raise AssertionError(f"{a.name}: bad family / stage")
        if a.level not in SWEEP_LEVELS or a.numerics not in NUMERICS:
            raise AssertionError(f"{a.name}: bad level / numerics")
        extra = set(a.not_built) | set(a.full_only) | set(a.gated)
        if not extra <= set(a.choices) or a.default in extra:
            raise AssertionError(f"{a.name}: marked values must be non-default choices")


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
    validator (``validate_attn_bwd_spec``), then the emission check
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
