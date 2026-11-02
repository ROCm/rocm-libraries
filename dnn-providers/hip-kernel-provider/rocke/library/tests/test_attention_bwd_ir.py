# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""IR, ABI and resource tests of the layout-general backward main kernel.

The shipped instance set covers every built base key (batched, direct,
``stage_vec = 8``, no mask or a runtime band; fp16 / bf16; d32 / d64 / d128)
on gfx942, gfx950, gfx1151 and gfx1201 in every correctness configuration of
the arch (the arch default geometry and the one-wave 16 x 16 configuration; on
RDNA the one-wave default and a two-wave tile), each as the plain instance,
the head-packed instance (short query spans of GQA groups) and the
``edge_tiles`` instance (three sequential q loops: edge, interior, edge; the
interior loop has no per-cell selects), plus the atomic dK / dV instances
(``dkv_mode = "atomic"``: ``h_k != h_v`` or ``g_split > 1``), plain and
head-packed, for both mask classes; the narrow-access instances
(``stage_vec = 1``, the arch's narrow tile) of every plain, head-packed and
atomic variant; every one of these variants again with ragged THD
sequences (``seq_mode = "thd"``); and the barrier-flavour / step-order
instances (``loop_sync = "lds_only"``, ``step_order = "head_outer"`` with one
q loop or with a head-group loop around the edge / interior / edge loops).
On gfx950 the 16x16x32 G1 / G3 atom (``atom_g13``, K-permuted G1 / G3 B) is
checked on plain, edge-tile, head-packed, atomic, narrow and THD variants for
lowering, structure, LDS barriers, matrix-op counts and the transpose read.

* lowering at every LLVM flavor; kernel signature equal to the
  ``rocke.attn_bwd.v3`` main list, no ``noalias``, inputs ``readonly``;
  unique kernel names;
* structure: K / V global reads only inside the liveness branch, no branch
  around a loop atomic, ``readfirstlane``'d q-loop trip counts with the
  accumulators initialised after them, no matrix op inside a branch, no
  q loop nested in another loop but the head-group loop of ``head_outer``
  with edge tiles;
* barrier flavour: under ``loop_sync = "lds_only"`` exactly the in-loop
  barriers are LDS-only (the kernel is otherwise its full twin op for op);
  step order: the head-outer wrap (q tile inner) or the head-group loop;
* atomic dK / dV: the epilogue adds ``2 * kN0 * D / (W * wave)`` fp32 values
  per lane inside the liveness branch, after the q loops; the CTA owns
  ``G / g_split`` query heads, and K / V are rebased on ``unit * G / gk`` and
  ``unit * G / gv``;
* grouped heads (direct mode): ``G * n_qt`` steps per q loop (runtime
  ``G``), head groups inner wrapping at ``G``, query head
  ``unit * G + group`` with Q / dO rebased on it in 64 bits in every step;
* padding: the per-batch SEQ_LEN counts are read once per CTA under the
  runtime ``has_len`` flag at element ``batch * len_stride`` and clamped to
  ``[0, S_max]``;
* THD: the offset tables are read once per CTA, outside any branch or loop
  (entries ``b`` and ``b + 1``, two 32-bit words each, the word index
  selected by the runtime ``off64``), and converted to tokens in 64 bits
  (``off * mult / div``); batched instances never read them;
* LDS barriers: every pair of LDS accesses that can race across waves
  (overlapping views, read / write or writes through aliasing views) is
  separated by a CTA-wide barrier on every path (zero-trip, and the wrap
  from one q step into the next); the q step ends with a barrier;
* counts per q step (per step variant): matrix ops per GEMM equal the
  stage-table model, dQ atomics per lane ``kM0 * D / (W * wave)`` at agent
  scope;
* workspace accesses (dQ, stats and, in atomic mode, dK / dV) rebased in
  i64; user-tensor accesses 16 bytes wide (``stage_vec = 8``) or one element
  wide with ``align <= 2`` (``stage_vec = 1``), the LDS, matrix-op and
  workspace accesses of the two being equal;
* no 128-bit transpose read; the gfx950 default reads its transposed
  operands with ``ds_read_tr16_b64``;
* stage order (``stage_order``, every value, on the plain, head-packed,
  edge-tile, atomic and narrow THD variants and with the LDS P / dS route,
  MFMA LDS accumulators, the ``agpr_fence`` / ``asm_g13`` routes, ``asm_g13``
  with ``asm_load_hoist`` and, on gfx950, the 16x16x32 G1 / G3 atom): the
  matrix ops issue as G0, G2, then G1 / G3 / G4 in the declared order with the
  model's counts (each chain traced to its carried dK / dV, its LDS image or
  a fresh zero), the dQ atomics follow the first G4 op and precede the next
  GEMM, the dS barrier sits right before G4 (relabel) or the P / dS barrier
  before all three (LDS route), no cross-wave LDS race; a non-default order
  salts the name and changes the IR but not the step's op census; the
  explicit default is the default kernel;
* the resource gate on every shipped instance (comgr + disassembly; zero
  scratch and spills, register budgets, LDS equal to the phase plan, no
  AGPR copies inside the q loop).
"""

from __future__ import annotations

import dataclasses
import re
from functools import cache

import pytest
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower

from kernels.common._attention_bwd_body import (
    DEFAULT_STAGE_ORDER,
    STAGE_ORDERS,
    BwdTileGeometry,
)
from kernels.common.attention_bwd import (
    INERT_KNOBS,
    AttnBwdSpec,
    _not_built,
    attn_bwd_arch_facts,
    attn_bwd_params,
    build_attn_bwd_main,
    lds_plan,
    validate_attn_bwd_spec,
)

from ._attention_bwd_test_utils import distinct_configs, kernel_signature
from .sdpa.bwd_kernel_cases import config_knobs

ARCHS = ("gfx942", "gfx950", "gfx1151", "gfx1201")
FLAVORS = ("llvm20", "llvm22", "llvm23")
DTYPES = ("fp16", "bf16")
HEADS = (32, 64, 128)
INPUTS = {"Q", "K", "V", "dO", "WS_LSE2", "WS_DSUM", "SEQ_Q", "SEQ_KV", "OFF_Q",
          "OFF_KV", "WORKLIST"}  # fmt: skip


# Spec knobs per instance variant (mask class, head packing, edge tiles).
VARIANTS = {
    "none": dict(mask_class="none"),
    "band": dict(mask_class="band"),
    "none_pack": dict(mask_class="none", head_pack=True),
    "band_pack": dict(mask_class="band", head_pack=True),
    "none_edge": dict(mask_class="none", edge_tiles=True),
    "band_edge": dict(mask_class="band", edge_tiles=True),
    "none_atomic": dict(mask_class="none", dkv_mode="atomic"),
    "band_atomic": dict(mask_class="band", dkv_mode="atomic"),
    "none_atomic_pack": dict(mask_class="none", dkv_mode="atomic", head_pack=True),
    "band_atomic_pack": dict(mask_class="band", dkv_mode="atomic", head_pack=True),
}
# The narrow-access instances (stage_vec = 1) of every base key, plain and
# head-packed (edge tiles are not a shipped narrow configuration).
VARIANTS.update(
    {
        f"sv1_{k}": dict(v, stage_vec=1)
        for k, v in list(VARIANTS.items())
        if "edge" not in k
    }
)
# Every variant again with ragged THD sequences.
VARIANTS.update(
    {f"thd_{k}": dict(v, seq_mode="thd") for k, v in list(VARIANTS.items())}
)
# Barrier flavour (LDS-only in-loop barriers) and step order (head groups
# outer: one flat loop, or a head-group loop around the edge / interior / edge
# loops), alone and combined, on direct, head-packed, atomic and THD variants.
SYNC_ORDER_VARIANTS = {
    "band_ldssync": dict(mask_class="band", loop_sync="lds_only"),
    "thd_none_atomic_pack_ldssync": dict(
        mask_class="none", dkv_mode="atomic", head_pack=True, seq_mode="thd",
        loop_sync="lds_only",
    ),  # fmt: skip
    "band_hout": dict(mask_class="band", step_order="head_outer"),
    "band_pack_hout": dict(mask_class="band", head_pack=True, step_order="head_outer"),
    "band_edge_hout": dict(mask_class="band", edge_tiles=True, step_order="head_outer"),
    "band_edge_atomic_pack_hout_ldssync": dict(
        mask_class="band", edge_tiles=True, dkv_mode="atomic", head_pack=True,
        step_order="head_outer", loop_sync="lds_only",
    ),  # fmt: skip
}
VARIANTS.update(SYNC_ORDER_VARIANTS)
# The THD work-list decode (worklist = True; runtime use_worklist) on a direct
# band, an atomic head-packed and an edge-tile instance.
WORKLIST_VARIANTS = {
    "thd_band_wl": dict(mask_class="band", seq_mode="thd", worklist=True),
    "thd_none_atomic_pack_wl": dict(
        mask_class="none", dkv_mode="atomic", head_pack=True, seq_mode="thd",
        worklist=True,
    ),  # fmt: skip
    "thd_band_edge_wl": dict(
        mask_class="band", edge_tiles=True, seq_mode="thd", worklist=True
    ),
}
VARIANTS.update(WORKLIST_VARIANTS)
DIRECT_VARIANTS = tuple(v for v in VARIANTS if "atomic" not in v)
ATOMIC_VARIANTS = tuple(v for v in VARIANTS if "atomic" in v)

INSTANCES = [
    (arch, cfg, d, dt, var)
    for arch in ARCHS
    for cfg in distinct_configs(arch)
    for d in HEADS
    for dt in DTYPES
    for var in VARIANTS
]

# Staging knobs on top of a variant: ``"<variant>+<knob>"`` instances.
# Every row-major buffer with the line XOR swizzle (absent buffers ignored).
ROW_XOR = (
    "dkv_out=xor,do=xor,ds=xor,k_res=xor,k_stage=xor,pt=xor,q=xor,v_res=xor,"
    "v_stage=xor"
)
KNOB_VALUES = {
    "lf": dict(qdo_stage="loads_first"),
    "joint": dict(qdo_stage="joint"),
    "top": dict(stats_load="step_top"),
    "jtop": dict(qdo_stage="joint", stats_load="step_top"),
    # layout levers: the XT transposes and the xor swizzle
    "xt": dict(transpose_source="xt_lds"),
    "xor": dict(lds_swizzle=ROW_XOR),
}
_KNOB_BASES = ("band", "band_pack", "band_edge", "thd_band_atomic", "sv1_band",
               "thd_sv1_none_atomic_pack")  # fmt: skip
# The 16x16x32 G1 / G3 atom (gfx950; K-permuted G1 / G3 B, on the transpose
# read the stacked read plan): ``"<variant>@wide"`` instances, block_m 32 where
# the configuration's tile has 16 rows. It changes the G1 / G3 matrix-op count
# (half the 16x16x16 twin's), so these are not staging-knob instances.
WIDE_SUFFIX = "@wide"
_WIDE_BASES = ("none", "band", "band_pack", "none_edge", "band_edge",
               "band_atomic_pack", "thd_band_pack", "thd_none_atomic", "sv1_band",
               "sv1_band_atomic_pack", "thd_sv1_band")  # fmt: skip
# Layout-lever instances whose production-comgr compile spills arch VGPRs:
# the validator refuses them (``LAYOUT_SPILLING_COMBOS``), so the IR suite does
# not enumerate them; test_validator_refuses_exactly_the_spilling_layouts pins
# the refusal to exactly these rows.
LAYOUT_REFUSED_ROWS = frozenset(
    (arch, "default", d, "fp16", var)
    for arch, d, var in (
        ("gfx942", 128, "band_edge+xor"),
        ("gfx950", 64, "band_pack+xor"),
        ("gfx1151", 64, "band+xt"),
        ("gfx1151", 64, "band_edge+xt"),
        ("gfx1151", 64, "thd_band_atomic+xt"),
        ("gfx1151", 64, "sv1_band+xt"),
        ("gfx1151", 64, "band_pack+xor"),
        ("gfx1151", 64, "band_edge+xor"),
        ("gfx1201", 64, "band_pack+xor"),
        ("gfx1201", 128, "band_pack+xor"),
        ("gfx1201", 128, "band_edge+xor"),
        ("gfx1201", 128, "thd_sv1_none_atomic_pack+xor"),
    )
)


def _ids(row):
    return "-".join(map(str, row))


def spec_of(arch, cfg, d, dt, var="none") -> AttnBwdSpec:
    if var.endswith(WIDE_SUFFIX):
        # "<variant>@wide": the 16x16x32 G1 / G3 atom (see WIDE_INSTANCES)
        spec = spec_of(arch, cfg, d, dt, var[: -len(WIDE_SUFFIX)])
        tile = validate_attn_bwd_spec(spec, arch)
        extra = {} if tile.block_m >= 32 else {"block_m": 32}
        return dataclasses.replace(spec, atom_g13="16x16x32", **extra)
    if "|" in var:
        # "base|knobs|order": a stage-order instance (see ORDER_ROWS)
        base, knobs, order = var.split("|")
        kw = dict(config_knobs(cfg, arch), **VARIANTS[base])
        kw.update(ORDER_KNOBS[knobs], stage_order=order)
        return AttnBwdSpec(head_size=d, dtype=dt, **kw)
    # "base+knob+...": a staging / stats knob instance (see KNOB_INSTANCES)
    base, *knobs = var.split("+")
    extra = {k: v for name in knobs for k, v in KNOB_VALUES[name].items()}
    return AttnBwdSpec(
        head_size=d, dtype=dt, **VARIANTS[base], **config_knobs(cfg, arch), **extra
    )


@cache
def kernel_of(arch, cfg, d, dt, var="none"):
    return build_attn_bwd_main(spec_of(arch, cfg, d, dt, var), arch=arch)


@cache
def ir_of(arch, cfg, d, dt, var="none", flavor="llvm22"):
    return lower(kernel_of(arch, cfg, d, dt, var), arch=arch, llvm_flavor=flavor)


def resolved_of(arch, cfg, d, dt, var="none"):
    return validate_attn_bwd_spec(spec_of(arch, cfg, d, dt, var), arch)


def _legal(arch, cfg, d, dt, var):
    try:
        validate_attn_bwd_spec(spec_of(arch, cfg, d, dt, var), arch)
    except ValueError:
        return False
    return True


def _resource_refused(row) -> bool:
    """The validator refuses the row as a recorded resource-gate failure
    (``RESOURCE_REFUSED``; pinned by test_attention_bwd_refusals.py)."""
    from kernels.common._attention_bwd_refusals import resource_refusal

    return resource_refusal(spec_of(*row).resolved(row[0]), row[0]) is not None


# Refused instances are not enumerated: no kernel is built for them.
INSTANCES = [r for r in INSTANCES if not _resource_refused(r)]


# Every arch's default configuration (d32 / d64 / d128, fp16), each knob value
# where it is legal ("joint" needs kM0 * D / 8 < threads: the CDNA wide tiles).
KNOB_INSTANCES = [
    row
    for arch in ARCHS
    for d in HEADS
    for base in _KNOB_BASES
    for knob in KNOB_VALUES
    for row in [(arch, "default", d, "fp16", f"{base}+{knob}")]
    if _legal(*row)
]


def _built(arch, cfg, d, dt, var) -> bool:
    try:
        r = validate_attn_bwd_spec(spec_of(arch, cfg, d, dt, var), arch)
    except ValueError:
        return False
    return not _not_built(r)


# Every gfx950 configuration where the wide G1 / G3 atom is legal and built.
WIDE_INSTANCES = [
    row
    for cfg in distinct_configs("gfx950")
    for d in HEADS
    for dt in DTYPES
    for base in _WIDE_BASES
    for row in [("gfx950", cfg, d, dt, f"{base}{WIDE_SUFFIX}")]
    if _built(*row)
]


# ---------------------------------------------------------------------------
# kernel-structure walkers
# ---------------------------------------------------------------------------


def _walk(ops, ancestors=()):
    for op in ops:
        yield op, ancestors
        for r in op.regions:
            yield from _walk(r.ops, ancestors + (op,))


def _mma_loops(kernel):
    """``(loop, ancestors)`` of every ``scf.for`` that holds a matrix op."""
    return [
        (op, anc)
        for op, anc in _walk(kernel.body.ops)
        if op.name == "scf.for"
        and any(o.name == "tile.mma" for o, _ in _walk(op.regions[0].ops))
    ]


def q_loops(kernel):
    """The q loops in program order: one, or edge / interior / edge
    (``edge_tiles``). A q loop is nested in no loop but the head-group loop of
    ``step_order = "head_outer"`` with edge tiles (:func:`group_loops`)."""
    loops = _mma_loops(kernel)
    inner = [(lp, anc) for lp, anc in loops if not _holds_mma_loop(lp, loops)]
    for _lp, anc in inner:
        outer = [a for a in anc if a.name == "scf.for"]
        assert len(outer) <= 1 and all(_is_group_loop(a) for a in outer)
    return [lp for lp, _anc in inner]


def _holds_mma_loop(loop, loops) -> bool:
    ids = {id(lp) for lp, _anc in loops}
    return any(id(o) in ids for o, _ in _walk(loop.regions[0].ops))


def _is_group_loop(loop) -> bool:
    """The head-group loop: its iter args are the dK / dV accumulators and
    the head group (named ``...grp``; under ``stage_table_atomics`` the
    deferred dQ partial follows it), and it holds q loops."""
    names = [m["name"] for m in loop.attrs["iter_args"]]
    grp = [n for n in names if not n.lstrip("%").startswith("bwd_gdq")]
    return bool(grp) and grp[-1].endswith("grp") and "git" in loop.attrs["iv"]


def group_loops(kernel):
    """Head-group loops around q loops (``head_outer`` with edge tiles)."""
    loops = _mma_loops(kernel)
    return [lp for lp, _anc in loops if _holds_mma_loop(lp, loops)]


def q_loop(kernel):
    """The first q loop."""
    return q_loops(kernel)[0]


def step_bodies(kernel):
    """Op lists of the q-step bodies (one per q loop)."""
    return [lp.regions[0].ops for lp in q_loops(kernel)]


def _last_op(ops):
    """Last op of a body before its yield."""
    return [op for op in ops if op.name != "scf.yield"][-1]


def _derived(kernel, names):
    """Value names derived from the given params through pointer adds."""
    out = {f"%{n}" for n in names}
    for op, _ in _walk(kernel.body.ops):
        if op.name == "tile.global_ptr_add" and op.operands[0].name in out:
            out.add(op.results[0].name)
        # an opaque SGPR copy of a pointer (the THD dK / dV slabs) is the pointer
        elif (
            op.name == "tile.inline_asm"
            and op.attrs.get("constraints") == "=s,0"
            and op.operands[0].name in out
        ):
            out.add(op.results[0].name)
    return out


# ---------------------------------------------------------------------------
# lowering, signature, names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row", INSTANCES + KNOB_INSTANCES + WIDE_INSTANCES, ids=_ids)
def test_instance_builds_and_lowers(row):
    ir = ir_of(*row)
    assert "alloca" not in ir and "addrspace(5)" not in ir
    assert "define amdgpu_kernel void @" in ir


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("arch", ARCHS)
def test_lowers_at_every_flavor(arch, flavor):
    for cfg in distinct_configs(arch):
        for var in (
            "none", "band_pack", "band_edge", "band_atomic_pack", "thd_band_pack",
            "thd_none_atomic", "sv1_band_atomic_pack", "thd_sv1_band",
        ):  # fmt: skip
            ir = ir_of(arch, cfg, 64, "bf16", var, flavor)
            assert "define amdgpu_kernel void @" in ir


@pytest.mark.parametrize("arch", ARCHS)
def test_kernarg_order_matches_params(arch):
    variants = (
        "none", "band_pack", "band_atomic", "thd_band", "thd_none_atomic",
        "sv1_band", "thd_sv1_none_atomic_pack",
    )  # fmt: skip
    for cfg, var in ((c, v) for c in distinct_configs(arch) for v in variants):
        sig = kernel_signature(ir_of(arch, cfg, 128, "fp16", var))
        assert [(n, k) for n, k, _ in sig] == list(attn_bwd_params("main"))
        for name, kind, text in sig:
            assert "noalias" not in text
            if kind == "Q":
                assert ("readonly" in text) == (name in INPUTS), text


@pytest.mark.parametrize("arch", ARCHS)
def test_xcd_remap_reads_the_remap_kernargs_only_when_built(arch):
    """``xcd_remap = False`` (every shipped instance) never reads ``xcd_n`` /
    ``xcd_chunk``; ``xcd_remap = True`` reads both, keeps the v3 signature and
    gets its own kernel name."""
    cfg = distinct_configs(arch)[0]
    for var in ("none", "band_atomic"):
        plain_spec = spec_of(arch, cfg, 64, "bf16", var)
        spec = dataclasses.replace(plain_spec, xcd_remap=True)
        plain = ir_of(arch, cfg, 64, "bf16", var)
        remap = lower(
            build_attn_bwd_main(spec, arch=arch), arch=arch, llvm_flavor="llvm22"
        )
        assert [(n, k) for n, k, _ in kernel_signature(remap)] == list(
            attn_bwd_params("main")
        )
        for name in ("xcd_n", "xcd_chunk"):
            uses = re.compile(rf"%{name}\b")
            assert len(uses.findall(plain)) == 1, (name, "plain")  # signature only
            assert len(uses.findall(remap)) > 1, (name, "remap")
        assert spec.kernel_name("main") != plain_spec.kernel_name("main")


@pytest.mark.parametrize("arch", ARCHS)
def test_worklist_reads_the_list_only_when_built(arch):
    """``worklist = False`` (every other instance) never reads ``WORKLIST`` /
    ``use_worklist``; ``worklist = True`` reads both (the list as i32 loads
    from the pointer, under the flag's 0/1-trip guard), keeps the v3
    signature and gets its own kernel name; the decode is the only change
    before the sequence decode (the q-loop body and the epilogue are the
    plain kernel's)."""
    cfg = distinct_configs(arch)[0]
    for plain_var, wl_var in (
        ("thd_band", "thd_band_wl"),
        ("thd_none_atomic_pack", "thd_none_atomic_pack_wl"),
        ("thd_band_edge", "thd_band_edge_wl"),
    ):
        plain = ir_of(arch, cfg, 64, "bf16", plain_var)
        wl = ir_of(arch, cfg, 64, "bf16", wl_var)
        assert [(n, k) for n, k, _ in kernel_signature(wl)] == list(
            attn_bwd_params("main")
        )
        for name in ("WORKLIST", "use_worklist"):
            uses = re.compile(rf"%{name}\b")
            assert len(uses.findall(plain)) == 1, (name, "plain")  # signature only
            assert len(uses.findall(wl)) > 1, (name, "worklist")
        loads = re.findall(
            r"getelementptr inbounds i32, ptr addrspace\(1\) %WORKLIST", wl
        )
        assert len(loads) == 3  # the count and the item's two words
        assert spec_of(arch, cfg, 64, "bf16", wl_var).kernel_name("main") != spec_of(
            arch, cfg, 64, "bf16", plain_var
        ).kernel_name("main")
        # the work-list kernel is the plain kernel's op sequence with one
        # contiguous run inserted: the decode (item index, guard loop)
        a = _op_names(kernel_of(arch, cfg, 64, "bf16", plain_var))
        b = _op_names(kernel_of(arch, cfg, 64, "bf16", wl_var))
        n = len(a)
        pre = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), n)
        suf = next((i for i, (x, y) in enumerate(zip(a[::-1], b[::-1])) if x != y), n)
        suf = min(suf, n - pre)  # an op equal on both sides of the run counts once
        assert pre + suf == n and len(b) > n, (pre, suf, n, len(b))
        inserted = b[pre : len(b) - suf]
        assert inserted.count("scf.for") == 1
        assert inserted.count("memref.global_load_typed") == 3
        assert not any(n.startswith("tile.") for n in inserted)


def _op_names(kernel):
    """Op names of ``kernel`` in emission order (regions inlined)."""
    out = []

    def walk(ops):
        for op in ops:
            out.append(op.name)
            for r in op.regions:
                walk(r.ops)

    walk(kernel.body.ops)
    return out
    names = {}
    for row in INSTANCES:
        arch = row[0]
        name = kernel_of(*row).name
        assert name == resolved_of(*row).kernel_name("main")
        assert names.setdefault((arch, name), row) == row, (name, row)


# ---------------------------------------------------------------------------
# build-time facts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ARCHS)
def test_lane_map_facts_and_relabel(arch):
    for cfg in distinct_configs(arch):
        geom = BwdTileGeometry.from_spec(resolved_of(arch, cfg, 64, "fp16"), arch)
        if geom.pt_route == "relabel":
            assert geom.relabel_plan.kind == "direct"
        assert geom.reader == (
            "tr16" if geom.transpose_source == "tr_read" else "plain"
        )


def test_illegal_geometry_raises_before_ir():
    r = resolved_of("gfx942", "default", 64, "fp16")
    unknown = dataclasses.replace(r)
    object.__setattr__(unknown, "transpose_source", "no_such_reader")
    with pytest.raises(ValueError, match="no operand reader"):
        BwdTileGeometry.from_spec(unknown, "gfx942")
    # every registered transpose source of the arch has a reader
    for ts in attn_bwd_arch_facts("gfx942").legal_transpose_sources:
        BwdTileGeometry.from_spec(dataclasses.replace(r, transpose_source=ts), "gfx942")
    # a transpose read on an arch without it
    with pytest.raises(ValueError):
        BwdTileGeometry.from_spec(
            r.__class__(**{**r.__dict__, "transpose_source": "tr_read"}), "gfx942"
        )


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row", INSTANCES + KNOB_INSTANCES + WIDE_INSTANCES, ids=_ids)
def test_structure(row):
    kernel = kernel_of(*row)
    r = resolved_of(*row)
    loops = q_loops(kernel)
    assert len(loops) == (3 if r.edge_tiles else 1)
    groups = group_loops(kernel)
    assert len(groups) == int(r.step_order == "head_outer" and r.edge_tiles)
    kv = _derived(kernel, ("K", "V"))
    kv_loads = 0
    guards = None  # branch conditions shared by every K / V load (by identity)
    for op, anc in _walk(kernel.body.ops):
        if op.name.startswith("memref.global_load") and op.operands[0].name in kv:
            # K / V are read only inside the liveness branch, never in a loop
            conds = {
                id(a.operands[0]): a.operands[0] for a in anc if a.name == "scf.if"
            }
            assert conds, op.name
            if guards is None:
                guards = conds
            else:
                guards = {k: v for k, v in guards.items() if k in conds}
            assert not any(lp in anc for lp in loops)
            kv_loads += 1
        if op.name == "tile.mma":
            assert not any(a.name in ("scf.if", "scf.if_else") for a in anc)
    assert kv_loads > 0
    # the shared guard is the CTA liveness test k0 < len_kv: a strict compare
    # of two different values (not a constant or a tautology)
    live = [
        c for c in (guards or {}).values()
        if c.op is not None and c.op.name == "arith.cmp"
        and c.op.attrs.get("pred") == "lt"
        and c.op.operands[0] is not c.op.operands[1]
    ]  # fmt: skip
    assert live, "K / V staging is not guarded by the kv-tile liveness test"
    order = {id(op): i for i, (op, _) in enumerate(_walk(kernel.body.ops))}
    first = (groups or loops)[0]
    first_upper = first.operands[1]
    for loop in groups:
        upper = loop.operands[1]
        assert upper.op is not None and upper.op.name == "tile.readfirstlane"
        assert loop.attrs.get("elide_trailing_barrier") is False
    for loop in loops:
        in_loop = list(_walk(loop.regions[0].ops))
        atomics = [
            (op, anc)
            for op, anc in in_loop
            if op.name == "memref.global_atomic_add_f32"
        ]
        assert atomics
        for _op, anc in atomics:
            assert not any(a.name in ("scf.if", "scf.if_else") for a in anc)
        # trip counts are readfirstlane'd; the trailing barrier is kept
        upper = loop.operands[1]
        assert upper.op is not None and upper.op.name == "tile.readfirstlane"
        assert loop.attrs.get("elide_trailing_barrier") is False
    # accumulators start after the exit test (iter args: dK / dV accumulators,
    # then the q tile and / or head group indices)
    metas = first.attrs["iter_args"]
    inits = first.operands[3:]
    accs = [v for v, m in zip(inits, metas) if re.search(r"d[kv]\d+$", m["name"])]
    assert len(accs) == len(inits) - (1 if groups else 2)
    for init in accs:
        assert order[id(init.op)] > order[id(first_upper.op)]
    if len(loops) == 3:
        # the interior loop skips the per-cell selects of the edge loops
        def selects(lp):
            return sum(op.name == "arith.select" for op, _ in _walk(lp.regions[0].ops))

        assert selects(loops[1]) < selects(loops[0]) == selects(loops[2])


@pytest.mark.parametrize(
    "row", [r for r in INSTANCES if r[4] in ATOMIC_VARIANTS], ids=_ids
)
def test_atomic_epilogue(row):
    """Atomic dK / dV: ``2 * kN0 * D / (W * wave)`` fp32 atomics per lane after
    the q loops, inside the liveness branch, into ``WS_DK`` / ``WS_DV``; no
    user dK / dV access; the direct instance of the same row has no epilogue
    atomic."""
    arch = row[0]
    r = resolved_of(*row)
    kernel = kernel_of(*row)
    wave = attn_bwd_arch_facts(arch).wave_size
    loops = q_loops(kernel)
    in_q = {id(op) for lp in loops for op, _ in _walk(lp.regions[0].ops)}
    ws = _derived(kernel, ("WS_DK", "WS_DV"))
    epi = []
    for op, anc in _walk(kernel.body.ops):
        if op.name != "memref.global_atomic_add_f32" or id(op) in in_q:
            continue
        assert op.operands[0].name in ws
        assert any(a.name == "scf.if" for a in anc)
        epi.append(op)
    assert len(epi) == 2 * r.block_n * r.head_size // (r.waves * wave)
    users = _derived(kernel, ("dK", "dV"))
    assert not any(
        op.operands and op.operands[0].name in users
        for op, _ in _walk(kernel.body.ops)
        if op.name.startswith("memref.global_")
    )
    direct = row[:4] + (row[4].replace("_atomic", ""),)
    if direct[4] in VARIANTS:
        dkernel = kernel_of(*direct)
    else:  # the direct twin of a variant without its own entry
        dspec = dataclasses.replace(spec_of(*row), dkv_mode="direct")
        dkernel = build_attn_bwd_main(dspec, arch=arch)
    in_q = {id(op) for lp in q_loops(dkernel) for op, _ in _walk(lp.regions[0].ops)}
    assert not [
        op
        for op, _ in _walk(dkernel.body.ops)
        if op.name == "memref.global_atomic_add_f32" and id(op) not in in_q
    ]


@pytest.mark.parametrize("arch", ARCHS)
def test_atomic_mode_heads_and_split(arch):
    """Atomic mode: the CTA's head unit and split come from ``y / g_split`` and
    ``y % g_split``; the q loop runs ``(G / g_split) * n_qt`` steps from query
    head ``unit * G + split * (G / g_split)``; K is rebased on ``unit * G /
    gk`` and V on ``unit * G / gv``."""
    kernel = kernel_of(arch, distinct_configs(arch)[0], 64, "fp16", "band_atomic")
    ops = [op for op, _ in _walk(kernel.body.ops)]

    def operands(op):
        return [o.name for o in op.operands]

    divs = [op for op in ops if op.name == "arith.div"]
    assert any(operands(op) == ["%G", "%g_split"] for op in divs)
    assert any(op.operands[1].name == "%gk" for op in divs)
    assert any(op.operands[1].name == "%gv" for op in divs)
    assert any(op.name == "arith.mod" and operands(op)[1] == "%g_split" for op in ops)
    for loop in q_loops(kernel):
        mul = loop.operands[1].op.operands[0].op
        per = [o.op for o in mul.operands if o.op is not None]
        assert any(
            p.name == "arith.div" and operands(p) == ["%G", "%g_split"] for p in per
        )


@pytest.mark.parametrize(
    "row",
    [r for r in INSTANCES if r[4] in ("none", "band", "band_edge")],
    ids=_ids,
)
def test_group_loop_sums_the_query_heads_of_a_kv_head(row):
    """Direct mode: the q loop runs ``G * n_qt`` steps of the CTA's kv head
    (runtime ``G``); head groups are inner and wrap at ``G``; each step's query
    head is ``unit * G + group``, and Q / dO are rebased on it in 64 bits
    inside the step."""
    kernel = kernel_of(*row)
    for loop in q_loops(kernel):
        trip = loop.operands[1].op
        assert trip.name == "tile.readfirstlane"
        mul = trip.operands[0].op
        assert mul.name == "arith.mul" and "%G" in [o.name for o in mul.operands]
        grp = loop.operands[-1]
        assert grp.op is not None and grp.op.name == "arith.constant"
        body = loop.regions[0].ops
        carried = {o.name for op in body for o in op.operands}
        grp_iv = [n for n in carried if n.endswith("grp")]
        assert len(grp_iv) == 1, grp_iv
        # the head group wraps at G
        wraps = [
            op for op in body if op.name == "arith.cmp" and op.operands[1].name == "%G"
        ]
        assert wraps
        # query head = unit * G + group (unit = the kv head of the CTA)
        heads = [
            op
            for op in body
            if op.name == "arith.add" and op.operands[1].name == grp_iv[0]
        ]
        assert heads
        base = heads[0].operands[0].op
        assert base.name == "arith.mul" and base.operands[1].name == "%G"
        head = heads[0].results[0].name
        # Q and dO slabs: the head is widened before it meets the head stride
        widened = {
            op.results[0].name
            for op in body
            if op.name == "arith.zext" and op.operands[0].name == head
        }
        assert widened
        rebased = {
            p.operands[0].name
            for p in body
            if p.name == "tile.global_ptr_add" and p.operands[0].name in ("%Q", "%dO")
        }
        assert rebased == {"%Q", "%dO"}


@pytest.mark.parametrize("arch", ARCHS)
def test_sequence_lengths_read_once_under_the_runtime_flag(arch):
    """Padding: SEQ_Q and SEQ_KV are each read once per CTA, outside the q
    loops, at element ``batch * len_stride``, inside a 0/1-trip loop on
    ``has_len`` (a null pointer is never dereferenced when the flag is 0),
    and the count is clamped to ``[0, S_max]`` before any use."""
    limits = {"%SEQ_Q": "%S_q_max", "%SEQ_KV": "%S_kv_max"}
    for cfg in distinct_configs(arch):
        for var in ("band", "band_pack", "none_atomic", "thd_band", "thd_none_atomic"):
            kernel = kernel_of(arch, cfg, 64, "fp16", var)
            in_q = {
                id(op) for lp in q_loops(kernel) for op, _ in _walk(lp.regions[0].ops)
            }
            ops = list(_walk(kernel.body.ops))
            found = {}
            for op, anc in ops:
                if not op.name.startswith("memref.global_load"):
                    continue
                ptr = op.operands[0].name
                if ptr not in limits:
                    continue
                found[ptr] = found.get(ptr, 0) + 1
                assert id(op) not in in_q
                guard = anc[-1]
                assert guard.name == "scf.for"
                trip = guard.operands[1].op
                assert trip.name == "arith.select"
                cond = trip.operands[0].op
                assert cond.name == "arith.cmp"
                assert "%has_len" in [o.name for o in cond.operands]
                idx = op.operands[1].op
                assert idx.name == "arith.mul"
                assert "%len_stride" in [o.name for o in idx.operands]
                users = [u for u, _ in ops if op.results[0] in u.operands]
                assert [u.name for u in users] == ["arith.smax"]
                clamp = [u for u, _ in ops if users[0].results[0] in u.operands]
                assert [u.name for u in clamp] == ["arith.smin"]
                assert clamp[0].operands[1].name == limits[ptr]
            assert found == {"%SEQ_Q": 1, "%SEQ_KV": 1}, (cfg, var, found)


def _producers(ops, value):
    """Ops reachable backwards from ``value`` through operands."""
    by_result = {id(r): op for op, _ in ops for r in op.results}
    seen, stack, out = set(), [value], []
    while stack:
        v = stack.pop()
        op = by_result.get(id(v))
        if op is None or id(op) in seen:
            continue
        seen.add(id(op))
        out.append(op)
        stack.extend(op.operands)
    return out


@pytest.mark.parametrize("arch", ARCHS)
def test_ragged_offsets_read_once_and_decoded_in_64_bits(arch):
    """THD: OFF_Q and OFF_KV are each read as four 32-bit words per CTA
    (entries ``b`` and ``b + 1``; the word index is selected by ``off64``,
    so an int32 table is never read past entry ``B``), outside every branch
    and loop; the token arithmetic (``off * mult / div``) is 64-bit; the
    batched instance of the same variant reads neither table."""
    mults = {"%OFF_Q": ("%q_mult", "%q_div"), "%OFF_KV": ("%kv_mult", "%kv_div")}
    for cfg in distinct_configs(arch):
        for var in ("band", "band_pack", "none_atomic_pack"):
            batched = kernel_of(arch, cfg, 64, "fp16", var)
            names = {o.name for op, _ in _walk(batched.body.ops) for o in op.operands}
            assert not names & set(mults), (cfg, var)
            kernel = kernel_of(arch, cfg, 64, "fp16", f"thd_{var}")
            ops = list(_walk(kernel.body.ops))
            found = {}
            for op, anc in ops:
                if not op.name.startswith("memref.global_load"):
                    continue
                ptr = op.operands[0].name
                if ptr not in mults:
                    continue
                found[ptr] = found.get(ptr, 0) + 1
                assert not anc, (ptr, [a.name for a in anc])
                assert op.results[0].type.name == "i32"
                idx = _producers(ops, op.operands[1])
                assert "%off64" in {o.name for p in idx for o in p.operands}
            assert found == {"%OFF_Q": 4, "%OFF_KV": 4}, (cfg, var, found)
            for ptr, (mult, div) in mults.items():
                users = [
                    op
                    for op, _ in ops
                    if any(o.name in (mult, div) for o in op.operands)
                ]
                assert users, (ptr, mult)
                for u in users:
                    assert u.name in ("arith.sext", "arith.zext"), u.name
                    assert u.results[0].type.name == "i64"
                # the token divisions take the widened divisor (i64)
                wide = {id(u.results[0]) for u in users if u.operands[0].name == div}
                divs = [
                    op
                    for op, _ in ops
                    if op.name == "arith.div" and id(op.operands[1]) in wide
                ]
                assert len(divs) == 2, (ptr, len(divs))
                assert all(op.results[0].type.name == "i64" for op in divs)


# ---------------------------------------------------------------------------
# LDS barriers: a cross-wave hazard model over the program order
# ---------------------------------------------------------------------------


_LDS_PREFIXES = ("tile.smem_", "tile.ds_read")
# CTA-wide barriers that drain LDS: the full one and the LDS-only one (no VMEM
# drain); both order LDS accesses across waves, which is all this model checks.
_BARRIERS = ("tile.sync", "tile.sync_lds_only")


def _is_lds(op) -> bool:
    return op.name.startswith(_LDS_PREFIXES) and op.name != "tile.smem_alloc"


@cache
def traced_kernel(arch, cfg, d, dt, var="none"):
    """Build the kernel and tag every LDS access with its view's span.

    Returns ``(kernel, tags)``; ``tags[id(op)] = (lo, hi, is_store, private)``
    where ``[lo, hi)`` is the accessed view in allocation elements and
    ``private`` marks the in-loop accumulator read-modify-write (each lane
    reads exactly the cells it wrote: no cross-wave hazard by construction).
    """
    return traced_spec(spec_of(arch, cfg, d, dt, var), arch)


def traced_spec(spec, arch, *, structural: bool = False):
    """:func:`traced_kernel` of any spec. Every LDS op a view method emits is
    tagged (a vector split at a padded row pitch emits several).

    ``structural``: build without the main kernel's recorded resource
    refusals (``main_kernel_refusals=False``), for a test of the emission
    structure of a spec the production compiler refuses on resources; every
    static rule still applies."""
    from kernels.common import _attention_bwd_body as body
    from kernels.common.attention_bwd import _emit_main
    from kernels.common._attention_bwd_frag import LdsView

    tags, state = {}, {"private": False}
    orig = {n: getattr(LdsView, n) for n in ("load", "store", "tr16_load")}
    orig_acc = {n: getattr(body.BwdTileStep, n) for n in ("_load_lds_acc", "_store_lds_acc")}  # fmt: skip

    def tagged(name):
        def f(self, b, *args, **kw):
            ops = b._region_stack[-1].ops
            n0 = len(ops)
            out = orig[name](self, b, *args, **kw)
            new = [op for op in ops[n0:] if _is_lds(op)]
            assert new, [op.name for op in ops[n0:]]
            hi = self.base + self.rows * self.pitch * self.scale
            for op in new:
                tags[id(op)] = (self.base, hi, name == "store", state["private"])
            return out

        return f

    def private(name):
        def f(self, *args, **kw):
            state["private"] = True
            try:
                return orig_acc[name](self, *args, **kw)
            finally:
                state["private"] = False

        return f

    try:
        for n in orig:
            setattr(LdsView, n, tagged(n))
        for n in orig_acc:
            setattr(body.BwdTileStep, n, private(n))
        if structural:
            kernel = _emit_main(spec, arch, main_kernel_refusals=False)
        else:
            kernel = build_attn_bwd_main(spec, arch=arch)
    finally:
        for n, f in orig.items():
            setattr(LdsView, n, f)
        for n, f in orig_acc.items():
            setattr(body.BwdTileStep, n, f)
    return kernel, tags


def _events(ops, tags, *, trips, in_branch=False):
    """Program-order events: ``("sync", None)`` or ``("lds", op)``.

    Every loop body is laid out ``trips`` times (0: the zero-trip path, 2:
    the wrap from one iteration into the next). Branch bodies are always
    included; a barrier inside a branch is not counted (not every wave
    reaches it).
    """
    out = []
    for op in ops:
        if op.name in _BARRIERS:
            if not in_branch:
                out.append(("sync", op))
        elif _is_lds(op):
            assert id(op) in tags, "an LDS access outside the LdsView methods"
            out.append(("lds", op))
        elif op.name == "scf.for":
            for _ in range(trips):
                out += _events(
                    op.regions[0].ops, tags, trips=trips, in_branch=in_branch
                )
        else:
            for r in op.regions:
                out += _events(r.ops, tags, trips=trips, in_branch=True)
    return out


def lds_hazards(kernel, tags):
    """Pairs of LDS accesses that can race across waves.

    Two accesses race when they overlap, no CTA-wide barrier lies between
    them on some path (zero-trip, or two iterations of each loop), they are
    not both the lane-private accumulator update, and they are a read and a
    write, or writes through two different (aliasing) views. Writes through
    one view are not a race: each thread owns the same cells of a view at
    every write site (copy chunks by ``tid``, C tiles by lane).
    """
    found = []
    for trips in (0, 2):
        ev = _events(kernel.body.ops, tags, trips=trips)
        open_ = []  # accesses since the last barrier
        for kind, op in ev:
            if kind == "sync":
                open_ = []
                continue
            lo, hi, st, priv = tags[id(op)]
            for prev in open_:
                plo, phi, pst, ppriv = tags[id(prev)]
                if not (lo < phi and plo < hi) or (priv and ppriv):
                    continue
                if (st != pst) or (st and (lo, hi) != (plo, phi)):
                    found.append((prev.name, plo, op.name, lo, trips))
            open_.append(op)
    return found


@pytest.mark.parametrize("row", INSTANCES + KNOB_INSTANCES + WIDE_INSTANCES, ids=_ids)
def test_lds_barriers_separate_cross_wave_accesses(row):
    kernel, tags = traced_kernel(*row)
    assert not lds_hazards(kernel, tags)[:4]
    # the q step ends with a barrier (the next step overwrites q / dO / dS)
    body = [op for op in q_loop(kernel).regions[0].ops if op.name != "scf.yield"]
    step_end = {"full": "tile.sync", "lds_only": "tile.sync_lds_only"}
    for ops in step_bodies(kernel):
        assert _last_op(ops).name == step_end[resolved_of(*row).loop_sync]
    # the q-loop body reads and writes LDS, so the model above saw it
    assert sum(id(op) in tags for op, _ in _walk(body)) > 0


# ---------------------------------------------------------------------------
# counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row", INSTANCES + KNOB_INSTANCES + WIDE_INSTANCES, ids=_ids)
def test_mma_and_atomic_counts_per_step(row):
    arch = row[0]
    r = resolved_of(*row)
    geom = BwdTileGeometry.from_spec(r, arch)
    model = geom.stage_table()
    m = model.mfma
    wave = attn_bwd_arch_facts(arch).wave_size
    for ops in step_bodies(kernel_of(*row)):
        mmas = [op for op, _ in _walk(ops) if op.name == "tile.mma"]
        # emission order: G0, G2, G1, G3, G4
        assert len(mmas) == m.g0 + m.g2 + m.g1 + m.g3 + m.g4
        ids = [op.attrs["op_id"] for op in mmas]
        bounds = (m.g0, m.g0 + m.g2, m.g0 + m.g2 + m.g1, m.g0 + m.g2 + m.g1 + m.g3)
        assert set(ids[: bounds[1]]) == {geom.g0.op.op_id}
        assert set(ids[bounds[1] : bounds[3]]) == {geom.g1.op.op_id}
        assert set(ids[bounds[3] :]) == {geom.g4.op.op_id}
        atomics = [
            op for op, _ in _walk(ops) if op.name == "memref.global_atomic_add_f32"
        ]
        assert len(atomics) == r.block_m * r.head_size // (r.waves * wave)
        assert len(atomics) == model.dq_atomics_per_lane


@pytest.mark.parametrize("arch", ARCHS)
def test_dq_atomics_are_agent_scope_and_native(arch):
    variants = ("none", "band_pack", "band_atomic", "none_atomic_pack")
    for cfg, var in ((c, v) for c in distinct_configs(arch) for v in variants):
        ir = ir_of(arch, cfg, 64, "fp16", var)
        rmw = [ln for ln in ir.splitlines() if "atomicrmw" in ln]
        assert rmw
        assert all("fadd" in ln and 'syncscope("agent") monotonic' in ln for ln in rmw)
        assert "cmpxchg" not in ir


# ---------------------------------------------------------------------------
# addressing and access width
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "var",
    [
        "none", "band_pack", "band_atomic", "none_atomic_pack", "thd_none",
        "thd_band_pack", "thd_band_atomic", "thd_none_atomic_pack",
        "sv1_band_atomic", "thd_sv1_none_atomic_pack",
    ],
)  # fmt: skip
@pytest.mark.parametrize("arch", ARCHS)
def test_workspace_rebase_is_64bit(arch, var):
    ir = ir_of(arch, distinct_configs(arch)[0], 64, "bf16", var)
    body = ir.split(") #0 {", 1)[1]
    used = ["WS_LSE2", "WS_DSUM", "WS_DQ"]
    unused = ["WORKLIST"] + ([] if var.startswith("thd_") else ["OFF_Q", "OFF_KV"])
    (used if "atomic" in var else unused).extend(["WS_DK", "WS_DV"])
    for name in used:
        uses = re.findall(rf"[^\n]*%{name}\b[^\n]*", body)
        assert uses, name
        for u in uses:
            assert "getelementptr inbounds i8" in u and "i64" in u, u
        assert not re.search(
            rf"getelementptr inbounds float, ptr addrspace\(1\) %{name}\b", ir
        )
    for name in unused:
        assert f"%{name}" not in body  # unused in this instance


def _user_accesses(ir, names):
    derived = set(names)
    out = []
    for line in ir.splitlines():
        m = re.match(
            r"\s*(%\S+) = getelementptr inbounds \S+, ptr addrspace\(1\) (%\S+),", line
        )
        if m and m.group(2).rstrip(",") in derived:
            derived.add(m.group(1))
            continue
        # an opaque SGPR copy of a pointer (the THD dK / dV slabs) is the pointer
        m = re.match(
            r'\s*(%\S+) = call ptr addrspace\(1\) asm "", "=s,0"\(ptr addrspace\(1\) '
            r"(%[^)\s]+)\)",
            line,
        )
        if m and m.group(2) in derived:
            derived.add(m.group(1))
            continue
        m = re.match(
            r"\s*%\S+ = load (\S+(?: x \S+>)?), ptr addrspace\(1\) (%\S+), align (\d+)",
            line,
        )
        if m and m.group(2) in derived:
            out.append(("load", m.group(1), int(m.group(3))))
        m = re.match(
            r"\s*store (\S+(?: x \S+>)?) \S+, ptr addrspace\(1\) (%\S+), align (\d+)",
            line,
        )
        if m and m.group(2) in derived:
            out.append(("store", m.group(1), int(m.group(3))))
    return out


USER_TENSORS = ("%Q", "%K", "%V", "%dO", "%dK", "%dV")
WS_TENSORS = ("%WS_LSE2", "%WS_DSUM", "%WS_DQ", "%WS_DK", "%WS_DV")


@pytest.mark.parametrize("var", ["band", "band_pack"])
@pytest.mark.parametrize("dtype,elem", [("fp16", "half"), ("bf16", "bfloat")])
@pytest.mark.parametrize("arch", ARCHS)
def test_stage_vec_8_access_width(arch, dtype, elem, var):
    ir = ir_of(arch, distinct_configs(arch)[-1], 128, dtype, var)
    acc = _user_accesses(ir, USER_TENSORS)
    kinds = {k for k, _t, _a in acc}
    assert kinds == {"load", "store"}
    assert all(t == f"<8 x {elem}>" and al == 16 for _k, t, al in acc), acc[:4]


@pytest.mark.parametrize("arch", ARCHS)
def test_thd_direct_forms_the_dkv_slabs_before_the_q_loops(arch):
    """``dkv_slab_hoist``, THD direct: the dK / dV slab pointers are two opaque
    SGPR copies formed outside the q loops and feed the epilogue stores; the
    default, batched and atomic instances have none (inert there: the IR is
    the default's apart from the kernel name)."""
    cfg = distinct_configs(arch)[-1]
    pat = 'asm "", "=s,0"(ptr addrspace(1)'

    def hoisted(var):
        spec = dataclasses.replace(
            spec_of(arch, cfg, 64, "bf16", var), dkv_slab_hoist=True
        )
        return spec, build_attn_bwd_main(spec, arch=arch)

    def text(spec, kernel):
        name = validate_attn_bwd_spec(spec, arch).kernel_name("main")
        ir = lower(kernel, arch=arch, llvm_flavor="llvm22")
        return ir.replace(name, "@K")

    for var, n in (("thd_band", 2), ("thd_band_atomic", 0), ("band", 0)):
        assert ir_of(arch, cfg, 64, "bf16", var).count(pat) == 0, var
        spec, kernel = hoisted(var)
        got = text(spec, kernel)
        assert got.count(pat) == n, var
        if n == 0:
            assert INERT_KNOBS["dkv_slab_hoist"](
                resolved_of(arch, cfg, 64, "bf16", var), arch
            )
            base = spec_of(arch, cfg, 64, "bf16", var)
            assert got == text(base, kernel_of(arch, cfg, 64, "bf16", var)), var
    spec, kernel = hoisted("thd_band")
    assert not INERT_KNOBS["dkv_slab_hoist"](validate_attn_bwd_spec(spec, arch), arch)
    in_q = {id(op) for lp in q_loops(kernel) for op, _ in _walk(lp.regions[0].ops)}
    pins = [
        op
        for op, _ in _walk(kernel.body.ops)
        if op.name == "tile.inline_asm" and op.attrs["constraints"] == "=s,0"
    ]
    assert len(pins) == 2 and not any(id(op) in in_q for op in pins)
    acc = _user_accesses(lower(kernel, arch=arch, llvm_flavor="llvm22"), ("%dK", "%dV"))
    assert acc and {k for k, _t, _a in acc} == {"store"}


_LDS_ACCESS = re.compile(
    r"^\s*(?:%\S+ = )?(load|store) (\S+(?: x \S+>)?)[^;]*ptr addrspace\(3\)"
    r".*align (\d+)"
)
_LDS_CALL = re.compile(r"call [^@]*(@llvm\.amdgcn\.ds\.[\w.]+)\(")


def _lds_and_workspace_accesses(ir):
    """Sorted ``(space, kind, type, align)`` of every LDS access, every LDS
    intrinsic call and every workspace load / store / atomic of a kernel."""
    out = []
    for line in ir.splitlines():
        m = _LDS_ACCESS.match(line)
        if m:
            out.append(("lds", m.group(1), m.group(2), int(m.group(3))))
        call = _LDS_CALL.search(line)
        if call and not line.lstrip().startswith("declare"):
            out.append(("lds", "call", call.group(1), 0))
        if "atomicrmw" in line:
            kind = line.split("atomicrmw", 1)[1].split()[0]
            out.append(("atomic", "rmw", kind, 0))
    ws = _user_accesses(ir, WS_TENSORS)
    out += [("ws", k, t, a) for k, t, a in ws]
    return sorted(out)


# Narrow instances per arch: every configuration, both dtypes, d32 / d64 /
# d128, plain and head-packed, direct and atomic, batched and THD.
SV1_ROWS = [
    r
    for r in INSTANCES
    if r[4] in ("sv1_band", "sv1_none_pack", "sv1_band_atomic_pack", "thd_sv1_band")
]


@pytest.mark.parametrize("row", SV1_ROWS, ids=_ids)
def test_stage_vec_1_access_width(row):
    """``stage_vec = 1``: no user-tensor access wider than one element and
    none with ``align > 2``; the LDS, matrix-op and workspace accesses equal
    those of the ``stage_vec = 8`` instance of the same geometry."""
    arch, dt = row[0], row[3]
    elem = {"fp16": "half", "bf16": "bfloat"}[dt]
    r = resolved_of(*row)
    assert r.stage_vec == 1
    ir = lower(build_attn_bwd_main(r, arch=arch), arch=arch, llvm_flavor="llvm22")
    acc = _user_accesses(ir, USER_TENSORS)
    kinds = {k for k, _t, _a in acc}
    assert kinds == ({"load"} if r.dkv_mode == "atomic" else {"load", "store"})
    assert all(t in (elem, "i16") and al <= 2 for _k, t, al in acc), acc[:4]
    wide = dataclasses.replace(r, stage_vec=8)
    wide_ir = lower(
        build_attn_bwd_main(wide, arch=arch), arch=arch, llvm_flavor="llvm22"
    )
    assert all(
        t == f"<8 x {elem}>" for _k, t, _a in _user_accesses(wide_ir, USER_TENSORS)
    )
    assert _lds_and_workspace_accesses(ir) == _lds_and_workspace_accesses(wide_ir)
    assert ir.count("@llvm.amdgcn.mfma") + ir.count("@llvm.amdgcn.wmma") == (
        wide_ir.count("@llvm.amdgcn.mfma") + wide_ir.count("@llvm.amdgcn.wmma")
    )


def test_stage_vec_1_defaults_to_the_narrow_tiles():
    """The narrow start points: 16 x 64 (32 x 64 at d32) on four waves with
    the plain LDS transpose on CDNA (d128 with the register-minimising
    scheduler); the one-wave 16 x 16 WMMA tile on RDNA."""
    for arch in ARCHS:
        for d in HEADS:
            r = resolved_of(arch, "default", d, "fp16", "sv1_band")
            if attn_bwd_arch_facts(arch).matrix_path == "mfma":
                assert (r.block_m, r.block_n, r.waves) == (32 if d == 32 else 16, 64, 4)
                assert r.transpose_source == "lds_plain"
                assert r.global_path == "vgpr" and r.ring_depth == 1
                # d128 keeps clear of the 256-VGPR file with the register-
                # minimising scheduler; the 16-byte tiles keep the default one
                assert r.scheduler_strategy == (
                    "iterative-minreg" if d == 128 else None
                )
                wide = resolved_of(arch, "default", d, "fp16", "band")
                assert wide.scheduler_strategy is None
            else:
                assert (r.block_m, r.block_n, r.waves) == (16, 16, 1)


@pytest.mark.parametrize("row", INSTANCES + KNOB_INSTANCES + WIDE_INSTANCES, ids=_ids)
def test_no_wide_transpose_read(row):
    ir = ir_of(*row)
    assert "tr16.b128" not in ir
    r = resolved_of(*row)
    if r.transpose_source == "tr_read":
        assert "@llvm.amdgcn.ds.read.tr16.b64(" in ir
    else:
        assert "ds.read.tr" not in ir


# ---------------------------------------------------------------------------
# barrier flavour and step order
# ---------------------------------------------------------------------------

LDS_SYNC_ROWS = [r for r in INSTANCES if VARIANTS[r[4]].get("loop_sync") == "lds_only"]
HEAD_OUTER_ROWS = [
    r for r in INSTANCES if VARIANTS[r[4]].get("step_order") == "head_outer"
]


def _twin(row, **kw):
    """The kernel of ``row`` with spec fields replaced."""
    spec = dataclasses.replace(spec_of(*row), **kw)
    return build_attn_bwd_main(spec, arch=row[0])


def _op_names(kernel, flavour=None):
    """Op names in walk order; ``flavour`` renames the LDS-only barrier."""
    out = []
    for op, _anc in _walk(kernel.body.ops):
        name = op.name
        if flavour is not None and name == "tile.sync_lds_only":
            name = flavour
        out.append(name)
    return out


def _barrier_waits(ir):
    """Immediates of the ``s_waitcnt`` that precedes each ``s_barrier``."""
    lines = [ln.strip() for ln in ir.splitlines()]
    out = []
    for i, ln in enumerate(lines):
        if ln == "call void @llvm.amdgcn.s.barrier()":
            m = re.match(
                r"call void @llvm\.amdgcn\.s\.waitcnt\(i32 (\d+)\)", lines[i - 1]
            )
            assert m, lines[i - 1]
            out.append(int(m.group(1)))
    return out


@pytest.mark.parametrize("row", LDS_SYNC_ROWS, ids=_ids)
def test_lds_only_loop_barriers(row):
    """``loop_sync = "lds_only"``: every barrier inside a q step is the LDS-only
    one and every barrier outside the q loops (staging, accumulator zeroing,
    epilogue) stays full; apart from the barrier flavour the kernel is the
    ``"full"`` twin op for op. In LLVM IR the twin's barriers share one drain
    (VMEM and LDS) and the variant replaces exactly the in-loop ones by a
    different (LDS-only) drain."""
    kernel = kernel_of(*row)
    loops = q_loops(kernel)
    in_q = {id(op) for lp in loops for op, _ in _walk(lp.regions[0].ops)}
    n_in = 0
    for op, _anc in _walk(kernel.body.ops):
        if op.name == "tile.sync_lds_only":
            assert id(op) in in_q
            n_in += 1
        elif op.name == "tile.sync":
            assert id(op) not in in_q
    # staging publication, the P / dS publication, the end of the step
    assert n_in == 3 * len(loops)
    full = _twin(row, loop_sync="full")
    assert _op_names(kernel, "tile.sync") == _op_names(full)
    assert "tile.sync_lds_only" not in _op_names(full)
    arch = row[0]
    waits = _barrier_waits(lower(kernel, arch=arch, llvm_flavor="llvm22"))
    full_waits = _barrier_waits(lower(full, arch=arch, llvm_flavor="llvm22"))
    assert len(waits) == len(full_waits) and len(set(full_waits)) == 1
    drain = full_waits[0]
    assert sum(w != drain for w in waits) == n_in
    assert len({w for w in waits if w != drain}) == 1


def _value_op(v):
    return v.op.name if v.op is not None else None


@pytest.mark.parametrize("row", HEAD_OUTER_ROWS, ids=_ids)
def test_head_outer_step_order(row):
    """``step_order = "head_outer"``. One q-tile range: one loop of ``n_grp *
    n_qt`` steps (the default's trip count) whose q tile wraps at the range
    end back to the range start and whose head group advances on the wrap.
    Edge tiles: a head-group loop (trip ``n_grp``, or 0 on an empty inverse
    band) around the edge / interior / edge loops, which carry the q tile
    only, take the group's query head from the group loop and have their trip
    counts (``readfirstlane``'d range lengths) computed before it."""
    r = resolved_of(*row)
    kernel = kernel_of(*row)
    loops = q_loops(kernel)
    groups = group_loops(kernel)
    default = _twin(row, step_order="qtile_outer")
    assert kernel.name != default.name
    if not r.edge_tiles:
        assert not groups and len(loops) == 1
        loop = loops[0]
        (dloop,) = q_loops(default)
        # same trip count expression as the default: n_grp * n_qt in an SGPR
        assert _value_op(loop.operands[1]) == "tile.readfirstlane"
        assert _value_op(loop.operands[1].op.operands[0]) == "arith.mul"
        names = [m["name"] for m in loop.attrs["iter_args"]]
        assert names[-2].endswith("qt") and names[-1].endswith("grp")
        qt_iv, grp_iv = loop.attrs["iter_args"][-2]["name"], names[-1]
        qt_init = loop.operands[3 + len(names) - 2]
        body = loop.regions[0].ops
        # the q tile wraps at qt_start + n_qt (the range end) ...
        wraps = [
            op for op in body
            if op.name == "arith.cmp" and op.attrs.get("pred") == "ge"
            and op.operands[0].op is not None
            and op.operands[0].op.name == "arith.add"
            and op.operands[0].op.operands[0].name == qt_iv
        ]  # fmt: skip
        assert len(wraps) == 1
        end = wraps[0].operands[1].op
        assert end.name == "arith.add" and end.operands[0] is qt_init
        # ... back to the range start, and the head group advances on the wrap
        sel = [
            op for op in body
            if op.name == "arith.select" and op.operands[0] is wraps[0].results[0]
        ]  # fmt: skip
        assert any(op.operands[1] is qt_init for op in sel)
        assert any(
            op.name == "arith.add" and op.operands[0].name == grp_iv
            and any(op.operands[1].op is s for s in sel)
            for op in body
        )  # fmt: skip
        # the default wraps the head group instead (never the q tile)
        dbody = dloop.regions[0].ops
        dq_iv = dloop.attrs["iter_args"][-2]["name"]
        assert not any(
            op.name == "arith.cmp" and op.operands[0].op is not None
            and op.operands[0].op.name == "arith.add"
            and op.operands[0].op.operands[0].name == dq_iv
            for op in dbody
        )  # fmt: skip
        return
    assert len(groups) == 1 and len(loops) == 3
    (group,) = groups
    order = {id(op): i for i, (op, _) in enumerate(_walk(kernel.body.ops))}
    in_group = {id(op) for op, _ in _walk(group.regions[0].ops)}
    assert all(id(lp) in in_group for lp in loops)
    # group trip: readfirstlane(select(0 < n_qt, n_grp, 0))
    trip = group.operands[1].op
    assert trip.name == "tile.readfirstlane"
    sel = trip.operands[0].op
    assert sel.name == "arith.select" and _value_op(sel.operands[0]) == "arith.cmp"
    zero = sel.operands[2].op
    assert zero.name == "arith.constant" and zero.attrs.get("value") == 0
    grp_iv = group.attrs["iter_args"][-1]["name"]
    assert grp_iv.endswith("grp")
    for lp in loops:
        upper = lp.operands[1].op
        assert upper.name == "tile.readfirstlane"
        assert _value_op(upper.operands[0]) == "arith.sub"
        assert order[id(upper)] < order[id(group)]
        names = [m["name"] for m in lp.attrs["iter_args"]]
        assert names[-1].endswith("qt") and not any(n.endswith("grp") for n in names)
    # the query head of the group is formed in the group body, not per step
    heads = [
        op for op in group.regions[0].ops
        if op.name == "arith.add" and any(o.name == grp_iv for o in op.operands)
        or op.name == "arith.mul" and any(o.name == grp_iv for o in op.operands)
    ]  # fmt: skip
    assert heads
    for lp in loops:
        assert not any(
            any(o.name == grp_iv for o in op.operands)
            and op.name in ("arith.add", "arith.mul")
            for op, _ in _walk(lp.regions[0].ops)
        )


@pytest.mark.parametrize("arch", ARCHS)
def test_new_values_change_the_ir_and_defaults_do_not(arch):
    """Each non-default barrier flavour / step order changes the emitted IR
    (beyond the kernel name); the spelled-out defaults emit the default IR."""
    cfg = distinct_configs(arch)[0]
    # (d128: the grouped head-outer form at d64 on the CDNA default tiles is a
    # recorded resource-gate failure, RESOURCE_REFUSED)
    base = spec_of(arch, cfg, 128, "fp16", "band_edge")

    def text(spec):
        name = validate_attn_bwd_spec(spec, arch).kernel_name("main")
        ir = lower(
            build_attn_bwd_main(spec, arch=arch), arch=arch, llvm_flavor="llvm22"
        )
        return ir.replace(name, "KNAME")

    ref = text(base)
    same = dataclasses.replace(base, loop_sync="full", step_order="qtile_outer")
    assert text(same) == ref
    for kw in (
        dict(loop_sync="lds_only"),
        dict(step_order="head_outer"),
        dict(step_order="head_outer", edge_tiles=False),
    ):
        alt = dataclasses.replace(base, **kw)
        assert text(alt) != text(dataclasses.replace(alt, **{
            k: AttnBwdSpec.__dataclass_fields__[k].default
            for k in kw if k in ("loop_sync", "step_order")
        }))  # fmt: skip


# ---------------------------------------------------------------------------
# resource gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "var", list(VARIANTS) + [f"+{k}" for k in KNOB_VALUES] + [WIDE_SUFFIX]
)
@pytest.mark.parametrize("arch", ARCHS)
def test_zero_scratch_no_spills_and_lds_equal_to_the_plan(arch, var):
    if var == WIDE_SUFFIX and arch != "gfx950":
        pytest.skip("the 16x16x32 G1 / G3 atom is gfx950-only")
    pytest.importorskip("rocke.runtime.comgr")
    from ._attention_bwd_resources import (
        GATE_FLAVOR,
        comgr_is_production,
        comgr_version,
        compile_or_skip,
        current_flavor,
        evaluate_resource_gate,
        find_objdump,
        pin_production_comgr,
    )

    # torch's bundled comgr, the production one, pinned by path (torch is not
    # imported); unpinned, a node's system comgr loads (the gate refuses it)
    pin_production_comgr()
    if find_objdump() is None:
        pytest.skip("llvm-objdump not found (needed for the q-loop AGPR copy rule)")
    # A verdict counts on the production flavor and comgr only (the predicate
    # of ResourceGateResult.valid): decided before compiling, so another
    # toolchain does not compile and disassemble every instance only to skip
    # (a disassembler time-out under load failed that discarded work).
    flavor, comgr = current_flavor(), comgr_version()
    if flavor != GATE_FLAVOR or not comgr_is_production(comgr):
        pytest.skip(
            f"resource gate is valid on {GATE_FLAVOR} with the production comgr "
            f"only (flavor {flavor}, comgr {comgr})"
        )
    facts = attn_bwd_arch_facts(arch)
    failures = []
    if var.startswith("+"):  # a staging knob over the variants it is tested on
        rows = [x for x in KNOB_INSTANCES if x[0] == arch and x[-1].endswith(var)]
    elif var == WIDE_SUFFIX:  # every wide G1 / G3 instance of the IR suite
        rows = [x for x in WIDE_INSTANCES if x[0] == arch]
        assert rows
    else:
        rows = [x for x in INSTANCES if x[0] == arch and x[-1] == var]
    for row in rows:
        r = resolved_of(*row)
        art = compile_or_skip(
            kernel_of(*row), arch=arch, capture_ir_text=False, backend="python"
        )
        res = evaluate_resource_gate(
            art.hsaco,
            arch=arch,
            lds_capacity_bytes=facts.lds_capacity_bytes,
            spec=r,
            ir_text=art.llvm_text,
        )
        if not res.valid:
            pytest.skip(
                f"resource gate is valid on llvm22 with the production comgr only "
                f"(flavor {res.flavor}, comgr {res.comgr})"
            )
        if res.failures:
            failures.append((row, res.failures))
        if res.warnings:
            failures.append((row, res.warnings, res.resources.sgpr))
        plan_bytes = lds_plan(r, arch).total_bytes
        if res.resources.lds_bytes > plan_bytes:
            failures.append((row, f"LDS {res.resources.lds_bytes} > plan {plan_bytes}"))
    assert not failures, failures


def test_validator_refuses_exactly_the_spilling_layouts():
    """The validator's layout-spill refusal hits exactly the recorded rows of
    the +xt / +xor product, and no other IR-suite row."""
    from kernels.common.attention_bwd import (
        LAYOUT_SPILLING_COMBOS,
        _layout_spill_key,
    )

    refused = set()
    for arch in ARCHS:
        for d in HEADS:
            for base in _KNOB_BASES:
                for knob in ("xt", "xor"):
                    row = (arch, "default", d, "fp16", f"{base}+{knob}")
                    try:
                        validate_attn_bwd_spec(spec_of(*row), arch)
                    except ValueError as exc:
                        if "LAYOUT_SPILLING_COMBOS" in str(exc):
                            refused.add(row)
    assert refused == LAYOUT_REFUSED_ROWS
    assert len(LAYOUT_SPILLING_COMBOS) == len(LAYOUT_REFUSED_ROWS)
    for row in INSTANCES + KNOB_INSTANCES + ORDER_ROWS:
        if _legal(row):
            key = _layout_spill_key(resolved_of(*row), row[0])
            assert key not in LAYOUT_SPILLING_COMBOS, row


# ---------------------------------------------------------------------------
# stage order: the issue order of G1, G3 and G4 inside a q step
# ---------------------------------------------------------------------------

# Spec knobs of other groups that change the P / dS hand-off (LDS route), the
# accumulators (LDS images on MFMA) or the G1 / G3 emission (AGPR routes).
ORDER_KNOBS = {
    "": {},
    "ptlds": dict(pt_route="lds"),
    "xt": dict(transpose_source="xt_lds"),
    "xtptlds": dict(transpose_source="xt_lds", pt_route="lds"),
    "acclds": dict(acc_in_lds=True, agpr_alloc=(0, 0)),
    "fence": dict(agpr_alloc=(128, 128), agpr_route="agpr_fence"),
    "asm": dict(agpr_alloc=(128, 128), agpr_route="asm_g13"),
    # the asm route with the second G1 / G3 GEMM's LDS operand reads hoisted
    # above the first one's asm cluster
    "asmhoist": dict(agpr_alloc=(128, 128), agpr_route="asm_g13", asm_load_hoist=True),
    # gfx950: the 16x16x32 G1 / G3 atom (K-permuted B), on the intrinsic and
    # on the hoisted asm route
    "wide": dict(atom_g13="16x16x32", block_m=32),
    "wideasmhoist": dict(
        atom_g13="16x16x32", block_m=32, agpr_alloc=(128, 128), agpr_route="asm_g13",
        asm_load_hoist=True,
    ),  # fmt: skip
}
# Order knobs legal on gfx950 only (the 16x16x32 atom).
ORDER_KNOBS_GFX950 = {"wide", "wideasmhoist"}
ORDER_BASES = ("band", "none_pack", "band_edge", "band_atomic_pack", "thd_sv1_none")


def _legal(row) -> bool:
    try:
        validate_attn_bwd_spec(spec_of(*row), row[0])
    except ValueError:
        return False
    return True


def _order_rows():
    rows = []
    for arch in ARCHS:
        for cfg in distinct_configs(arch):
            for order in STAGE_ORDERS:
                for base in ORDER_BASES:
                    rows.append((arch, cfg, 64, "fp16", f"{base}||{order}"))
                for d in (32, 128):
                    rows.append((arch, cfg, d, "bf16", f"band||{order}"))
                for knobs in ORDER_KNOBS:
                    if not knobs:
                        continue
                    for d in (64, 128):
                        rows.append((arch, cfg, d, "fp16", f"band|{knobs}|{order}"))
    return [r for r in rows if _legal(r)]


ORDER_ROWS = _order_rows()
NON_DEFAULT_ORDER_ROWS = [
    r for r in ORDER_ROWS if not r[4].endswith("|" + DEFAULT_STAGE_ORDER)
]
_CARRIED = re.compile(r"^%bwd_q(?:i_|e_)?(dk|dv)\d+$")


def _with_order(row, order):
    base, knobs, _ = row[4].split("|")
    return row[:4] + (f"{base}|{knobs}|{order}",)


def test_order_rows_cover_every_value_and_knob_point():
    for arch in ARCHS:
        rows = [r for r in ORDER_ROWS if r[0] == arch]
        assert {r[4].rsplit("|", 1)[1] for r in rows} == set(STAGE_ORDERS)
        knobs = {r[4].split("|")[1] for r in rows}
        mfma = attn_bwd_arch_facts(arch).matrix_path == "mfma"
        # every hand-off, accumulator, G1 / G3 route and atom this arch can
        # build
        rdna = {"", "ptlds", "xt", "xtptlds"}
        want = set(ORDER_KNOBS) if mfma else rdna
        if arch != "gfx950":
            want -= ORDER_KNOBS_GFX950
        assert knobs == want, (arch, knobs)


def _matrix_ops(ops):
    """Matrix instructions of a step body in program order: ``(op, index of
    the accumulator input)`` per ``tile.mma``, and once per MFMA line of an
    AGPR-form asm cluster (input 0 is the cluster's first accumulator)."""
    out = []
    for op, _ in _walk(ops):
        if op.name == "tile.mma":
            out.append((op, 2))
        elif op.name == "tile.inline_asm" and "mfma" in op.attrs["template"]:
            out += [(op, 0)] * op.attrs["template"].count("mfma")
    return out


def _classify(ops, tags, acc_bases):
    """GEMM label of every matrix op of a step body, in program order.

    An accumulator chain is followed back to its first input: a carried dV /
    dK block argument (register accumulators: ``g1`` / ``g3``), an fp32 image
    read from the dV / dK LDS buffer (``acc_in_lds``: ``g1`` / ``g3``), or a
    fresh zero (``g04``: G0, G2 or G4).
    """
    prod = {}
    for op, _ in _walk(ops):
        for res in op.results:
            prod[res.name] = op
    mats = _matrix_ops(ops)
    mat_ids = {id(op): idx for op, idx in mats}
    labels = []
    for op, idx in mats:
        v = op.operands[idx]
        while v.name in prod and id(prod[v.name]) in mat_ids:
            src = prod[v.name]
            v = src.operands[mat_ids[id(src)]]
        m = _CARRIED.match(v.name)
        if m:
            labels.append({"dv": "g1", "dk": "g3"}[m.group(1)])
            continue
        # LDS accumulator image: vec_pack <- vec_extract <- LDS load
        label = "g04"
        src = prod.get(v.name)
        if src is not None and acc_bases:
            seen = set()
            stack = list(src.operands)
            while stack:
                x = stack.pop()
                p = prod.get(x.name)
                if p is None or id(p) in seen:
                    continue
                seen.add(id(p))
                if id(p) in tags:
                    lo = tags[id(p)][0]
                    if lo in acc_bases:
                        label = acc_bases[lo]
                        break
                    continue
                stack.extend(p.operands)
        labels.append(label)
    return mats, labels


def _acc_bases(r, arch):
    """LDS element offsets of the dV / dK accumulator images (``acc_in_lds``)."""
    if not r.acc_in_lds:
        return {}
    plan = lds_plan(r, arch)
    return {
        plan.buffer("dv_acc").offset // 2: "g1",
        plan.buffer("dk_acc").offset // 2: "g3",
    }


@pytest.mark.parametrize("row", ORDER_ROWS, ids=_ids)
def test_stage_order_issue_order_barriers_and_atomics(row):
    """Matrix ops issue as G0, G2, then G1 / G3 / G4 in ``stage_order`` with
    the model's counts; the dQ atomics follow the first G4 matrix op and
    precede the next GEMM; the dS barrier (relabel route) sits right before
    G4, the P / dS barrier (LDS route) before all three; the step has three
    barriers and no cross-wave LDS race."""
    arch = row[0]
    order = row[4].rsplit("|", 1)[1]
    r = resolved_of(*row)
    assert r.stage_order == order
    geom = BwdTileGeometry.from_spec(r, arch)
    model = geom.stage_table()
    m = model.mfma
    count = {"g1": m.g1, "g3": m.g3, "g4": m.g4}
    units = [order[i : i + 2] for i in (0, 2, 4)]
    kernel, tags = traced_kernel(*row)
    assert not lds_hazards(kernel, tags)[:4]
    bases = _acc_bases(r, arch)
    for ops in step_bodies(kernel):
        # the step ends with its barrier (then only the agpr_fence pins)
        body = [op for op in ops if op.name != "scf.yield"]
        end = max(i for i, op in enumerate(body) if op.name == "tile.sync")
        assert all(
            op.name == "tile.inline_asm" and op.attrs["constraints"] == "=a,0"
            for op in body[end + 1 :]
        )
        assert (end == len(body) - 1) == (r.agpr_route != "agpr_fence")
        mats, labels = _classify(ops, tags, bases)
        head = m.g0 + m.g2
        assert labels[:head] == ["g04"] * head
        tail = [("g4" if x == "g04" else x) for x in labels[head:]]
        assert tail == [u for u in units for _ in range(count[u])], tail
        # program-order positions of the matrix ops, atomics and barriers
        pos = {id(op): i for i, (op, _) in enumerate(_walk(ops))}
        mat_pos = [pos[id(op)] for op, _ in mats]
        first = {}
        at = head
        for u in units:
            first[u] = mat_pos[at]
            at += count[u]
        atomics = [
            pos[id(op)]
            for op, _ in _walk(ops)
            if op.name == "memref.global_atomic_add_f32"
        ]
        assert len(atomics) == model.dq_atomics_per_lane
        nxt = units.index("g4") + 1
        stop = first[units[nxt]] if nxt < 3 else len(pos)
        assert all(first["g4"] < a < stop for a in atomics)
        syncs = [pos[id(op)] for op in ops if op.name == "tile.sync"]
        # xt_lds under relabel: the Q^T / dO^T images are published by one
        # more barrier before the first G1 / G3 (the dS barrier when G4 is
        # first, the P / dS barrier on the LDS route)
        xt_sync = (
            r.transpose_source == "xt_lds"
            and r.pt_route == "relabel"
            and units[0] != "g4"
        )
        assert len(syncs) == 3 + xt_sync
        if xt_sync:
            assert mat_pos[head - 1] < syncs[1] < first[units[0]]
        mid = syncs[-2]
        assert mat_pos[head - 1] < mid  # after G2 and the hand-off
        if r.pt_route == "relabel":
            # nothing but the dS barrier between the previous GEMM and G4
            assert max(p for p in mat_pos if p < first["g4"]) < mid < first["g4"]
        else:
            assert mid < min(first.values())


@pytest.mark.parametrize("row", NON_DEFAULT_ORDER_ROWS, ids=_ids)
def test_stage_order_changes_only_the_issue_order(row):
    """A non-default order salts the name and changes the IR; the step holds
    the same ops (by kind and matrix op id) as the default order's step, but
    for the barrier that publishes the XT images (``xt_lds`` under relabel:
    absent when G4, and its dS barrier, come first)."""
    base = _with_order(row, DEFAULT_STAGE_ORDER)
    k, k0 = kernel_of(*row), kernel_of(*base)
    assert k.name != k0.name
    assert ir_of(*row) != ir_of(*base)
    r = resolved_of(*row)
    xt_relabel = r.transpose_source == "xt_lds" and r.pt_route == "relabel"

    def census(kernel):
        return [
            sorted(
                (op.name, str(op.attrs.get("op_id", "")))
                for op, _ in _walk(ops)
                if not (xt_relabel and op.name == "tile.sync")
            )
            for ops in step_bodies(kernel)
        ]

    assert census(k) == census(k0)
    assert "tr16.b128" not in ir_of(*row)


@pytest.mark.parametrize("arch", ARCHS)
def test_default_stage_order_is_the_omitted_field(arch):
    """The explicit default order is the default spec: same name, same IR."""
    for cfg in distinct_configs(arch):
        for var in ("band", "none_atomic_pack", "thd_sv1_band"):
            spec = spec_of(arch, cfg, 64, "fp16", var)
            explicit = dataclasses.replace(spec, stage_order=DEFAULT_STAGE_ORDER)
            assert explicit == spec
            assert ir_of(arch, cfg, 64, "fp16", var) == lower(
                build_attn_bwd_main(explicit, arch=arch),
                arch=arch,
                llvm_flavor="llvm22",
            )


# ---------------------------------------------------------------------------
# staging knobs: Q / dO issue order and the stats load placement
# ---------------------------------------------------------------------------


def _staging(ops):
    """Ops of a q-step body up to its first top-level barrier (the staging),
    in program order (``_walk`` order), and the rest."""
    head = []
    for i, op in enumerate(ops):
        if op.name == "tile.sync":
            return head, ops[i:]
        head.append(op)
    raise AssertionError("q step without a staging barrier")


def _loads_from(ops, derived):
    return [
        (i, op, anc)
        for i, (op, anc) in enumerate(_walk(ops))
        if op.name.startswith("memref.global_load")
        and op.operands
        and op.operands[0].name in derived
    ]


def _lds_stores(ops):
    return [
        (i, op, anc)
        for i, (op, anc) in enumerate(_walk(ops))
        if op.name.startswith("tile.smem_store")
    ]


def _knob_rows(knob):
    return [r for r in KNOB_INSTANCES if r[4].endswith(f"+{knob}")]


def _plain(row):
    return row[:4] + (row[4].split("+")[0],)


@pytest.mark.parametrize("row", KNOB_INSTANCES, ids=_ids)
def test_staging_knob_changes_the_kernel(row):
    """Every non-default staging value emits a different kernel (name salted,
    IR different) from its default twin, with the same matrix-op and dQ-atomic
    counts and the same set of LDS / workspace access kinds."""
    plain = _plain(row)
    assert resolved_of(*row).kernel_name() != resolved_of(*plain).kernel_name()
    ir, ir0 = ir_of(*row), ir_of(*plain)
    assert ir.replace(resolved_of(*row).kernel_name(), "K") != ir0.replace(
        resolved_of(*plain).kernel_name(), "K"
    )
    for tok in ("@llvm.amdgcn.mfma", "@llvm.amdgcn.wmma", "atomicrmw"):
        assert ir.count(tok) == ir0.count(tok), tok


@pytest.mark.parametrize("row", _knob_rows("lf"), ids=_ids)
def test_loads_first_issues_every_staging_load_before_a_store(row):
    """``qdo_stage = "loads_first"``: in every q step all Q / dO global loads
    precede the first LDS store of the staging; per tile and thread there are
    ``ceil(kM0 * D / 8 / threads)`` chunk loads, as in the default (whose first
    store comes before its last load)."""
    r = resolved_of(*row)
    wave = attn_bwd_arch_facts(row[0]).wave_size
    chunks = -(-(r.block_m * r.head_size // 8) // (r.waves * wave))
    for kernel, first in ((kernel_of(*row), True), (kernel_of(*_plain(row)), False)):
        qdo = _derived(kernel, ("Q", "dO"))
        for ops in step_bodies(kernel):
            head, _rest = _staging(ops)
            loads = _loads_from(head, qdo)
            stores = _lds_stores(head)
            per_chunk = 1 if r.stage_vec == 8 else 8  # element loads when narrow
            assert len(loads) == 2 * chunks * per_chunk, (len(loads), chunks)
            assert len(stores) == 2 * chunks
            assert (loads[-1][0] < stores[0][0]) == first


@pytest.mark.parametrize("row", _knob_rows("joint") + _knob_rows("jtop"), ids=_ids)
def test_joint_staging_is_one_wave_uniform_pass(row):
    """``qdo_stage = "joint"``: the staging is two branches, Q then dO, each
    with exactly one chunk load and one LDS store per thread; the branch
    conditions come from the ``readfirstlane``'d wave index (wave-uniform), and
    no staging load sits outside them."""
    kernel = kernel_of(*row)
    q, do = _derived(kernel, ("Q",)), _derived(kernel, ("dO",))
    for ops in step_bodies(kernel):
        head, _rest = _staging(ops)
        ifs = [op for op in head if op.name == "scf.if"]
        assert len(ifs) == 2
        for br, derived in zip(ifs, (q, do)):
            inner = br.regions[0].ops
            loads = _loads_from(inner, q | do)
            assert loads and all(op.operands[0].name in derived for _i, op, _a in loads)
            assert len(_lds_stores(inner)) == 1
            # the condition depends on the wave index, not on the thread id
            seen, todo = set(), [br.operands[0]]
            names = set()
            while todo:
                v = todo.pop()
                if id(v) in seen or v.op is None:
                    continue
                seen.add(id(v))
                names.add(v.op.name)
                todo.extend(v.op.operands)
            assert "tile.readfirstlane" in names
        outside = _loads_from([op for op in head if op.name != "scf.if"], q | do)
        assert not outside


@pytest.mark.parametrize("row", _knob_rows("top") + _knob_rows("jtop"), ids=_ids)
def test_step_top_issues_the_stats_loads_first(row):
    """``stats_load = "step_top"``: every lse2 / Dsum load of a q step is issued
    before its first Q / dO load (ahead of the staging and G0 / G2); the
    default issues them after the first matrix op. Same number of loads."""
    for kernel, top in ((kernel_of(*row), True), (kernel_of(*_plain(row)), False)):
        stats = _derived(kernel, ("WS_LSE2", "WS_DSUM"))
        qdo = _derived(kernel, ("Q", "dO"))
        counts = []
        for ops in step_bodies(kernel):
            st = _loads_from(ops, stats)
            first_qdo = _loads_from(ops, qdo)[0][0]
            first_mma = next(
                i for i, (op, _a) in enumerate(_walk(ops)) if op.name == "tile.mma"
            )
            assert st
            if top:
                assert st[-1][0] < first_qdo
            else:
                assert st[0][0] > first_mma
            counts.append(len(st))
        if top:
            want = counts
        else:
            assert counts == want


# ---------------------------------------------------------------------------
# G4 dS key slicing (g4_slicing, block_k4) and G0 / G2 as asm (s_in_agpr)
# ---------------------------------------------------------------------------

# Knob points the slicing values are crossed with: the LDS P / dS route, K^T
# from LDS (the K^T slices stream with dS), and asm_g13 with s_in_agpr (every
# GEMM but G4 an asm cluster).
G4_KNOBS = {
    "": {},
    "ptlds": dict(pt_route="lds"),
    "ktlds": dict(kt_source="lds"),
    "asms": dict(agpr_alloc=(128, 128), agpr_route="asm_g13", s_in_agpr=True),
}
G4_BASES = ("band", "band_edge", "band_atomic_pack", "thd_sv1_none")
G4_VALUES = ("slice", "slice_db", "slice_db_g3")


def _g4_spec(arch, d, base, knobs, value, k4=None):
    kw = dict(config_knobs("default", arch), **VARIANTS[base], **G4_KNOBS[knobs])
    spec = AttnBwdSpec(head_size=d, dtype="fp16", g4_slicing=value, **kw)
    if k4 is None:  # two G4 atom-K steps per slice where kN0 allows it
        r = validate_attn_bwd_spec(dataclasses.replace(spec, g4_slicing="step"), arch)
        k4a = BwdTileGeometry.from_spec(r, arch).g4.atom_k
        k4 = 2 * k4a if r.block_n % (2 * k4a) == 0 else k4a
    return dataclasses.replace(spec, block_k4=k4)


def _g4_rows():
    rows = []
    for arch in ("gfx942", "gfx950"):
        for d in (64, 128):
            for base in G4_BASES:
                for knobs in G4_KNOBS:
                    for value in G4_VALUES:
                        try:
                            spec = _g4_spec(arch, d, base, knobs, value)
                            validate_attn_bwd_spec(spec, arch)
                        except ValueError:
                            continue
                        rows.append((arch, d, base, knobs, value))
    return rows


G4_ROWS = _g4_rows()


def test_g4_rows_cover_every_value_and_knob_point():
    for arch in ("gfx942", "gfx950"):
        rows = [r for r in G4_ROWS if r[0] == arch]
        assert {r[4] for r in rows} == set(G4_VALUES)
        assert {r[3] for r in rows} == set(G4_KNOBS), arch
        assert {r[2] for r in rows} == set(G4_BASES), arch


def _ds_reads(ops, tags, ds_lo):
    """Program-order positions (in ``_walk(ops)``) of the reads of the dS tile."""
    out = []
    for i, (op, _) in enumerate(_walk(ops)):
        t = tags.get(id(op))
        if t is not None and not t[2] and t[0] == ds_lo:
            out.append(i)
    return out


@pytest.mark.parametrize("row", G4_ROWS, ids=_ids)
def test_g4_slicing_order_barriers_and_counts(row):
    """Key slicing changes only the issue order inside G4 (and, for
    ``slice_db_g3``, reads G4's first dS slice before G3 behind a barrier
    moved there): the step holds the ops of the ``step`` kernel (by kind and
    matrix op id), the matrix ops issue as G0, G2, G1, G3, G4 with the model's
    counts, three barriers per step, no cross-wave LDS race; under the relabel
    route the dS reads before the first G4 MMA (before the first G3 MMA for
    ``slice_db_g3``) are one atom-K step (step), one slice (slice) or two
    slices (slice_db)."""
    arch, d, base, knobs, value = row
    spec = _g4_spec(arch, d, base, knobs, value)
    r = validate_attn_bwd_spec(spec, arch)
    step = dataclasses.replace(spec, g4_slicing="step")
    kernel, tags = traced_spec(spec, arch)
    k0, tags0 = traced_spec(step, arch)
    assert not lds_hazards(kernel, tags)[:4]
    geom = BwdTileGeometry.from_spec(r, arch)
    m = geom.stage_table().mfma
    plan = lds_plan(r, arch)
    ds_lo = plan.buffer("ds").offset // 2
    n_sl = r.block_n // r.block_k4
    steps_k = r.block_n // geom.g4.atom_k
    bodies, bodies0 = step_bodies(kernel), step_bodies(k0)
    assert len(bodies) == len(bodies0)

    def census(ops):
        return sorted((op.name, str(op.attrs.get("op_id", ""))) for op, _ in _walk(ops))

    for ops, ops0 in zip(bodies, bodies0):
        assert census(ops) == census(ops0)
        mats, labels = _classify(ops, tags, {})
        head = m.g0 + m.g2
        assert labels[:head] == ["g04"] * head
        tail = [("g4" if x == "g04" else x) for x in labels[head:]]
        assert tail == ["g1"] * m.g1 + ["g3"] * m.g3 + ["g4"] * m.g4
        pos = {id(op): i for i, (op, _) in enumerate(_walk(ops))}
        mat_pos = [pos[id(op)] for op, _ in mats]
        first_g3, first_g4 = mat_pos[head + m.g1], mat_pos[head + m.g1 + m.g3]
        syncs = [pos[id(op)] for op in ops if op.name == "tile.sync"]
        assert len(syncs) == 3 and _last_op(ops).name == "tile.sync"
        mid = syncs[1]
        if r.pt_route != "relabel":
            assert mid < mat_pos[head]  # the P / dS barrier precedes G1 / G3 / G4
            continue
        reads = _ds_reads(ops, tags, ds_lo)
        reads0 = _ds_reads(ops0, tags0, ds_lo)
        assert reads and len(reads) == len(reads0)
        per_step = len(reads) // steps_k
        per_slice = len(reads) // n_sl
        if r.g4_slicing == "slice_db_g3":
            # the dS barrier sits after G1, before G3; G4's first slice is read
            # between the barrier and the first G3 MMA
            assert mat_pos[head + m.g1 - 1] < mid < first_g3
            assert sum(mid < p < first_g3 for p in reads) == per_slice
            assert sum(p < first_g4 for p in reads) == min(2, n_sl) * per_slice
        else:
            assert max(p for p in mat_pos if p < first_g4) < mid < first_g4
            want = {
                "step": per_step,
                "slice": per_slice,
                "slice_db": min(2, n_sl) * per_slice,
            }[r.g4_slicing]
            assert sum(p < first_g4 for p in reads) == want
        assert all(p > mid for p in reads)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_g4_slicing_differs_from_step_and_keeps_the_ir_census(arch):
    """Every non-step value at two or more slices is a distinct kernel with
    the step kernel's MFMA, LDS-read and atomic counts in the IR text."""
    texts = {}
    for value in ("step",) + G4_VALUES:
        spec = _g4_spec(arch, 128, "band", "", value)
        r = validate_attn_bwd_spec(spec, arch)
        assert r.g4_slicing == value and r.block_n // r.block_k4 >= 2
        ir = lower(
            build_attn_bwd_main(spec, arch=arch), arch=arch, llvm_flavor="llvm22"
        )
        texts[value] = ir.replace(r.kernel_name("main"), "K")
    assert len(set(texts.values())) == 4

    def count(ir):
        return (
            len(re.findall(r"call <4 x float> @llvm\.amdgcn\.mfma", ir)),
            len(
                re.findall(r"load <\d+ x (?:half|bfloat|i16)>, ptr addrspace\(3\)", ir)
            ),
            ir.count("atomicrmw"),
            ir.count("@llvm.amdgcn.s.barrier"),
        )

    want = count(texts["step"])
    assert all(count(t) == want for t in texts.values())


def test_g4_first_slice_without_the_moved_barrier_is_a_race(monkeypatch):
    """Mutation control of ``slice_db_g3``: with the dS barrier left before G4
    while G4's first slice is read under G3, the cross-wave model finds the dS
    write / read race (the shipped placement has none)."""
    from kernels.common import _attention_bwd_body as body

    for arch in ("gfx942", "gfx950"):
        spec = _g4_spec(arch, 128, "band", "", "slice_db_g3")
        kernel, tags = traced_spec(spec, arch)
        assert not lds_hazards(kernel, tags)
        with monkeypatch.context() as mp:
            mp.setattr(body.BwdTileStep, "_ds_reader", property(lambda self: "g4"))
            bad, bad_tags = traced_spec(spec, arch)
        assert lds_hazards(bad, bad_tags)


# The new levers under the resource gate (rules 5 and 6): G4 key slicing on the
# default tiles (K^T from registers and from LDS), and asm_g13 with s_in_agpr
# on the S / dP-in-AGPR start points at both cluster granularities.
_GATE_ASM_S = {
    "gfx942": [
        (128, dict(block_m=16, block_n=128), (192, 192)),
        (64, dict(block_m=32, block_n=128), (128, 128)),
    ],
    "gfx950": [
        (128, dict(block_m=32, block_n=128), (192, 192)),
        (128, dict(block_m=16, block_n=192), (256, 256)),
    ],
}


def _gate_new_rows():
    rows = []
    for arch in ("gfx942", "gfx950"):
        for d in (64, 128):
            for knobs in ("", "ktlds"):
                for value in G4_VALUES:
                    rows.append(
                        pytest.param(
                            arch,
                            _g4_spec(arch, d, "band", knobs, value),
                            id=f"{arch}-d{d}-g4-{value}{'-' + knobs if knobs else ''}",
                        )
                    )
        for d, tile, alloc in _GATE_ASM_S[arch]:
            for cluster in ("k_step", "gemm"):
                spec = AttnBwdSpec(
                    head_size=d, dtype="bf16", waves=4, warp_grid_g4=(1, 4),
                    block_k4=32, kv_residency="reg", kt_source="reg",
                    agpr_alloc=alloc, agpr_route="asm_g13", s_in_agpr=True,
                    asm_cluster=cluster, **tile,
                )  # fmt: skip
                tag = f"{tile['block_m']}x{tile['block_n']}"
                rows.append(
                    pytest.param(arch, spec, id=f"{arch}-d{d}-{tag}-asm_s-{cluster}")
                )
    return rows


@pytest.mark.parametrize("arch,spec", _gate_new_rows())
def test_new_levers_pass_the_resource_gate(arch, spec):
    """Zero scratch and spills, register budgets, LDS within the plan, and in
    every matrix loop no AGPR copy beyond the declared plan: none for key
    slicing (VGPR form), the S / dP reads only for asm_g13 with s_in_agpr,
    whose G0 / G2 / G1 / G3 MFMAs write AGPRs and G4's write VGPRs."""
    pytest.importorskip("rocke.runtime.comgr")
    from rocke.helpers.compile import compile_kernel

    from ._attention_bwd_resources import (
        GATE_FLAVOR,
        comgr_is_production,
        comgr_version,
        current_flavor,
        evaluate_resource_gate,
        find_objdump,
        pin_production_comgr,
    )

    pin_production_comgr()  # torch's bundled comgr, by path; torch not imported
    if find_objdump() is None:
        pytest.skip("llvm-objdump not found (needed for the q-loop AGPR copy rule)")
    flavor, comgr = current_flavor(), comgr_version()
    if flavor != GATE_FLAVOR or not comgr_is_production(comgr):
        pytest.skip(
            f"resource gate is valid on {GATE_FLAVOR} with the production comgr "
            f"only (flavor {flavor}, comgr {comgr})"
        )
    r = validate_attn_bwd_spec(spec, arch)
    art = compile_kernel(
        build_attn_bwd_main(spec, arch=arch), arch=arch, capture_ir_text=False,
        backend="python",
    )  # fmt: skip
    facts = attn_bwd_arch_facts(arch)
    res = evaluate_resource_gate(
        art.hsaco, arch=arch, lds_capacity_bytes=facts.lds_capacity_bytes, spec=r
    )
    assert res.valid
    print(f"[gate-new] {arch} {r.kernel_name()} {res.record()}")
    assert not res.failures, res.record()
    assert not res.warnings, res.record()
    assert res.resources.lds_bytes <= lds_plan(r, arch).total_bytes
    if r.s_in_agpr:
        geom = BwdTileGeometry.from_spec(r, arch)
        asm = 2 * geom.g0.mma_per_wave + 2 * geom.g1.mma_per_wave
        for lp in res.loops:
            assert lp["mfma_dst"] == {"agpr": asm, "vgpr": geom.g4.mma_per_wave}, lp
