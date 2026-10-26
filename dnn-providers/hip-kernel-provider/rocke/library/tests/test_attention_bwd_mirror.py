# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Mirror pairing (``lb_order = "mirror"``): two kv tiles per main-kernel CTA.

CPU / IR tests:

* the plan builds the pairing: it sets the spec switch ``lb_mirror``, the
  runtime ``pair = 1`` and the grid ``ceil(n_kv_tiles / 2)``, and still
  refuses an unmasked request and an unpaired grid that does not overfill the
  device (more CTAs than ``num_cus``);
  the switch is a salted bool and is the only spec change;
* a ``lb_mirror`` kernel reads the ``pair`` kernarg, runs the second kv tile
  inside one CTA-uniform branch on ``n_pass == 2`` that starts with a full
  barrier and repeats the first tile's q loops and epilogue under loop names
  of its own; a kernel without the switch does not read ``pair``;
* the registry and the sweep: ``lb_order = "mirror"`` is built, ``lb_mirror``
  is a plan-owned axis, the sweep's candidate spec sets it, and stage R pairs
  only masked slices, with the device CU count as the request hint.

The decode itself (every kv tile once, odd and even tile counts, pairing
before the order) is covered by ``test_attention_bwd_worklist``; device runs
are in ``test_attention_bwd_sched_numeric``; the resource gate of the paired
kernels against their one-tile twins is :func:`test_mirror_resource_gate`.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower

from kernels.common.attention_bwd import (
    AttnBwdSpec,
    attn_bwd_arch_facts,
    build_attn_bwd_main,
    validate_attn_bwd_spec,
)
from kernels.common.attention_bwd_plan import (
    LB_ORDERS,
    MIRROR_PAIRING_BUILT,
    AttnBwdPolicy,
    AttnBwdRequest,
    FEATURE_GROUPS,
    attn_bwd_plan,
)

from ._attention_bwd_test_utils import distinct_configs
from .sdpa.bwd_kernel_cases import config_knobs

ARCHS = ("gfx942", "gfx950", "gfx1151", "gfx1201")


def _policy(**tuning):
    return AttnBwdPolicy(
        declared_groups=frozenset(FEATURE_GROUPS),
        waive_unverified=True,
        tuning=lambda *a: dict(tuning),
    )


def _req(**kw):
    base = dict(b=2, h_q=8, h_k=8, h_v=8, s_q=4096, s_kv=4096, d_qk=128, d_v=128)
    base.update(kw)
    return AttnBwdRequest(dtype="bf16", **base)


def test_mirror_is_built_and_planned():
    assert MIRROR_PAIRING_BUILT
    pol = _policy(lb_order="mirror")
    for arch in ("gfx942", "gfx950"):
        plan = attn_bwd_plan(_req(causal=True, num_cus=8), arch, policy=pol)
        spec = plan.specs["main"]
        assert spec.lb_mirror is True
        main = plan.launches[1]
        assert main.scalars["pair"] == 1
        assert main.scalars["lb_order"] == LB_ORDERS["mirror"]
        n = math.ceil(4096 / spec.block_n)
        assert main.scalars["n_kv_tiles"] == n
        assert main.grid[0] == math.ceil(n / 2)
        nat = attn_bwd_plan(_req(causal=True, num_cus=8), arch, policy=_policy())
        assert nat.specs["main"].lb_mirror is False
        assert dataclasses.replace(spec, lb_mirror=False) == nat.specs["main"]
        assert nat.launches[1].scalars["pair"] == 0


def test_mirror_refusals():
    pol = _policy(lb_order="mirror")
    with pytest.raises(ValueError, match="mask"):
        attn_bwd_plan(_req(num_cus=8), "gfx942", policy=pol)
    with pytest.raises(ValueError, match="num_cus"):
        attn_bwd_plan(_req(causal=True), "gfx942", policy=pol)
    small = _req(causal=True, b=1, h_q=1, h_k=1, h_v=1, s_q=256, s_kv=256)
    with pytest.raises(ValueError, match="fills"):
        attn_bwd_plan(dataclasses.replace(small, num_cus=1024), "gfx942", policy=pol)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_mirror_fill_guard_is_the_unpaired_grid(arch):
    """Pairing is refused exactly when the unpaired grid runs in one wave
    (at most ``num_cus`` CTAs) and accepted above it, also where the paired
    grid itself is under ``num_cus`` (a multi-wave unpaired grid)."""
    pol = _policy(lb_order="mirror")
    for kw in (
        dict(b=8, h_q=32, h_k=8, h_v=8, s_q=512, s_kv=512),
        dict(b=2, h_q=8, h_k=8, h_v=8, s_q=1536, s_kv=1536),  # odd tile count
        dict(b=1, h_q=4, h_k=4, h_v=4, s_q=1024, s_kv=4096, causal_bottom_right=True),
    ):
        mask = {} if "causal_bottom_right" in kw else {"causal": True}
        req = _req(**{**kw, **mask})
        plan = attn_bwd_plan(dataclasses.replace(req, num_cus=1), arch, policy=pol)
        main = plan.launches[1]
        n = main.scalars["n_kv_tiles"]
        rest = main.grid[1] * main.grid[2]
        unpaired, paired = n * rest, main.grid[0] * rest
        assert main.grid[0] == math.ceil(n / 2) and paired < unpaired
        for cus in (unpaired, unpaired + 1, 4 * unpaired):
            with pytest.raises(ValueError, match="overfills"):
                attn_bwd_plan(dataclasses.replace(req, num_cus=cus), arch, policy=pol)
        for cus in (unpaired - 1, paired + 1, paired, 1):
            p = attn_bwd_plan(dataclasses.replace(req, num_cus=cus), arch, policy=pol)
            assert p.launches == plan.launches and p.specs == plan.specs


def test_lb_mirror_is_a_salted_bool():
    base = AttnBwdSpec(head_size=64)
    on = AttnBwdSpec(head_size=64, lb_mirror=True)
    assert on.kernel_name("main") != base.kernel_name("main")
    assert ("lb_mirror", True) in on.non_default_fields()
    with pytest.raises(ValueError, match="lb_mirror"):
        AttnBwdSpec(head_size=64, lb_mirror=1)


def _walk(ops, anc=()):
    for op in ops:
        yield op, anc
        for r in op.regions:
            yield from _walk(r.ops, anc + (op,))


MIRROR_VARIANTS = {
    "band": dict(mask_class="band"),
    "band_edge": dict(mask_class="band", edge_tiles=True),
    "band_edge_hout": dict(mask_class="band", edge_tiles=True, step_order="head_outer"),
    "band_atomic_pack": dict(mask_class="band", dkv_mode="atomic", head_pack=True),
    "thd_band_slab": dict(mask_class="band", seq_mode="thd", dkv_slab_hoist=True),
    "band_xcd": dict(mask_class="band", xcd_remap=True),
}
ROWS = [
    (arch, cfg, d, var)
    for arch in ARCHS
    for cfg in distinct_configs(arch)
    for d in (32, 64, 128)
    for var in MIRROR_VARIANTS
]


def _spec(arch, cfg, d, var, **kw):
    return AttnBwdSpec(
        head_size=d, dtype="fp16", **config_knobs(cfg, arch), **MIRROR_VARIANTS[var],
        **kw,
    )  # fmt: skip


def _legal(spec, arch):
    try:
        validate_attn_bwd_spec(spec, arch)
    except ValueError:
        return False
    return True


@pytest.mark.parametrize("row", ROWS, ids=lambda r: "-".join(map(str, r)))
def test_mirror_kernel_runs_the_second_tile(row):
    arch, cfg, d, var = row
    spec = _spec(arch, cfg, d, var, lb_mirror=True)
    if not _legal(spec, arch):
        pytest.skip("not legal on this target (or refused: RESOURCE_REFUSED)")
    kernel = build_attn_bwd_main(spec, arch=arch)
    top = kernel.body.ops
    ifs = [op for op in top if op.name == "scf.if"]
    second = []
    for op in ifs:
        ops = op.regions[0].ops
        if ops and ops[0].name == "tile.sync":
            second.append(op)
    assert len(second) == 1, [op.name for op in top]
    # the condition: n_pass == 2 of the paired decode
    cond = second[0].operands[0]
    assert cond.op.name == "arith.cmp"
    inner = [o for o, _ in _walk(second[0].regions[0].ops)]
    loops = [o for o in inner if o.name == "scf.for"]
    assert loops and all(
        "m1_" in o.attrs["iv"] for o in loops if "bwd" in o.attrs["iv"]
    )
    assert any(o.name == "tile.mma" for o in inner)
    # the same q loops and epilogue as the first tile
    outer = [o for o, _ in _walk(top) if not any(o is x for x in inner)]
    n_mma = sum(o.name == "tile.mma" for o in inner)
    assert n_mma == sum(o.name == "tile.mma" for o in outer)
    n_atom = sum(o.name.startswith("memref.global_atomic") for o in inner)
    assert n_atom == sum(o.name.startswith("memref.global_atomic") for o in outer)
    ir = lower(kernel, arch=arch, llvm_flavor="llvm22")
    assert _reads(ir, "%pair")
    plain_spec = dataclasses.replace(spec, lb_mirror=False)
    if _legal(plain_spec, arch):  # (no kernel for a recorded gate failure)
        plain = build_attn_bwd_main(plain_spec, arch=arch)
        assert not _reads(lower(plain, arch=arch, llvm_flavor="llvm22"), "%pair")


@pytest.mark.parametrize("row", ROWS, ids=lambda r: "-".join(map(str, r)))
def test_mirror_lds_barriers_separate_cross_wave_accesses(row):
    """The LDS hazard model of ``test_attention_bwd_ir`` on each kv tile of a
    paired CTA: the first tile's code, and the second tile's code, which
    starts with a full barrier (the model does not count barriers inside a
    branch, so the CTA-uniform second-tile branch is checked as its own
    program after that barrier)."""
    from types import SimpleNamespace

    from . import test_attention_bwd_ir as T

    arch, cfg, d, var = row
    spec = _spec(arch, cfg, d, var, lb_mirror=True)
    if not _legal(spec, arch):
        pytest.skip("not legal on this target (or refused: RESOURCE_REFUSED)")
    kernel, tags = T.traced_spec(spec, arch)
    top = kernel.body.ops
    k = next(
        i
        for i, op in enumerate(top)
        if op.name == "scf.if"
        and op.regions[0].ops
        and op.regions[0].ops[0].name == "tile.sync"
    )
    for ops in (top[:k], top[k].regions[0].ops):
        part = SimpleNamespace(body=SimpleNamespace(ops=ops))
        assert not T.lds_hazards(part, tags)[:4]


def _reads(ir: str, name: str) -> bool:
    """Whether a kernarg is used in the body (not only declared)."""
    body = ir.split("{", 1)[1]
    return any(name + "," in ln or name + ")" in ln or ln.rstrip().endswith(name)
               for ln in body.splitlines())  # fmt: skip


def test_registry_and_sweep():
    from benchmarks.common import attention_bwd_rocke_arm as arm
    from benchmarks.common import attention_bwd_sweep as sweep
    from dispatch.attention_bwd.tuning_common import backward_axis

    lb = backward_axis("lb_order")
    assert "mirror" in lb.choices and "mirror" not in lb.not_built
    ax = backward_axis("lb_mirror")
    assert ax.plan_owned and ax.family == 7 and ax.default is False
    assert sweep.DEVICE_NUM_CUS == arm.DEVICE_NUM_CUS
    assert "num_cus" in arm.REQUEST_HINTS
    sl = sweep.SliceKey("gfx942", "general", 64)
    c = sweep.Candidate.make(sl, {}, {"lb_order": "mirror"})
    assert c.spec().lb_mirror is True
    assert (
        sweep.Candidate.make(sl, {}, {"lb_order": "reverse"}).spec().lb_mirror is False
    )
    rel = sweep.Release.parse("all", "production")
    vals = sweep._runtime_values(sl, [], rel)
    mirror = [v for v in vals if v.get("lb_order") == "mirror"]
    assert mirror and all(v["num_cus"] == sweep.DEVICE_NUM_CUS for v in mirror)
    none = dataclasses.replace(sl, mask_class="none")
    assert not [v for v in sweep._runtime_values(none, [], rel)
                if v.get("lb_order") == "mirror"]  # fmt: skip


# A paired CTA needs the second tile's kernarg-derived scalars after the first
# tile; where they exceed the SGPR file the backend spills SGPRs to VGPR lanes.
# Those paired specs fail the resource gate on the production compiler and are
# refused by the validator (RESOURCE_REFUSED, pinned by
# test_attention_bwd_refusals.py); the gate rows are the legal ones.
MIRROR_GATE_ROWS = [
    r
    for r in ROWS
    if r[0] in ("gfx942", "gfx950")
    and r[3] in ("band", "band_edge", "band_atomic_pack", "thd_band_slab")
    and _legal(_spec(*r, lb_mirror=True), r[0])
]


def test_mirror_gate_rows_remain():
    """The refusals leave paired gate rows on both CDNA targets."""
    for arch in ("gfx942", "gfx950"):
        assert any(r[0] == arch for r in MIRROR_GATE_ROWS), arch


@pytest.mark.parametrize("row", MIRROR_GATE_ROWS, ids=lambda r: "-".join(map(str, r)))
def test_mirror_resource_gate(row):
    """The resource gate (production comgr) passes for the paired kernel and
    its one-tile twin: no failure and no crossed SGPR watch level."""
    pytest.importorskip("rocke.runtime.comgr")
    from ._attention_bwd_resources import (
        compile_or_skip,
        disassemble,
        evaluate_resource_gate,
        find_objdump,
        pin_production_comgr,
    )

    pin_production_comgr()  # torch's bundled comgr, by path; torch not imported
    if find_objdump() is None:
        pytest.skip("llvm-objdump not found")
    arch, cfg, d, var = row
    spec = _spec(arch, cfg, d, var, lb_mirror=True)
    facts = attn_bwd_arch_facts(arch)
    for s in (spec, dataclasses.replace(spec, lb_mirror=False)):
        r = validate_attn_bwd_spec(s, arch)
        art = compile_or_skip(
            build_attn_bwd_main(s, arch=arch), arch=arch, capture_ir_text=False,
            backend="python",
        )  # fmt: skip
        isa = disassemble(art.hsaco, arch=arch)
        res = evaluate_resource_gate(
            art.hsaco, arch=arch, lds_capacity_bytes=facts.lds_capacity_bytes, spec=r,
            isa_text=isa,
        )  # fmt: skip
        if not res.valid:
            pytest.skip(
                f"not the production comgr / flavor ({res.flavor}, {res.comgr})"
            )
        assert not res.failures and not res.warnings, (s.lb_mirror, res.record())


def _device_plan_refusal(case, arch, cfg, overrides, request=None):
    """The validator's refusal of the main kernel a device test launches (the
    harness plan, then the build's check on the device arch), or ``None``."""
    from ._attention_bwd_harness import plan_case

    try:
        plan = plan_case(
            case, arch=arch, config=cfg, overrides=overrides, request=request
        )
        validate_attn_bwd_spec(plan.specs["main"], arch)
    except ValueError as exc:
        return exc, None
    return None, plan


@pytest.mark.parametrize("arch", ARCHS)
def test_device_mirror_refusals_are_exactly_the_recorded_cells(arch):
    """``MIRROR_REFUSED`` of the device test is exactly the set of cells whose
    paired kernel the validator refuses (a resource refusal), a cell whose
    paired kernel is legal has a legal twin, and every variant but counted
    waits with edge tiles keeps a cell."""
    from . import test_attention_bwd_sched_numeric as S
    from .sdpa.bwd_kernel_cases import selected_configs

    refused, ran = {}, set()
    for c in S.CASES:
        for var, (extra, req) in S.MIRROR_VARIANTS.items():
            if var in S._MIRROR_MFMA_ONLY and arch not in S._MFMA:
                continue
            for cfg in selected_configs(arch):
                ov = {"lb_order": "mirror", **extra}
                rq = {"num_cus": 1, **req}
                if ov.get("xcd_chunk") == "gx":
                    exc, plan = _device_plan_refusal(c, arch, cfg, ov, rq)
                    if plan is not None:
                        ov["xcd_chunk"] = plan.launches[1].grid[0]
                exc, _plan = _device_plan_refusal(c, arch, cfg, ov, rq)
                if exc is not None:
                    if "RESOURCE_REFUSED" in str(exc):
                        refused[(arch, c.id, var, cfg)] = str(exc)
                    else:  # a lever the configuration cannot take
                        assert var in S._MIRROR_COMPOSE, (arch, c.id, var, cfg, exc)
                    continue
                twin = {k: v for k, v in extra.items() if k != "xcd_chunk"}
                texc, _ = _device_plan_refusal(c, arch, cfg, twin or None)
                assert texc is None, (arch, c.id, var, cfg, texc)
                ran.add(var)
    want = {k: v for k, v in S.MIRROR_REFUSED.items() if k[0] == arch}
    assert set(refused) == set(want)
    for k, why in want.items():
        assert f"({why}:" in refused[k], (k, refused[k])
    variants = {
        v for v in S.MIRROR_VARIANTS if arch in S._MFMA or v not in S._MIRROR_MFMA_ONLY
    }
    # counted waits with edge tiles: refused on every cell (every tile)
    assert "mirror_counted_edge" not in ran
    keep = variants - set(S._MIRROR_COMPOSE) - {"mirror_counted_edge"}
    assert ran >= keep, keep - ran


@pytest.mark.parametrize("arch", ("gfx942", "gfx950"))
def test_device_counted_refusals_are_exactly_the_recorded_cells(arch):
    """``COUNTED_REFUSED`` of the device test is exactly the set of cells whose
    counted kernel or full twin the validator refuses, and every counted
    variant keeps a cell."""
    from . import test_attention_bwd_sched_numeric as S
    from .sdpa.bwd_kernel_cases import selected_configs

    refused, ran = {}, set()
    for c in list(S.CASES) + [S.PACK_CASE]:
        for var, kw in S.COUNTED_VARIANTS.items():
            if "g_split" in kw and (c.h_q // c.h_k) % 2:
                continue
            for cfg in selected_configs(arch):
                full_kw = {k: v for k, v in kw.items() if k != "counted_issue"}
                exc, _ = _device_plan_refusal(c, arch, cfg, full_kw or None)
                which = "twin"
                if exc is None:
                    which = "counted"
                    exc, _ = _device_plan_refusal(
                        c, arch, cfg, {"loop_sync": "counted", **kw}
                    )
                if exc is None:
                    ran.add(var)
                    continue
                if "RESOURCE_REFUSED" in str(exc):
                    refused[(arch, c.id, var, cfg)] = (which, str(exc))
    want = {k: v for k, v in S.COUNTED_REFUSED.items() if k[0] == arch}
    assert set(refused) == set(want)
    for k, why in want.items():
        which, rule = why.split(": ", 1)
        assert refused[k][0] == which and f"({rule}:" in refused[k][1], (k, refused[k])
    assert ran == set(S.COUNTED_VARIANTS), set(S.COUNTED_VARIANTS) - ran
