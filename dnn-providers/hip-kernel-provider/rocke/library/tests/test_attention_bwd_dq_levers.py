# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""dQ-path levers of the backward main kernel (CPU: spec, IR, plan).

* ``dq_issue``: ``"fragment"`` emits the same ops as ``"block"`` (equal IR
  when a wave owns one G4 fragment) and issues each fragment's atomics before
  the next fragment's MMAs;
* ``ptr_noalias``: exactly the declared pointers carry ``noalias``; ``"all"``
  is refused in direct mode;
* ``q_rotate``: the q loops keep their trip counts and wrap inside their range;
* ``lb_order = "interleave"``: the plan sets ``lb_interleave`` and the
  kernarg value; the decode is checked in ``test_attention_bwd_worklist``.

The device checks (bit-identical dK / dV, tolerance tier for ``q_rotate``) are
in ``test_attention_bwd_numeric``.
"""

from __future__ import annotations

import dataclasses
import re

import pytest
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower

from kernels.common._attention_bwd_body import BwdTileGeometry
from kernels.common.attention_bwd import (
    AttnBwdSpec,
    build_attn_bwd_main,
    validate_attn_bwd_spec,
)
from kernels.common.attention_bwd_plan import AttnBwdRequest, attn_bwd_plan

from ._attention_bwd_test_utils import distinct_configs
from .sdpa.bwd_kernel_cases import config_knobs, config_policy
from .test_attention_bwd_ir import _walk, step_bodies

ARCHS = ("gfx942", "gfx950", "gfx1151", "gfx1201")
WS = {"WS_LSE2", "WS_DSUM", "WS_DQ", "WS_DK", "WS_DV"}


def _spec(arch, cfg, d=64, **kw):
    return AttnBwdSpec(head_size=d, dtype="bf16", **config_knobs(cfg, arch), **kw)


def _ir(spec, arch):
    return lower(build_attn_bwd_main(spec, arch=arch), arch=arch, llvm_flavor="llvm22")


def _unnamed(ir: str, spec, arch) -> str:
    return ir.replace(validate_attn_bwd_spec(spec, arch).kernel_name("main"), "K")


def _g4_frags(spec, arch) -> int:
    g4 = BwdTileGeometry.from_spec(validate_attn_bwd_spec(spec, arch), arch).g4
    return g4.frags_m * g4.frags_n


# ---------------------------------------------------------------------------
# spec values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        dict(dq_issue="row"),
        dict(ptr_noalias=True),
        dict(ptr_noalias=0),
        dict(ptr_noalias="inputs"),
        dict(q_rotate=1),
        dict(lb_interleave=None),
    ],
)
def test_bad_values_raise(kw):
    with pytest.raises(ValueError):
        AttnBwdSpec(head_size=64, **kw)


def test_defaults_are_off_and_unsalted():
    s = AttnBwdSpec(head_size=64)
    assert (s.dq_issue, s.ptr_noalias, s.q_rotate, s.lb_interleave) == (
        "block", False, False, False,
    )  # fmt: skip
    names = {
        AttnBwdSpec(head_size=64, **kw).kernel_name("main")
        for kw in (
            {},
            dict(dq_issue="fragment"),
            dict(ptr_noalias="workspace"),
            dict(q_rotate=True),
            dict(lb_interleave=True),
        )
    }
    assert len(names) == 5  # every non-default value is salted


@pytest.mark.parametrize("arch", ARCHS)
def test_noalias_all_needs_atomic_mode(arch):
    with pytest.raises(ValueError, match="ptr_noalias='all'"):
        validate_attn_bwd_spec(AttnBwdSpec(head_size=64, ptr_noalias="all"), arch)
    validate_attn_bwd_spec(
        AttnBwdSpec(head_size=64, ptr_noalias="all", dkv_mode="atomic"), arch
    )


# ---------------------------------------------------------------------------
# ptr_noalias
# ---------------------------------------------------------------------------


def _noalias_params(ir):
    head = re.search(r"define amdgpu_kernel void @\S+\((.*?)\) #0", ir).group(1)
    ptrs = [p for p in head.split(", ") if p.startswith("ptr")]
    assert len(ptrs) == 16
    return {p.split("%")[-1] for p in ptrs if " noalias " in p}


@pytest.mark.parametrize("arch", ARCHS)
def test_noalias_marks_exactly_the_declared_pointers(arch):
    cfg = distinct_configs(arch)[0]
    for dkv, value, want in (
        ("direct", False, set()),
        ("atomic", False, set()),
        ("direct", "workspace", WS),
        ("atomic", "workspace", WS),
        ("atomic", "all", None),
    ):
        spec = _spec(arch, cfg, dkv_mode=dkv, ptr_noalias=value)
        got = _noalias_params(_ir(spec, arch))
        if want is None:
            assert len(got) == 16
        else:
            assert got == want, (dkv, value, got)
        if value:
            # the attribute is the only difference from the default instance
            base = dataclasses.replace(spec, ptr_noalias=False)
            a = _unnamed(_ir(spec, arch), spec, arch).replace(" noalias", "")
            assert a == _unnamed(_ir(base, arch), base, arch)


# ---------------------------------------------------------------------------
# dq_issue
# ---------------------------------------------------------------------------


def _step_ops(kernel):
    return [[op for op, _ in _walk(ops)] for ops in step_bodies(kernel)]


def _count(ops, name):
    return sum(o.name == name for o in ops)


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("value", ["fragment", "fragment_regs"])
@pytest.mark.parametrize("var", [{}, dict(edge_tiles=True), dict(head_pack=True)])
def test_dq_issue_per_fragment(arch, value, var):
    seen_multi = False
    for cfg in distinct_configs(arch):
        for d in (32, 64, 128):
            blk = _spec(arch, cfg, d, **var)
            frg = dataclasses.replace(blk, dq_issue=value)
            r = validate_attn_bwd_spec(blk, arch)
            g4 = BwdTileGeometry.from_spec(r, arch).g4
            ir_b, ir_f = _ir(blk, arch), _ir(frg, arch)
            if r.acc_in_lds or (value == "fragment" and _g4_frags(blk, arch) == 1):
                # LDS accumulators (always per fragment), or one fragment per
                # wave: the same IR apart from the name
                assert _unnamed(ir_f, frg, arch) == _unnamed(ir_b, blk, arch)
                continue
            kb = build_attn_bwd_main(blk, arch=arch)
            kf = build_attn_bwd_main(frg, arch=arch)
            for ob, of in zip(_step_ops(kb), _step_ops(kf), strict=True):
                for name in ("tile.mma", "memref.global_atomic_add_f32"):
                    assert _count(ob, name) == _count(of, name), name
                if value == "fragment_regs" and g4.frags_m == 1:
                    # dS read once per K step, as in the block order (the
                    # packed sink adds one scalar constant per call)
                    def kinds(ops):
                        return sorted(o.name for o in ops if o.name != "arith.constant")

                    assert kinds(ob) == kinds(of)
                if _g4_frags(blk, arch) == 1:
                    continue
                seen_multi = True
                names = [o.name for o in of]
                g4_last = max(
                    i for i, o in enumerate(of)
                    if o.name == "tile.mma" and o.attrs["op_id"] == g4.op.op_id
                )  # fmt: skip
                first_atomic = names.index("memref.global_atomic_add_f32")
                assert first_atomic < g4_last  # atomics interleave with G4
    if arch in ("gfx942", "gfx950"):
        assert seen_multi  # the debug tile owns several G4 fragments per wave


# ---------------------------------------------------------------------------
# q_rotate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize(
    "var", [{}, dict(edge_tiles=True), dict(head_pack=True), dict(dkv_mode="atomic")]
)
def test_q_rotate_keeps_the_loops(arch, var):
    for cfg in distinct_configs(arch):
        base = _spec(arch, cfg, **var)
        rot = dataclasses.replace(base, q_rotate=True)
        kb = build_attn_bwd_main(base, arch=arch)
        kr = build_attn_bwd_main(rot, arch=arch)
        sb, sr = _step_ops(kb), _step_ops(kr)
        assert len(sb) == len(sr) == (3 if var.get("edge_tiles") else 1)
        for ob, orr in zip(sb, sr):
            nb = sorted(o.name for o in ob)
            nr = sorted(o.name for o in orr)
            # one extra wrap compare + select per loop, nothing else
            extra = list(nr)
            for n in nb:
                extra.remove(n)
            assert sorted(extra) == ["arith.cmp", "arith.select"], extra
        _ir(rot, arch)  # lowers


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("edge", [False, True])
def test_q_rotate_with_the_head_outer_order(arch, edge):
    """``step_order = "head_outer"``: one loop runs each head group's q tiles
    from the rotated start, wrapping at the range end; a group ends when its
    q tiles are back at the start (one compare in place of the range-end
    test). With edge tiles a group loop runs one loop per range (each: the
    wrap compare + select)."""
    for cfg in distinct_configs(arch):
        # d128: the grouped head-outer form at d32 / d64 on the CDNA default
        # tiles is a recorded resource-gate failure (RESOURCE_REFUSED)
        base = _spec(arch, cfg, d=128, step_order="head_outer", edge_tiles=edge)
        rot = dataclasses.replace(base, q_rotate=True)
        sb = _step_ops(build_attn_bwd_main(base, arch=arch))
        sr = _step_ops(build_attn_bwd_main(rot, arch=arch))
        assert len(sb) == len(sr) == (3 if edge else 1)
        want = ["arith.cmp", "arith.select"] if edge else ["arith.cmp"]
        for ob, orr in zip(sb, sr):
            extra = sorted(o.name for o in orr)
            for n in sorted(o.name for o in ob):
                extra.remove(n)
            assert sorted(extra) == want, extra
        _ir(rot, arch)  # lowers


# ---------------------------------------------------------------------------
# lb_order = interleave (plan)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ("gfx942", "gfx950"))
def test_plan_interleave_sets_the_decode_switch(arch):
    req = AttnBwdRequest(
        b=2, h_q=4, h_k=4, h_v=4, s_q=300, s_kv=300, d_qk=64, d_v=64, causal=True
    )
    nat = attn_bwd_plan(req, arch, policy=config_policy("default", arch))
    il = attn_bwd_plan(
        req, arch, policy=config_policy("default", arch, {"lb_order": "interleave"})
    )
    assert not nat.specs["main"].lb_interleave
    assert il.specs["main"].lb_interleave
    assert il.runtime["lb_order"] == "interleave"
    assert il.launches[1].scalars["lb_order"] == 3
    assert il.launches[1].kernel != nat.launches[1].kernel
    assert il.launches[1].grid == nat.launches[1].grid
    assert (
        dataclasses.replace(il.specs["main"], lb_interleave=False) == nat.specs["main"]
    )
