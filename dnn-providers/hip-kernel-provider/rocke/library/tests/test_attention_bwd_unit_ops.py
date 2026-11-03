# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Value-neutral work the backward main kernel does not emit (CPU, IR only).

* Multipliers that are exactly 1.0 under the compiled scale placement
  (``static_scale``) are not applied: their kernargs stay in the ABI but no
  op reads them; every other multiplier is applied, and ``"runtime"``
  applies all three.
* Interior q steps (``edge_tiles``: every row of the step is ``< len_q``) do
  not clamp rows to ``len_q - 1`` in the Q / dO staging or the dQ atomics:
  no op of the interior q loop reads the q-side clamp bound, while the edge
  loops still do. ``interior_clamp`` keeps those clamps (a register
  allocation lever; inert without edge tiles).

The values are unchanged in both cases (``x * 1.0 == x``; ``min(q, last) ==
q`` for ``q <= last``); the device tests compare the edge-tile and the
single-loop kernels bit for bit.
"""

from __future__ import annotations

import re
from dataclasses import replace

import pytest
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower

from kernels.common import _attention_bwd_addr as addr_mod
from kernels.common.attention_bwd import (
    INERT_KNOBS,
    STATIC_SCALES,
    AttnBwdSpec,
    build_attn_bwd_main,
    static_unit_mults,
    validate_attn_bwd_spec,
)

from .sdpa.bwd_kernel_cases import config_knobs
from .test_attention_bwd_ir import _walk, q_loops

ARCHS = ("gfx942", "gfx950", "gfx1151", "gfx1201")
MULTS = ("ds_mult", "dk_mult", "dq_mult")


def _spec(arch, cfg="default", **kw):
    knobs = config_knobs(cfg, arch)
    if knobs is None:
        pytest.skip(f"configuration {cfg} is not defined on {arch}")
    return AttnBwdSpec(head_size=64, dtype="bf16", **{**knobs, **kw})


def _operand_names(kernel) -> set[str]:
    return {o.name for op, _ in _walk(kernel.body.ops) for o in op.operands}


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("dkv_mode", ("direct", "atomic"))
@pytest.mark.parametrize("static_scale", STATIC_SCALES)
def test_unit_multipliers_are_not_read(arch, dkv_mode, static_scale):
    spec = _spec(arch, dkv_mode=dkv_mode, static_scale=static_scale)
    used = _operand_names(build_attn_bwd_main(spec, arch=arch))
    unit = static_unit_mults(static_scale, dkv_mode)
    assert {m for m in MULTS if f"%{m}" in used} == set(MULTS) - unit
    if static_scale == "runtime":
        assert not unit


def test_unit_multiplier_table():
    # the placements of attention_bwd_plan._scale_mults, by kernarg
    assert static_unit_mults("fold_ds", "direct") == {"dk_mult", "dq_mult"}
    assert static_unit_mults("fold_ds", "atomic") == {"dk_mult", "dq_mult"}
    assert static_unit_mults("convert", "direct") == {"ds_mult", "dq_mult"}
    assert static_unit_mults("convert", "atomic") == set(MULTS)
    assert static_unit_mults("dq_before_atomic", "direct") == {"ds_mult"}
    assert static_unit_mults("dq_before_atomic", "atomic") == {"ds_mult", "dk_mult"}
    with pytest.raises(ValueError):
        static_unit_mults("fold", "direct")


EDGE_VARIANTS = {
    "band": dict(mask_class="band"),
    "none": dict(mask_class="none"),
    "band_atomic": dict(mask_class="band", dkv_mode="atomic"),
    "thd_band": dict(mask_class="band", seq_mode="thd"),
    "band_pack": dict(mask_class="band", head_pack=True),
    "thd_band_token_major": dict(
        mask_class="band", seq_mode="thd", ws_layout="token_major"
    ),
}


def _q_bound_reads(arch, spec, monkeypatch):
    """Ops of each q loop that read the q-side clamp bound ``len_q - 1``.

    Built without the main kernel's recorded resource refusals (the probe
    checks the emitted clamps, not the code object; every static rule still
    applies)."""
    from kernels.common.attention_bwd import _emit_main

    bounds = []
    real = addr_mod.last_row

    def last_row(b, length):
        out = real(b, length)
        bounds.append(out)
        return out

    monkeypatch.setattr(addr_mod, "last_row", last_row)
    kernel = _emit_main(spec, arch, main_kernel_refusals=False)
    assert len(bounds) == 2  # the q side, then the kv side
    q_last = bounds[0]
    return [
        sum(any(o is q_last for o in op.operands) for op, _ in _walk(lp.regions[0].ops))
        for lp in q_loops(kernel)
    ]


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("cfg", ("debug", "default"))
@pytest.mark.parametrize("var", sorted(EDGE_VARIANTS))
def test_interior_q_steps_do_not_clamp_rows(arch, cfg, var, monkeypatch):
    spec = _spec(arch, cfg, edge_tiles=True, **EDGE_VARIANTS[var])
    edge0, interior, edge1 = _q_bound_reads(arch, spec, monkeypatch)
    assert interior == 0
    assert edge0 > 0 and edge1 > 0  # the probe sees them


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("var", sorted(EDGE_VARIANTS))
def test_interior_clamp_keeps_the_interior_row_clamps(arch, var, monkeypatch):
    """``interior_clamp``: the interior loop clamps its rows like the edge
    loops (as many reads of the q-side clamp bound as each edge loop: the Q /
    dO staging, packed staging and dQ sink clamps); the edge loops are
    unchanged; without edge tiles the knob is inert (``INERT_KNOBS``, and the
    same IR apart from the kernel name)."""
    spec = _spec(arch, "default", edge_tiles=True, **EDGE_VARIANTS[var])
    clamped = _q_bound_reads(arch, replace(spec, interior_clamp=True), monkeypatch)
    edge0, interior, edge1 = clamped
    assert interior > 0 and edge0 > 0 and edge1 > 0
    assert interior == edge0 == edge1, clamped
    # the edge loops are not touched by the knob
    plain = _q_bound_reads(arch, spec, monkeypatch)
    assert (plain[0], plain[2]) == (edge0, edge1) and plain[1] == 0
    inert = INERT_KNOBS["interior_clamp"]
    assert not inert(validate_attn_bwd_spec(spec, arch), arch)
    base = _spec(arch, "default", **EDGE_VARIANTS[var])
    on = replace(base, interior_clamp=True)
    assert inert(validate_attn_bwd_spec(on, arch), arch)
    assert on.kernel_name("main") != base.kernel_name("main")
    ir = [
        re.sub(
            r"rocke_attn_bwd_main_\w+",
            "K",
            lower(build_attn_bwd_main(s, arch=arch), arch=arch, llvm_flavor="llvm22"),
        )
        for s in (base, on)
    ]
    assert ir[0] == ir[1]
