# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Structural test: the warp-tiled GDN/KDA decode issues every global load
before any floating-point math (ALGORITHM.md §4.3, "Why loads first").

The golden hashes only say that the IR moved; the numeric tests pass in either
order. This test pins the order itself: inside the ``active`` region, in
program order, the last global load must come before the first float op. An
emitter that loads a tensor after the math has started (the interleaved order
the simple path keeps) fails it.

Single-wave ``fuse_conv`` is exempt by design where ``late_row_inputs`` says
so: one wave owns all ``DV`` rows, and hoisting their inputs as well would
spill, so it loads each row's ``v`` conv inputs, the state and the norm
inputs where the parent order did (ALGORITHM.md §4.3, "One exception").
With ``conv_once`` at ``DK = DV = 128``, ``WTK`` 4 or 8 and no
``waves_per_eu`` floor it loads first like every other tile. The predicate
and the order on both sides of it are pinned below; a separate test pins that
the exemption is still in force at the knob-free exempt tile: some loads come
before the first float op and some after it. The no-new-spill check of the
tiles that load first and the exempt tile's zero-scratch contract are in
``test_gdn_decode_fused.py``.

Reachability: the registered GDN tiles come from dispatch. The KDA tuned tiles
at every state dtype, the raw-gate and no-L2-norm variants, the fused modes
and the knobs are built as specs directly here. Dispatch routes f32 state and
the fused modes (``tests/dispatch/gdn/test_gfx950_wiring.py``,
``TestStateDtypeDispatch`` and ``TestFusedDispatch``) but sets no knob, so the
knob cases guard direct ``build_gdn_decode`` callers and the paths dispatch
will route later.

Builds IR only: no lowering, no GPU, no comgr.
"""

from __future__ import annotations

import dataclasses as dc
import sys
from pathlib import Path

import pytest

_LIB_ROOT = str(Path(__file__).resolve().parent.parent)
if sys.path and sys.path[0] != _LIB_ROOT:
    sys.path.insert(0, _LIB_ROOT)

from dispatch.gdn import GdnDecodeRequest, dispatch_gdn_decode_all  # noqa: E402
from dispatch.gdn.gfx950 import (  # noqa: E402
    _TUNED_TILES_KDA,
    FUSED_DEFAULT_TILE,
    KDA_FUSED_SPEC_ID,
)
from kernels.gfx950.gdn_decode import (  # noqa: E402
    GdnDecodeSpec,
    build_gdn_decode,
    is_valid_spec,
    late_row_inputs,
)

_ARCH = "gfx950"
_FLOAT_OPS = {
    "arith.fadd",
    "arith.fsub",
    "arith.fmul",
    "arith.fma",
    "arith.fneg",
    "arith.fmin",
    "arith.fmax",
    "arith.fcmp",
}


def _is_global_load(name: str) -> bool:
    return "global_load" in name or "buffer_load" in name


def _is_float_math(name: str) -> bool:
    return name in _FLOAT_OPS or name.startswith("math.")


def _flatten(ops):
    for op in ops:
        yield op
        for region in op.regions:
            yield from _flatten(region.ops)


def _active_region_ops(kernel):
    """Ops of the ``scf_if(active)`` body, nested regions inlined in order."""
    guards = [op for op in kernel.body.ops if op.name == "scf.if"]
    assert len(guards) == 1, [op.name for op in kernel.body.ops]
    return [op.name for op in _flatten(guards[0].regions[0].ops)]


# Every tile knob on that applies to the spec it is added to; the knob code
# must keep the loads-first order too.
_TILE_KNOBS = dict(
    state_load_hint="streaming",
    state_store_hint="streaming",
    dpp_reduce=True,
    xcd_remap=True,
    stream_rows=True,
    interleave_cols=True,
    waves_per_eu=4,
)


def _specs():
    """(checked, exempt): specs that must load first, and the exempt tiles."""
    out, exempt = {}, {}
    for result in dispatch_gdn_decode_all(GdnDecodeRequest(batch=16, arch=_ARCH)):
        out[f"registered_{result.candidate.spec_id}"] = result.spec
    base = GdnDecodeSpec()
    kda_tiles = [(tile, spec_id) for _, tile, spec_id in _TUNED_TILES_KDA]
    kda_tiles.append((FUSED_DEFAULT_TILE, KDA_FUSED_SPEC_ID))
    for tile, spec_id in kda_tiles:
        for state_dtype in ("bf16", "f16", "f32"):
            spec = dc.replace(
                base,
                gate_kind="kda",
                state_dtype=state_dtype,
                num_warps=tile[0],
                warp_threads_k=tile[1],
                blocks_per_v_dim=tile[2],
            )
            out[f"tuned_{spec_id}_st{state_dtype}"] = spec
            out[f"tuned_{spec_id}_st{state_dtype}_knobs"] = dc.replace(
                spec, **_TILE_KNOBS
            )
    out["kda_raw_gate"] = dc.replace(base, gate_kind="kda", fuse_gate=False)
    out["no_l2norm"] = dc.replace(base, use_qk_l2norm=False)
    for gate in ("gdn", "kda"):
        for state_dtype in ("bf16", "f16", "f32"):
            for conv, norm in ((True, False), (False, True), (True, True)):
                for nw in (1, 4):
                    tag = ("_cv" if conv else "") + ("_rn" if norm else "")
                    spec = dc.replace(
                        base,
                        num_k_heads=16,
                        num_v_heads=16,
                        gate_kind=gate,
                        state_dtype=state_dtype,
                        num_warps=nw,
                        warp_threads_k=16,
                        blocks_per_v_dim=1,
                        fuse_conv=conv,
                        fuse_out_norm=norm,
                    )
                    case = f"fused_{gate}_st{state_dtype}{tag}_w{nw}"
                    if conv and nw == 1:
                        # the exempt tile (late_row_inputs; module docstring)
                        # has its own test
                        exempt[case] = spec
                        continue
                    out[case] = spec
                    # the fused knobs with every lane loading its own norm
                    # gate, then with the once-per-row LDS products
                    lane = dc.replace(spec, conv_once=conv, out_lds=norm, **_TILE_KNOBS)
                    out[f"{case}_knobs"] = lane
                    if conv and norm:
                        out[f"{case}_knobs_ngo"] = dc.replace(lane, norm_gate_once=True)
    # single-wave fuse_conv where late_row_inputs lets it load first
    one_wave_knobs = dict(_TILE_KNOBS, waves_per_eu=0)
    for gate in ("gdn", "kda"):
        for state_dtype in ("bf16", "f16", "f32"):
            for norm in (False, True):
                for wtk in (4, 8):
                    spec = dc.replace(
                        base,
                        num_k_heads=16,
                        num_v_heads=16,
                        gate_kind=gate,
                        state_dtype=state_dtype,
                        num_warps=1,
                        warp_threads_k=wtk,
                        blocks_per_v_dim=1,
                        fuse_conv=True,
                        fuse_out_norm=norm,
                        conv_once=True,
                    )
                    case = f"fused_{gate}_st{state_dtype}_cv{'_rn' if norm else ''}_w1k{wtk}_co"
                    out[case] = spec
                    knobs = dc.replace(spec, out_lds=norm, **one_wave_knobs)
                    out[f"{case}_knobs"] = knobs
                    if norm:
                        out[f"{case}_knobs_ngo"] = dc.replace(
                            knobs, norm_gate_once=True
                        )
    return out, exempt


_SPECS, _EXEMPT_SPECS = _specs()


def _load_and_math_indices(spec):
    names = _active_region_ops(build_gdn_decode(spec, arch=_ARCH))
    loads = [i for i, n in enumerate(names) if _is_global_load(n)]
    maths = [i for i, n in enumerate(names) if _is_float_math(n)]
    return names, loads, maths


def test_case_sets_are_populated():
    """An empty dispatch result would silently drop every registered case."""
    assert any(c.startswith("registered_") for c in _SPECS)
    assert len(_EXEMPT_SPECS) == 2 * 3 * 2  # gate x state dtype x norm


def _late_loads(spec):
    """Global loads issued after the first float op, and that op's index."""
    names, loads, maths = _load_and_math_indices(spec)
    assert loads and maths, spec.kernel_name()
    return [
        names[i] for i in loads if i > maths[0]
    ], f"{names[maths[0]]} at op {maths[0]}"


@pytest.mark.parametrize("case", sorted(_SPECS))
def test_every_global_load_precedes_the_first_float_op(case):
    assert not late_row_inputs(_SPECS[case]), case
    late, first_math = _late_loads(_SPECS[case])
    assert not late, (
        f"{case}: {len(late)} global load(s) after the first float op "
        f"({first_math}): {late[:4]}"
    )


@pytest.mark.parametrize("case", sorted(_EXEMPT_SPECS))
def test_exempt_tile_loads_some_inputs_first_and_rows_late(case):
    """The exempt tile issues some loads before the first float op and the
    per-row ones after it. If every load moved first, the exemption is gone
    and the tile would spill at ``(1,16,1)`` (ALGORITHM.md §4.3)."""
    assert late_row_inputs(_EXEMPT_SPECS[case]), case
    _, loads, maths = _load_and_math_indices(_EXEMPT_SPECS[case])
    assert maths, case
    early = [i for i in loads if i < maths[0]]
    late = [i for i in loads if i > maths[0]]
    assert early, f"{case}: no global load before the first float op"
    assert late, f"{case}: exemption not in force, every load comes first"


# late_row_inputs at the single-wave fuse_conv tile and its neighbours:
# (overrides of _ONE_WAVE, keeps the late order). Single-wave conv_once loads
# first only at DK = DV = 128, WTK 4 or 8 and no waves_per_eu floor; every
# other single-wave fuse_conv spec keeps the exception.
_ONE_WAVE = dict(
    num_k_heads=16,
    num_v_heads=16,
    gate_kind="kda",
    state_dtype="f32",
    num_warps=1,
    warp_threads_k=8,
    blocks_per_v_dim=1,
    fuse_conv=True,
    fuse_out_norm=True,
    conv_once=True,
)
_PREDICATE_CASES = {
    "co_wtk8": ({}, False),
    "co_wtk4": (dict(warp_threads_k=4), False),
    "co_wtk8_gdn_bf16": (dict(gate_kind="gdn", state_dtype="bf16"), False),
    "co_wtk4_conv_only": (dict(warp_threads_k=4, fuse_out_norm=False), False),
    "co_wtk1": (dict(warp_threads_k=1), True),
    "co_wtk2": (dict(warp_threads_k=2), True),
    "co_wtk16": (dict(warp_threads_k=16), True),
    "co_wtk8_wpe1": (dict(waves_per_eu=1), True),
    "co_wtk8_wpe4": (dict(waves_per_eu=4), True),
    "co_wtk8_d64": (dict(head_k_dim=64, head_v_dim=64), True),
    "co_wtk8_d256": (dict(head_k_dim=256, head_v_dim=256), True),
    "no_co_wtk8": (dict(conv_once=False), True),
    "no_co_wtk4": (dict(conv_once=False, warp_threads_k=4), True),
    "two_waves_no_co": (dict(conv_once=False, num_warps=2), False),
    "norm_only": (dict(conv_once=False, fuse_conv=False), False),
}


@pytest.mark.parametrize("case", sorted(_PREDICATE_CASES))
def test_late_row_inputs_predicate_and_load_order(case):
    overrides, keeps_late = _PREDICATE_CASES[case]
    spec = dc.replace(GdnDecodeSpec(), **dict(_ONE_WAVE, **overrides))
    assert is_valid_spec(spec, arch=_ARCH)[0], is_valid_spec(spec, arch=_ARCH)[1]
    assert late_row_inputs(spec) is keeps_late, case
    late, first_math = _late_loads(spec)
    # the predicate is what the emitter does: late loads exactly where it says
    assert bool(late) is keeps_late, (case, first_math, late[:4])
