# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Structural test: the warp-tiled GDN/KDA decode issues every global load
before any floating-point math (ALGORITHM.md §4.3, "Why loads first").

The golden hashes only say that the IR moved; the numeric tests pass in either
order. This test pins the order itself: inside the ``active`` region, in
program order, the last global load must come before the first float op. An
emitter that loads a tensor after the math has started (the interleaved order
the simple path keeps) fails it.

One tile is exempt by design: single-wave ``fuse_conv`` owns all ``DV`` rows,
and hoisting their inputs as well would spill, so it loads each row's ``v``
conv inputs, the state and the norm inputs where the parent order did
(ALGORITHM.md §4.3, "One exception"). A separate test pins that the exemption
is still in force there: some loads come before the first float op and some
after it. Its zero-scratch contract is pinned in ``test_gdn_decode_fused.py``.

Reachability: the registered GDN tiles come from dispatch. The KDA tuned tiles
at every state dtype, the raw-gate and no-L2-norm variants and the fused modes
are built as specs directly. Dispatch does not route f32 state or the fused
modes today (``tests/dispatch/gdn/test_gfx950_wiring.py``,
``TestDeferredSpecFieldsAreUnreachable``), so those cases guard direct
``build_gdn_decode`` callers and the paths dispatch will route later.

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
from dispatch.gdn.gfx950 import _TUNED_TILES_KDA  # noqa: E402
from kernels.gfx950.gdn_decode import GdnDecodeSpec, build_gdn_decode  # noqa: E402

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


def _specs():
    """(checked, exempt): specs that must load first, and the exempt tiles."""
    out, exempt = {}, {}
    for result in dispatch_gdn_decode_all(GdnDecodeRequest(batch=16, arch=_ARCH)):
        out[f"registered_{result.candidate.spec_id}"] = result.spec
    base = GdnDecodeSpec()
    for _, tile, spec_id in _TUNED_TILES_KDA:
        for state_dtype in ("bf16", "f16", "f32"):
            out[f"tuned_{spec_id}_st{state_dtype}"] = dc.replace(
                base,
                gate_kind="kda",
                state_dtype=state_dtype,
                num_warps=tile[0],
                warp_threads_k=tile[1],
                blocks_per_v_dim=tile[2],
            )
    out["kda_raw_gate"] = dc.replace(base, gate_kind="kda", fuse_gate=False)
    out["no_l2norm"] = dc.replace(base, use_qk_l2norm=False)
    for gate in ("gdn", "kda"):
        for state_dtype in ("bf16", "f16", "f32"):
            for conv, norm in ((True, False), (False, True), (True, True)):
                for nw in (1, 4):
                    tag = ("_cv" if conv else "") + ("_rn" if norm else "")
                    # the exempt tile (module docstring) has its own test
                    dst = exempt if conv and nw == 1 else out
                    dst[f"fused_{gate}_st{state_dtype}{tag}_w{nw}"] = dc.replace(
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


@pytest.mark.parametrize("case", sorted(_SPECS))
def test_every_global_load_precedes_the_first_float_op(case):
    names, loads, maths = _load_and_math_indices(_SPECS[case])
    assert loads and maths, case
    late = [names[i] for i in loads if i > maths[0]]
    assert not late, (
        f"{case}: {len(late)} global load(s) after the first float op "
        f"({names[maths[0]]} at op {maths[0]}): {late[:4]}"
    )


@pytest.mark.parametrize("case", sorted(_EXEMPT_SPECS))
def test_exempt_tile_loads_some_inputs_first_and_rows_late(case):
    """The exempt tile issues some loads before the first float op and the
    per-row ones after it. If every load moved first, the exemption is gone
    and the tile would spill at ``(1,16,1)`` (ALGORITHM.md §4.3)."""
    _, loads, maths = _load_and_math_indices(_EXEMPT_SPECS[case])
    assert maths, case
    early = [i for i in loads if i < maths[0]]
    late = [i for i in loads if i > maths[0]]
    assert early, f"{case}: no global load before the first float op"
    assert late, f"{case}: exemption not in force, every load comes first"
