#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""C++/Python lowerer byte-identity gate for the attention backward kernels.

This is the lowerer axis of the engine parity rule: Python builds each
kernel, both engines lower it, and the ``.ll`` must match byte for byte.
(The builder axis, C++ twins of the builders, is gated by the parity pairs
under ``library/tests/parity``.)
Same outcome classification as ``test_attention_ir_cpp_parity.py``.

The cases are what the dispatcher can select:

* every entry of the per-arch general tuned tables and of the dense tuned
  table, planned under the routed (shipped) policy;
* the default knobs (no tuned table) on the same general keys, and the dense
  keys with dense defaults on gfx942 and gfx950;
* every unified tuned page tile, in direct and atomic page mode;
* knob values that are registered but in no shipped table and emit an op
  no other case does (``_KNOB_CASES``).

Each case lowers every kernel of its plan (prep, main, dq, convert, reduce,
dense, page_*), once per LLVM flavor in ``GOLDEN_FLAVORS``.
"""

from __future__ import annotations

import pytest

# Registered knob values outside every shipped table that need an op no
# shipped case emits (tile.buffer_load): (arch, knob, value, general table key
# index). The key is one where the knob is not inert (on the other keys the
# plan leaves the op out). Pinned here, not swept, so the gate stays fast.
_KNOB_CASES = (
    ("gfx942", "ws_bounds", "buffer", 1),
    ("gfx950", "ws_bounds", "buffer", 0),
)
# The op the knob cases exist for. Each must emit it: a retuned or reordered
# table can put the pinned index on a key where the knob is inert.
_KNOB_OP = "tile.buffer_load"


def _flavors():
    from rocke.core.ir_golden import GOLDEN_FLAVORS

    return GOLDEN_FLAVORS


def _general_requests(arch, key):
    from dispatch.attention_bwd import tuned_tables

    return tuned_tables.general_key_requests(arch, key)


def _first_plan(reqs, plan_one):
    last = None
    for req in reqs:
        try:
            return plan_one(req)
        except ValueError as e:
            last = e
    raise AssertionError(f"no plannable representative: {last}")


def _cases():
    """``[(case_id, family, arch, recipe)]``; plans are made per flavor."""
    from dispatch.attention_bwd import routing
    from dispatch.attention_bwd.dense_tuning import DENSE_BWD_TUNED
    from dispatch.attention_bwd.unified_tuning import UNIFIED_TUNED_PAGE_TILES

    out = []
    for arch, mod in routing.ARCH_TUNING.items():
        for i, key in enumerate(mod.GENERAL_TUNED_TABLE):
            out.append((f"general/{arch}/tuned{i}", "general", arch, ("tuned", key)))
            out.append(
                (f"general/{arch}/default{i}", "general", arch, ("default", key))
            )
        for i, (_a, dk) in enumerate(mod.DENSE_TUNED):
            out.append((f"dense/{arch}/tuned{i}", "dense", arch, ("tuned", dk)))
    for i, (_a, dk) in enumerate(DENSE_BWD_TUNED):
        for arch in ("gfx942", "gfx950"):
            out.append((f"dense/{arch}/default{i}", "dense", arch, ("default", dk)))
    for (arch, d, page), name in UNIFIED_TUNED_PAGE_TILES.items():
        for mode in ("direct", "atomic"):
            cid = f"unified/{arch}/d{d}_p{page}_{name}_{mode}"
            out.append((cid, "unified", arch, ("tuned", (d, page, mode))))
    for arch, knob, value, ki in _KNOB_CASES:
        cid = f"general/{arch}/{knob}={value}"
        out.append((cid, "general", arch, ("knob", (knob, value, ki))))
    return out


def _plan(family, arch, recipe):
    from dispatch.attention_bwd import routing, tuned_tables
    from kernels.common.attention_bwd_plan import shipped_policy

    kind, key = recipe
    if family == "general":
        if kind == "knob":
            knob, value, ki = key
            policy = tuned_tables._gate_policy(tuning=lambda *_: {knob: value})
            key = list(routing.ARCH_TUNING[arch].GENERAL_TUNED_TABLE)[ki]
        else:
            policy = routing.routed_policy() if kind == "tuned" else shipped_policy()
        return _first_plan(
            _general_requests(arch, key),
            lambda r: routing.plan_attention_bwd(
                r, arch, policy=policy, route="general"
            ),
        )
    if family == "dense":
        if kind == "tuned":
            policy = routing.routed_policy()
        else:
            policy = routing.dense_policy(
                tuned_tables._gate_policy(), {(arch, key): {}}
            )
        return _first_plan(
            tuned_tables.dense_key_requests(arch, key),
            lambda r: routing.plan_attention_bwd(r, arch, policy=policy, route="dense"),
        )
    from kernels.common.attention_bwd_unified_plan import AttnBwdUnifiedRequest

    d, page, mode = key
    req = AttnBwdUnifiedRequest(
        num_seqs=3, h_q=8, h_kv=2, d_qk=d, d_v=d, total_q=100,
        max_seqlen_q=60, max_seqlen_kv=300, page=page, num_blocks=50,
        max_blocks_per_seq=-(-300 // page), block_table_injective=mode == "direct",
    )  # fmt: skip
    return routing.plan_attention_bwd_unified(req, arch)


def _builders(family):
    if family == "unified":
        from kernels.common.attention_bwd_unified_run import _BUILDERS
    else:
        from kernels.common.attention_bwd_run import _BUILDERS
    return _BUILDERS


def _walk(ops):
    for op in ops:
        yield op
        for r in op.regions:
            yield from _walk(r.ops)


def _first_diff(py: str, cpp: str) -> str:
    for i, (a, b) in enumerate(zip(py.splitlines(), cpp.splitlines())):
        if a != b:
            return f"line {i + 1}:\n    py : {a[:200]}\n    cpp: {b[:200]}"
    return f"length: py {py.count(chr(10))} lines, cpp {cpp.count(chr(10))} lines"


def _params():
    try:
        cases = _cases()
    except ImportError as e:  # pragma: no cover
        return [pytest.param(None, marks=pytest.mark.skip(reason=str(e)))]
    params = []
    for cid, family, arch, recipe in cases:
        params.append(pytest.param((family, arch, recipe), id=cid))
    return params


@pytest.mark.parametrize("case", _params())
def test_attention_bwd_ir_cpp_python_byte_identity(case, monkeypatch):
    try:
        from rocke.core.backend import BackendError
        from rocke.helpers.compile import _lower_llvm_via_backend
    except ImportError as e:  # pragma: no cover
        pytest.skip(f"backend lowering unavailable: {e}")
    family, arch, recipe = case
    builders = _builders(family)
    monkeypatch.setenv("ROCKE_CPP_STRICT", "1")
    mism, compared, knob_ops = [], 0, 0
    for flavor in _flavors():
        # Plan and build under the flavor being lowered: admission can read it.
        monkeypatch.setenv("ROCKE_LLVM_FLAVOR", flavor)
        plan = _plan(family, arch, recipe)
        seen = set()
        for step in plan.launches:
            if step.kernel in seen:
                continue
            seen.add(step.kernel)
            kernel = builders[step.stage](plan.specs[step.stage], arch=arch)
            if recipe[0] == "knob":
                knob_ops += sum(op.name == _KNOB_OP for op in _walk(kernel.body.ops))
            py = _lower_llvm_via_backend(kernel, arch=arch, backend="python", spec=None)
            try:
                cpp = _lower_llvm_via_backend(
                    kernel, arch=arch, backend="cpp", spec=None
                )
            except BackendError as e:
                pytest.skip(f"C++ engine not importable: {str(e)[:200]}")
            compared += 1
            if py != cpp:
                mism.append(f"{flavor}/{step.kernel} {_first_diff(py, cpp)}")
    assert not mism, "attention bwd cpp/python IR byte-mismatch:\n  " + "\n  ".join(
        mism
    )
    assert compared, "no backward kernel was compared"
    if recipe[0] == "knob":
        assert knob_ops, f"{recipe[1]}: no {_KNOB_OP} emitted; re-pick the table key"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
