# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The spec-shaped ``supports_tiled_2d_spec`` must agree with the dispatcher gate.

``hkp_pack`` pre-flights a spec through ``supports_tiled_2d_spec``; the dispatcher
decides routing through ``supports_native_unified_attention_tiled``. If the wrapper
drops a spec field the gate forwards (the gfx942 wide-K / ring flags), pack time
sizes a different kernel than the one dispatch builds and can wrongly reject (or
admit) a spec. This runs both on specs the dispatcher itself produces.

The 4-warp fast cohort (``_gfx942_4warp_fast``) is excluded: its flag is not a
spec field, so that builder is intentionally not aliased in ``hkp_pack``.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import itertools

import pytest

import kernels.common.attention_unified as au
from kernels.common.attention_unified import _tiled_spec_from_problem


def _problem(**kw):
    base = dict(
        total_q=4096,
        num_seqs=1,
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=16,
        max_seqlen_q=4096,
        max_seqlen_k=4096,
        dtype="bf16",
        sliding_window=0,
    )
    base.update(kw)
    return au.UnifiedAttentionProblem(**base)


_GRID = list(itertools.product((64, 128, 256), ("bf16", "fp16"), (16, 64), (0, 4096)))


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_2d_spec_wrapper_matches_dispatcher_gate(arch, monkeypatch):
    monkeypatch.setattr(au, "_RESOLVED_ATTENTION_ARCH", arch)
    mod = importlib.import_module(f"kernels.{arch}.attention_tiled_2d")
    checked = 0
    for head_size, dtype, block_size, window in _GRID:
        p = _problem(
            head_size=head_size,
            dtype=dtype,
            block_size=block_size,
            sliding_window=window,
        )
        if arch == "gfx942" and au._gfx942_4warp_fast(p, arch):
            continue
        gate_ok, gate_reason = au.supports_native_unified_attention_tiled(p, arch)
        try:
            spec = _tiled_spec_from_problem(p, arch)
        except (RuntimeError, ValueError):
            # Over-budget or invalid spec shape; skip rather than fail.
            continue
        wrap_ok, wrap_reason = mod.supports_tiled_2d_spec(spec, arch=arch)
        assert wrap_ok == gate_ok, (
            f"{arch} D{head_size} {dtype} bs{block_size} sw{window}: "
            f"gate=({gate_ok}, {gate_reason!r}) wrapper=({wrap_ok}, {wrap_reason!r})"
        )
        checked += 1
    assert checked, "no problem in the grid produced a comparable spec"


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_2d_wrapper_forwards_all_predicate_params(arch):
    """Every parameter of ``supports_tiled_2d`` that has a matching spec field
    must be forwarded by ``_supports_tiled_2d_spec``.

    This is a structural guard against parameter drift: if someone adds a new
    field to the spec and a matching parameter to ``supports_tiled_2d`` without
    updating the wrapper, this test fails immediately rather than silently
    shipping a kernel the predicate did not validate.

    ``use_d256_fast`` is excluded because it is derived from the builder alias,
    not carried by the spec.
    """
    mod = importlib.import_module(f"kernels.{arch}.attention_tiled_2d")
    predicate = mod.supports_tiled_2d
    # gfx942 delegates through an internal _supports_tiled_2d_spec; gfx950
    # has no such indirection.  Inspect the implementation that contains the
    # actual forwarding calls in either case.
    wrapper_impl = getattr(mod, "_supports_tiled_2d_spec", mod.supports_tiled_2d_spec)

    predicate_params = set(inspect.signature(predicate).parameters)
    predicate_params -= {"use_d256_fast"}  # builder-derived, not a spec field

    # Resolve the spec type — the annotation may be a string (forward ref) so
    # use typing.get_type_hints to get the actual class.
    import typing

    hints = typing.get_type_hints(mod.supports_tiled_2d_spec)
    spec_cls = hints["spec"]
    spec_fields = {f.name for f in dataclasses.fields(spec_cls)}

    wrapper_src = inspect.getsource(wrapper_impl)
    for param in predicate_params:
        if param in spec_fields:
            assert param in wrapper_src, (
                f"{arch}: supports_tiled_2d parameter {param!r} has a matching "
                f"spec field on {spec_cls.__name__} but does not appear in "
                f"_supports_tiled_2d_spec — add it to the wrapper"
            )
