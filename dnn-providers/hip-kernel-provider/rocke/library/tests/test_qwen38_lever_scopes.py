# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Per-stage / per-phase lever scopes of the Rust direct compiler (``dlevers``) in the Qwen3.8 mono builders:
``FrameworkKnobs.stage_levers`` (one scope per stage of a layer kernel) and ``FusedKnobs.phase_levers`` (one per
phase of the fused MoE body). CPU only: the IR the builders emit."""

from __future__ import annotations

import collections
import dataclasses

import pytest
from rocke.core.ir_serialize import canonicalize

from kernels.gfx950.qwen38_mono import moe_fused as MF
from kernels.gfx950.qwen38_mono import stage_moe_fused as SMF
from kernels.gfx950.qwen38_mono import toy_chain as T
from kernels.gfx950.qwen38_mono.framework import FrameworkKnobs


def _scopes(kernel) -> collections.Counter:
    c = collections.Counter()

    def walk(region):
        for op in region.ops:
            c[op.attrs.get("dlevers")] += 1
            for r in op.regions:
                walk(r)

    walk(kernel.body)
    return c


def _ops(kernel):
    out = []

    def walk(region):
        for op in region.ops:
            out.append(op)
            for r in op.regions:
                walk(r)

    walk(kernel.body)
    return out


def test_stage_levers_scope_each_stages_ops_and_nothing_else():
    spec = T.ToyChainSpec(depth=2, work=8)  # stages g0, g1, out
    plain = T.build_toy_chain(spec)
    kernel = T.build_toy_chain(dataclasses.replace(spec, knobs=FrameworkKnobs(stage_levers="")))
    assert canonicalize(kernel) == canonicalize(plain)
    scoped = T.build_toy_chain(dataclasses.replace(spec, knobs=FrameworkKnobs(stage_levers="g1=nofence+loadorder;out=-x")))
    c = _scopes(scoped)
    assert set(c) == {None, "nofence,loadorder", "-x"}
    assert 0 < c["nofence,loadorder"] < sum(c.values())
    # the same ops as the plain build, in the same order: a scope only marks them
    assert [op.name for op in _ops(scoped)] == [op.name for op in _ops(plain)]
    assert scoped.name == plain.name  # not in the name
    with pytest.raises(ValueError):
        T.build_toy_chain(dataclasses.replace(spec, knobs=FrameworkKnobs(stage_levers="no_such_stage=nofence")))
    with pytest.raises(ValueError):
        FrameworkKnobs(stage_levers="nofence")


def test_phase_levers_scope_the_code_between_phase_points():
    k = SMF.best_knobs(4, "best_plain")
    plain = MF.build_fused_suffix(4, k)
    empty = MF.build_fused_suffix(4, dataclasses.replace(k, phase_levers=""))
    assert canonicalize(plain) == canonicalize(empty)
    scoped = MF.build_fused_suffix(4, dataclasses.replace(k, phase_levers="router=a;dn=nofence+loadorder;combine=-nofence"))
    c = _scopes(scoped)
    assert set(c) == {None, "a", "nofence,loadorder", "-nofence"}
    assert [op.name for op in _ops(scoped)] == [op.name for op in _ops(plain)]
    # the ops after the last phase point (the return) are outside every scope
    assert scoped.body.ops[-1].attrs.get("dlevers") is None
    # every phase stamp point opens the next phase: with stamps, the stamp after the dn phase is in combine's scope
    stamped = MF.build_fused_suffix(4, dataclasses.replace(k, phase_stamps=True, phase_levers="dn=x;combine=y"))
    assert {"x", "y"} <= set(_scopes(stamped))
    with pytest.raises(ValueError):
        dataclasses.replace(k, phase_levers="nosuchphase=x")
    with pytest.raises(ValueError):
        dataclasses.replace(k, phase_levers="dn")
