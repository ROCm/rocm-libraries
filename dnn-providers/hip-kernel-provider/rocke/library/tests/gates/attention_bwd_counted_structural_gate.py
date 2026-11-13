# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Exhaustive host-only structural gate for counted backward synchronization.

This module is intentionally not named ``test_*.py``. Ordinary pytest discovery
runs fast unit contracts; CTest and ``library/tests/run_all.py`` invoke this
exhaustive matrix explicitly.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower

from kernels.common._attention_bwd_body import BwdTileGeometry
from kernels.common._attention_bwd_counted import COUNTED_ISSUES, COUNTED_VMCNT_MAX
from kernels.common.attention_bwd import build_attn_bwd_main, validate_attn_bwd_spec

from tests._attention_bwd_counted_cases import (
    DTYPES,
    HEADS,
    MFMA_ARCHS,
    counted_rows,
    is_legal_counted_row,
    require_explicit_gate_entrypoint,
    row_digest,
    row_id,
    spec_of,
)
from tests._attention_bwd_counted_ir import (
    assert_counted_ir_model,
    predecessors,
    q_loops,
    walk_ops,
)
from tests._attention_bwd_counted_model import ir_events
from tests._attention_bwd_lds_model import LDS_BARRIERS, lds_hazards, traced_spec
from tests._attention_bwd_test_utils import distinct_configs

require_explicit_gate_entrypoint()

_ROWS = counted_rows()
_EXPECTED_ROWS = 817
_EXPECTED_ROW_DIGEST = (
    "191aab5bcee46a997b47e01cb2a88303055aa673e9308b97a2751dd74f6bd912"
)


def test_gate_manifest_is_stable():
    """Make a lost or duplicated matrix row an explicit reviewed change."""
    assert len(_ROWS) == _EXPECTED_ROWS
    assert len(set(_ROWS)) == len(_ROWS)
    assert row_digest(_ROWS) == _EXPECTED_ROW_DIGEST


@pytest.mark.parametrize("row", _ROWS, ids=row_id)
def test_counted_ir_and_lds_models(row):
    """Validate waits and cross-wavefront LDS ordering on one instrumented build."""
    arch, _, _, _, _, issue = row
    spec = spec_of(*row)
    resolved = validate_attn_bwd_spec(spec, arch)
    assert (
        BwdTileGeometry.from_spec(resolved, arch).counted_atomics <= COUNTED_VMCNT_MAX
    )

    kernel, tags = traced_spec(spec, arch)
    assert_counted_ir_model(kernel, spec, arch, issue)

    previous = predecessors(kernel)
    bare = [
        op for op, _ in walk_ops(kernel.body.ops) if op.name == "tile.s_barrier_bare"
    ]
    assert bare
    for barrier in bare:
        wait = previous[id(barrier)][-1]
        assert wait.name == "tile.s_waitcnt" and wait.attrs["lgkmcnt"] == 0

    top = kernel.body.ops
    branch = next(
        (
            index
            for index, op in enumerate(top)
            if op.name == "scf.if"
            and op.regions[0].ops
            and op.regions[0].ops[0].name == "tile.sync"
        ),
        None,
    )
    # The mirror tile sits in a uniform branch. The generic hazard model
    # deliberately ignores branch-local barriers because it cannot prove
    # uniformity, so validate the pre-branch and mirror-tile regions separately.
    parts = [top] if branch is None else [top[:branch], top[branch].regions[0].ops]
    barriers = LDS_BARRIERS + ("tile.s_barrier_bare",)
    for ops in parts:
        part = SimpleNamespace(body=SimpleNamespace(ops=ops))
        assert not lds_hazards(part, tags, barriers=barriers)[:4]


@pytest.mark.parametrize("arch", MFMA_ARCHS)
@pytest.mark.parametrize("variant", ("band", "band_hout", "band_pack", "band_ptlds"))
def test_counted_step_is_the_full_step(arch, variant):
    """Counted barriers do not change matrix-op order or dQ atomics."""
    for config in distinct_configs(arch):
        full_kernel = None
        for issue in COUNTED_ISSUES:
            row = (arch, config, 64, "fp16", variant, issue)
            if not is_legal_counted_row(row):
                continue
            spec = spec_of(*row)
            counted_kernel = build_attn_bwd_main(spec, arch=arch)
            if full_kernel is None:
                full = dataclasses.replace(
                    spec, loop_sync="full", counted_issue="early"
                )
                full_kernel = build_attn_bwd_main(full, arch=arch)
            for counted_loop, full_loop in zip(
                q_loops(counted_kernel), q_loops(full_kernel), strict=True
            ):
                counted_mma = [
                    op.attrs["op_id"]
                    for op, _ in walk_ops(counted_loop.regions[0].ops)
                    if op.name == "tile.mma"
                ]
                full_mma = [
                    op.attrs["op_id"]
                    for op, _ in walk_ops(full_loop.regions[0].ops)
                    if op.name == "tile.mma"
                ]
                assert counted_mma == full_mma
                counted_atomics = [
                    kind
                    for kind, _ in ir_events(counted_loop.regions[0].ops)
                    if kind == "A"
                ]
                full_atomics = [
                    kind
                    for kind, _ in ir_events(full_loop.regions[0].ops)
                    if kind == "A"
                ]
                assert len(counted_atomics) == len(full_atomics) > 0


def test_counted_bare_barrier_is_required_by_lds_model():
    """The representative counted loop races when its bare barriers are ignored."""
    spec = spec_of("gfx942", "default", 64, "fp16", "band_edge", "early")
    kernel, tags = traced_spec(spec, "gfx942")
    assert lds_hazards(kernel, tags, barriers=LDS_BARRIERS)
    assert not lds_hazards(
        kernel, tags, barriers=LDS_BARRIERS + ("tile.s_barrier_bare",)
    )


def test_counted_ir_lowers_on_every_mfma_target():
    for arch in MFMA_ARCHS:
        for issue in COUNTED_ISSUES:
            kernel = build_attn_bwd_main(
                spec_of(arch, "default", 128, "bf16", "band_edge", issue),
                arch=arch,
            )
            llvm = lower(kernel, arch=arch, llvm_flavor="llvm22")
            assert "@llvm.amdgcn.s.barrier()" in llvm


def test_mirror_edge_rows_are_refused():
    """Every counted mirror-plus-edge matrix cell is refused explicitly."""
    from kernels.common._attention_bwd_refusals import resource_refusal

    for arch in MFMA_ARCHS:
        for config in distinct_configs(arch):
            for head in HEADS:
                for dtype in DTYPES:
                    for issue in COUNTED_ISSUES:
                        row = (arch, config, head, dtype, "band_mirror_edge", issue)
                        spec = spec_of(*row)
                        reason = resource_refusal(spec.resolved(arch), arch)
                        assert reason is not None and not is_legal_counted_row(row), row
