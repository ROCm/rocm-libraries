# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Milestone 0: the external emit RECORDER + `b`-witness gate + addressing round-trip.

Pure CPU (builds IR + inspects it; no compile/run), so it stays in the CPU suite. Target kernel is
the toy LDS-staged GEMM (single wave, single atom, single buffer, identity store/read, Python-unrolled
K-loop -> constant origins) -- the isolation boundary that proves the spine before CRC's symbolic
origins / cooperative scope / swizzle.
"""
from __future__ import annotations

import pytest

import rocke.helpers.tiling.emit as emit_mod
from rocke.helpers.tiling import load_fragment  # noqa: F401 -- bound so restoration is exercised here
from rocke.helpers.tiling import tiling_recorder as tr
from rocke.helpers.tiling.kernels import tiling_gemm_interleaved_demo as demo
from rocke.helpers.tiling.mma.mma_operation import TileMma


def _record_toy():
    (kernel, _mma), pipeline = tr.record_build(demo.build_lds_staged_gemm, 64, 64, 64)
    return kernel, pipeline


def _serialize(region):
    """Structural fingerprint of an IR region: op name + attrs + operand/result names, recursive."""
    return [
        (
            op.name,
            tuple(sorted((k, repr(v)) for k, v in op.attrs.items())),
            tuple(v.name for v in op.operands),
            tuple(v.name for v in op.results),
            [_serialize(r) for r in op.regions],
        )
        for op in region.ops
    ]


def test_records_exact_node_sequence():
    _kernel, pipeline = _record_toy()
    got = [(n.kind, n.space) for n in pipeline.transactions]
    k_iter = [
        ("load", "global"), ("load", "global"),
        ("store", "lds"), ("store", "lds"),
        ("load", "lds"), ("load", "lds"),
    ]
    expected = [("fill", "reg")] + k_iter * 4 + [("store", "global")]
    assert got == expected
    assert not pipeline.ops  # the toy uses a raw b.mma, not the TileMma verb -> no recorded op

    store_a = pipeline.transactions[3]
    assert store_a.space == "lds" and store_a.strides == (16, 1) and store_a.origin == (0, 0)
    assert store_a.register_count == 4 and store_a.vw == 4 and store_a.op_fanout == 1
    assert store_a.dtype_name == "f16"

    c_store = pipeline.transactions[-1]
    assert c_store.space == "global" and c_store.vw == 1 and c_store.op_fanout == 4  # scalar global store


def test_witness_reconciles_memory_and_flags_direct_mma():
    kernel, pipeline = _record_toy()
    rep = tr.witness(pipeline, kernel, raise_on_gap=False)

    assert rep.mem_expected == rep.mem_counted == 28 and rep.mem_ok
    assert rep.histogram["memref.global_load_vN"] == 8
    assert rep.histogram["memref.global_store_typed"] == 4
    assert rep.histogram["tile.smem_store_vN"] == 8
    assert rep.histogram["tile.smem_load_vN"] == 8

    # The toy calls b.mma directly (bypassing the TileMma verb) -> 4 unaccounted tile.mma. The witness
    # must catch this LOUDLY rather than draw a silently-short pipeline.
    assert rep.mma_counted == 4 and rep.mma_expected == 0 and not rep.mma_ok and not rep.ok
    with pytest.raises(tr.CoverageError, match="mma expected=0 counted=4"):
        tr.witness(pipeline, kernel)


def test_addressing_roundtrip_passes_on_toy():
    _kernel, pipeline = _record_toy()
    verified = tr.verify_roundtrip(pipeline)
    names = sorted(pipeline.spaces[s] for s in verified)
    assert len(verified) == 2
    assert names[0].startswith("%lds_a") and names[1].startswith("%lds_b")


def test_roundtrip_catches_a_corrupted_read():
    """A read encoding that reads an address the store never wrote must be caught (the gate is real,
    not vacuous). Swap the lds_a read's tile_desc for the lds_b read's -> its addresses no longer
    match what the lds_a store wrote."""
    import dataclasses

    _kernel, pipeline = _record_toy()
    a_read = next(t for t in pipeline.transactions
                  if t.space_name.startswith("%lds_a") and t.kind == "load")
    # Corrupt the recorded lds_a read with a shifted origin so its addresses miss the store's set.
    bad = dataclasses.replace(a_read, origin=(1, 0))
    pipeline.nodes[pipeline.nodes.index(a_read)] = bad
    with pytest.raises(AssertionError, match="round-trip"):
        tr.verify_roundtrip(pipeline)


def test_recorded_build_is_byte_identical():
    kernel_rec, _pipeline = _record_toy()
    kernel_plain, _ = demo.build_lds_staged_gemm(64, 64, 64)
    assert _serialize(kernel_rec.body) == _serialize(kernel_plain.body)


def test_verbs_restored_after_normal_build():
    orig_emit = emit_mod.load_fragment
    orig_call = TileMma.__call__
    orig_demo = demo.load_fragment
    _record_toy()
    assert emit_mod.load_fragment is orig_emit
    assert TileMma.__call__ is orig_call
    assert demo.load_fragment is orig_demo  # the build fn's module globals are restored too


def test_verbs_restored_on_exception():
    orig_emit = emit_mod.load_fragment
    orig_call = TileMma.__call__

    def _boom():
        raise RuntimeError("boom during build")

    with pytest.raises(RuntimeError, match="boom during build"):
        tr.record_build(_boom)
    assert emit_mod.load_fragment is orig_emit
    assert TileMma.__call__ is orig_call
