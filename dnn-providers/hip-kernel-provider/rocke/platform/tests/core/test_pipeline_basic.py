# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for the pipeline="basic" feature (split load_global/store_lds).

Covers:
  - SchedulePolicy.for_pipeline("basic") attributes
  - CoalescedTileLoader.load_global() / .store_lds() IR structure

The convolution consumers of pipeline="basic" (is_valid_spec rejecting it with
async_dma=True; build_implicit_gemm_conv emitting a bounded runtime scf.for
K-loop with buffer_load_vN and smem_store_vN ops) are tested in
library/tests/test_conv_platform_regressions.py.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten(region, out):
    for op in region.ops:
        out.append(op)
        for r in op.regions:
            _flatten(r, out)
    return out


# ---------------------------------------------------------------------------
# SchedulePolicy
# ---------------------------------------------------------------------------


class TestSchedulePolicyBasic:
    def test_for_pipeline_basic_returns_emit_hints_false(self):
        from rocke.helpers.schedule import SchedulePolicy

        pol = SchedulePolicy.for_pipeline("basic")
        assert pol.emit_hints is False

    def test_for_pipeline_basic_name(self):
        from rocke.helpers.schedule import SchedulePolicy

        pol = SchedulePolicy.for_pipeline("basic")
        assert pol.name == "basic"

    def test_for_pipeline_basic_not_interwave(self):
        """pipeline="basic" must not use interwave or intrawave scheduling modes;
        emit_hints=False means no sched_group_barrier hints are emitted."""
        from rocke.helpers.schedule import SchedulePolicy

        pol = SchedulePolicy.for_pipeline("basic")
        assert pol.mode not in ("interwave", "intrawave")

    def test_for_pipeline_basic_no_setprio(self):
        from rocke.helpers.schedule import SchedulePolicy

        pol = SchedulePolicy.for_pipeline("basic")
        assert pol.setprio_level is None

    def test_unknown_pipeline_raises(self):
        from rocke.helpers.schedule import SchedulePolicy

        with pytest.raises(ValueError, match="unknown schedule policy"):
            SchedulePolicy.for_pipeline("notapipeline")


# ---------------------------------------------------------------------------
# CoalescedTileLoader split-load helpers
# ---------------------------------------------------------------------------


def _make_loader_and_builder():
    """Return (b, loader, rsrc, tid, smem) for a 64x32 tile, 256 threads."""
    from rocke.core.ir import IRBuilder, F16, PtrType
    from rocke.helpers.loads import CoalescedTileLoader

    b = IRBuilder("split_load_test")
    b.kernel.attrs["max_workgroup_size"] = 256

    loader = CoalescedTileLoader.from_tile(
        tile_rows=64, tile_cols=32, block_size=256, max_vec=4
    )
    rsrc_ptr = b.param("rsrc", PtrType(F16, "global"))
    rsrc = b.buffer_rsrc(rsrc_ptr, b.const_i32(1024 * 1024))
    tid = b.param("tid", None)
    smem = b.smem_alloc(F16, [loader.tile_rows, loader.tile_cols], name_hint="tile")
    return b, loader, rsrc, tid, smem


def _flat_desc(b, row, col, loader):
    off = b.add(b.mul(row, b.const_i32(loader.tile_cols)), col)
    return off, None


class TestCoalescedTileLoaderSplit:
    def test_load_global_returns_correct_count(self):
        b, loader, rsrc, tid, smem = _make_loader_and_builder()

        staged = loader.load_global(
            b,
            tid=tid,
            descriptor=lambda b, r, c: _flat_desc(b, r, c, loader),
            rsrc=rsrc,
        )
        assert len(staged) == loader.vecs_per_thread

    def test_load_global_emits_buffer_load_ops(self):
        """load_global() must emit tile.buffer_load_vN but NOT tile.smem_store_vN."""
        b, loader, rsrc, tid, smem = _make_loader_and_builder()

        loader.load_global(
            b,
            tid=tid,
            descriptor=lambda b, r, c: _flat_desc(b, r, c, loader),
            rsrc=rsrc,
        )
        op_names = [op.name for op in b.kernel.body.ops]
        load_ops = [n for n in op_names if "buffer_load" in n]
        store_ops = [n for n in op_names if "smem_store" in n]

        assert (
            len(load_ops) == loader.vecs_per_thread
        ), f"expected {loader.vecs_per_thread} buffer_load ops, got {load_ops}"
        assert store_ops == [], "load_global() must not emit any smem_store ops"

    def test_store_lds_emits_smem_store_ops(self):
        """store_lds() must emit tile.smem_store_vN for each staged vec."""
        b, loader, rsrc, tid, smem = _make_loader_and_builder()

        staged = loader.load_global(
            b,
            tid=tid,
            descriptor=lambda b, r, c: _flat_desc(b, r, c, loader),
            rsrc=rsrc,
        )
        count_before = len(b.kernel.body.ops)
        loader.store_lds(b, smem_dst=smem, staged=staged)
        new_ops = [op.name for op in b.kernel.body.ops][count_before:]

        smem_stores = [n for n in new_ops if "smem_store" in n]
        assert (
            len(smem_stores) == loader.vecs_per_thread
        ), f"expected {loader.vecs_per_thread} smem_store ops, got {smem_stores}"

    def test_split_load_sequence_matches_fused_load(self):
        """load_global + store_lds must emit the same types (and count) of
        buffer_load_vN and smem_store_vN as the monolithic load()."""
        from rocke.core.ir import IRBuilder, F16, PtrType
        from rocke.helpers.loads import CoalescedTileLoader

        loader = CoalescedTileLoader.from_tile(
            tile_rows=64, tile_cols=32, block_size=256, max_vec=4
        )

        def _setup():
            b = IRBuilder("cmp")
            b.kernel.attrs["max_workgroup_size"] = 256
            rsrc_ptr = b.param("rsrc", PtrType(F16, "global"))
            rsrc = b.buffer_rsrc(rsrc_ptr, b.const_i32(1024 * 1024))
            tid = b.param("tid", None)
            smem = b.smem_alloc(
                F16, [loader.tile_rows, loader.tile_cols], name_hint="tile"
            )
            return b, rsrc, tid, smem

        def _desc(b, row, col):
            return _flat_desc(b, row, col, loader)

        # Fused path
        b_fused, rsrc_f, tid_f, smem_f = _setup()
        loader.load(b_fused, tid=tid_f, smem_dst=smem_f, descriptor=_desc, rsrc=rsrc_f)
        fused_loads = sum(
            1 for op in b_fused.kernel.body.ops if "buffer_load" in op.name
        )
        fused_stores = sum(
            1 for op in b_fused.kernel.body.ops if "smem_store" in op.name
        )

        # Split path
        b_split, rsrc_s, tid_s, smem_s = _setup()
        staged = loader.load_global(b_split, tid=tid_s, descriptor=_desc, rsrc=rsrc_s)
        loader.store_lds(b_split, smem_dst=smem_s, staged=staged)
        split_loads = sum(
            1 for op in b_split.kernel.body.ops if "buffer_load" in op.name
        )
        split_stores = sum(
            1 for op in b_split.kernel.body.ops if "smem_store" in op.name
        )

        assert split_loads == fused_loads
        assert split_stores == fused_stores

    def test_load_global_raises_without_rsrc_when_use_buffer_rsrc(self):
        from rocke.core.ir import IRBuilder, F16
        from rocke.helpers.loads import CoalescedTileLoader

        b = IRBuilder("err_test")
        b.kernel.attrs["max_workgroup_size"] = 256
        loader = CoalescedTileLoader.from_tile(
            tile_rows=64, tile_cols=32, block_size=256, max_vec=4
        )
        tid = b.param("tid", None)

        with pytest.raises(ValueError, match="use_buffer_rsrc=True requires rsrc"):
            loader.load_global(
                b,
                tid=tid,
                descriptor=lambda b, r, c: (b.const_i32(0), None),
                rsrc=None,
            )


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
