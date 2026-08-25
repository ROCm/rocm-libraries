# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU lane for the chunkwise KDA family (``kernels/gfx950/kda_chunkwise.py``).

Covers the three specs' admission rules and takes each builder through comgr,
asserting the emitted code object fits the resource budget. No GPU: the
numeric contract lives in ``test_kda_chunkwise_gfx950_numeric.py``.

The rejection cases are the interesting half. Each one encodes a structural
assumption the emitted IR depends on -- a lane mapping, an ``ds_read_b128``
alignment contract, or an occupancy target -- so a spec that violates one must
be refused rather than silently miscompiled or quietly slow.
"""

from __future__ import annotations

import pytest

from kernels.gfx950.kda_chunkwise import (
    KdaChunkFusedSpec,
    KdaChunkPrepSpec,
    KdaChunkScanSpec,
    KdaTileSpec,
    build_kda_chunk_fused,
    build_kda_chunk_prep,
    build_kda_chunk_scan,
    is_valid_fused_spec,
    is_valid_scan_spec,
    is_valid_spec,
)

ARCH = "gfx950"


def _compile_or_skip(kernel, *, arch: str = ARCH):
    """Compile through comgr, skipping only when the toolchain is missing.

    A failed compile is a real defect and propagates; an absent toolchain is an
    environment fact and skips.
    """
    try:
        from rocke.helpers.compile import compile_kernel
    except Exception as e:  # noqa: BLE001  # pragma: no cover - env-dependent
        pytest.skip(f"comgr toolchain unavailable: {e}")
    try:
        return compile_kernel(kernel, arch=arch, capture_ir_text=False)
    except ImportError as e:  # pragma: no cover - env-dependent
        pytest.skip(f"comgr toolchain unavailable: {e}")


def _tile(**kw) -> KdaTileSpec:
    return KdaTileSpec(**kw)


class TestPrepSpec:
    def test_default_is_admitted(self):
        ok, why = is_valid_spec(KdaChunkPrepSpec(), arch=ARCH)
        assert ok, why

    def test_lds_within_half_budget(self):
        """The prep kernel's whole optimization story is 2 workgroups per CU."""
        assert KdaChunkPrepSpec().lds_bytes() <= 160 * 1024 // 2

    @pytest.mark.parametrize(
        "kw,needle",
        [
            # C is the M and N extent of every C x C product, so it is pinned to
            # the atom rather than free.
            (dict(chunk=64), "atom"),
            # A thread owns one (half-chunk, channel) column of the cumsum.
            (dict(block_size=128), "block_size"),
            # Odd rows of a bf16 C x C tile would land off a 16 B boundary and
            # silently break the rank update's ds_read_b128 alignment.
            (dict(pad_cb=4), "8"),
            # A block step writes back its own rows out of a contiguous run of 8
            # accumulator slots.
            (dict(solve_block=4), "8"),
            (dict(solve_block=12), "divide"),
        ],
    )
    def test_rejections(self, kw, needle):
        ok, why = is_valid_spec(KdaChunkPrepSpec(tile=_tile(**kw)), arch=ARCH)
        assert not ok, f"{kw} should be rejected"
        assert needle in why, f"{kw}: unhelpful reason {why!r}"

    def test_unsupported_arch_and_dtype(self):
        assert not is_valid_spec(KdaChunkPrepSpec(), arch="gfx942")[0]
        assert not is_valid_spec(KdaChunkPrepSpec(dtype="fp16"), arch=ARCH)[0]

    def test_builds_and_fits(self):
        spec = KdaChunkPrepSpec()
        art = _compile_or_skip(build_kda_chunk_prep(spec))
        assert art.hsaco_bytes > 0

    def test_invalid_spec_raises(self):
        with pytest.raises(ValueError, match="invalid kda_chunk_prep spec"):
            build_kda_chunk_prep(KdaChunkPrepSpec(tile=_tile(chunk=64)))


class TestFusedSpec:
    def test_default_is_admitted(self):
        ok, why = is_valid_fused_spec(KdaChunkFusedSpec(), arch=ARCH)
        assert ok, why

    def test_c32_16x16_inner_panels_are_admitted(self):
        tile = _tile(block_size=512, scan_atom_m=16, tile_atom_m=16)
        spec = KdaChunkFusedSpec(tile=tile)
        ok, why = is_valid_fused_spec(spec, arch=ARCH)
        assert ok, why
        assert "ta16" in spec.kernel_name()

    def test_c16_padded_atom_schedule_is_admitted(self):
        tile = _tile(
            chunk=16,
            block_size=512,
            pad_cb=16,
            tile_atom_m=16,
            scan_atom_m=16,
        )
        spec = KdaChunkFusedSpec(tile=tile)
        ok, why = is_valid_fused_spec(spec, arch=ARCH)
        assert ok, why
        assert spec.lds_bytes() == 87_104

        # K=32 consumes 16 real chunk columns plus 16 explicit zero columns.
        bad = KdaChunkFusedSpec(
            tile=_tile(
                chunk=16,
                block_size=512,
                pad_cb=8,
                tile_atom_m=16,
                scan_atom_m=16,
            )
        )
        ok, why = is_valid_fused_spec(bad, arch=ARCH)
        assert not ok
        assert "zero pad" in why

    def test_unknown_tile_atom_is_rejected(self):
        ok, why = is_valid_spec(KdaChunkPrepSpec(tile=_tile(tile_atom_m=24)), arch=ARCH)
        assert not ok
        assert "tile_atom_m" in why

    def test_head_v_must_match_wave_count(self):
        """Each wave owns one atom-row band of the state and nothing else.

        That single rule partitions all five products in the scan body, which is
        what keeps the state in registers with no cross-wave reduction, so a
        ``head_v`` the waves do not cover exactly has no valid emission.
        """
        ok, why = is_valid_fused_spec(KdaChunkFusedSpec(head_v=64), arch=ARCH)
        assert not ok
        assert "head_v" in why

    def test_overlay_budget_and_prefetch_exclusion(self):
        """The explicit pool aliases only buffers with disjoint lifetimes.

        Barrier-free input prefetch deliberately writes the staging tiles while
        the scan is live, so combining it with that aliasing would corrupt the
        state mirror/residual. Keep the invalid combination out of codegen.
        """
        tile = KdaTileSpec()
        bad = KdaChunkFusedSpec(tile=tile, overlay_lds=True)
        ok, why = is_valid_fused_spec(bad, arch=ARCH)
        assert not ok
        assert "prefetch_inputs=False" in why

        overlay = KdaChunkFusedSpec(tile=tile, prefetch_inputs=False, overlay_lds=True)
        ok, why = is_valid_fused_spec(overlay, arch=ARCH)
        assert ok, why
        assert overlay.lds_bytes() == 88_704
        assert (
            overlay.lds_bytes()
            < KdaChunkFusedSpec(tile=tile, prefetch_inputs=False).lds_bytes()
        )

    def test_builds_and_fits(self):
        art = _compile_or_skip(build_kda_chunk_fused(KdaChunkFusedSpec()))
        assert art.hsaco_bytes > 0


class TestScanSpec:
    def test_default_is_admitted(self):
        ok, why = is_valid_scan_spec(KdaChunkScanSpec(), arch=ARCH)
        assert ok, why

    def test_lds_leaves_room_for_two_workgroups(self):
        """The split path only earns back its tile traffic at 2 WG/CU.

        The scan is a latency-bound chain of small matmuls; at one workgroup per
        CU there is no second workgroup to cover it. So the occupancy target is a
        spec-level rejection rule here, not a tuning note.
        """
        spec = KdaChunkScanSpec()
        assert spec.lds_bytes() <= 160 * 1024 // spec.min_occupancy
        ok, why = is_valid_scan_spec(KdaChunkScanSpec(min_occupancy=3), arch=ARCH)
        assert not ok
        assert "workgroups per CU" in why

    def test_head_v_must_match_wave_count(self):
        ok, why = is_valid_scan_spec(KdaChunkScanSpec(head_v=64), arch=ARCH)
        assert not ok
        assert "head_v" in why

    def test_staging_alignment_rejections(self):
        """Staging is ds_write_b128 throughout, so both pitches stay 8-aligned."""
        for kw in (dict(pad_dk=4), dict(pad_cb=4)):
            ok, why = is_valid_scan_spec(KdaChunkScanSpec(tile=_tile(**kw)), arch=ARCH)
            assert not ok, kw
            assert "8" in why

    def test_builds_and_fits(self):
        art = _compile_or_skip(build_kda_chunk_scan(KdaChunkScanSpec()))
        assert art.hsaco_bytes > 0


class TestSpecNaming:
    def test_names_are_distinct_and_carry_the_shape(self):
        names = {
            KdaChunkPrepSpec().kernel_name(),
            KdaChunkFusedSpec().kernel_name(),
            KdaChunkScanSpec().kernel_name(),
        }
        assert len(names) == 3
        for n in names:
            assert "dk128" in n and "dv128" in n and "c32" in n

    def test_off_default_knobs_reach_the_name(self):
        """Two specs that emit different code must not share a cache key."""
        a = KdaChunkPrepSpec().kernel_name()
        b = KdaChunkPrepSpec(tile=_tile(solve_block=32)).kernel_name()
        assert a != b
        c = KdaChunkPrepSpec(tile=_tile(pad_cb=16)).kernel_name()
        assert a != c
        assert "pcb16" in c
