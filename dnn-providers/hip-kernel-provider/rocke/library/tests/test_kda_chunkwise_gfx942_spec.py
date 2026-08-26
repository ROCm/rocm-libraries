# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU lane for the gfx942 chunkwise KDA split/fused kernel family."""

from __future__ import annotations

import pytest

from kernels.gfx942.kda_chunkwise import (
    LDS_LIMIT,
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

ARCH = "gfx942"


def _compile_or_skip(kernel):
    """Compile through comgr, skipping only when the toolchain is absent."""
    try:
        from rocke.helpers.compile import compile_kernel
    except Exception as exc:  # noqa: BLE001  # pragma: no cover
        pytest.skip(f"comgr toolchain unavailable: {exc}")
    try:
        return compile_kernel(kernel, arch=ARCH, capture_ir_text=False)
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"comgr toolchain unavailable: {exc}")


def _tile(**kwargs) -> KdaTileSpec:
    return KdaTileSpec(**kwargs)


class TestGfx942Defaults:
    def test_native_geometry(self):
        tile = KdaTileSpec()
        assert tile.chunk == 16
        assert tile.block_size == 256
        assert tile.tile_atom_m == 16
        assert tile.scan_atom_m == 16

    @pytest.mark.parametrize(
        ("spec", "validator"),
        [
            (KdaChunkPrepSpec(), is_valid_spec),
            (KdaChunkFusedSpec(), is_valid_fused_spec),
            (KdaChunkScanSpec(), is_valid_scan_spec),
        ],
    )
    def test_default_specs_are_admitted(self, spec, validator):
        ok, why = validator(spec, arch=ARCH)
        assert ok, why
        assert spec.lds_bytes() <= LDS_LIMIT

    def test_one_partition_is_64_value_channels(self):
        assert KdaChunkFusedSpec().head_v == 64
        assert KdaChunkScanSpec().head_v == 64
        assert KdaChunkFusedSpec().tile.num_waves == 4

    def test_gfx950_is_rejected(self):
        assert not is_valid_spec(KdaChunkPrepSpec(), arch="gfx950")[0]
        assert not is_valid_fused_spec(KdaChunkFusedSpec(), arch="gfx950")[0]
        assert not is_valid_scan_spec(KdaChunkScanSpec(), arch="gfx950")[0]


class TestPrepSpec:
    @pytest.mark.parametrize(
        ("kwargs", "needle"),
        [
            ({"chunk": 64}, "atom"),
            ({"block_size": 128}, "block_size"),
            ({"pad_cb": 4}, "8"),
            ({"solve_block": 4}, "8"),
            ({"solve_block": 12}, "divide"),
            ({"tile_atom_m": 24}, "tile_atom_m"),
        ],
    )
    def test_structural_rejections(self, kwargs, needle):
        ok, why = is_valid_spec(
            KdaChunkPrepSpec(tile=_tile(**kwargs)), arch=ARCH
        )
        assert not ok
        assert needle in why

    def test_c32_exceeds_gfx942_lds(self):
        spec = KdaChunkPrepSpec(
            tile=_tile(chunk=32, tile_atom_m=16, scan_atom_m=16)
        )
        ok, why = is_valid_spec(spec, arch=ARCH)
        assert not ok
        assert "LDS" in why

    def test_invalid_dtype(self):
        assert not is_valid_spec(
            KdaChunkPrepSpec(dtype="fp16"), arch=ARCH
        )[0]


class TestFusedSpec:
    def test_full_dv128_workgroup_is_rejected(self):
        """gfx942 uses two DV64 workgroups rather than one DV128 group."""
        ok, why = is_valid_fused_spec(
            KdaChunkFusedSpec(head_v=128), arch=ARCH
        )
        assert not ok
        assert "head_v" in why

    def test_overlay_and_prefetch_cannot_alias(self):
        bad = KdaChunkFusedSpec(overlay_lds=True)
        ok, why = is_valid_fused_spec(bad, arch=ARCH)
        assert not ok
        assert "prefetch_inputs=False" in why

class TestScanSpec:
    def test_scan_uses_one_lds_residency(self):
        spec = KdaChunkScanSpec()
        assert spec.min_occupancy == 1
        assert spec.lds_bytes() <= LDS_LIMIT
        ok, why = is_valid_scan_spec(
            KdaChunkScanSpec(min_occupancy=2), arch=ARCH
        )
        assert not ok
        assert "workgroups per CU" in why

    def test_wave_partition_must_cover_value_slice(self):
        ok, why = is_valid_scan_spec(
            KdaChunkScanSpec(head_v=128), arch=ARCH
        )
        assert not ok
        assert "head_v" in why

    @pytest.mark.parametrize("kwargs", [{"pad_dk": 4}, {"pad_cb": 4}])
    def test_staging_alignment(self, kwargs):
        ok, why = is_valid_scan_spec(
            KdaChunkScanSpec(tile=_tile(**kwargs)), arch=ARCH
        )
        assert not ok
        assert "8" in why


@pytest.mark.parametrize(
    "builder,spec",
    [
        (build_kda_chunk_prep, KdaChunkPrepSpec()),
        (build_kda_chunk_fused, KdaChunkFusedSpec()),
        (build_kda_chunk_scan, KdaChunkScanSpec()),
    ],
)
def test_builds_and_compiles(builder, spec):
    artifact = _compile_or_skip(builder(spec))
    assert artifact.hsaco_bytes > 0


def test_names_are_distinct_and_encode_gfx942_geometry():
    names = {
        KdaChunkPrepSpec().kernel_name(),
        KdaChunkFusedSpec().kernel_name(),
        KdaChunkScanSpec().kernel_name(),
    }
    assert len(names) == 3
    for name in names:
        assert "dk128" in name
        assert "dv64" in name
        assert "c16" in name
