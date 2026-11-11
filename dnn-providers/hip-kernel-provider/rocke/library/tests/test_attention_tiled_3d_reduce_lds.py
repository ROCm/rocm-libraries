# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The split-KV reduce kernel's segment bound, sourced from the tools, not by hand.

``supports_tiled_reduce_spec`` refuses a ``num_segments`` whose LDS factor cache
(``split_kv_reduce_lds_bytes``) does not fit ``ArchTarget.lds_capacity_bytes``. These cases hold
that formula to what comgr actually allocates (the code object's
``group_segment_fixed_size``), and hold the C++ host's mirror,
``GFX950_UNIFIED_3D_MAX_SEGMENTS``, to the same arch bound.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest

# An installed test tree carries a copy of the two headers under ``hkp_packs/``
# (the rocKE umbrella CMake project installs it); a source checkout reads the provider's.
_PACKS_DIRS = (
    Path(__file__).resolve().parent / "hkp_packs",
    Path(__file__).resolve().parents[3]
    / "src"
    / "engines"
    / "kernel_ingestor_engine"
    / "packs",
)
_GEOMETRY_HEADERS = (
    "Gfx950AttentionDenseGeometry.hpp",
    "Gfx950AttentionUnified3DGeometry.hpp",
)


def _packs_dir() -> Path:
    for d in _PACKS_DIRS:
        if all((d / h).is_file() for h in _GEOMETRY_HEADERS):
            return d
    pytest.skip(
        "hip-kernel-provider pack headers are neither installed nor in a checkout"
    )


def _reduce(num_segments: int):
    from kernels import (
        UnifiedAttentionReduceTiledSpec,
        build_unified_attention_reduce_tiled,
    )

    return build_unified_attention_reduce_tiled(
        UnifiedAttentionReduceTiledSpec(
            head_size=128,
            num_query_heads=16,
            num_kv_heads=2,
            dtype="bf16",
            num_segments=num_segments,
        )
    )


def _compiled_lds_bytes(kernel, arch: str) -> int:
    """``group_segment_fixed_size`` of ``kernel`` compiled for ``arch``; skips when the
    comgr toolchain or the HSACO reader is unavailable."""
    try:
        from rocke.helpers.compile import compile_kernel

        art = compile_kernel(kernel, arch=arch, capture_ir_text=False)
    except (ImportError, OSError, RuntimeError) as e:  # pragma: no cover
        pytest.skip(f"comgr toolchain unavailable: {e}")
    from rocke.analysis.isa import analyze_hsaco

    with tempfile.NamedTemporaryFile(suffix=".hsaco") as fh:
        fh.write(bytes(art.hsaco))
        fh.flush()
        try:
            lds = analyze_hsaco(Path(fh.name)).resources.lds_bytes
        except (FileNotFoundError, RuntimeError) as e:  # pragma: no cover
            pytest.skip(f"HSACO introspection tool unavailable: {e}")
    if lds is None:  # pragma: no cover - metadata shape drift
        pytest.skip("could not parse group_segment_fixed_size from HSACO")
    return lds


@pytest.mark.parametrize("num_segments", [2, 3, 5, 16, 1000])
def test_reduce_lds_formula_is_what_comgr_allocates(num_segments):
    """Counts that are not a multiple of four also hold the pool's 16-byte rounding."""
    from kernels.common.attention_split_kv_reduce import split_kv_reduce_lds_bytes

    assert _compiled_lds_bytes(
        _reduce(num_segments), "gfx950"
    ) == split_kv_reduce_lds_bytes(num_segments)


def _cpp_int(header: str, name: str) -> int:
    text = (_packs_dir() / header).read_text()
    match = re.search(rf"inline constexpr int64_t {name} = (\d+);", text)
    assert match, f"{name} not found as a literal in {header}"
    return int(match.group(1))


def test_cpp_segment_bound_mirrors_the_arch_bound():
    """The C++ host bound is gfx950's LDS capacity over the reduce's bytes per segment,
    each restated in C++, and equals the Python gate's bound."""
    from rocke.core.arch.target import ArchTarget

    from kernels.gfx950.attention_tiled_3d import reduce_max_segments

    cap = _cpp_int("Gfx950AttentionDenseGeometry.hpp", "GFX950_LDS_CAPACITY_BYTES")
    per_segment = _cpp_int(
        "Gfx950AttentionUnified3DGeometry.hpp",
        "GFX950_UNIFIED_3D_REDUCE_LDS_BYTES_PER_SEGMENT",
    )
    assert cap == ArchTarget.from_gfx("gfx950").lds_capacity_bytes
    assert cap // per_segment == reduce_max_segments("gfx950")
    text = (_packs_dir() / "Gfx950AttentionUnified3DGeometry.hpp").read_text()
    assert (
        "GFX950_UNIFIED_3D_MAX_SEGMENTS\n"
        "    = GFX950_LDS_CAPACITY_BYTES / GFX950_UNIFIED_3D_REDUCE_LDS_BYTES_PER_SEGMENT;"
    ) in text
