# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The split-KV reduce kernel's segment bound, sourced from the tools, not by hand.

``supports_tiled_reduce_spec`` refuses a ``num_segments`` whose LDS factor cache
(``reduce_lds_bytes``) does not fit ``ArchTarget.lds_capacity_bytes``. These cases hold
that formula to what comgr actually allocates (the code object's
``group_segment_fixed_size``), and hold the C++ host's mirror,
``GFX950_UNIFIED_3D_MAX_SEGMENTS``, to the same arch bound.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest

_PACKS = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "engines"
    / "kernel_ingestor_engine"
    / "packs"
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


@pytest.mark.parametrize("num_segments", [16, 1000])
def test_reduce_lds_formula_is_what_comgr_allocates(num_segments):
    from kernels.gfx950.attention_tiled_3d import reduce_lds_bytes

    assert _compiled_lds_bytes(_reduce(num_segments), "gfx950") == reduce_lds_bytes(
        num_segments
    )


def _cpp_int(header: str, name: str) -> int:
    text = (_PACKS / header).read_text()
    match = re.search(rf"inline constexpr int64_t {name} = (\d+);", text)
    assert match, f"{name} not found as a literal in {header}"
    return int(match.group(1))


def test_cpp_segment_bound_mirrors_the_arch_bound():
    """The C++ host bound is gfx950's LDS capacity over the reduce's bytes per segment,
    each restated in C++ from the Python source of truth."""
    from rocke.core.arch.target import ArchTarget

    from kernels.gfx950 import attention_tiled_3d as mod

    assert (
        _cpp_int("Gfx950AttentionDenseGeometry.hpp", "GFX950_LDS_CAPACITY_BYTES")
        == ArchTarget.from_gfx("gfx950").lds_capacity_bytes
    )
    assert _cpp_int(
        "Gfx950AttentionUnified3DGeometry.hpp",
        "GFX950_UNIFIED_3D_REDUCE_LDS_BYTES_PER_SEGMENT",
    ) == mod.reduce_lds_bytes(1)
    text = (_PACKS / "Gfx950AttentionUnified3DGeometry.hpp").read_text()
    assert (
        "GFX950_UNIFIED_3D_MAX_SEGMENTS\n"
        "    = GFX950_LDS_CAPACITY_BYTES / GFX950_UNIFIED_3D_REDUCE_LDS_BYTES_PER_SEGMENT;"
    ) in text
