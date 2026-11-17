# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""C++ builder/binding comparisons against the independent Python builder."""

from __future__ import annotations

from dataclasses import asdict, replace

import pytest
from rocke.core.ir_serialize import serialize
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.instances.common.peer_tile_copy import (
    PeerTileCopySpec,
    build_peer_tile_copy,
    is_valid_spec,
    peer_tile_copy_grid,
)

engine = pytest.importorskip("rocke_engine")


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize(
    "geometry", [(4, 64, 64), (3, 17, 64), (7, 33, 128), (9, 65, 256), (1, 1, 256)]
)
def test_independent_cpp_builder_matches_python(arch, dtype, geometry) -> None:
    spec = PeerTileCopySpec(*geometry, dtype=dtype)
    values = asdict(spec)
    kernel = build_peer_tile_copy(spec, arch)
    assert engine.peer_tile_copy_serialize_ir(values, arch=arch) == serialize(kernel)
    assert engine.peer_tile_copy_lower_llvm(
        values, arch=arch
    ) == _lower_kernel_to_llvm_python(kernel, arch=arch)
    assert engine.peer_tile_copy_verify(values, arch=arch) == []
    assert engine.peer_tile_copy_is_valid(values, arch=arch) == is_valid_spec(
        spec, arch
    )
    assert engine.peer_tile_copy_kernel_name(values) == spec.kernel_name()


@pytest.mark.parametrize(
    "change",
    [
        {"dtype": "f32"},
        {"dtype": None},
        {"tile_rows": 0},
        {"tile_cols": -1},
        {"block_size": 32},
        {"tile_rows": 4097},
        {"name": ""},
        {"name": "a" * 256},
    ],
)
def test_cpp_validation_reasons_match_python(change) -> None:
    spec = replace(PeerTileCopySpec(), **change)
    assert engine.peer_tile_copy_is_valid(asdict(spec)) == is_valid_spec(spec)
    with pytest.raises(RuntimeError, match="peer tile copy build failed"):
        engine.peer_tile_copy_lower_llvm(asdict(spec))


@pytest.mark.parametrize("shape", [(0, 0), (0, 7), (7, 0), (1, 1), (5, 67)])
def test_cpp_grid_matches_python(shape) -> None:
    spec = PeerTileCopySpec()
    assert engine.peer_tile_copy_grid(*shape, asdict(spec)) == peer_tile_copy_grid(
        *shape, spec
    )


@pytest.mark.parametrize("shape", [(-1, 0), (1 << 62, 1)])
def test_cpp_grid_rejects_invalid_axes(shape) -> None:
    with pytest.raises(ValueError):
        engine.peer_tile_copy_grid(*shape, asdict(PeerTileCopySpec()))
