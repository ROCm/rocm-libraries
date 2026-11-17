# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Python reference for peer tile-copy geometry, masks, and i64 memory indices."""

from __future__ import annotations

from _emit_common import run_emit
from rocke.instances.common.peer_tile_copy import PeerTileCopySpec, build_peer_tile_copy


def _spec(index: int) -> tuple[PeerTileCopySpec, str]:
    configs = [
        (4, 64, 64, "bf16", "gfx950"),
        (4, 64, 64, "bf16", "gfx942"),
        (3, 17, 64, "f16", "gfx950"),
        (3, 17, 64, "f16", "gfx942"),
        (7, 33, 128, "bf16", "gfx950"),
        (7, 33, 128, "bf16", "gfx942"),
        (9, 65, 256, "f16", "gfx950"),
        (9, 65, 256, "f16", "gfx942"),
        (1, 1, 256, "bf16", "gfx950"),
        (1, 1, 256, "bf16", "gfx942"),
    ]
    if not 0 <= index < len(configs):
        raise SystemExit(f"unknown config index {index}")
    rows, cols, block, dtype, arch = configs[index]
    return PeerTileCopySpec(rows, cols, block, dtype), arch


if __name__ == "__main__":
    raise SystemExit(run_emit(_spec, build_peer_tile_copy))
