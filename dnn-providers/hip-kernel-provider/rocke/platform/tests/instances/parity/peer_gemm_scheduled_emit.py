# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Standalone oracle for finite scheduled peer GEMM work; no GPU runtime."""

from __future__ import annotations

from _emit_common import run_emit
from rocke.instances.common.peer_gemm import (
    PeerGemmScheduleSpec,
    PeerGemmSpec,
    build_scheduled_peer_gemm,
)


def _spec(index: int) -> tuple[PeerGemmScheduleSpec, str]:
    shapes = [
        (16, 16, 16, 1, "f16"),
        (16, 16, 16, 2, "bf16"),
        (32, 16, 32, 3, "f16"),
        (16, 32, 64, 8, "bf16"),
        (32, 32, 64, 4, "f16"),
        (32, 32, 32, 3, "bf16"),
    ]
    if not 0 <= index < 72:
        raise SystemExit(f"unknown config index {index}")
    config = index // 2
    cap = (0, 1, 3)[config % 3]
    order = ("source_grouped", "tile_interleaved")[(config // 3) % 2]
    gemm = PeerGemmSpec(*shapes[config // 6])
    return PeerGemmScheduleSpec(gemm, order, cap), "gfx942" if index % 2 else "gfx950"


if __name__ == "__main__":
    raise SystemExit(run_emit(_spec, build_scheduled_peer_gemm))
