# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Standalone Python oracle for pointer-selected peer GEMM and existing MFMA reuse."""

from __future__ import annotations

from _emit_common import run_emit
from rocke.instances.common.peer_gemm import PeerGemmSpec, build_peer_gemm


def _spec(index: int) -> tuple[PeerGemmSpec, str]:
    configs = [
        (16, 16, 16, 1, "f16"),
        (16, 16, 16, 2, "bf16"),
        (32, 16, 32, 3, "f16"),
        (16, 32, 64, 8, "bf16"),
        (32, 32, 64, 4, "f16"),
        (32, 32, 32, 3, "bf16"),
    ]
    if not 0 <= index < len(configs) * 2:
        raise SystemExit(f"unknown config index {index}")
    return PeerGemmSpec(*configs[index // 2]), "gfx942" if index % 2 else "gfx950"


if __name__ == "__main__":
    raise SystemExit(run_emit(_spec, build_peer_gemm))
