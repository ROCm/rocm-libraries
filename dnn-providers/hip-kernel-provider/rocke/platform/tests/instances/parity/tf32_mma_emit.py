# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
from rocke.instances.gfx942.tf32_mma_probe import (
    PREPARATIONS,
    Tf32MmaProbeSpec,
    build_tf32_mma_probe,
)
from _emit_common import run_emit


def _spec(idx):
    if not 0 <= idx < 10:
        raise SystemExit(f"unknown config index {idx}")
    return Tf32MmaProbeSpec(16 if idx < 5 else 32, PREPARATIONS[idx % 5])


if __name__ == "__main__":
    raise SystemExit(
        run_emit(_spec, lambda spec, arch: build_tf32_mma_probe(spec), arch="gfx942")
    )
