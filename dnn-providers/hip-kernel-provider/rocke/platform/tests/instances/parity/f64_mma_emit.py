# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
from _emit_common import run_emit
from rocke.instances.common.f64_mma_probe import F64MmaProbeSpec, build_f64_mma_probe

# (m, k, arch): every shape on each target whose catalog lists the f64 atoms.
# gfx942 comes first so that its blessed config indices 0-5 stay stable; a new
# target appends.
_CONFIGS = tuple(
    (m, k, arch)
    for arch in ("gfx942", "gfx90a", "gfx950")
    for m, k in ((16, 4), (16, 8), (16, 64), (4, 4), (4, 8), (4, 64))
)


def _spec(idx):
    if not 0 <= idx < len(_CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    m, k, arch = _CONFIGS[idx]
    return F64MmaProbeSpec(m, k), arch


def main() -> int:
    return run_emit(_spec, build_f64_mma_probe)


if __name__ == "__main__":
    raise SystemExit(main())
