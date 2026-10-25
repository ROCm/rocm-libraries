# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
from _emit_common import run_emit
from rocke.instances.common.f64_mma_probe import F64MmaProbeSpec, build_f64_mma_probe

# (m, k, arch): the gfx942 rows first, then the same atoms on gfx90a / gfx950.
_CONFIGS = (
    (16, 4, "gfx942"),
    (16, 8, "gfx942"),
    (16, 64, "gfx942"),
    (4, 4, "gfx942"),
    (4, 8, "gfx942"),
    (4, 64, "gfx942"),
    (16, 8, "gfx90a"),
    (4, 8, "gfx90a"),
    (16, 8, "gfx950"),
    (4, 8, "gfx950"),
)


def _spec(idx):
    if not 0 <= idx < len(_CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    m, k, arch = _CONFIGS[idx]
    return F64MmaProbeSpec(m, k), arch


def main() -> int:
    return run_emit(_spec, lambda spec, arch: build_f64_mma_probe(spec, arch))


if __name__ == "__main__":
    raise SystemExit(main())
