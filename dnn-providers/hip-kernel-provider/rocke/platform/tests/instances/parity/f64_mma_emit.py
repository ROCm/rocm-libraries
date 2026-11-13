# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
from _emit_common import run_emit
from rocke.instances.gfx942.f64_mma_probe import F64MmaProbeSpec, build_f64_mma_probe

_CONFIGS = ((16, 4), (16, 8), (16, 64), (4, 4), (4, 8), (4, 64))


def _spec(idx):
    if not 0 <= idx < len(_CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    return F64MmaProbeSpec(*_CONFIGS[idx])


def main() -> int:
    return run_emit(_spec, lambda spec, arch: build_f64_mma_probe(spec), arch="gfx942")


if __name__ == "__main__":
    raise SystemExit(main())
