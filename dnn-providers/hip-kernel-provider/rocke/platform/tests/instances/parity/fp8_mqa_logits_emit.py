#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Python reference emitter for gfx942 FP8-logits byte identity.
from rocke.instances.gfx942.fp8_mqa_logits import (
    Fp8MqaLogitsSpec,
    build_fp8_mqa_logits,
)
from _emit_common import run_emit


def _spec(idx: int):
    if idx == 0:
        spec = Fp8MqaLogitsSpec()
    elif idx == 1:
        spec = Fp8MqaLogitsSpec(waves_per_block=2)
    elif idx == 2:
        spec = Fp8MqaLogitsSpec(head_dim=64)
    else:
        raise SystemExit(f"unknown config index {idx}")
    return spec, "gfx942"


def main() -> int:
    return run_emit(
        _spec,
        build_fp8_mqa_logits,
        usage="usage: fp8_mqa_logits_emit.py <config_index 0..2> [mode]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
