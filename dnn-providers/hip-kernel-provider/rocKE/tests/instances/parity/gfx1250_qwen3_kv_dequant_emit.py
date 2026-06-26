#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx1250_qwen3_kv_dequant_emit.py -- Python reference emitter for
# the gfx1250 Qwen3 KV dequant-smoke parity harness.
from rocke.instances.gfx1250.qwen3_kv_cache import (
    Qwen3KvDequantSpec,
    build_qwen3_kv_dequant_smoke,
)
from _emit_common import run_emit


def _spec(idx: int) -> Qwen3KvDequantSpec:
    if idx == 0:
        return Qwen3KvDequantSpec(kv_storage_dtype="fp8e4m3")
    if idx == 1:
        return Qwen3KvDequantSpec(kv_storage_dtype="bf8e5m2")
    if idx == 2:
        return Qwen3KvDequantSpec(kv_storage_dtype="fp8e4m3", output_dtype="fp16")
    if idx == 3:
        return Qwen3KvDequantSpec(kv_storage_dtype="bf8e5m2", output_dtype="fp16")
    raise SystemExit(f"unknown config index {idx}")


def _build(spec, arch="gfx1250"):
    return build_qwen3_kv_dequant_smoke(spec, arch=arch)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: gfx1250_qwen3_kv_dequant_emit.py <config_index 0..3>\n",
        arch="gfx1250",
    )


if __name__ == "__main__":
    raise SystemExit(main())
