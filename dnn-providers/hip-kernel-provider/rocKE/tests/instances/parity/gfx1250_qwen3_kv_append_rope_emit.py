#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx1250_qwen3_kv_append_rope_emit.py -- Python reference emitter
# for the gfx1250 Qwen3 KV append/RoPE parity harness.
from rocke.instances.gfx1250.qwen3_kv_cache import (
    Qwen3KvAppendRopeSpec,
    build_qwen3_kv_append_rope,
)
from _emit_common import run_emit


def _spec(idx: int) -> Qwen3KvAppendRopeSpec:
    if idx == 0:
        return Qwen3KvAppendRopeSpec()
    if idx == 1:
        return Qwen3KvAppendRopeSpec(kv_storage_dtype="fp8e4m3")
    if idx == 2:
        return Qwen3KvAppendRopeSpec(kv_storage_dtype="bf8e5m2")
    if idx == 3:
        return Qwen3KvAppendRopeSpec(use_rope=False)
    if idx == 4:
        return Qwen3KvAppendRopeSpec(input_dtype="fp16", kv_storage_dtype="fp8e4m3")
    if idx == 5:
        return Qwen3KvAppendRopeSpec(
            num_kv_heads=8, kv_storage_dtype="bf8e5m2", use_rope=False
        )
    raise SystemExit(f"unknown config index {idx}")


def _build(spec, arch="gfx1250"):
    return build_qwen3_kv_append_rope(spec, arch=arch)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: gfx1250_qwen3_kv_append_rope_emit.py <config_index 0..5>\n",
        arch="gfx1250",
    )


if __name__ == "__main__":
    raise SystemExit(main())
