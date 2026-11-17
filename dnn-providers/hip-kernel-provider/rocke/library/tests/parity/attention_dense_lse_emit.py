#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/attention_dense_lse_emit.py -- Python reference emitter for the
# dense-attention LSE helpers in kernels/common/_lse_store.py: lse_store,
# dense_row_has_key and zero_keyless_row.
#
# The gfx942 and gfx950 dense builders call these from their epilogues. Each
# config builds the smallest kernel that reaches one branch of them (causal or
# not, diagonal shift, window, right bound, O store width, dtype), so a C/Python
# divergence shows up against a one-screen kernel. Byte-compared against
# attention_dense_lse_emit.c.
#
# Every builder call is its own statement in the same order the C emitter
# issues it: each call consumes an SSA id.
#
# arch = gfx950, llvm_flavor = AUTO, matching the C side.
from kernels.common._lse_store import dense_row_has_key, lse_store, zero_keyless_row

from rocke.core.ir import BF16, F16, F32, I32, IRBuilder, KernelDef, PtrType

from _emit_common import run_emit

_DTYPES = {"bf16": BF16, "fp16": F16}

# (causal, diagonal, window, right_bound, dtype, o_store_width)
_CONFIGS = (
    (True, 0, 0, 0, "bf16", 4),  # 0: plain causal
    (False, 0, 0, 0, "fp16", 4),  # 1: full attention
    (True, 64, 0, 0, "bf16", 2),  # 2: bottom-right diagonal shift
    (True, 0, 128, 0, "bf16", 1),  # 3: causal window, scalar O store
    (True, 0, 128, 64, "fp16", 4),  # 4: two-sided band
    (False, 0, 128, 0, "bf16", 2),  # 5: left-only window
    (True, 192, 0, 0, "fp16", 4),  # 6: diagonal shift, fp16
    (True, 0, 0, 32, "bf16", 1),  # 7: lookahead only
)


def _spec(idx: int):
    if not 0 <= idx < len(_CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    # run_emit reads a tuple as (spec, arch), so the config tuple is wrapped.
    return _CONFIGS[idx], "gfx950"


def _build(cfg, *, arch: str = "gfx950") -> KernelDef:
    causal, diagonal, window, right_bound, dtype_name, width = cfg
    dtype = _DTYPES[dtype_name]
    b = IRBuilder("attention_dense_lse")
    lse = b.param("lse", PtrType(F32, "global"))
    o = b.param("o", PtrType(dtype, "global"))
    kv_len = b.param("kv_len", I32)
    q_base = b.param("q_base", I32)
    m = b.param("m", F32)
    row_sum = b.param("l", F32)
    tid = b.thread_id_x()
    query_row = b.add(q_base, tid)
    has_key = dense_row_has_key(
        b,
        query_row,
        kv_len,
        causal=causal,
        diagonal=diagonal,
        window=window,
        right_bound=right_bound,
    )
    lse_store(b, lse, tid, m, row_sum, row_has_mass=has_key)
    val = b.cast_f32_to(m, dtype)
    packed = b.vec_pack([val] * width, dtype)
    packed = zero_keyless_row(b, packed, has_key, dtype, width)
    stride = b.const_i32(width)
    off = b.mul(tid, stride)
    b.global_store_vN(o, off, packed, width, align=2 * width)
    b.ret()
    return b.kernel


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: attention_dense_lse_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
