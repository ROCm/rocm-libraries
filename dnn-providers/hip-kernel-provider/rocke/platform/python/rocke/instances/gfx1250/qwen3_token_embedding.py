# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 token-embedding gather kernel.

The model-input op: gather rows of the embedding table for a batch of token ids.
``out[t, :] = table[input_ids[t], :]`` for ``table`` of shape
``[vocab, hidden]``. Pure vectorised copy (no compute), so it is arch-neutral
(builds + runs on gfx1250 and on CDNA wave64); kept under the gfx1250 namespace
as part of the Qwen3-30B-A3B day-0 operator set.

Each thread copies one ``vec``-wide chunk of one token's hidden row; the grid is
``ceil(tokens * hidden/vec / block_size)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ...core.ir import BF16, F16, I32, IRBuilder, KernelDef, PtrType, Type

__all__ = [
    "Qwen3TokenEmbeddingSpec",
    "build_qwen3_token_embedding",
    "qwen3_token_embedding_grid",
    "qwen3_token_embedding_signature",
]


def _dtype_ir(dtype: str) -> Type:
    if dtype in ("fp16", "f16"):
        return F16
    if dtype == "bf16":
        return BF16
    raise ValueError(f"unsupported dtype {dtype!r} (expected fp16/bf16)")


def _require_supported(arch: str) -> Tuple[bool, str]:
    from ...core.arch import ArchTarget

    try:
        ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    return True, "supported"


@dataclass(frozen=True)
class Qwen3TokenEmbeddingSpec:
    """Token-embedding row gather. ``hidden`` must be a multiple of ``vec``."""

    hidden: int = 2048
    dtype: str = "bf16"
    vec: int = 8
    block_size: int = 256
    name: str = "rocke_gfx1250_qwen3_token_embedding"

    def __post_init__(self) -> None:
        if self.dtype not in ("fp16", "f16", "bf16"):
            raise ValueError("dtype must be fp16/bf16")
        if self.vec not in (1, 2, 4, 8):
            raise ValueError("vec must be 1/2/4/8")
        if self.hidden <= 0 or self.hidden % self.vec != 0:
            raise ValueError("hidden must be a positive multiple of vec")

    def kernel_name(self) -> str:
        from ...helpers.spec import kernel_name_join

        return kernel_name_join(
            self.name, f"h{self.hidden}", self.dtype, f"v{self.vec}"
        )


def qwen3_token_embedding_grid(
    num_tokens: int, spec: Qwen3TokenEmbeddingSpec
) -> Tuple[int, int, int]:
    vpr = spec.hidden // spec.vec
    total = num_tokens * vpr
    gx = (total + spec.block_size - 1) // spec.block_size
    return (gx, 1, 1)


def qwen3_token_embedding_signature(spec: Qwen3TokenEmbeddingSpec):
    from ...helpers.spec import SignatureBuilder

    dt = spec.dtype if spec.dtype != "f16" else "fp16"
    return (
        SignatureBuilder()
        .ptr("input_ids", "i32")
        .ptr("table", dt)
        .ptr("out", dt)
        .scalar("num_tokens", "i32")
        .build()
    )


def build_qwen3_token_embedding(
    spec: Qwen3TokenEmbeddingSpec, *, arch: str = "gfx1250"
) -> KernelDef:
    ok, reason = _require_supported(arch)
    if not ok:
        raise NotImplementedError(reason)

    dt = _dtype_ir(spec.dtype)
    H = spec.hidden
    vec = spec.vec
    vpr = H // vec
    bs = spec.block_size
    align = vec * 2  # bf16/fp16 = 2 bytes

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = bs

    input_ids = b.param(
        "input_ids", PtrType(I32, "global"), noalias=True, readonly=True, align=16
    )
    table = b.param(
        "table", PtrType(dt, "global"), noalias=True, readonly=True, align=16
    )
    out = b.param("out", PtrType(dt, "global"), noalias=True, writeonly=True, align=16)
    num_tokens = b.param("num_tokens", I32)

    c_vpr = b.const_i32(vpr)
    c_vec = b.const_i32(vec)
    c_H = b.const_i32(H)

    tid = b.add(b.mul(b.block_id_x(), b.const_i32(bs)), b.thread_id_x())
    total = b.mul(num_tokens, c_vpr)

    with b.scf_if(b.cmp_lt(tid, total)):
        token = b.div(tid, c_vpr)
        vcol = b.mod(tid, c_vpr)
        col = b.mul(vcol, c_vec)
        tok_id = b.global_load_i32(input_ids, token)
        src = b.add(b.mul(tok_id, c_H), col)
        dst = b.add(b.mul(token, c_H), col)
        v = b.global_load_vN(table, src, dt, vec, align=align)
        b.global_store_vN(out, dst, v, vec, align=align)

    return b.kernel
