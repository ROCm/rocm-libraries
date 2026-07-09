# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared Qwen3-30B-A3B shape fixtures for gfx1250 day-0 operator work.

The gfx950 Qwen walkthrough is benchmark-oriented and imports torch at module
load time. This file is intentionally small and GPU-free so operator sandbox
agents can import the same model geometry in no-GPU tests, build-only checks,
and remote gfx1250 validation scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Literal, Mapping, Tuple


ARCH = "gfx1250"
ISA = "amdgcn-amd-amdhsa--gfx1250"
DTYPE = "bf16"
WAVE_SIZE = 32

BATCH = 2
HIDDEN = 2048
HIDDEN_SIZE = HIDDEN
MOE_INTER = 768
NUM_EXPERTS = 128
TOPK = 8
NHEAD_Q = 32
NHEAD_K = 4
HEAD_DIM = 64
BLOCK_SIZE = 16
QKV_PROJ = (NHEAD_Q + 2 * NHEAD_K) * HEAD_DIM
ATTENTION_WIDTH = NHEAD_Q * HEAD_DIM

ROUTER_BLOCK_SIZE = 128
SORT_BLOCK_SIZE = 128
NORM_BLOCK_SIZE = 256
NORM_VEC = 8
STATIC_SLOT_SIZE = BATCH * TOPK
RDQUANT_OUT_DTYPES: Tuple[str, ...] = ("i8", "fp8e4m3", "bf8e5m2")
DECODE_KV_LENS: Tuple[int, ...] = (512, 1024, 2048, 4096)
PREFILL_Q_LENS: Tuple[int, ...] = (64, 128)

AttentionMode = Literal["prefill_2d", "decode_3d"]
KvStorageDType = Literal["bf16", "fp8e4m3", "bf8e5m2"]


@dataclass(frozen=True)
class Qwen3A3BConfig:
    batch: int = 2
    hidden: int = 2048
    moe_intermediate: int = 768
    num_experts: int = 128
    topk: int = 8
    num_query_heads: int = 32
    num_kv_heads: int = 4
    head_dim: int = 64
    block_size: int = 16
    dtype: str = DTYPE
    arch: str = ARCH
    isa: str = ISA

    @property
    def qkv_width(self) -> int:
        return (self.num_query_heads + 2 * self.num_kv_heads) * self.head_dim

    @property
    def attention_width(self) -> int:
        return self.num_query_heads * self.head_dim

    @property
    def active_moe_pairs(self) -> int:
        return self.batch * self.topk

    def with_overrides(self, **kwargs: int | str) -> "Qwen3A3BConfig":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class GemmShape:
    name: str
    m: int
    n: int
    k: int
    dtype: str = "bf16"
    layout: str = "RCR"
    role: str = "dense"
    op: str = "dense"

    @property
    def M(self) -> int:
        return self.m

    @property
    def N(self) -> int:
        return self.n

    @property
    def K(self) -> int:
        return self.k


@dataclass(frozen=True)
class AttentionShape:
    name: str
    mode: AttentionMode
    total_q: int
    num_seqs: int
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    block_size: int
    max_seqlen_q: int
    max_seqlen_k: int
    dtype: str = "bf16"
    kv_storage_dtype: str = "bf16"

    @property
    def batch(self) -> int:
        return self.num_seqs

    @property
    def head_dim(self) -> int:
        return self.head_size

    @property
    def q_len(self) -> int:
        return self.max_seqlen_q

    @property
    def kv_len(self) -> int:
        return self.max_seqlen_k

    @property
    def num_queries_per_kv(self) -> int:
        return self.num_query_heads // self.num_kv_heads

    @property
    def num_kv_blocks(self) -> int:
        return (self.kv_len + self.block_size - 1) // self.block_size


@dataclass(frozen=True)
class MoeShape:
    tokens: int
    experts: int
    topk: int
    hidden: int
    intermediate: int
    dtype: str = "bf16"
    arch: str = ARCH
    router_block_size: int = ROUTER_BLOCK_SIZE
    sort_block_size: int = SORT_BLOCK_SIZE
    static_slot_size: int = STATIC_SLOT_SIZE

    @property
    def active_pairs(self) -> int:
        return self.tokens * self.topk

    @property
    def total_pairs(self) -> int:
        return self.active_pairs

    def to_fused_moe_forward_spec(self, *, use_static_offsets: bool = True):
        from rocke.instances.common.fused_moe_e2e import FusedMoeForwardSpec

        return FusedMoeForwardSpec(
            tokens=self.tokens,
            experts=self.experts,
            topk=self.topk,
            hidden=self.hidden,
            intermediate=self.intermediate,
            dtype=self.dtype,
            sort_block_size=self.sort_block_size,
            router_block_size=self.router_block_size,
            arch=self.arch,
            use_static_offsets=use_static_offsets,
            static_slot_size=self.static_slot_size if use_static_offsets else None,
            preshuffle_w_gate_up_interleaved=True,
            active_tile_skip_gemms=True,
        )


@dataclass(frozen=True)
class QwenOpShape:
    """One normalized operator shape from a single decode layer."""

    name: str
    kind: str
    dims: Mapping[str, int | str]
    gate: str
    fallback_backend: str
    graph_capture_safe: bool
    operator_dependency: str


def _dims(**kwargs: int | str) -> Mapping[str, int | str]:
    return dict(kwargs)


def qwen3_30b_a3b_config(batch: int = 2) -> Qwen3A3BConfig:
    return Qwen3A3BConfig(batch=batch)


def decode_gemm_shapes(batch: int = 2) -> Tuple[GemmShape, ...]:
    cfg = qwen3_30b_a3b_config(batch)
    return (
        GemmShape(
            "qkv_decode_bf16",
            cfg.batch,
            cfg.qkv_width,
            cfg.hidden,
            role="attention",
            op="qkv",
        ),
        GemmShape(
            "o_decode_bf16",
            cfg.batch,
            cfg.hidden,
            cfg.attention_width,
            role="attention",
            op="o",
        ),
        GemmShape(
            "router_decode_bf16",
            cfg.batch,
            cfg.num_experts,
            cfg.hidden,
            role="moe",
            op="router",
        ),
    )


def prefill_gemm_shapes(total_tokens: int, batch: int = 2) -> Tuple[GemmShape, ...]:
    cfg = qwen3_30b_a3b_config(batch)
    return (
        GemmShape(
            "qkv_prefill_bf16",
            total_tokens,
            cfg.qkv_width,
            cfg.hidden,
            role="attention",
            op="prefill_qkv",
        ),
        GemmShape(
            "o_prefill_bf16",
            total_tokens,
            cfg.hidden,
            cfg.attention_width,
            role="attention",
            op="prefill_o",
        ),
        GemmShape(
            "router_prefill_bf16",
            total_tokens,
            cfg.num_experts,
            cfg.hidden,
            role="moe",
            op="prefill_router",
        ),
    )


def decode_attention_shapes(
    batch: int = 2, kv_lens: Iterable[int] = (512, 1024, 2048, 4096)
) -> Tuple[AttentionShape, ...]:
    cfg = qwen3_30b_a3b_config(batch)
    return tuple(
        AttentionShape(
            name=f"decode_kv{kv_len}",
            mode="decode_3d",
            total_q=cfg.batch,
            num_seqs=cfg.batch,
            num_query_heads=cfg.num_query_heads,
            num_kv_heads=cfg.num_kv_heads,
            head_size=cfg.head_dim,
            block_size=cfg.block_size,
            max_seqlen_q=1,
            max_seqlen_k=kv_len,
            dtype=cfg.dtype,
        )
        for kv_len in kv_lens
    )


def prefill_attention_shape(
    seq_len: int, batch: int = 1, *, kv_storage_dtype: str = "bf16"
) -> AttentionShape:
    cfg = qwen3_30b_a3b_config(batch)
    total_q = batch * seq_len
    return AttentionShape(
        name=f"prefill_s{seq_len}_b{batch}_{kv_storage_dtype}",
        mode="prefill_2d",
        total_q=total_q,
        num_seqs=batch,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_size=cfg.head_dim,
        block_size=cfg.block_size,
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
        dtype=cfg.dtype,
        kv_storage_dtype=kv_storage_dtype,
    )


def moe_shape(batch: int = 2) -> MoeShape:
    cfg = qwen3_30b_a3b_config(batch)
    return MoeShape(
        tokens=cfg.batch,
        experts=cfg.num_experts,
        topk=cfg.topk,
        hidden=cfg.hidden,
        intermediate=cfg.moe_intermediate,
        dtype=cfg.dtype,
    )


QWEN3_30B_A3B_CONFIG = Qwen3A3BConfig()
QWEN3_30B_A3B_DECODE = MoeShape(
    tokens=BATCH, experts=NUM_EXPERTS, topk=TOPK, hidden=HIDDEN, intermediate=MOE_INTER
)
QWEN3_30B_A3B_DECODE_MOE = QWEN3_30B_A3B_DECODE


def qwen3_decode_attention_shapes(
    *,
    kv_lens: Iterable[int] = DECODE_KV_LENS,
    kv_storage_dtype: KvStorageDType = "bf16",
) -> Tuple[AttentionShape, ...]:
    return tuple(
        replace(shape, kv_storage_dtype=kv_storage_dtype)
        for shape in decode_attention_shapes(kv_lens=kv_lens)
    )


def qwen3_prefill_attention_shapes(
    *,
    q_lens: Iterable[int] = PREFILL_Q_LENS,
    kv_storage_dtype: KvStorageDType = "bf16",
) -> Tuple[AttentionShape, ...]:
    return tuple(
        prefill_attention_shape(q_len, batch=1, kv_storage_dtype=kv_storage_dtype)
        for q_len in q_lens
    )


def qwen3_kv_dequant_shapes() -> Tuple[AttentionShape, ...]:
    return (
        *qwen3_decode_attention_shapes(kv_lens=(512,), kv_storage_dtype="fp8e4m3"),
        *qwen3_decode_attention_shapes(kv_lens=(512,), kv_storage_dtype="bf8e5m2"),
    )


def layer_shapes(
    config: Qwen3A3BConfig = QWEN3_30B_A3B_CONFIG, *, kv_len: int = 1024
) -> Tuple[QwenOpShape, ...]:
    num_kv_blocks = (kv_len + config.block_size - 1) // config.block_size
    return (
        QwenOpShape(
            "input_rmsnorm",
            "rmsnorm_add",
            _dims(M=config.batch, N=config.hidden, dtype=config.dtype),
            "norm",
            "aiter.rmsnorm2d_fwd_with_add_or_torch",
            True,
            "gfx1250 graph-safe RMSNorm custom op/kernel",
        ),
        QwenOpShape(
            "qkv_proj",
            "dense_gemm",
            _dims(
                M=config.batch,
                N=config.qkv_width,
                K=config.hidden,
                dtype=config.dtype,
                layout="RCR",
            ),
            "gemm",
            "ATOM LinearBase / torch.matmul",
            False,
            "gfx1250 bf16 skinny GEMM operator branch",
        ),
        QwenOpShape(
            "decode_attention",
            "decode_attention",
            _dims(
                batch=config.batch,
                nhead_q=config.num_query_heads,
                nhead_k=config.num_kv_heads,
                head_dim=config.head_dim,
                block_size=config.block_size,
                kv_len=kv_len,
                num_kv_blocks=num_kv_blocks,
                dtype=config.dtype,
            ),
            "attention",
            "AITER unified_attention / ATOM decode path",
            False,
            "gfx1250 paged decode attention operator branch",
        ),
        QwenOpShape(
            "o_proj",
            "dense_gemm",
            _dims(
                M=config.batch,
                N=config.hidden,
                K=config.attention_width,
                dtype=config.dtype,
                layout="RCR",
            ),
            "gemm",
            "ATOM LinearBase / torch.matmul",
            False,
            "gfx1250 bf16 skinny GEMM operator branch",
        ),
        QwenOpShape(
            "post_attention_rmsnorm",
            "rmsnorm_add",
            _dims(M=config.batch, N=config.hidden, dtype=config.dtype),
            "norm",
            "aiter.rmsnorm2d_fwd_with_add_or_torch",
            True,
            "gfx1250 graph-safe RMSNorm custom op/kernel",
        ),
        QwenOpShape(
            "router_topk",
            "router_topk",
            _dims(
                tokens=config.batch,
                experts=config.num_experts,
                topk=config.topk,
                dtype="fp32",
            ),
            "routing",
            "aiter.moe_fused_gate_or_torch_topk",
            True,
            "gfx1250 graph-safe router/topk custom op/kernel",
        ),
        QwenOpShape(
            "moe_e2e",
            "moe_e2e",
            _dims(
                tokens=config.batch,
                experts=config.num_experts,
                topk=config.topk,
                hidden=config.hidden,
                intermediate=config.moe_intermediate,
                dtype=config.dtype,
            ),
            "moe",
            "aiter.fused_moe / ATOM MoE path",
            False,
            "gfx1250 fused MoE operator branch",
        ),
    )


QWEN3_30B_A3B_SHAPES = layer_shapes(QWEN3_30B_A3B_CONFIG)
ALL_BF16_GEMM_SHAPES = decode_gemm_shapes() + prefill_gemm_shapes(total_tokens=128)
LOWBIT_BLOCK_SCALED_GEMM_SHAPES = tuple(
    replace(shape, dtype=dtype)
    for shape in ALL_BF16_GEMM_SHAPES
    for dtype in ("fp8e4m3", "bf8e5m2")
)
EXPERT_BLOCK_SCALED_GEMM_SHAPES = tuple(
    GemmShape(name, STATIC_SLOT_SIZE, n, k, dtype=dtype, role="moe", op=op)
    for name, n, k, op in (
        ("expert_gate", MOE_INTER, HIDDEN, "expert_gate"),
        ("expert_up", MOE_INTER, HIDDEN, "expert_up"),
        ("expert_down", HIDDEN, MOE_INTER, "expert_down"),
    )
    for dtype in ("fp8e4m3", "bf8e5m2")
)
ALL_GEMM_SHAPES = (
    ALL_BF16_GEMM_SHAPES
    + LOWBIT_BLOCK_SCALED_GEMM_SHAPES
    + EXPERT_BLOCK_SCALED_GEMM_SHAPES
)


def shape_by_name(name: str) -> GemmShape:
    for shape in ALL_GEMM_SHAPES:
        if shape.name == name:
            return shape
    raise KeyError(name)


def bf16_universal_gemm_spec(shape: GemmShape):
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
    )

    return UniversalGemmSpec(
        name=f"gfx1250_qwen_{shape.name}",
        tile=TileSpec(
            tile_m=16,
            tile_n=16,
            tile_k=32,
            warp_m=1,
            warp_n=1,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=32,
        ),
        trait=TraitSpec(
            pipeline="mem", epilogue="default", pad_m=True, pad_n=True, pad_k=True
        ),
        data=DataSpec(
            dtype_a="bf16",
            dtype_b="bf16",
            dtype_c="bf16",
            dtype_acc="fp32",
            layout=shape.layout,
        ),
        wave_size=WAVE_SIZE,
    )


def add_rmsnorm_rdquant_specs(
    shape: MoeShape = QWEN3_30B_A3B_DECODE,
) -> Tuple[object, ...]:
    from rocke.instances.common.add_rmsnorm2d_rdquant import AddRmsnorm2DRdquantSpec

    return tuple(
        AddRmsnorm2DRdquantSpec(
            n_per_block=shape.hidden,
            dtype=shape.dtype,
            out_dtype=out_dtype,  # type: ignore[arg-type]
            block_size=NORM_BLOCK_SIZE,
            vec=NORM_VEC,
            wave_size=WAVE_SIZE,
        )
        for out_dtype in RDQUANT_OUT_DTYPES
    )


def topk_softmax_spec(shape: MoeShape = QWEN3_30B_A3B_DECODE):
    from rocke.instances.common.topk_softmax import TopkSoftmaxSpec

    return TopkSoftmaxSpec(
        n_per_row=shape.experts,
        k=shape.topk,
        dtype="f32",
        out_dtype="f32",
        block_size=shape.router_block_size,
    )


def moe_sorting_spec(shape: MoeShape = QWEN3_30B_A3B_DECODE):
    from rocke.instances.common.moe_sorting import MoeSortingSpec

    return MoeSortingSpec(
        tokens=shape.tokens,
        topk=shape.topk,
        experts=shape.experts,
        block_size=shape.sort_block_size,
    )


def fused_moe_forward_spec(shape: MoeShape = QWEN3_30B_A3B_DECODE):
    return shape.to_fused_moe_forward_spec(use_static_offsets=True)


def quantization_modes_for_day0(include_int4: bool = False) -> Tuple[str, ...]:
    modes = ["bf16", "fp8e4m3", "bf8e5m2"]
    if include_int4:
        modes.extend(["i4_fp8", "i4_bf8"])
    return tuple(modes)


def attention_problem_kwargs(shape: AttentionShape) -> Dict[str, object]:
    return {
        "total_q": shape.total_q,
        "num_seqs": shape.num_seqs,
        "num_query_heads": shape.num_query_heads,
        "num_kv_heads": shape.num_kv_heads,
        "head_size": shape.head_size,
        "block_size": shape.block_size,
        "max_seqlen_q": shape.max_seqlen_q,
        "max_seqlen_k": shape.max_seqlen_k,
        "dtype": shape.dtype,
        "q_dtype": shape.dtype,
        "use_fp8": shape.kv_storage_dtype in ("fp8e4m3", "bf8e5m2"),
    }


__all__ = [
    "ARCH",
    "AttentionShape",
    "ATTENTION_WIDTH",
    "BATCH",
    "BLOCK_SIZE",
    "DECODE_KV_LENS",
    "DTYPE",
    "EXPERT_BLOCK_SCALED_GEMM_SHAPES",
    "GemmShape",
    "HEAD_DIM",
    "HIDDEN",
    "HIDDEN_SIZE",
    "ISA",
    "MOE_INTER",
    "MoeShape",
    "NHEAD_K",
    "NHEAD_Q",
    "NORM_BLOCK_SIZE",
    "NORM_VEC",
    "NUM_EXPERTS",
    "PREFILL_Q_LENS",
    "QKV_PROJ",
    "QWEN3_30B_A3B_CONFIG",
    "QWEN3_30B_A3B_DECODE",
    "QWEN3_30B_A3B_DECODE_MOE",
    "QWEN3_30B_A3B_SHAPES",
    "QwenOpShape",
    "Qwen3A3BConfig",
    "RDQUANT_OUT_DTYPES",
    "ROUTER_BLOCK_SIZE",
    "SORT_BLOCK_SIZE",
    "STATIC_SLOT_SIZE",
    "TOPK",
    "WAVE_SIZE",
    "ALL_BF16_GEMM_SHAPES",
    "ALL_GEMM_SHAPES",
    "LOWBIT_BLOCK_SCALED_GEMM_SHAPES",
    "add_rmsnorm_rdquant_specs",
    "attention_problem_kwargs",
    "bf16_universal_gemm_spec",
    "decode_attention_shapes",
    "decode_gemm_shapes",
    "fused_moe_forward_spec",
    "layer_shapes",
    "moe_shape",
    "moe_sorting_spec",
    "prefill_attention_shape",
    "prefill_gemm_shapes",
    "quantization_modes_for_day0",
    "qwen3_decode_attention_shapes",
    "qwen3_kv_dequant_shapes",
    "qwen3_prefill_attention_shapes",
    "qwen3_30b_a3b_config",
    "shape_by_name",
    "topk_softmax_spec",
]
