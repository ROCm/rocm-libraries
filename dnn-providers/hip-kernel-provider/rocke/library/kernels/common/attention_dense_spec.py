# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Architecture-neutral spec contract for dense attention kernels.

The base type contains only problem fields, geometry, and policies implemented by
both gfx942 and gfx950. Architecture-specific codegen knobs belong on concrete
subclasses in the owning kernel modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from rocke.core.ir import BF16, F16
from rocke.helpers.spec import kernel_name_join


_DTYPE_IR = {"bf16": BF16, "fp16": F16}

# Shared query/KV geometry only. LDS layout choices are architecture-specific.
DENSE_TILE_GEOMETRIES = MappingProxyType(
    {
        "default": MappingProxyType({"block_m": 256, "block_n": 64}),
        "bm128": MappingProxyType({"block_m": 128, "block_n": 64}),
    }
)
DEFAULT_DENSE_TILE_GEOMETRY = DENSE_TILE_GEOMETRIES["default"]

_COMMON_PERSIST_DECODES = frozenset({"auto", "qb_major", "hkv_major"})


@dataclass(frozen=True)
class AttentionDenseSpec:
    """Shared compile-time problem and geometry for dense attention."""

    # Problem shape and semantics.
    batch: int
    seqlen_q: int
    seqlen_kv: int
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    causal: bool = True
    dtype: str = "bf16"
    sliding_window: int = 0
    ragged: bool = False
    varlen: bool = False

    # Geometry and common implementation policy.
    block_m: int = DEFAULT_DENSE_TILE_GEOMETRY["block_m"]
    block_n: int = DEFAULT_DENSE_TILE_GEOMETRY["block_n"]
    waves_per_eu: int = 2
    lds_k_group_pad: int = 8
    persistent: bool = False
    num_persistent: int = 256
    interleave: bool = False
    persist_decode: str = "auto"
    # Historical shared naming/behavior flag. Kept in the base for compatibility;
    # architecture-specific migration can move it independently in a follow-up.
    lazy_rescale: bool = True

    # Problem modes currently implemented only by a subset of architectures.
    # They remain shared semantic fields so supports_* can reject unsupported
    # requests explicitly; unlike codegen knobs, they never silently no-op.
    paged: bool = False
    block_size: int = 0
    num_kv_blocks: int = 0
    use_sinks: bool = False

    def supported_persist_decodes(self) -> frozenset[str]:
        """Decode values the concrete kernel type can actually emit."""
        return _COMMON_PERSIST_DECODES

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPE_IR:
            raise ValueError(
                f"dtype must be one of {sorted(_DTYPE_IR)}, got {self.dtype}"
            )
        if self.block_m <= 0:
            raise ValueError(f"block_m must be positive, got {self.block_m}")
        if self.block_n <= 0 or self.block_n % 32 != 0:
            raise ValueError(
                f"block_n must be a positive multiple of 32, got {self.block_n}"
            )
        if self.head_size not in (64, 128):
            raise ValueError(f"head_size must be 64 or 128, got {self.head_size}")
        if self.lds_k_group_pad < 0 or self.lds_k_group_pad % 8 != 0:
            raise ValueError(
                "lds_k_group_pad must be a non-negative multiple of 8 bf16 "
                "elements (16 bytes) so the K group pitch stays "
                f"ds_read_b128-aligned, got {self.lds_k_group_pad}"
            )

        if self.ragged:
            if self.seqlen_q <= 0 or self.seqlen_kv <= 0:
                raise ValueError("ragged requires positive seqlen_q/seqlen_kv")
            if self.seqlen_q != self.seqlen_kv:
                raise ValueError(
                    "ragged is self-attention only (seqlen_q == seqlen_kv), got "
                    f"{self.seqlen_q} != {self.seqlen_kv}"
                )
            if self.varlen:
                raise ValueError("ragged is not supported with varlen")
            if self.sliding_window > 0:
                raise ValueError("ragged is not supported with sliding_window")
        else:
            if self.seqlen_q % self.block_m != 0:
                raise ValueError(
                    f"seqlen_q must be a multiple of block_m={self.block_m}, "
                    f"got {self.seqlen_q}"
                )
            if self.seqlen_kv % self.block_n != 0:
                raise ValueError(
                    f"seqlen_kv must be a multiple of block_n={self.block_n}, "
                    f"got {self.seqlen_kv}"
                )

        if self.num_kv_heads == 0 or self.num_query_heads % self.num_kv_heads:
            raise ValueError(
                f"num_query_heads ({self.num_query_heads}) must be a positive "
                f"multiple of num_kv_heads ({self.num_kv_heads})"
            )
        if self.persistent and self.num_persistent <= 0:
            raise ValueError(
                f"num_persistent must be positive, got {self.num_persistent}"
            )
        if self.persist_decode not in self.supported_persist_decodes():
            raise ValueError(
                f"persist_decode must be one of "
                f"{sorted(self.supported_persist_decodes())}, "
                f"got {self.persist_decode!r}"
            )
        if self.sliding_window < 0:
            raise ValueError(f"sliding_window must be >= 0, got {self.sliding_window}")
        if self.sliding_window > 0:
            if not self.causal:
                raise ValueError("sliding_window>0 requires causal=True")
            if self.sliding_window % self.block_n:
                raise ValueError(
                    f"sliding_window ({self.sliding_window}) must be a multiple "
                    f"of block_n={self.block_n}"
                )
        if self.varlen:
            if self.persistent:
                raise ValueError("varlen is not supported with persistent=True")
            if not self.causal:
                raise ValueError("varlen requires causal=True")
        if not 1 <= self.waves_per_eu <= 8:
            raise ValueError(f"waves_per_eu must be in [1, 8], got {self.waves_per_eu}")

        if self.paged:
            if self.block_size <= 0:
                raise ValueError("paged=True requires block_size > 0")
            if self.block_size & (self.block_size - 1):
                raise ValueError(
                    f"paged block_size ({self.block_size}) must be a power of two"
                )
            if self.block_n % self.block_size:
                raise ValueError(
                    f"block_n ({self.block_n}) must be a multiple of page "
                    f"block_size ({self.block_size})"
                )
            rows_per_wave = self.block_n // self.num_waves
            if self.block_size < rows_per_wave or self.block_size % rows_per_wave:
                raise ValueError(
                    f"paged block_size ({self.block_size}) must be >= and a "
                    f"multiple of ROWS_PER_WAVE ({rows_per_wave})"
                )
            if self.num_kv_blocks <= 0:
                raise ValueError("paged=True requires num_kv_blocks > 0")
            cache_bytes = (
                self.num_kv_blocks
                * self.block_size
                * self.num_kv_heads
                * self.head_size
                * 2
            )
            if cache_bytes > 2**31 - 1:
                raise ValueError(
                    f"paged cache {cache_bytes} B exceeds i32 addressing (2 GiB)"
                )
            if self.batch != 1:
                raise ValueError("paged multi-sequence (batch>1) not yet implemented")
            if self.varlen:
                raise ValueError("paged varlen not yet implemented (single-seq only)")
            if self.persistent:
                raise ValueError(
                    "paged + persistent not yet implemented "
                    "(persistent builder is contiguous-only)"
                )
            if self.head_size != 128:
                raise ValueError("paged not yet implemented for head_size != 128")
            if self.dtype not in ("fp16", "bf16"):
                raise ValueError(f"paged not yet implemented for dtype={self.dtype}")
            if self.sliding_window <= 0:
                raise ValueError(
                    "paged not yet implemented for plain-causal "
                    "(sliding_window>0 only)"
                )
        if self.use_sinks and self.paged:
            raise ValueError("use_sinks is not yet supported with paged KV")
        if self.use_sinks and self.varlen:
            raise ValueError("use_sinks is not yet supported with varlen")

    @property
    def num_waves(self) -> int:
        return self.block_m // 32

    @property
    def dtype_ir(self):
        return _DTYPE_IR[self.dtype]

    @property
    def num_queries_per_kv(self) -> int:
        return self.num_query_heads // self.num_kv_heads

    @property
    def resolved_persist_decode(self) -> str:
        """Resolve the common auto policy to hkv-major or qb-major."""
        if self.persist_decode != "auto":
            return self.persist_decode
        gqa = self.num_queries_per_kv
        nqb = (self.seqlen_q + self.block_m - 1) // self.block_m
        per_hkv = gqa * nqb * self.batch
        if gqa > 1 and per_hkv >= 2 * self.num_persistent:
            return "hkv_major"
        return "qb_major"

    def _layout_name_parts(self) -> tuple[str, ...]:
        return ()

    def _algorithm_name_parts(self) -> tuple[str, ...]:
        return ("lazyrs",) if self.lazy_rescale else ()

    def _persist_decode_name_part(self) -> str:
        return "hkvmaj" if self.resolved_persist_decode == "hkv_major" else ""

    def kernel_name(self) -> str:
        parts = [
            "rocke_attention_dense",
            f"d{self.head_size}",
            f"hq{self.num_query_heads}",
            f"kv{self.num_kv_heads}",
            f"bn{self.block_n}",
            self.dtype,
        ]
        if self.block_m != DEFAULT_DENSE_TILE_GEOMETRY["block_m"]:
            parts.append(f"bm{self.block_m}")
        if 128 // self.head_size > 1:
            parts.append(f"kpad{self.lds_k_group_pad}")
        parts.extend(self._layout_name_parts())
        parts.extend(
            [
                f"sq{self.seqlen_q}",
                f"sk{self.seqlen_kv}",
                "causal" if self.causal else "full",
            ]
        )
        if self.ragged:
            parts.append("ragged")
        if self.sliding_window > 0:
            parts.append(f"swa{self.sliding_window}")
        if self.use_sinks:
            parts.append("sinks")
        if self.varlen:
            parts.append("varlen")
        if self.paged:
            parts.extend((f"pgd{self.block_size}", f"nb{self.num_kv_blocks}"))
        parts.extend(self._algorithm_name_parts())
        if self.persistent:
            parts.append(f"persist{self.num_persistent}")
            decode = self._persist_decode_name_part()
            if decode:
                parts.append(decode)
            if self.interleave:
                parts.append("intl")
        return kernel_name_join(*parts)


def attention_dense_cache_key(spec: AttentionDenseSpec, *, arch: str) -> tuple:
    """Cache identity containing every frozen spec field and the architecture."""
    if not arch:
        raise ValueError("attention dense cache identity requires an explicit arch")
    return (arch, spec)


__all__ = [
    "AttentionDenseSpec",
    "DEFAULT_DENSE_TILE_GEOMETRY",
    "DENSE_TILE_GEOMETRIES",
    "attention_dense_cache_key",
]
