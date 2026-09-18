# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Architecture-neutral spec contract for dense attention kernels.

The base type contains only problem fields, geometry, and policies implemented by
both gfx942 and gfx950. Architecture-specific codegen knobs belong on concrete
subclasses in the owning kernel modules.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as _dataclass_fields
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

# Signed 32-bit ceiling for tensor extents. See ``check_dense_spec_preflight``
# check 4 for why the SIGNED bound binds even though the buffer-resource
# num_records field is unsigned in hardware.
INT32_LIMIT = 2**31


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

    @property
    def runtime_param_fields(self) -> tuple[str, ...]:
        """Spec fields this kernel reads as runtime kernel params instead of
        baking into the body, so they must NOT split cache identity -- one
        compiled kernel serves every value of them.

        Empty here: the base contract is that the whole problem shape is baked.
        A subclass whose body emits a field as a param declares it by overriding
        this, and ``attention_dense_cache_key`` excludes it. Declaring a field
        that the body still bakes is a cache collision, which is why the
        declaration lives on the spec that owns the body rather than in the
        shared key function.

        NOTE: this does not yet drive the symbol name. ``batch`` has never
        appeared in the dense name on any path, so a spec that bakes batch and
        one that does not are homonyms. Harmless for in-process dispatch, where
        the cache key is the identity; a blocker for AOT packaging and for
        per-batch specialization, where the symbol IS the identity.
        """
        return ()

    def _layout_name_parts(self) -> tuple[str, ...]:
        return ()

    def _shape_name_parts(self) -> tuple[str, ...]:
        """Name tokens for the baked problem shape. A subclass whose kernel takes
        the shape as runtime params (so one kernel serves every batch/seqlen)
        overrides this to drop sq/sk from the symbol name."""
        return (f"sq{self.seqlen_q}", f"sk{self.seqlen_kv}")

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
        parts.extend(self._shape_name_parts())
        parts.append("causal" if self.causal else "full")
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
    """Cache identity: the arch, the concrete spec type, and every spec field
    that affects codegen.

    Fields the spec declares in ``runtime_param_fields`` are excluded -- its
    kernel reads them at runtime, so one compiled binary serves every value and
    they must not split identity. This is what collapses the AOT
    batch x seqlen instance explosion on the arches that opt in. Specs that
    declare nothing (the default) keep every field, so their identity is
    unchanged.

    The shape is the same on both paths, so callers never branch on it. The
    class object rather than its name keeps subclass identity
    (``Gfx942AttentionDenseSpec`` is not ``Gfx950AttentionDenseSpec``) without
    stringly-typing it; if a stable on-disk identity is ever needed, that wants
    ``__qualname__`` or a digest instead.
    """
    if not arch:
        raise ValueError("attention dense cache identity requires an explicit arch")
    skip = frozenset(spec.runtime_param_fields)
    rest = tuple(
        (f.name, getattr(spec, f.name))
        for f in _dataclass_fields(spec)
        if f.name not in skip
    )
    return (arch, type(spec), rest)


def check_dense_spec_preflight(spec: AttentionDenseSpec) -> tuple[bool, str]:
    """Return ``(ok, reason)`` for the checks every dense body shares.

    Each arch's ``supports_attention_dense`` calls this and then adds its own scope
    (dtype / head-size sets, deferred modes, private knobs, LDS budget, the tile
    divisibility its own DMA imposes). A check belongs HERE only if it reads
    base-spec fields only AND its verdict is the same for every dense body -- which
    is why the LDS budget stays per-arch (it needs the arch capacity and the body's
    tile math) and the mode rejections stay per-arch (gfx942 defers varlen, gfx950
    ships it).

    The four checks, in a load-bearing order:

    1. Re-run the dataclass validators, so a hand-built spec (or one smuggled past
       the frozen ctor) is rejected with a structured reason instead of an exception
       escaping this ``(bool, str)`` API. The CONCRETE type is reconstructed, not
       the base, so an arch's private knob validators run too; ``ZeroDivisionError``
       is caught alongside ``ValueError`` because ``__post_init__`` evaluates
       ``seqlen_kv % block_n`` BEFORE it validates ``block_n > 0``.
    2. Positive extents. Every dataclass validator is a divisibility test and
       Python's ``%`` is sign-following (``-256 % 256 == 0``, ``8 % -1 == 0``), so
       zero and negative shapes pass all of them. ``num_query_heads == 0`` is the
       worst: ``gqa = Hq // Hkv == 0`` emits ``sdiv i32 %hq, 0`` into the kernel.
    3. ``block_n`` divides the query tile. The causal KV clamp uses
       ``n_per = block_m // block_n``, a FLOOR, so an indivisible ``block_n``
       silently drops every key past the last whole sub-tile, and
       ``block_n > block_m`` makes ``n_per`` 0 -> zero-trip loop -> ``l == 0`` ->
       ``rcp(0)`` -> NaN. Neither fails loudly.
    4. 32-bit addressing. Offsets are built from IRBuilder add/mul, which lower to
       ``add nsw`` / ``mul nsw`` i32 -- signed overflow is UB, not a wrap, so LLVM
       may poison the whole address chain rather than merely read the wrong place.
       The buffer-resource ``num_records`` field is unsigned in hardware, but it is
       emitted through ``const_i32`` (no range check) and the voffset feeding it is
       signed i32 arithmetic, so the signed bound is the binding one on both paths.

    Order matters between 2 and 4: a negative extent makes the products in 4
    vacuously true, so the sign check has to run first.

    On a kernel that takes the shape as runtime params (see
    ``runtime_param_fields``), check 4 is the ONLY defense. There the extent is a
    device-side ``mul`` of two kernargs rather than a constant folded at emission,
    so there is nothing in the IR to inspect -- and because those fields no longer
    split the launcher-cache key, the check has to run per launch rather than per
    compile. It does: every caller reaches ``supports`` before the cache lookup.
    """
    try:
        type(spec)(**{f.name: getattr(spec, f.name) for f in _dataclass_fields(spec)})
    except (ValueError, ZeroDivisionError) as e:
        return False, f"invalid {type(spec).__name__}: {e}"

    for name in (
        "batch",
        "seqlen_q",
        "seqlen_kv",
        "num_query_heads",
        "num_kv_heads",
        "head_size",
    ):
        value = getattr(spec, name)
        if value <= 0:
            return False, f"{name} must be positive, got {value}"

    if spec.block_m % spec.block_n != 0:
        return False, (
            f"block_n must divide the {spec.block_m}-row query tile (got "
            f"block_n={spec.block_n}; the spec also requires block_n % 32 == 0, so "
            f"use 32, 64, 128 or 256). Load-bearing for causal=True, where "
            f"n_per = {spec.block_m} // block_n floors and drops keys"
        )

    kv_bytes = spec.batch * spec.seqlen_kv * spec.num_kv_heads * spec.head_size * 2
    if kv_bytes >= INT32_LIMIT:
        return False, (
            f"K/V extent is {kv_bytes} B, at or past the 32-bit buffer-resource "
            f"limit ({INT32_LIMIT} B)"
        )
    qo_elems = spec.batch * spec.seqlen_q * spec.num_query_heads * spec.head_size
    if qo_elems >= INT32_LIMIT:
        return False, (
            f"Q/O extent is {qo_elems} elements, at or past the 32-bit addressing "
            f"limit ({INT32_LIMIT})"
        )
    return True, ""


__all__ = [
    "AttentionDenseSpec",
    "DEFAULT_DENSE_TILE_GEOMETRY",
    "DENSE_TILE_GEOMETRIES",
    "INT32_LIMIT",
    "attention_dense_cache_key",
    "check_dense_spec_preflight",
]
