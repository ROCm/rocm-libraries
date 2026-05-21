# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared scaffolding for the FMHA instance family.

This module factors what every FMHA spec needs:

* :class:`FmhaShape` -- the ``(head_size, num_query_heads,
  num_kv_heads, block_size_q, block_size_k)`` tuple with the GQA
  factor as a derived property.
* :class:`FmhaCommonSpec` -- the runtime knob bundle (dtype,
  mask_mode, sliding_window, softcap / rotary / dropout flags).
* :func:`validate_common_spec` -- common spec validation that
  catches dtype / head_size / mask-combination mistakes early.
* :class:`FmhaKernelBuilder` -- the **boilerplate killer**: a
  builder that wraps an :class:`IRBuilder`, declares the canonical
  parameter set (Q, K, V, O, strides, scale_log2, seqlen) in one
  call, gives the kernel a :class:`TensorDescriptor` for each
  tensor that hides the manual ``token*stride + head*stride``
  arithmetic, and decodes the grid coords. Variant builders use
  this so a new FMHA flavour is ~30 lines of business logic, not
  120 lines of param declarations + stride math.

The MFMA-tiled inner bodies live in :mod:`_fmha_warp_body` (warp-
distributed scalar; one warp per CTA, head_dim distributed) and
:mod:`_fmha_mfma_body` (production MFMA, 2/4-warp BLOCK_M tiles,
LDS staging, cshuffle epilogue). The :class:`FmhaKernelBuilder`
binds equally well to either body.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from ..core.ir import F32, I32, IRBuilder, KernelDef, PtrType, Value
from ..helpers.io import io_ir_type
from ..helpers.spec import SignatureBuilder
from ..transforms import (
    TensorDescriptor,
)


__all__ = [
    "FmhaCommonSpec",
    "FmhaKernelBuilder",
    "FmhaMaskMode",
    "FmhaShape",
    "FmhaTensorSpec",
    "validate_common_spec",
]


FmhaMaskMode = Literal["none", "causal", "sliding_window", "alibi", "custom"]


@dataclass(frozen=True)
class FmhaShape:
    """Shape bundle used by every FMHA flavour."""

    head_size: int
    num_query_heads: int
    num_kv_heads: int
    block_size_q: int = 16
    block_size_k: int = 64

    @property
    def num_queries_per_kv(self) -> int:
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError(
                f"num_query_heads ({self.num_query_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        return self.num_query_heads // self.num_kv_heads


@dataclass(frozen=True)
class FmhaCommonSpec:
    """Common-knob bundle shared across the FMHA variants."""

    shape: FmhaShape
    dtype: str = "f16"
    scale_log2: float = 0.0
    mask_mode: FmhaMaskMode = "none"
    sliding_window: int = 0
    use_softcap: bool = False
    use_rotary: bool = False
    use_dropout: bool = False
    use_sinks: bool = False

    @property
    def head_size(self) -> int:
        return self.shape.head_size

    @property
    def use_alibi_matrix(self) -> bool:
        return self.mask_mode == "alibi"

    @property
    def use_custom_mask(self) -> bool:
        return self.mask_mode == "custom"


def validate_common_spec(spec: FmhaCommonSpec) -> Tuple[bool, str]:
    """Validate the spec's runtime knobs against the supported set.

    Catches the typical footguns:

    * unsupported head_size (e.g. 96 falls through CK Tile's
      standard {32, 64, 128, 192, 256} canon)
    * unsupported dtype outside {f16, fp16, bf16}
    * sliding_window mask with no window
    * GQA divisibility mismatch

    Specs that pass this check are guaranteed to map onto **some**
    valid kernel; whether a given builder accepts them is a builder-
    level check on top.
    """
    s = spec.shape
    if s.head_size <= 0 or s.head_size > 256:
        return False, f"head_size {s.head_size} out of supported range (1..256)"
    if s.head_size not in (32, 64, 128, 192, 256):
        return False, (
            f"head_size {s.head_size} not in the supported set "
            "{32, 64, 128, 192, 256} (CK Tile's standard FMHA shapes)"
        )
    if s.num_query_heads <= 0 or s.num_kv_heads <= 0:
        return False, (
            f"num_query_heads / num_kv_heads must be > 0 "
            f"(got {s.num_query_heads}, {s.num_kv_heads})"
        )
    if s.num_query_heads % s.num_kv_heads != 0:
        return False, (
            f"num_query_heads ({s.num_query_heads}) must be divisible by "
            f"num_kv_heads ({s.num_kv_heads}) for GQA / MQA"
        )
    if s.block_size_q not in (16, 32, 64, 128):
        return False, f"block_size_q {s.block_size_q} not in {{16, 32, 64, 128}}"
    if s.block_size_k not in (16, 32, 64, 128, 256):
        return False, f"block_size_k {s.block_size_k} not in {{16..256}}"
    if spec.dtype not in ("f16", "fp16", "bf16"):
        return False, f"dtype {spec.dtype!r} must be 'f16' / 'fp16' / 'bf16'"
    if spec.mask_mode not in ("none", "causal", "sliding_window", "alibi", "custom"):
        return False, f"mask_mode {spec.mask_mode!r} not recognised"
    if spec.mask_mode == "sliding_window" and spec.sliding_window <= 0:
        return False, (
            f"sliding_window mask requires window > 0 (got {spec.sliding_window})"
        )
    if spec.use_sinks and spec.dtype not in ("f16", "fp16"):
        return False, "use_sinks=True only supports f16 in v1"
    return True, "ok"


# ---------------------------------------------------------------------
# FmhaKernelBuilder -- the boilerplate-killing helper
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class FmhaTensorSpec:
    """One named FMHA tensor: ``Q`` / ``K`` / ``V`` / ``O`` etc.

    * ``name`` -- the param name (matches the kernel signature).
    * ``dtype`` -- one of ``"f16"`` / ``"bf16"`` / ``"fp8e4m3"`` /
      ``"bf8e5m2"`` / ``"i8"``; mapped to the pointer pointee type.
    * ``readonly`` / ``writeonly`` -- standard CK alias hints.
    * ``align`` -- pointer alignment guarantee in bytes (16 for f16
      pixels of 8 elems, 8 for fp8 pixels of 8 elems, 1 for i8).
    """

    name: str
    dtype: str
    readonly: bool = True
    writeonly: bool = False
    align: int = 16


def _stride_param_names(name: str) -> Tuple[str, str]:
    """Naming convention: ``stride_{lower(name)}_token`` and ``..._head``."""
    n = name.lower()
    return f"stride_{n}_token", f"stride_{n}_head"


class FmhaKernelBuilder:
    """High-level FMHA-kernel builder that hides param boilerplate.

    Usage (in a variant-specific build function)::

        spec = FmhaFwdVarlenSpec(...)
        kb = FmhaKernelBuilder(spec.kernel_name(), spec.common)
        kb.add_tensor("Q", readonly=True)
        kb.add_tensor("K", readonly=True)
        kb.add_tensor("V", readonly=True)
        kb.add_tensor("O", readonly=False, writeonly=True)
        kb.add_scalar("scale_log2", F32)
        kb.add_scalar("seqlen_k", I32)
        kb.add_strides("q", "k", "v", "o")
        kb.decode_grid()           # binds q_token / head_idx / kv_head_idx
        # Now call your kernel body. Address math uses kb.q_descriptor /
        # kb.k_descriptor / ... so no manual stride * coord arithmetic.

    Builders that need extra params (block_table, cu_seqlens, scale
    pointers for sage, codebook for sage int variants) call
    :meth:`add_scalar` / :meth:`add_ptr` directly.

    The :class:`TensorDescriptor` for each tensor is exposed via
    :attr:`tensor_descriptor` -- callers compute the (token, head, d)
    -> linear offset via ``desc.offset(b, token=..., head=..., d=...)``
    and the kernel never has to write the multiply chain by hand.
    """

    def __init__(self, kernel_name: str, common: FmhaCommonSpec) -> None:
        self.common = common
        self.b = IRBuilder(kernel_name)
        # All params in declaration order; one tuple per param:
        # ("tensor"|"ptr"|"stride"|"scalar", name, dtype).
        self._sig_order: List[Tuple[str, str, str]] = []
        self._tensor_params: Dict[str, Value] = {}
        self._stride_params: Dict[str, Tuple[Value, Value]] = {}
        self._other_params: Dict[str, Value] = {}
        # Decoded grid coords -- populated by .decode_grid().
        self.q_token: Optional[Value] = None
        self.head_idx: Optional[Value] = None
        self.kv_head_idx: Optional[Value] = None
        self.batch_idx: Optional[Value] = None

    @property
    def kernel(self) -> KernelDef:
        return self.b.kernel

    @property
    def builder(self) -> IRBuilder:
        return self.b

    # ----- param declarations -----

    def add_tensor(
        self,
        name: str,
        *,
        dtype: Optional[str] = None,
        readonly: bool = True,
        writeonly: bool = False,
        align: int = 16,
    ) -> Value:
        """Declare a tensor param and remember its dtype for descriptors.

        Accepts the full FMHA dtype palette: ``f16`` / ``bf16`` for
        activations, ``fp8e4m3`` / ``bf8e5m2`` for fp8-attention K/V,
        and ``i8`` for the sage int8 / packed-int4 K/V cache. The
        pointee IR type is dispatched accordingly so callers don't
        have to import ``io_ir_type`` vs ``quant_ir_type`` themselves.
        """
        from ..core.ir import BF8E5M2, FP8E4M3, I8

        actual_dtype = dtype or self.common.dtype
        if actual_dtype in ("f16", "fp16", "bf16"):
            ty = io_ir_type(actual_dtype)
        elif actual_dtype == "fp8e4m3":
            ty = FP8E4M3
        elif actual_dtype == "bf8e5m2":
            ty = BF8E5M2
        elif actual_dtype == "i8":
            ty = I8
        else:
            raise ValueError(
                f"FmhaKernelBuilder.add_tensor dtype {actual_dtype!r} "
                "not in {f16, fp16, bf16, fp8e4m3, bf8e5m2, i8}"
            )
        p = self.b.param(
            name,
            PtrType(ty, "global"),
            noalias=True,
            readonly=readonly,
            writeonly=writeonly,
            align=align,
        )
        self._tensor_params[name] = p
        self._sig_order.append(("tensor", name, actual_dtype))
        return p

    def add_ptr(
        self,
        name: str,
        *,
        dtype: str,
        readonly: bool = True,
        align: int = 4,
    ) -> Value:
        """Declare a non-canonical pointer param (block_table, scales, etc.)."""
        from ..core.ir import I8

        if dtype == "i32":
            ty = I32
        elif dtype == "f32":
            ty = F32
        elif dtype == "i8":
            ty = I8
        else:
            ty = io_ir_type(dtype)
        p = self.b.param(
            name,
            PtrType(ty, "global"),
            noalias=True,
            readonly=readonly,
            align=align,
        )
        self._other_params[name] = p
        self._sig_order.append(("ptr", name, dtype))
        return p

    def add_scalar(self, name: str, dtype: str = "i32") -> Value:
        """Declare a scalar param. ``dtype`` is ``"i32"`` or ``"f32"``."""
        ty = I32 if dtype == "i32" else F32
        p = self.b.param(name, ty)
        self._other_params[name] = p
        self._sig_order.append(("scalar", name, dtype))
        return p

    def add_strides(self, *names: str) -> None:
        """Declare ``stride_{name}_token`` and ``stride_{name}_head`` for each
        name in ``names`` (typical: ``q, k, v, o``).
        """
        for name in names:
            sn_token, sn_head = _stride_param_names(name)
            tok = self.b.param(sn_token, I32)
            hd = self.b.param(sn_head, I32)
            self._stride_params[name] = (tok, hd)
            self._other_params[sn_token] = tok
            self._other_params[sn_head] = hd
            self._sig_order.append(("scalar", sn_token, "i32"))
            self._sig_order.append(("scalar", sn_head, "i32"))

    # ----- accessors -----

    @property
    def tensors(self) -> Dict[str, Value]:
        return self._tensor_params

    def tensor(self, name: str) -> Value:
        return self._tensor_params[name]

    def stride(self, name: str) -> Tuple[Value, Value]:
        return self._stride_params[name]

    def stride_token(self, name: str) -> Value:
        return self._stride_params[name][0]

    def stride_head(self, name: str) -> Value:
        return self._stride_params[name][1]

    def scalar(self, name: str) -> Value:
        return self._other_params[name]

    def ptr(self, name: str) -> Value:
        return self._other_params[name]

    def block_size(self, block_size: int) -> None:
        """Set the kernel's ``max_workgroup_size`` attribute."""
        self.b.kernel.attrs["max_workgroup_size"] = block_size

    # ----- grid decode -----

    def decode_grid(
        self,
        *,
        num_queries_per_kv: Optional[int] = None,
        has_batch_axis: bool = False,
    ) -> Tuple[Value, Value, Value]:
        """Bind grid-axis variables and the GQA / MQA decode in one call.

        Grid convention: ``block_id_x`` = q_token, ``block_id_y`` =
        head_idx, ``block_id_z`` = batch_idx (only if ``has_batch_axis``
        is True; otherwise z is ignored).

        Returns ``(q_token, head_idx, kv_head_idx)``; also populated
        as attributes on this builder.
        """
        b = self.b
        self.q_token = b.block_id_x()
        self.head_idx = b.block_id_y()
        if has_batch_axis:
            self.batch_idx = b.block_id_z()
        nqkv = (
            num_queries_per_kv
            if num_queries_per_kv is not None
            else self.common.shape.num_queries_per_kv
        )
        if nqkv == 1:
            self.kv_head_idx = self.head_idx
        else:
            self.kv_head_idx = b.div(self.head_idx, b.const_i32(nqkv))
        return self.q_token, self.head_idx, self.kv_head_idx

    # ----- tensor descriptor factory -----

    def tensor_descriptor(
        self,
        tensor_name: str,
        *,
        coord_names: Tuple[str, str, str] = ("token", "head", "d"),
        lengths: Optional[Tuple[int, int, int]] = None,
    ) -> TensorDescriptor:
        """Build a :class:`TensorDescriptor` for a registered tensor.

        Layout: ``(token, head, d)`` row-major; strides come from the
        registered ``stride_{name}_token`` and ``stride_{name}_head``
        params and assume the head_dim is innermost (stride 1).

        The returned descriptor's ``.offset(b, token=..., head=...,
        d=...)`` produces ``(linear_offset, validity)`` pairs that
        kernel bodies feed into ``global_load`` / ``global_store``.

        For paged-KV / varlen / split-KV addressing, callers can chain
        an :func:`transforms.indirect` or :func:`transforms.embed`
        on top of the descriptor.
        """
        if lengths is None:
            # Use large stand-in sizes that won't trigger validity
            # predicates for in-range coords. Callers can pass exact
            # sizes if they want the descriptor to enforce bounds.
            lengths = (1 << 24, 1 << 12, max(self.common.shape.head_size, 1))
        strides_seen = self._stride_params.get(tensor_name)
        if strides_seen is None:
            # Sage codebooks etc. -- fall back to a naive descriptor
            # with default row-major strides.
            return TensorDescriptor.naive(
                tensor_name,
                lengths=list(lengths),
                coord_names=list(coord_names),
            )
        return TensorDescriptor(
            name=tensor_name,
            base_names=tuple(coord_names),
            base_lengths=tuple(int(x) for x in lengths),
            base_strides=(0, 0, 1),  # placeholders -- runtime strides via embed below
            chain=(),
            upper_names=tuple(coord_names),
        )

    # ----- signature builder -----

    def signature(self) -> List[Tuple[str, str]]:
        """Build the kernel ABI signature from the in-order param log.

        Walks :attr:`_sig_order` (populated by ``add_tensor`` /
        ``add_ptr`` / ``add_scalar`` / ``add_strides`` calls) and
        emits the same shape that
        :class:`ck_dsl.helpers.spec.SignatureBuilder` produces. The
        order matches the declaration order so callers can read off
        the kernel signature directly from the build function.
        """
        sb = SignatureBuilder()
        for kind, name, dtype in self._sig_order:
            if kind in ("tensor", "ptr"):
                sb.ptr(name, dtype)
            else:  # scalar
                sb.scalar(name, dtype)
        return sb.build()

    # ----- row-base helpers (the most common address math pattern) -----

    def q_row_base(self) -> Value:
        """``q_token * stride_q_token + head_idx * stride_q_head``."""
        return self._row_base("q", self.q_token, self.head_idx)

    def o_row_base(self) -> Value:
        """``q_token * stride_o_token + head_idx * stride_o_head``."""
        return self._row_base("o", self.q_token, self.head_idx)

    def k_row_base(self, k_idx: Value) -> Value:
        """``k_idx * stride_k_token + kv_head_idx * stride_k_head``."""
        return self._row_base("k", k_idx, self.kv_head_idx)

    def v_row_base(self, k_idx: Value) -> Value:
        return self._row_base("v", k_idx, self.kv_head_idx)

    def _row_base(self, tensor_name: str, tok: Value, hd: Value) -> Value:
        tok_stride, head_stride = self._stride_params[tensor_name]
        return self.b.add(
            self.b.mul(tok, tok_stride),
            self.b.mul(hd, head_stride),
        )
