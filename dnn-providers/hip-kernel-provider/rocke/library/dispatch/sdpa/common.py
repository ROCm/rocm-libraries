# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``SdpaRequest``: one frozen request that represents a hipDNN SDPA forward node.

Every field of the hipDNN SDPA forward node has a home here, including the ones
no rocKE kernel implements. A field that is valid but unimplemented is reported
by :meth:`SdpaRequest.features` and declined by candidates through the existing
``Capability.check`` contract; it is never dropped. ``__post_init__`` rejects only
requests the hipDNN frontend would also reject (``SdpaFwdNode::pre_validate_node``)
plus what a direct Python caller could break (ranks, strides, literals).

The request is normalized on construction, so one meaning has one spelling and
:meth:`SdpaRequest.normalized` is a stable cache key:

- a band bound is ``-1`` (unbounded) or ``>= 0``; hipDNN's resolvers reject
  anything below ``-1``, so this does too;
- the scale has exactly one source, and ``scale`` is ``1.0`` unless that source is
  ``"value"``; a baked ``1.0`` is the default source, because hipDNN reads an
  unset scale as 1.0;
- a bias is right-aligned to rank 4, and every size-1 bias dim has stride 0;
- an empty ``SdpaPaged`` or ``SdpaFp8``, or a zero-probability ``SdpaDropout``
  with no tensors, is ``None``.

Deprecated causal flags do not exist here: :func:`dispatch.sdpa.hipdnn.from_hipdnn`
folds them into ``(diagonal, left_bound, right_bound)`` before construction.

Shapes and strides live only on the :class:`SdpaTensor` operands, so the two
cannot disagree. All logical dims are BHSD, as in hipDNN; layout is a property
of the strides.
"""

from __future__ import annotations

import operator
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Literal, TypeAlias

from rocke.core.arch import base_arch_from_target_id
from rocke.dispatch.core import OperatorRequest
from rocke.dispatch.tuning.identity import normalize_knobs

__all__ = [
    "SDPA_ABI_VERSION",
    "SDPA_DIM_VOCABULARY",
    "SDPA_DTYPES",
    "SDPA_FEATURES",
    "Dtype",
    "SdpaBias",
    "SdpaDropout",
    "SdpaFp8",
    "SdpaPaged",
    "SdpaRequest",
    "SdpaTensor",
]

SDPA_ABI_VERSION = "rocke-sdpa-stride/v1"

Dtype: TypeAlias = Literal[
    "fp16",
    "bf16",
    "fp32",
    "fp64",
    "fp8_e4m3",
    "fp8_e4m3_fnuz",
    "fp8_e5m2",
    "fp8_e5m2_fnuz",
    "fp8_e8m0",
    "fp6_e2m3",
    "fp6_e3m2",
    "fp4_e2m1",
    "int4",
    "int8",
    "uint8",
    "int32",
    "int64",
    "bool",
]

# Every hipDNN ``DataType`` except UNSET, so an operand of any dtype the schema
# can carry is representable (seq-len and page tables are int32, block masks
# and dropout masks may be bool or uint8).
SDPA_DTYPES: tuple[str, ...] = Dtype.__args__

_FP8_DTYPES = frozenset(d for d in SDPA_DTYPES if d.startswith("fp8"))
# hipDNN reads ragged offsets as 4- or 8-byte integers (``RaggedTensor.hpp``).
_RAGGED_OFFSET_DTYPES = ("int32", "int64")
_DIAGONALS = ("top_left", "bottom_right")
_SCALE_SOURCES = ("default", "value", "tensor")
_IMPLEMENTATIONS = ("auto", "composite", "unified")
_BASE_ALIGNMENT = 16

SDPA_DIM_VOCABULARY = (
    "batch",
    "h_q",
    "h_k",
    "h_v",
    "s_q",
    "s_kv",
    "d_qk",
    "d_v",
    "gqa_ratio",
)

# Every name :meth:`SdpaRequest.features` can report. A candidate admits a
# request only when its ``supports_features`` covers what the request reports.
SDPA_FEATURES = frozenset(
    {
        "gqa",
        "kv_heads_differ",
        "d_v_differs",
        "dtype_mixed",
        "compute_dtype",
        "layout_bshd",
        "layout_bhsd",
        "layout_strided",
        "kv_layout_differs",
        "o_layout_differs",
        "batch_stride",
        "d_stride",
        "unaligned",
        "scale_tensor",
        "causal_tl",
        "causal_br",
        "window",
        "window_br",
        "window_right",
        "bias",
        "bias_broadcast",
        "alibi",
        "padded",
        "ragged",
        "dropout",
        "paged",
        "block_mask",
        "sinks",
        "fp8",
        "fp8_s_o",
        "lse",
        "max_sum_exp",
        "mma_core_mode",
        "implementation_hint",
    }
)


def _check_dtype(dtype: object, what: str) -> None:
    if dtype not in SDPA_DTYPES:
        raise ValueError(f"{what} dtype {dtype!r} is not one of {SDPA_DTYPES}")


def _check_choice(value: object, choices: tuple[str, ...], what: str) -> None:
    if value not in choices:
        raise ValueError(f"{what} must be one of {choices}, got {value!r}")


def _as_int(value: object, what: str) -> int:
    """``value`` as an ``int``; a float, bool or string is malformed, not rounded."""
    if isinstance(value, bool):
        raise ValueError(f"{what} must be an integer, got {value!r}")
    try:
        return operator.index(value)
    except TypeError:
        raise ValueError(f"{what} must be an integer, got {value!r}") from None


def _check_bool(value: object, what: str) -> None:
    """``bool("false")`` is ``True``, so a non-bool flag is malformed, not coerced."""
    if not isinstance(value, bool):
        raise ValueError(f"{what} must be a bool, got {value!r}")


@dataclass(frozen=True)
class SdpaTensor:
    """One operand: hipDNN ``TensorAttributes`` minus the uid, name and value.

    ``dims`` and ``strides`` are in element units and in logical order (BHSD
    for Q/K/V/O/Stats). ``ragged_offset`` is the descriptor of the tensor that
    ``ragged_offset_tensor_uid`` names (its uid dropped): when set, the operand
    is packed per batch, its batch stride is not used, and a kernel reads
    ``B + 1`` offsets of that dtype. ``alignment`` is the byte alignment hipDNN
    guarantees for the base pointer. A by-value tensor's runtime ``value`` is
    deliberately not stored: seeds and offsets change per call and must not
    split the cache key.
    """

    dtype: Dtype
    dims: tuple[int, ...]
    strides: tuple[int, ...]
    ragged_offset: SdpaTensor | None = None
    ragged_offset_multiplier: int = 1
    by_value: bool = False
    alignment: int = _BASE_ALIGNMENT

    def __post_init__(self) -> None:
        _check_dtype(self.dtype, "tensor")
        _check_bool(self.by_value, "by_value")
        dims = tuple(_as_int(d, "tensor dims") for d in self.dims)
        strides = tuple(_as_int(s, "tensor strides") for s in self.strides)
        if len(dims) != len(strides):
            raise ValueError(f"dims {dims} and strides {strides} differ in rank")
        if any(d <= 0 for d in dims):
            raise ValueError(f"dims must be positive, got {dims}")
        if any(s < 0 for s in strides):
            raise ValueError(f"strides must be non-negative, got {strides}")
        multiplier = _as_int(self.ragged_offset_multiplier, "ragged_offset_multiplier")
        alignment = _as_int(self.alignment, "alignment")
        if multiplier < 1:
            raise ValueError(f"ragged_offset_multiplier must be >= 1, got {multiplier}")
        if alignment < 0:
            raise ValueError(f"alignment must be non-negative, got {alignment}")
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "strides", strides)
        object.__setattr__(self, "ragged_offset_multiplier", multiplier)
        object.__setattr__(self, "alignment", alignment)
        self._validate_ragged_offset()

    def _validate_ragged_offset(self) -> None:
        """The hipDNN frontend and ``RaggedTensor`` rules for the offset operand."""
        offset = self.ragged_offset
        if offset is None:
            if self.ragged_offset_multiplier != 1:
                raise ValueError(
                    "ragged_offset_multiplier "
                    f"{self.ragged_offset_multiplier} needs a ragged_offset"
                )
            return
        if not isinstance(offset, SdpaTensor):
            raise ValueError(
                f"ragged_offset must be an SdpaTensor, got {type(offset).__name__}"
            )
        if offset.ragged:
            raise ValueError("a ragged_offset cannot itself be ragged")
        if offset.dtype not in _RAGGED_OFFSET_DTYPES:
            raise ValueError(
                f"ragged_offset dtype must be one of {_RAGGED_OFFSET_DTYPES}, "
                f"got {offset.dtype!r}"
            )
        if len(self.dims) < 2:
            raise ValueError(f"a ragged tensor needs rank >= 2, got dims {self.dims}")
        if offset.volume != self.dims[0] + 1:
            raise ValueError(
                f"ragged_offset needs B + 1 = {self.dims[0] + 1} elements, "
                f"got dims {offset.dims}"
            )

    @property
    def ragged(self) -> bool:
        """The operand is packed per batch through ``ragged_offset``."""
        return self.ragged_offset is not None

    @property
    def volume(self) -> int:
        volume = 1
        for d in self.dims:
            volume *= d
        return volume


@dataclass(frozen=True)
class SdpaBias:
    """Additive bias (hipDNN ``attn_mask``), right-aligned onto ``(B, H_q, S_q, S_kv)``.

    Construction pads the dims to rank 4 with leading size-1 dims and sets every
    size-1 dim's stride to 0, so a rank-2 ``(S_q, S_kv)`` bias and a rank-4
    ``(1, 1, S_q, S_kv)`` bias are the same request.
    """

    tensor: SdpaTensor

    def __post_init__(self) -> None:
        t = self.tensor
        rank = len(t.dims)
        if not 1 <= rank <= 4:
            raise ValueError(f"bias rank must be in [1, 4], got {rank}")
        # Padding would move a ragged bias's batch axis off its B + 1 offsets.
        if t.ragged and rank != 4:
            raise ValueError(f"a ragged bias must be rank 4, got rank {rank}")
        pad = 4 - rank
        dims = (1,) * pad + t.dims
        strides = tuple(
            0 if d == 1 else s for d, s in zip(dims, (0,) * pad + t.strides)
        )
        if (dims, strides) != (t.dims, t.strides):
            object.__setattr__(self, "tensor", replace(t, dims=dims, strides=strides))


@dataclass(frozen=True)
class SdpaDropout:
    """Dropout: probability plus the RNG inputs, the explicit mask, and the debug dump."""

    probability: float
    seed: SdpaTensor | None = None
    offset: SdpaTensor | None = None
    mask: SdpaTensor | None = None
    scale: SdpaTensor | None = None
    rng_dump: SdpaTensor | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "probability", float(self.probability))


@dataclass(frozen=True)
class SdpaPaged:
    """Paged KV. Any one of the three hipDNN fields makes the request paged.

    Each is optional because the schema makes each optional; the page-table
    format itself is not mapped yet, so every candidate declines ``paged``.
    """

    page_table_k: SdpaTensor | None = None
    page_table_v: SdpaTensor | None = None
    max_seq_len_kv: int | None = None


@dataclass(frozen=True)
class SdpaFp8:
    """FP8 (de)quantization operands.

    hipDNN indexes the Q and K descales by the KV head of K and the V descale by
    the KV head of V; each is a scalar or ``(B, H_kv, 1, 1)``.
    """

    descale_q: SdpaTensor | None = None
    descale_k: SdpaTensor | None = None
    descale_v: SdpaTensor | None = None
    descale_s: SdpaTensor | None = None
    scale_s: SdpaTensor | None = None
    scale_o: SdpaTensor | None = None
    amax_s: SdpaTensor | None = None
    amax_o: SdpaTensor | None = None

    @property
    def has_s_o(self) -> bool:
        """Softmax / output quantization is requested (not just Q/K/V descale)."""
        return any(
            t is not None
            for t in (
                self.descale_s,
                self.scale_s,
                self.scale_o,
                self.amax_s,
                self.amax_o,
            )
        )


def _layout_class(t: SdpaTensor) -> str:
    """``"bshd"``, ``"bhsd"`` or ``"strided"`` from the H/S/D strides.

    A size-1 axis has no meaningful stride, so it matches any value. When both
    packed layouts match (``H == 1`` or ``S == 1``) the result is ``"bshd"``,
    the layout the dense kernels take natively.
    """
    _, h, s, d = t.dims
    actual = t.strides[1:]

    def matches(expected: tuple[int, int, int]) -> bool:
        return all(
            size == 1 or stride == want
            for size, stride, want in zip((h, s, d), actual, expected)
        )

    if matches((d, h * d, 1)):
        return "bshd"
    if matches((s * d, d, 1)):
        return "bhsd"
    return "strided"


def _has_batch_stride(t: SdpaTensor) -> bool:
    """Batch stride differs from the packed ``H * S * D`` (ignored when ragged or B == 1)."""
    b, h, s, d = t.dims
    return not t.ragged and b > 1 and t.strides[0] != h * s * d


@dataclass(frozen=True)
class SdpaRequest(OperatorRequest):
    """A hipDNN SDPA forward node, losslessly, as a dispatch request.

    See the module docstring for the normalization rules and
    ``SDPA_FEATURES`` for what :meth:`features` reports.

    For candidates and for lowering to ``dispatch.attention.AttentionRequest``:

    - :meth:`normalized` has no ``layout`` key (layout is the ``layout_*``
      features), so a candidate's ``Capability.layouts`` must stay empty or it
      declines every request.
    - hipDNN ``left_bound = W`` keeps ``W + 1`` keys; ``AttentionBand`` /
      ``sliding_window`` ``W`` sees ``W`` keys. Lower with ``left_bound + 1``.
    - ``causal_tl`` / ``causal_br`` / ``window*`` here correspond to
      ``ATTENTION_FEATURES`` ``causal`` / ``causal_bottom_right`` / ``band``.
    """

    arch: str

    # Required operands
    q: SdpaTensor
    k: SdpaTensor
    v: SdpaTensor
    o: SdpaTensor
    compute_dtype: Dtype = "fp32"

    # Scale: exactly one source. "default" means 1.0, as in cuDNN and hipDNN.
    scale_source: Literal["default", "value", "tensor"] = "default"
    scale: float = 1.0
    scale_tensor: SdpaTensor | None = None

    # Mask, normalized (deprecated causal flags already folded in)
    diagonal: Literal["top_left", "bottom_right"] = "top_left"
    left_bound: int = -1
    right_bound: int = -1
    alibi: bool = False
    bias: SdpaBias | None = None
    block_mask: SdpaTensor | None = None
    sink_token: SdpaTensor | None = None

    # Sequence lengths
    padding_mask: bool = False
    seq_len_q: SdpaTensor | None = None
    seq_len_kv: SdpaTensor | None = None

    # Optional subsystems
    dropout: SdpaDropout | None = None
    paged: SdpaPaged | None = None
    fp8: SdpaFp8 | None = None

    # Outputs
    stats: SdpaTensor | None = None
    max: SdpaTensor | None = None
    sum_exp: SdpaTensor | None = None

    # Hints
    mma_core_mode: Dtype | None = None
    implementation: Literal["auto", "composite", "unified"] = "auto"

    # Dispatch selectors, same meaning as AttentionRequest
    op: str = "sdpa_fwd"
    algorithm: str = "auto"
    spec_id: str = "auto"
    tuning_id: str = "auto"
    # The knob overrides ``tuning_id`` stands for, as sorted (name, value) pairs.
    tuning_knobs: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        self._normalize()
        self._validate()

    def _normalize(self) -> None:
        # Same canonical form as AttentionRequest (dispatch.attention.common.canonical_arch),
        # so "gfx950:sramecc+" passes Capability.check's exact arch match.
        object.__setattr__(
            self, "arch", base_arch_from_target_id(self.arch.strip().lower())
        )
        object.__setattr__(self, "left_bound", _as_int(self.left_bound, "left_bound"))
        object.__setattr__(
            self, "right_bound", _as_int(self.right_bound, "right_bound")
        )
        object.__setattr__(self, "tuning_knobs", normalize_knobs(self.tuning_knobs))
        if self.scale_source != "value":
            object.__setattr__(self, "scale", 1.0)
        object.__setattr__(self, "scale", float(self.scale))
        if self.scale_source == "value" and self.scale == 1.0:
            object.__setattr__(self, "scale_source", "default")
        # A group with nothing set requests nothing; as from_hipdnn reads it, it
        # is absent, so it reports no feature and does not split the cache key.
        for name, empty in (
            ("paged", SdpaPaged()),
            ("fp8", SdpaFp8()),
            ("dropout", SdpaDropout(0.0)),
        ):
            if getattr(self, name) == empty:
                object.__setattr__(self, name, None)

    def _validate(self) -> None:
        """Frontend rules (``SdpaFwdNode::pre_validate_node``) plus Python-caller ones."""
        _check_bool(self.alibi, "alibi")
        _check_bool(self.padding_mask, "padding_mask")
        _check_dtype(self.compute_dtype, "compute")
        if self.mma_core_mode is not None:
            _check_dtype(self.mma_core_mode, "mma_core_mode")
        _check_choice(self.diagonal, _DIAGONALS, "diagonal")
        _check_choice(self.scale_source, _SCALE_SOURCES, "scale_source")
        _check_choice(self.implementation, _IMPLEMENTATIONS, "implementation")
        if self.left_bound < -1 or self.right_bound < -1:
            raise ValueError(
                "left_bound and right_bound must be >= -1 (-1 is unbounded), got "
                f"left_bound={self.left_bound}, right_bound={self.right_bound}"
            )

        for name in ("q", "k", "v", "o"):
            rank = len(getattr(self, name).dims)
            if rank != 4:
                raise ValueError(f"{name} must be rank 4 (BHSD), got rank {rank}")
        q, k, v, o = self.q.dims, self.k.dims, self.v.dims, self.o.dims
        if not q[0] == k[0] == v[0]:
            raise ValueError(f"batch mismatch: Q {q[0]}, K {k[0]}, V {v[0]}")
        if q[3] != k[3]:
            raise ValueError(f"head dim mismatch: Q {q[3]} vs K {k[3]}")
        if k[2] != v[2]:
            raise ValueError(f"seq_kv mismatch: K {k[2]} vs V {v[2]}")
        if q[1] % k[1] or q[1] % v[1]:
            raise ValueError(
                f"h_q={q[1]} must be divisible by h_k={k[1]} and h_v={v[1]}"
            )
        if o != (q[0], q[1], q[2], v[3]):
            raise ValueError(
                f"O dims {o} must be (B, H_q, S_q, D_v) = {(q[0], q[1], q[2], v[3])}"
            )

        if self.scale_source == "tensor":
            if self.scale_tensor is None:
                raise ValueError('scale_source="tensor" needs scale_tensor')
            if self.scale_tensor.volume != 1:
                raise ValueError(
                    f"scale tensor must have volume 1, got dims {self.scale_tensor.dims}"
                )
        elif self.scale_tensor is not None:
            raise ValueError(
                "scale_tensor and a baked scale are mutually exclusive; "
                f'set scale_source="tensor" (got {self.scale_source!r})'
            )

        if self.bias is not None:
            # Last two dims are the frontend's rule. B and H are checked too: the
            # hipDNN reference indexes a non-broadcast bias dim with the output
            # index, so any other extent reads out of range.
            want = (q[0], q[1], q[2], k[2])
            got = self.bias.tensor.dims
            for axis, name in enumerate(("batch", "h_q", "s_q", "s_kv")):
                if got[axis] not in (1, want[axis]):
                    raise ValueError(
                        f"bias {name} dim {got[axis]} must be {want[axis]} or 1 "
                        f"(bias dims {got}, target {want})"
                    )

    # Derived, read-only views
    @property
    def dtype(self) -> str:
        """Q's dtype; what ``Capability.dtypes`` gates on."""
        return self.q.dtype

    @property
    def batch(self) -> int:
        return self.q.dims[0]

    @property
    def h_q(self) -> int:
        return self.q.dims[1]

    @property
    def h_k(self) -> int:
        return self.k.dims[1]

    @property
    def h_v(self) -> int:
        return self.v.dims[1]

    @property
    def s_q(self) -> int:
        return self.q.dims[2]

    @property
    def s_kv(self) -> int:
        return self.k.dims[2]

    @property
    def d_qk(self) -> int:
        return self.q.dims[3]

    @property
    def d_v(self) -> int:
        return self.v.dims[3]

    def dims(self) -> dict[str, int]:
        return {
            "batch": self.batch,
            "h_q": self.h_q,
            "h_k": self.h_k,
            "h_v": self.h_v,
            "s_q": self.s_q,
            "s_kv": self.s_kv,
            "d_qk": self.d_qk,
            "d_v": self.d_v,
            "gqa_ratio": self.h_q // self.h_k,
        }

    def _operands(self) -> tuple[SdpaTensor, ...]:
        """Every tensor operand present on the request, required and optional.

        Optional groups (bias, dropout, paged, fp8) are read through their
        dataclass fields, so a tensor added to one of them is covered without
        touching this method. A ragged operand's offset tensor is an operand
        too: a kernel loads it.
        """
        found: list[SdpaTensor] = [self.q, self.k, self.v, self.o]
        for value in (
            self.scale_tensor,
            self.block_mask,
            self.sink_token,
            self.seq_len_q,
            self.seq_len_kv,
            self.stats,
            self.max,
            self.sum_exp,
            self.bias,
            self.dropout,
            self.paged,
            self.fp8,
        ):
            if isinstance(value, SdpaTensor):
                found.append(value)
            elif value is not None:
                found.extend(
                    t
                    for f in fields(value)
                    if isinstance(t := getattr(value, f.name), SdpaTensor)
                )
        found.extend(t.ragged_offset for t in tuple(found) if t.ragged)
        return tuple(found)

    def features(self) -> frozenset[str]:
        active: set[str] = set()
        primaries = (self.q, self.k, self.v, self.o)

        if self.h_q != self.h_k or self.h_q != self.h_v:
            active.add("gqa")
        if self.h_k != self.h_v:
            active.add("kv_heads_differ")
        if self.d_qk != self.d_v:
            active.add("d_v_differs")
        if any(t.dtype != self.q.dtype for t in primaries[1:]):
            active.add("dtype_mixed")
        if self.compute_dtype != "fp32":
            active.add("compute_dtype")

        layout = _layout_class(self.q)
        active.add(f"layout_{layout}")
        if any(_layout_class(t) != layout for t in (self.k, self.v)):
            active.add("kv_layout_differs")
        if _layout_class(self.o) != layout:
            active.add("o_layout_differs")
        if any(_has_batch_stride(t) for t in primaries):
            active.add("batch_stride")
        if any(t.dims[3] > 1 and t.strides[3] != 1 for t in primaries):
            active.add("d_stride")
        # Every operand counts, not just Q/K/V/O: a kernel that vector-loads an
        # under-aligned bias, or reads a ragged mask, is wrong on that operand too.
        # A by-value tensor is a host pointer, which hipDNN exempts from alignment.
        operands = self._operands()
        if any(t.alignment < _BASE_ALIGNMENT for t in operands if not t.by_value):
            active.add("unaligned")

        if self.scale_source == "tensor":
            active.add("scale_tensor")

        active |= self._mask_features()
        if self.bias is not None:
            active.add("bias")
            target = (self.batch, self.h_q, self.s_q, self.s_kv)
            if any(d == 1 and t > 1 for d, t in zip(self.bias.tensor.dims, target)):
                active.add("bias_broadcast")
        if self.alibi:
            active.add("alibi")
        if self.block_mask is not None:
            active.add("block_mask")
        if self.sink_token is not None:
            active.add("sinks")

        if (
            self.padding_mask
            or self.seq_len_q is not None
            or self.seq_len_kv is not None
        ):
            active.add("padded")
        if any(t.ragged for t in operands):
            active.add("ragged")
        if self.dropout is not None:
            active.add("dropout")
        if self.paged is not None:
            active.add("paged")
        if self.fp8 is not None or any(
            t.dtype in _FP8_DTYPES for t in (self.q, self.k, self.v)
        ):
            active.add("fp8")
        if self.fp8 is not None and self.fp8.has_s_o:
            active.add("fp8_s_o")

        if self.stats is not None:
            active.add("lse")
        if self.max is not None or self.sum_exp is not None:
            active.add("max_sum_exp")
        if self.mma_core_mode is not None:
            active.add("mma_core_mode")
        if self.implementation != "auto":
            active.add("implementation_hint")
        return frozenset(active)

    def _mask_features(self) -> set[str]:
        """Classify the band. ``causal_br`` on a square problem is ``causal_tl``.

        Any other bounded band is a ``window``; ``window_br`` says its diagonal
        actually moves, and ``window_right`` says it keeps keys right of the
        diagonal, which the production ``dist < W`` windows never do.

        A bottom-right diagonal moves when ``s_q != s_kv``, and also whenever the
        lengths are per sequence (padded, ragged or paged): the offset is then
        ``s_kv[b] - s_q[b]``, which differs between batches even when the
        maximum extents are equal.
        """
        left, right = self.left_bound, self.right_bound
        if left == -1 and right == -1:
            return set()
        per_sequence = (
            self.padding_mask
            or self.seq_len_q is not None
            or self.seq_len_kv is not None
            or self.q.ragged
            or self.k.ragged
            or self.paged is not None
        )
        moved = self.diagonal == "bottom_right" and (
            self.s_q != self.s_kv or per_sequence
        )
        if left == -1 and right == 0:
            return {"causal_br" if moved else "causal_tl"}
        active = {"window"}
        if moved:
            active.add("window_br")
        if right != 0:
            active.add("window_right")
        return active

    def normalized(self) -> dict:
        """Every field as plain JSON-able data, plus the derived ``dtype``.

        Round-trips through :meth:`from_normalized`.
        """
        payload = asdict(self)
        payload["dtype"] = self.dtype
        return payload

    @classmethod
    def from_normalized(cls, payload: dict[str, Any]) -> SdpaRequest:
        """Rebuild a request from :meth:`normalized` (or its JSON round trip)."""
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in payload:
                continue
            value = payload[f.name]
            if isinstance(value, dict):
                value = _rebuild(_NESTED.get(f.name, SdpaTensor), value)
            kwargs[f.name] = value
        return cls(**kwargs)


_NESTED: dict[str, type] = {
    "bias": SdpaBias,
    "dropout": SdpaDropout,
    "paged": SdpaPaged,
    "fp8": SdpaFp8,
}


def _tensor_from(payload: dict[str, Any]) -> SdpaTensor:
    offset = payload.get("ragged_offset")
    if isinstance(offset, dict):
        payload = {**payload, "ragged_offset": _tensor_from(offset)}
    return SdpaTensor(**payload)


def _rebuild(cls: type, payload: dict[str, Any]) -> Any:
    if cls is SdpaTensor:
        return _tensor_from(payload)
    return cls(
        **{
            key: _tensor_from(value) if isinstance(value, dict) else value
            for key, value in payload.items()
        }
    )
