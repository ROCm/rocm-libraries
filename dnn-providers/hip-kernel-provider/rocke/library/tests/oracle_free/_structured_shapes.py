# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shapes, encodings and V exposures of the structured-logit check.

Everything here is independent of the kernel's arithmetic: the logical shape of
a launch and its mask, exact FP16/BF16 encoding of inputs and targets, the
legal key interval of a query row, the V profiles that expose probability
columns, and the container of raw output bits. There is no attention
implementation in this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from functools import cache
from types import MappingProxyType

import numpy as np
from rocke.numeric.bf16 import bf16_to_f32, f32_to_bf16

_DTYPES = frozenset({"fp16", "bf16"})
_MASKS = frozenset({"causal", "window"})
_FORMAT = {
    "fp16": (10, 5, 15),
    "bf16": (7, 8, 127),
}
BASIS_OFFSETS = (0, 64, 128, 192)
# Head h scales its V profile by 2**h, so heads are distinguishable: a kernel
# that reads the wrong V head, uses a wrong head stride, or writes one output
# head everywhere moves every legal coordinate by at least a factor of two. A
# power of two is exact in both dtypes. The cap keeps outputs well inside the
# FP16 range.
_MAX_HEADS = 8


class MaskMutation(str, Enum):
    """A legal key set that is wrong by one key at a mask boundary."""

    INCLUDE_FUTURE_KEY = "include_future_key"
    EXCLUDE_LAST_KEY = "exclude_last_key"
    INCLUDE_OLD_WINDOW_KEY = "include_old_window_key"
    EXCLUDE_OLDEST_WINDOW_KEY = "exclude_oldest_window_key"


@dataclass(frozen=True)
class DenseShape:
    """Logical dense-attention shape, dtype and mask of one launch.

    Query head ``h`` reads KV head ``h``: the check gives every head its own
    logit, so it needs as many KV heads as query heads."""

    batch: int
    seqlen_q: int
    seqlen_k: int
    nhead_q: int
    nhead_k: int
    head_dim: int
    dtype: str
    mask: str
    window: int = 0

    def __post_init__(self) -> None:
        for name in (
            "batch",
            "seqlen_q",
            "seqlen_k",
            "nhead_q",
            "nhead_k",
            "head_dim",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.nhead_q != self.nhead_k:
            raise ValueError(
                "nhead_q must equal nhead_k: each head carries its own logit"
            )
        if self.nhead_k > _MAX_HEADS:
            raise ValueError(
                f"nhead_k must be at most {_MAX_HEADS} for distinct head scales"
            )
        if self.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}")
        if self.mask not in _MASKS:
            raise ValueError(f"mask must be one of {sorted(_MASKS)}")
        if self.mask == "window":
            if self.window <= 0:
                raise ValueError("window mask requires window > 0")
        elif self.window:
            raise ValueError("window must be zero unless mask='window'")

    @property
    def output_shape(self) -> tuple[int, int, int, int]:
        """Return the runner's logical ``[B, Sq, Hq, D]`` output shape."""
        return (self.batch, self.seqlen_q, self.nhead_q, self.head_dim)

    @property
    def key_shape(self) -> tuple[int, int, int, int]:
        """Return the runner's logical ``[B, Sk, Hkv, D]`` K/V shape."""
        return (self.batch, self.seqlen_k, self.nhead_k, self.head_dim)

    @property
    def head_scale_exponents(self) -> tuple[int, ...]:
        """Per query head ``h``, the exponent of its V scale ``2**h``."""
        return tuple(range(self.nhead_q))

    @property
    def case_id(self) -> str:
        """Return a stable identifier used to close calibration scope."""
        mask = f"window{self.window}" if self.mask == "window" else self.mask
        return (
            f"{self.dtype}-{mask}-b{self.batch}-sq{self.seqlen_q}-"
            f"sk{self.seqlen_k}-hq{self.nhead_q}-hk{self.nhead_k}-d{self.head_dim}"
        )


# The two shapes of the quick matrix: top-left causal in FP16 and a sliding
# window of 64 keys in BF16.
QUICK_SHAPES = (
    DenseShape(1, 256, 256, 4, 4, 64, "fp16", "causal"),
    DenseShape(1, 256, 256, 4, 4, 64, "bf16", "window", 64),
)


@dataclass(frozen=True)
class SpecFingerprint:
    """Proof-relevant fields of the dispatch-resolved dense subject."""

    path: str
    persistent: bool
    runtime_shape: bool
    head_dim: int
    block_m: int
    block_n: int
    use_exp2_fast: bool
    use_sinks: bool
    emit_lse: bool


def legal_interval(
    case: DenseShape,
    query: int,
    mutation: MaskMutation | str | None = None,
) -> tuple[int, int]:
    """Return inclusive analyzer legal bounds, or ``(0, -1)`` when empty."""
    if not 0 <= query < case.seqlen_q:
        raise IndexError(query)
    selected = None if mutation is None else MaskMutation(mutation)
    causal_only = (MaskMutation.INCLUDE_FUTURE_KEY, MaskMutation.EXCLUDE_LAST_KEY)
    if selected is not None and (selected in causal_only) != (case.mask == "causal"):
        raise ValueError(f"mutation {selected.value} does not apply to {case.mask}")

    upper_shift = 0
    if selected == MaskMutation.INCLUDE_FUTURE_KEY:
        upper_shift = 1
    elif selected == MaskMutation.EXCLUDE_LAST_KEY:
        upper_shift = -1
    upper = min(query + upper_shift, case.seqlen_k - 1)

    lower = 0
    if case.mask == "window":
        window = case.window
        if selected == MaskMutation.INCLUDE_OLD_WINDOW_KEY:
            window += 1
        elif selected == MaskMutation.EXCLUDE_OLDEST_WINDOW_KEY:
            window -= 1
        lower = max(0, query - window + 1)
    return (lower, upper) if lower <= upper else (0, -1)


def _round_ratio_to_even(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    twice = 2 * remainder
    return quotient + int(
        twice > denominator or (twice == denominator and quotient & 1)
    )


@cache
def target_bits(numerator: int, denominator: int, dtype: str) -> int:
    """Round one nonnegative exact rational to FP16 or BF16 with RNE."""
    if dtype not in _FORMAT:
        raise ValueError(dtype)
    if numerator < 0 or denominator <= 0:
        raise ValueError("target ratio requires numerator >= 0 and denominator > 0")
    if numerator == 0:
        return 0

    fraction_bits, exponent_bits, bias = _FORMAT[dtype]
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if numerator < denominator << exponent:
            exponent -= 1
    elif numerator << -exponent < denominator:
        exponent -= 1

    min_exponent = 1 - bias
    max_exponent = (1 << exponent_bits) - 2 - bias
    if exponent < min_exponent:
        shift = fraction_bits - min_exponent
        rounded = _round_ratio_to_even(numerator << shift, denominator)
        if rounded >= 1 << fraction_bits:
            return 1 << fraction_bits
        return rounded

    shift = exponent - fraction_bits
    if shift >= 0:
        rounded = _round_ratio_to_even(numerator, denominator << shift)
    else:
        rounded = _round_ratio_to_even(numerator << -shift, denominator)
    if rounded == 1 << (fraction_bits + 1):
        rounded >>= 1
        exponent += 1
    if exponent > max_exponent:
        return ((1 << exponent_bits) - 1) << fraction_bits
    encoded_exponent = exponent + bias
    return (encoded_exponent << fraction_bits) | (rounded - (1 << fraction_bits))


def decode_bits(bits: np.ndarray | int, dtype: str) -> np.ndarray:
    """Decode raw FP16/BF16 bits to FP64 without losing signed-zero input bits."""
    if dtype not in _DTYPES:
        raise ValueError(dtype)
    raw = np.asarray(bits, dtype=np.uint16)
    if dtype == "fp16":
        return raw.copy().view(np.float16).astype(np.float64)
    # The shared codec returns at least 1-D; keep a scalar input scalar.
    return bf16_to_f32(raw).reshape(raw.shape).astype(np.float64)


def encode_bits(values: np.ndarray, dtype: str) -> np.ndarray:
    """Encode FP32 values as raw FP16/BF16 bits, refusing any value that would round.

    Exactness keeps the subject's inputs independent of conversion rounding: the
    expectations assume the kernel reads exactly these values. Each decoded value
    must reproduce its FP32 source bit-for-bit, so signed zero survives and NaN or
    out-of-range inputs are rejected."""
    if dtype not in _DTYPES:
        raise ValueError(dtype)
    source = np.ascontiguousarray(values, dtype=np.float32)
    # NaN and infinity survive the bit round trip below, so reject them first.
    nonfinite = ~np.isfinite(source)
    if np.any(nonfinite):
        first = tuple(int(index) for index in np.argwhere(nonfinite)[0])
        raise ValueError(
            f"{int(np.count_nonzero(nonfinite))} value(s) are not exactly "
            f"representable in {dtype}, first {float(source[first])!r} at {first}"
        )
    if dtype == "fp16":
        with np.errstate(over="ignore"):
            bits = source.astype(np.float16).view(np.uint16)
    else:
        # Round-to-nearest-even, but an accepted value never rounds: the
        # round-trip check below rejects anything that is not exact.
        bits = f32_to_bf16(source).reshape(source.shape)
    round_trip = decode_bits(bits, dtype).astype(np.float32)
    inexact = round_trip.view(np.uint32) != source.view(np.uint32)
    if np.any(inexact):
        first = tuple(int(index) for index in np.argwhere(inexact)[0])
        raise ValueError(
            f"{int(np.count_nonzero(inexact))} value(s) are not exactly "
            f"representable in {dtype}, first {float(source[first])!r} at {first}"
        )
    return np.ascontiguousarray(bits)


def basis_offset(name: str) -> int:
    """The first key a ``basis@<offset>`` exposure selects."""
    prefix = "basis@"
    if not name.startswith(prefix):
        raise ValueError(f"unknown exposure {name!r}")
    text = name[len(prefix) :]
    try:
        offset = int(text)
    except ValueError as error:
        raise ValueError(f"invalid basis exposure {name!r}") from error
    if offset < 0 or str(offset) != text:
        raise ValueError(f"invalid basis exposure {name!r}")
    return offset


def exposure_values(shape: DenseShape, exposure: str) -> np.ndarray:
    """V ``[B, Sk, Hkv, D]`` of one exposure, scaled ``2**h`` per head.

    ``basis@o`` sets ``V[k, h, d]`` for ``k = o + d`` only, so each output
    dimension exposes one probability column. ``folded`` sets it for
    ``d = k mod D``, so each dimension exposes the sum of one residue class."""
    scales = np.ldexp(np.float32(1.0), np.arange(shape.nhead_k)).astype(np.float32)
    v = np.zeros(shape.key_shape, dtype=np.float32)
    if exposure == "folded":
        for key in range(shape.seqlen_k):
            v[:, key, :, key % shape.head_dim] = scales
        return v
    offset = basis_offset(exposure)
    for dim in range(shape.head_dim):
        key = offset + dim
        if key < shape.seqlen_k:
            v[:, key, :, dim] = scales
    return v


@dataclass(frozen=True, eq=False)
class Observation:
    """Named raw-bit subject outputs from basis and folded exposures."""

    exposures: tuple[tuple[str, np.ndarray], ...]

    def __post_init__(self) -> None:
        if not self.exposures:
            raise ValueError("an observation requires at least one exposure")
        normalized: list[tuple[str, np.ndarray]] = []
        names: set[str] = set()
        for name, values in self.exposures:
            if name in names:
                raise ValueError(f"duplicate exposure {name!r}")
            if name != "folded":
                basis_offset(name)
            raw = np.asarray(values)
            if raw.dtype != np.uint16:
                raise ValueError(f"exposure {name!r} must contain raw uint16 bits")
            copy = np.ascontiguousarray(raw).copy()
            copy.setflags(write=False)
            normalized.append((name, copy))
            names.add(name)
        object.__setattr__(self, "exposures", tuple(normalized))

    @classmethod
    def from_mapping(cls, exposures: Mapping[str, np.ndarray]) -> Observation:
        """Copy a mapping into one immutable observation."""
        return cls(tuple(exposures.items()))

    @property
    def by_name(self) -> Mapping[str, np.ndarray]:
        """Return a read-only name-to-bits view."""
        return MappingProxyType(dict(self.exposures))


def _observed_basis_offsets(observation: Observation) -> tuple[int, ...]:
    return tuple(
        sorted(
            basis_offset(name) for name, _ in observation.exposures if name != "folded"
        )
    )


def validate_observation(
    case: DenseShape, observation: Observation, *, require_complete: bool
) -> None:
    """Check an observation against the shape; with ``require_complete`` it must
    hold the folded exposure and basis exposures that cover every key once."""
    for name, values in observation.exposures:
        if values.shape != case.output_shape:
            raise ValueError(
                f"exposure {name!r} shape {values.shape} != {case.output_shape}"
            )
    if not require_complete:
        return
    names = observation.by_name
    if "folded" not in names:
        raise ValueError("a gated observation requires a folded exposure")
    offsets = _observed_basis_offsets(observation)
    exposed = [
        offset + dim
        for offset in offsets
        for dim in range(case.head_dim)
        if offset + dim < case.seqlen_k
    ]
    if exposed != list(range(case.seqlen_k)):
        raise ValueError("basis exposures must expose every key exactly once")
