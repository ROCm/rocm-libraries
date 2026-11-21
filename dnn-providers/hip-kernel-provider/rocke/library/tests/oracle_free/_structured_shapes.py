# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shapes, masks and V exposures of the structured-logit check.

Everything here is independent of the kernel's arithmetic: the logical shape of
a launch and its mask, the legal key interval of a query row, and the V
profiles that expose probability columns. There is no attention implementation
in this module.

What does not depend on the logits has one home in this directory,
``_analytical.py``, and is read from there: the FP16/BF16 formats, the exact
encoding and decoding of raw bits, the exact rounding of a rational target, and
the container of raw output bits with its exposure names and coverage check.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ._analytical import (
    _DTYPES,
    UniformObservation,
    _basis_offset,
    _validate_observation,
)

# The container of raw output bits, its exposure-name parser and its coverage
# check, under the names this check uses. Nothing in them depends on the logits.
Observation = UniformObservation
basis_offset = _basis_offset
validate_observation = _validate_observation

_MASKS = frozenset({"causal", "window"})
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
