# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Validate non-paged KV views and pack tiled-decode byte-stride arguments.

The public logical shape is [B, H_kv, capacity, D]. Both BSHD and BHSD physical
storage work without a copy. Only positive, non-overlapping, vector-aligned
views with unit D stride are admitted. K and V may use different layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index


@dataclass(frozen=True)
class KvTensorLayout:
    shape: tuple[int, int, int, int]
    strides: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if len(self.shape) != 4 or len(self.strides) != 4:
            raise ValueError("strided KV requires rank-4 [B,H,capacity,D] tensors")
        try:
            shape = tuple(index(x) for x in self.shape)
            strides = tuple(index(x) for x in self.strides)
        except TypeError as exc:
            raise ValueError("KV shape and strides must be integers") from exc
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "strides", strides)
        if any(x <= 0 for x in shape) or any(x <= 0 for x in strides):
            raise ValueError("KV extents and strides must be positive")
        if strides[3] != 1:
            raise ValueError("strided KV requires unit D stride")
        if any(n > 1 and (s * 2) % 16 for n, s in zip(shape[:3], strides[:3])):
            raise ValueError("strided KV outer strides must preserve 16-byte alignment")
        # Sufficient non-overlap check for dense axis permutations and padding.
        # Reject unusual interleavings instead of claiming arbitrary-view support.
        span = 1
        for stride, extent in sorted((s, n) for n, s in zip(shape, strides) if n > 1):
            if stride < span:
                raise ValueError("strided KV view may overlap")
            span += (extent - 1) * stride
        if span * 2 > (1 << 63) - 1:
            raise ValueError("strided KV address span exceeds signed 64-bit bytes")
        if self.head_span_bytes > 0x7FFF0000 or strides[2] * 2 > 0x7FFF0000:
            raise ValueError("strided KV per-head span exceeds the buffer-offset limit")
        if any(s * 2 > (1 << 63) - 1 for s in strides[:2]):
            raise ValueError("strided KV base strides exceed signed 64-bit bytes")

    @property
    def head_span_bytes(self) -> int:
        return 2 * ((self.shape[2] - 1) * self.strides[2] + self.shape[3])

    @property
    def storage_span_bytes(self) -> int:
        return 2 * (1 + sum((n - 1) * s for n, s in zip(self.shape, self.strides)))

    def arguments(self, prefix: str) -> dict[str, int]:
        return {
            f"{prefix}_stride_batch_bytes": self.strides[0] * 2,
            f"{prefix}_stride_head_bytes": self.strides[1] * 2,
            f"{prefix}_stride_token_bytes": self.strides[2] * 2,
            f"{prefix}_span_bytes": self.head_span_bytes,
        }

    @classmethod
    def from_tensor(cls, tensor) -> KvTensorLayout:
        if str(tensor.dtype) not in ("torch.float16", "torch.bfloat16"):
            raise ValueError("strided KV supports fp16/bf16 tensors")
        layout = cls(tuple(tensor.shape), tuple(tensor.stride()))
        if int(tensor.data_ptr()) % 16:
            raise ValueError("strided KV base pointer must be 16-byte aligned")
        available = tensor.untyped_storage().nbytes() - tensor.storage_offset() * 2
        if layout.storage_span_bytes > available:
            raise ValueError("strided KV view exceeds its backing storage")
        return layout


@dataclass(frozen=True)
class StridedKvCacheLayout:
    key: KvTensorLayout
    value: KvTensorLayout

    def __post_init__(self) -> None:
        if self.key.shape != self.value.shape:
            raise ValueError("K and V must have the same logical shape")

    def arguments(self) -> dict[str, int]:
        return {**self.key.arguments("k"), **self.value.arguments("v")}

    @classmethod
    def from_tensors(cls, key, value) -> StridedKvCacheLayout:
        if key.dtype != value.dtype or key.device != value.device:
            raise ValueError("K and V must have matching dtype and device")
        return cls(KvTensorLayout.from_tensor(key), KvTensorLayout.from_tensor(value))
