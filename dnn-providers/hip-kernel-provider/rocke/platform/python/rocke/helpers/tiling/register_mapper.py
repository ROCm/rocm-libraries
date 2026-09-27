# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Register mapper -- port of CK TileDistrEncRegMap (pure-int, no IR).

INTERNAL machinery (not in the top-level public surface). Given a
:class:`~rocke.helpers.tiling.encoding.WarpDistributionEncoding`, computes the fragment dimensions and
the concrete ``(lane_index, register_index) -> matrix coordinate`` mapping in pure Python (a
builder-free mirror of rocke's ``calculate_x``). Used by:

- validation: ``num_lanes`` must equal the wave size; ``num_vector_items`` the per-lane
  register count (this closes the silent half-fragment hazard),
- reflection/visualization (forward + inverse maps),
- the oracle: the C-tile forward map must reproduce ``MfmaAtom.lane_to_output``.

Lane / vector decomposition rule (verified against the MFMA layout and the compact
unmerge-merge descriptor): the merged lane index and register index each split across
their contributing buckets with the **last-listed contributor changing fastest**, and each
X coordinate is reconstructed mixed-radix with the innermost H level at stride 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from .encoding import WarpDistributionEncoding

__all__ = ["RegisterMapper", "LaneRegister"]

@dataclass(frozen=True)
class LaneRegister:
    """A (lane, per-lane register slot) location within a wave's fragment."""

    lane: int
    register: int

class RegisterMapper:
    """Pure-int register/lane <-> matrix-coordinate mapper for one warp encoding."""

    def __init__(self, encoding: WarpDistributionEncoding) -> None:
        self._encoding = encoding
        lane_majors = encoding.lane_to_rh_major[0] if encoding.lane_to_rh_major else ()
        lane_minors = encoding.lane_to_rh_minor[0] if encoding.lane_to_rh_minor else ()
        self._lane_buckets = tuple(zip(lane_majors, lane_minors))
        self._lane_lengths = tuple(
            self._bucket_length(major, minor) for major, minor in self._lane_buckets
        )
        self._register_buckets = tuple(
            zip(encoding.register_to_rh_major, encoding.register_to_rh_minor)
        )
        self._register_lengths = tuple(
            self._bucket_length(major, minor) for major, minor in self._register_buckets
        )

    def _bucket_length(self, major: int, minor: int) -> int:
        if major == 0:
            return self._encoding.replication_lengths[minor]
        return self._encoding.hierarchical_lengths[major - 1][minor]

    @property
    def num_lanes(self) -> int:
        """Lanes the encoding spans (must equal the wave size)."""
        return prod(self._lane_lengths) if self._lane_lengths else 1

    @property
    def num_vector_items(self) -> int:
        """Per-lane register slots (the fragment's per-lane element count)."""
        return prod(self._register_lengths) if self._register_lengths else 1

    @property
    def num_repeat(self) -> int:
        """Replication factor (RDNA3 repeat); 1 when there is no replication."""
        lengths = self._encoding.replication_lengths
        return lengths[0] if lengths else 1

    @property
    def matrix_major_size(self) -> int:
        """Extent of the first X dim (rows / M for C; M for A)."""
        return prod(self._encoding.hierarchical_lengths[0])

    @property
    def matrix_minor_size(self) -> int:
        """Extent of the second X dim (cols / N for C; K for A/B)."""
        return prod(self._encoding.hierarchical_lengths[1])

    @staticmethod
    def _split_last_fastest(index: int, lengths: tuple[int, ...]) -> list[int]:
        parts = [0] * len(lengths)
        remainder = index
        for position in reversed(range(len(lengths))):
            parts[position] = remainder % lengths[position]
            remainder //= lengths[position]
        return parts

    def matrix_coordinates(self, lane: int, register: int) -> tuple[int, ...]:
        """Return the matrix coordinate (per X dim) held by ``(lane, register)``.

        For a C encoding this is ``(row, col)``; for an A encoding ``(m, k)``.
        """
        if not 0 <= lane < self.num_lanes:
            raise ValueError(
                f"lane out of range -- lane={lane}, expected 0..{self.num_lanes - 1}"
            )
        if not 0 <= register < self.num_vector_items:
            raise ValueError(
                f"register out of range -- register={register}, "
                f"expected 0..{self.num_vector_items - 1}"
            )
        contributor: dict[tuple[int, int], int] = {}
        for bucket, value in zip(
            self._lane_buckets, self._split_last_fastest(lane, self._lane_lengths)
        ):
            contributor[bucket] = value
        for bucket, value in zip(
            self._register_buckets,
            self._split_last_fastest(register, self._register_lengths),
        ):
            contributor[bucket] = value

        coordinates: list[int] = []
        for x_dim, levels in enumerate(self._encoding.hierarchical_lengths):
            coordinate = 0
            stride = 1
            for level in reversed(range(len(levels))):
                coordinate += contributor.get((x_dim + 1, level), 0) * stride
                stride *= levels[level]
            coordinates.append(coordinate)
        return tuple(coordinates)

    def inverse_map(self) -> dict[tuple[int, ...], LaneRegister]:
        """Matrix coordinate -> first ``(lane, register)`` that holds it."""
        mapping: dict[tuple[int, ...], LaneRegister] = {}
        for lane in range(self.num_lanes):
            for register in range(self.num_vector_items):
                key = self.matrix_coordinates(lane, register)
                mapping.setdefault(key, LaneRegister(lane=lane, register=register))
        return mapping
