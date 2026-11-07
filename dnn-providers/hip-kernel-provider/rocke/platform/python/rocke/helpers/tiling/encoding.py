# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""WarpDistributionEncoding -- the foundational coordinate-transform encoding.

This is the SUBSTRATE type the whole tiling layer sits on: the raw
``(replication, hierarchical, lane, register)`` mapping that every other module speaks in --
the descriptors, the authoring surface (:func:`~rocke.helpers.tiling.layouts.make_tile_desc`), the MMA
calculators (:mod:`rocke.helpers.tiling.mma.warp_encoding`), the register mapper, the IR verbs, and
reflection all consume or produce it. It lives at the package root (not under ``mma/``) precisely
because it is not MMA-specific -- an MMA operand layout is just one producer of it.

The encoding is field-compatible with rocke's ``TileDistributionEncoding`` (same six fields), so
the C encoding can be compared against ``make_c_warp_dstr_encoding`` field-for-field.

Field glossary: ``replication_lengths`` = Rs; ``hierarchical_lengths`` = Hs;
``lane_to_rh_major/minor`` = Ps2RHs_major/minor; ``register_to_rh_major/minor`` =
Ys2RHs_major/minor. Major 0 = R bucket; major 1..N = X-dim (major-1); minor = level.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

__all__ = [
    "LaneTerm",
    "WarpDistributionEncoding",
    "lane_terms",
    "register_coord_offsets",
]


@dataclass(frozen=True)
class WarpDistributionEncoding:
    """Raw coordinate-transform encoding; mirrors rocke's fields.

    Validated as a bijection in ``__post_init__``: every hierarchical (H) and replication
    (R) bucket is referenced by exactly one lane (P) sub-entry or register (Y) entry, and
    every bucket is covered. This is the structural correctness net.
    """

    replication_lengths: tuple[int, ...]
    hierarchical_lengths: tuple[tuple[int, ...], ...]
    lane_to_rh_major: tuple[tuple[int, ...], ...]
    lane_to_rh_minor: tuple[tuple[int, ...], ...]
    register_to_rh_major: tuple[int, ...]
    register_to_rh_minor: tuple[int, ...]

    def __post_init__(self) -> None:
        self._validate_bijection()

    def bucket_length(self, major: int, minor: int) -> int:
        """Length of a bucket: major 0 -> replication level; major>=1 -> X-dim level."""
        if major == 0:
            return self.replication_lengths[minor]
        return self.hierarchical_lengths[major - 1][minor]

    def _bucket_in_range(self, kind: str, index: int, major: int, minor: int) -> None:
        if major == 0:
            if not (0 <= minor < len(self.replication_lengths)):
                raise ValueError(
                    f"{kind}{index} R-bucket out of range -- (major=0, minor={minor}), "
                    f"replication has {len(self.replication_lengths)} levels"
                )
            return
        if not (1 <= major <= len(self.hierarchical_lengths)):
            raise ValueError(
                f"{kind}{index} major out of range -- major={major}, "
                f"expected 0 (R) or 1..{len(self.hierarchical_lengths)} (X dims)"
            )
        levels = self.hierarchical_lengths[major - 1]
        if not (0 <= minor < len(levels)):
            raise ValueError(
                f"{kind}{index} H-bucket out of range -- (major={major}, minor={minor}), "
                f"X-dim {major - 1} has {len(levels)} levels"
            )

    def _validate_bijection(self) -> None:
        if len(self.lane_to_rh_major) != len(self.lane_to_rh_minor):
            raise ValueError("lane_to_rh major/minor rank mismatch")
        if len(self.register_to_rh_major) != len(self.register_to_rh_minor):
            raise ValueError("register_to_rh major/minor rank mismatch")

        seen: set[tuple[int, int]] = set()

        def claim(kind: str, index: int, major: int, minor: int) -> None:
            self._bucket_in_range(kind, index, major, minor)
            key = (major, minor)
            if key in seen:
                raise ValueError(
                    f"bucket (major={major}, minor={minor}) referenced by more than one "
                    f"lane/register entry -- the distribution is not a bijection"
                )
            seen.add(key)

        for lane_index, (majors, minors) in enumerate(
            zip(self.lane_to_rh_major, self.lane_to_rh_minor)
        ):
            if len(majors) != len(minors):
                raise ValueError(
                    f"lane P{lane_index} major/minor sub-sequence mismatch"
                )
            for major, minor in zip(majors, minors):
                claim("P", lane_index, major, minor)
        for reg_index, (major, minor) in enumerate(
            zip(self.register_to_rh_major, self.register_to_rh_minor)
        ):
            claim("Y", reg_index, major, minor)

        for x_dim, levels in enumerate(self.hierarchical_lengths):
            for level in range(len(levels)):
                if (x_dim + 1, level) not in seen:
                    raise ValueError(
                        f"H bucket has no contributor -- X-dim {x_dim} level {level} is "
                        f"unreferenced; every element must map to a lane or register"
                    )
        for level in range(len(self.replication_lengths)):
            if (0, level) not in seen:
                raise ValueError(
                    f"R bucket has no contributor -- replication level {level} is "
                    f"unreferenced"
                )


def register_coord_offsets(
    encoding: WarpDistributionEncoding, register_index: int
) -> tuple[int, ...]:
    """The register's COMPILE-TIME contribution to each X coordinate (lane buckets contribute 0),
    computed in pure Python. A coordinate is its lane part (the same for every register) plus this
    offset, so consecutive registers' addresses differ by a constant the backend can fold.
    """
    register_buckets = list(
        zip(encoding.register_to_rh_major, encoding.register_to_rh_minor)
    )
    register_lengths = [encoding.bucket_length(*bucket) for bucket in register_buckets]
    remainder = register_index
    register_values = [0] * len(register_lengths)
    for position in reversed(range(len(register_lengths))):
        register_values[position] = remainder % register_lengths[position]
        remainder //= register_lengths[position]
    reg_const = dict(zip(register_buckets, register_values))
    offsets: list[int] = []
    for x_dim, levels in enumerate(encoding.hierarchical_lengths):
        coord = 0
        stride = 1
        for level in reversed(range(len(levels))):
            value = reg_const.get((x_dim + 1, level))
            if value is not None:
                coord += value * stride
            stride *= levels[level]
        offsets.append(coord)
    return tuple(offsets)


@dataclass(frozen=True)
class LaneTerm:
    """The per-lane part of one X coordinate: the values the lanes give it are ``0 <= v < extent``,
    each a multiple of ``step``. ``extent == 1`` means the coordinate has no per-lane part (every
    lane holds the same value)."""

    extent: int
    step: int


def lane_terms(encoding: WarpDistributionEncoding) -> tuple[LaneTerm, ...]:
    """The exact per-lane term of each X coordinate, register contributions excluded.

    Each lane bucket of a coordinate takes every value in ``[0, length)`` independently (the lane
    id is split mixed-radix, every digit reduced modulo its length), so the term ranges over
    ``[0, 1 + sum((length - 1) * stride))`` and is a multiple of the gcd of those strides. With
    the register offsets it covers the axis's whole extent: ``extent - 1 + max(offset) == shape - 1``.
    """
    lane_buckets = {
        (major, minor)
        for majors, minors in zip(encoding.lane_to_rh_major, encoding.lane_to_rh_minor)
        for major, minor in zip(majors, minors)
    }
    terms = []
    for x_dim, levels in enumerate(encoding.hierarchical_lengths):
        extent, step, stride = 1, 0, 1
        for level in reversed(range(len(levels))):
            length = encoding.bucket_length(x_dim + 1, level)
            if (x_dim + 1, level) in lane_buckets and length > 1:
                extent += (length - 1) * stride
                step = gcd(step, stride)
            stride *= levels[level]
        terms.append(LaneTerm(extent, step or 1))
    return tuple(terms)
