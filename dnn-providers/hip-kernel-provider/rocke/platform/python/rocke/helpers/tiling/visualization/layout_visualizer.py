# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Text reflection + visualization for warp encodings (the anti-anti-pattern).

A layout is understood by *seeing* it, never by decoding integer sequences. This module
renders an encoding as ASCII maps and a structured description. Text-first so it works in
the REPL, CI, and inside error messages; a richer (matplotlib/HTML) renderer is post-M1.

- :func:`describe` -- structured (machine-readable) reflection, for humans and agents.
- :func:`render_forward_map` -- lane x register -> matrix coordinate grid.
- :func:`render_inverse_map` -- matrix coordinate -> (lane, register).
"""

from __future__ import annotations

from ..encoding import WarpDistributionEncoding
from ..register_mapper import RegisterMapper

__all__ = ["describe", "render_forward_map", "render_inverse_map"]

def describe(encoding: WarpDistributionEncoding) -> dict[str, object]:
    """Structured, machine-readable reflection of an encoding (for humans and agents)."""
    mapper = RegisterMapper(encoding)
    return {
        "num_lanes": mapper.num_lanes,
        "num_vector_items": mapper.num_vector_items,
        "num_repeat": mapper.num_repeat,
        "matrix_major_size": mapper.matrix_major_size,
        "matrix_minor_size": mapper.matrix_minor_size,
        "hierarchical_lengths": encoding.hierarchical_lengths,
        "replication_lengths": encoding.replication_lengths,
    }

def render_forward_map(
    encoding: WarpDistributionEncoding,
    *,
    axis_names: tuple[str, str] = ("major", "minor"),
) -> str:
    """Render the lane x register -> (axis0, axis1) coordinate grid as ASCII.

    ``axis_names`` labels the two matrix axes, e.g. ``("row", "col")`` for a C encoding or
    ``("m", "k")`` for an A encoding.
    """
    mapper = RegisterMapper(encoding)
    header = f"lane \\ reg | " + " | ".join(
        f"r{register}" for register in range(mapper.num_vector_items)
    )
    lines = [
        f"forward map ({axis_names[0]},{axis_names[1]}) -- "
        f"{mapper.num_lanes} lanes x {mapper.num_vector_items} regs",
        header,
        "-" * len(header),
    ]
    for lane in range(mapper.num_lanes):
        cells = []
        for register in range(mapper.num_vector_items):
            coordinate = mapper.matrix_coordinates(lane, register)
            cells.append(",".join(str(value) for value in coordinate))
        lines.append(f"{lane:>9} | " + " | ".join(cells))
    return "\n".join(lines)

def render_inverse_map(
    encoding: WarpDistributionEncoding,
    *,
    axis_names: tuple[str, str] = ("major", "minor"),
) -> str:
    """Render the matrix coordinate -> (lane, register) grid as ASCII."""
    mapper = RegisterMapper(encoding)
    inverse = mapper.inverse_map()
    lines = [
        f"inverse map -- {axis_names[0]} 0..{mapper.matrix_major_size - 1} x "
        f"{axis_names[1]} 0..{mapper.matrix_minor_size - 1} (lane,reg); "
        f"each element replicated {mapper.num_repeat}x",
    ]
    for major in range(mapper.matrix_major_size):
        cells = []
        for minor in range(mapper.matrix_minor_size):
            location = inverse[(major, minor)]
            cells.append(f"L{location.lane}r{location.register}")
        lines.append(f"{major:>3} | " + " | ".join(cells))
    return "\n".join(lines)
