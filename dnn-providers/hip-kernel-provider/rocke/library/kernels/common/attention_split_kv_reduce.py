# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The split-KV reduce spec that pairs with a tiled 3D segment spec.

One home for which segment fields the reduce kernel reads. The dispatcher
(``dispatch.attention.tuning_specs.make_explicit_attention_3d_specs``) builds its
launched pair through :func:`split_kv_reduce_spec`, and the gfx950 module export
(``kernels.gfx950.attention_tiled_3d.reduce_spec_for``) derives the reduce it packs
through it too, so the pairing cannot drift between them.
"""

from __future__ import annotations


def split_kv_reduce_spec(reduce_type, segment, *, emit_lse: bool = False):
    """``reduce_type`` built from segment spec ``segment``.

    Every reduce field but ``emit_lse`` is a segment field. ``emit_lse`` is the
    problem's (the output the caller asked for), not the segment's, so the caller
    passes it. It is forwarded unless it is ``False``, so a reduce type without the
    field builds as before when LSE is off, and the reduce type still validates any
    other value.
    """
    return reduce_type(
        head_size=segment.head_size,
        num_query_heads=segment.num_query_heads,
        num_kv_heads=segment.num_kv_heads,
        dtype=segment.dtype,
        num_segments=segment.num_segments,
        waves_per_eu=segment.waves_per_eu,
        **({} if emit_lse is False else {"emit_lse": emit_lse}),
    )
