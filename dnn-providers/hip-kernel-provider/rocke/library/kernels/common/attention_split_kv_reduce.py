# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The split-KV reduce spec that pairs with a tiled 3D segment spec, and its LDS size.

One home for which segment fields the reduce kernel reads. The dispatcher
(``dispatch.attention.tuning_specs.make_explicit_attention_3d_specs``) builds its
launched pair through :func:`split_kv_reduce_spec`, and the gfx950 module export
(``kernels.gfx950.attention_tiled_3d.reduce_spec_for``) derives the reduce it packs
through it too, so the pairing cannot drift between them. Both LDS gates (the
strided-KV dispatcher's and ``supports_tiled_reduce_spec``) size the reduce through
:func:`split_kv_reduce_lds_bytes`.
"""

from __future__ import annotations

# The reduce's only LDS allocation caches one f32 softmax factor per segment; the
# lowering aligns the pool to 16 bytes.
_LDS_BYTES_PER_SEGMENT = 4
_LDS_POOL_ALIGN = 16


def split_kv_reduce_lds_bytes(num_segments: int) -> int:
    """LDS the reduce allocates for ``num_segments`` segments."""
    size = num_segments * _LDS_BYTES_PER_SEGMENT
    return (size + _LDS_POOL_ALIGN - 1) & ~(_LDS_POOL_ALIGN - 1)


def split_kv_reduce_max_segments(lds_capacity_bytes: int) -> int:
    """The most segments whose reduce fits ``lds_capacity_bytes``."""
    return (lds_capacity_bytes & ~(_LDS_POOL_ALIGN - 1)) // _LDS_BYTES_PER_SEGMENT


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


# The reduce builder assumes one wave64 owns ``head_size // 64`` output dims per thread.
REDUCE_THREADS = 64


def tiled_reduce_unsupported_reason(spec, *, arch: str):
    """Why the tiled split-KV reduce kernel cannot be built for ``spec`` on ``arch``.

    Returns ``None`` when it can. The one copy of the rules both arches'
    ``supports_tiled_reduce_spec`` apply: ``arch`` must be a wave64 attention
    target (the kernel is one 64-thread CTA whose ``wave64_reduce_max/sum``
    butterfly assumes a single wave64), ``dtype`` fp16/bf16, ``head_size`` a
    positive multiple of 64, and the one f32 LDS factor per segment must fit
    ``arch``'s LDS.
    """
    from rocke.core.arch.target import ArchTarget

    from .attention_arch import validate_tiled_attention_arch

    arch_ok, arch_reason = validate_tiled_attention_arch(arch)
    if not arch_ok:
        return f"tiled reduce kernel: {arch_reason}"
    if spec.dtype not in ("fp16", "bf16"):
        return f"tiled reduce kernel currently supports fp16/bf16 (got {spec.dtype!r})"
    if spec.head_size <= 0 or spec.head_size % REDUCE_THREADS != 0:
        return (
            "tiled reduce kernel needs head_size divisible by "
            f"{REDUCE_THREADS} (got {spec.head_size})"
        )
    # ``arch`` passed validate_tiled_attention_arch, so it has a catalog row.
    target = ArchTarget.from_gfx(arch)
    lds_bytes = split_kv_reduce_lds_bytes(spec.num_segments)
    if not target.fits_lds(lds_bytes):
        return (
            f"tiled reduce kernel caches one f32 LDS factor per segment; "
            f"num_segments={spec.num_segments} needs {lds_bytes} B, over {arch}'s "
            f"{target.lds_capacity_bytes} B LDS (at most "
            f"{split_kv_reduce_max_segments(target.lds_capacity_bytes)})"
        )
    return None
