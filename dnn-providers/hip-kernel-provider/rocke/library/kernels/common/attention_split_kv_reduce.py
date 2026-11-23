# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The split-KV reduce spec that pairs with a tiled 3D segment spec, and its LDS size.

One home for which segment fields the reduce kernel reads. The dispatcher
(``dispatch.attention.tuning_specs.make_explicit_attention_3d_specs``) builds its
launched pair through :func:`split_kv_reduce_spec`, and the gfx950 module export
(``kernels.gfx950.attention_tiled_3d.reduce_spec_for``) derives the reduce it packs
through it too, so the pairing cannot drift between them. Both LDS gates (the
strided-KV dispatcher's and :func:`supports_split_kv_reduce`) size the reduce through
:func:`split_kv_reduce_lds_bytes`.

:func:`supports_split_kv_reduce` is the reduce kernel's one pack-time predicate. It
leaves ``spec`` unannotated because each arch has its own reduce spec class, and
``hkp_pack`` only accepts a predicate whose first parameter is annotated with the
builder's spec class. So an arch cannot re-export it as-is (``hkp_pack`` would refuse
the reduce builder at pack time); it must wrap it in a predicate typed with its own
spec, under the name its builder's predicate alias expects (gfx950 and gfx942 each
wrap it as ``supports_tiled_reduce_spec``).
"""

from __future__ import annotations

from typing import Tuple

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


def reduce_max_segments(arch: str) -> int:
    """The most segments whose factor cache fits ``arch``'s LDS
    (``ArchTarget.lds_capacity_bytes``). The C++ host mirrors gfx950's as
    ``GFX950_UNIFIED_3D_MAX_SEGMENTS``."""
    from rocke.core.arch.target import ArchTarget

    return split_kv_reduce_max_segments(ArchTarget.from_gfx(arch).lds_capacity_bytes)


# The reduce builder assumes one wave64 owns ``head_size // 64`` output dims per thread.
_REDUCE_THREADS = 64


def supports_split_kv_reduce(spec, *, arch: str) -> Tuple[bool, str]:
    """Whether the split-KV reduce kernel can be built for reduce spec ``spec`` on ``arch``.

    The reduce kernel has no dispatcher gate of its own: it is only ever built beside the
    3D segment kernel, whose ``supports_tiled_3d`` decides routing. This predicate lets
    ``hkp_pack`` validate it as a companion at pack time, and states what the builder
    otherwise assumes: ``head_size`` splits evenly across the 64 lanes, the heads group
    evenly, and the segment-factor cache fits in LDS. The kernel is one 64-thread CTA
    whose ``wave64_reduce_max/sum`` butterfly assumes a single wave64, so ``arch`` is
    gated like the segment kernel's: ``validate_tiled_attention_arch`` admits only the
    wave64 gfx950 and gfx942 targets and refuses wave32 RDNA.

    Not a predicate ``hkp_pack`` can resolve on its own: ``spec`` is unannotated, so
    each arch wraps this in a predicate whose ``spec`` is typed with its reduce spec
    class (see the module docstring).
    """
    from rocke.core.arch.target import ArchTarget

    from ._fmha_common import check_head_counts
    from .attention_arch import validate_tiled_attention_arch

    arch_ok, arch_reason = validate_tiled_attention_arch(arch)
    if not arch_ok:
        return False, f"tiled reduce kernel: {arch_reason}"
    if spec.dtype not in ("fp16", "bf16"):
        return (
            False,
            f"tiled reduce kernel currently supports fp16/bf16 (got {spec.dtype!r})",
        )
    if spec.head_size <= 0 or spec.head_size % _REDUCE_THREADS != 0:
        return (
            False,
            "tiled reduce kernel needs head_size divisible by "
            f"{_REDUCE_THREADS} (got {spec.head_size})",
        )
    ok, reason = check_head_counts(spec.num_query_heads, spec.num_kv_heads)
    if not ok:
        return False, f"tiled reduce kernel: {reason}"
    # ``arch`` passed validate_tiled_attention_arch, so it has a catalog row.
    target = ArchTarget.from_gfx(arch)
    lds_bytes = split_kv_reduce_lds_bytes(spec.num_segments)
    if not target.fits_lds(lds_bytes):
        return (
            False,
            f"tiled reduce kernel caches one f32 LDS factor per segment; "
            f"num_segments={spec.num_segments} needs {lds_bytes} B, over {arch}'s "
            f"{target.lds_capacity_bytes} B LDS (at most {reduce_max_segments(arch)})",
        )
    return True, "supported"


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
