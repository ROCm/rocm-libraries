# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""RDNA WMMA attention tuning rules and the per-arch WMMA spaces.

The axes are data in :mod:`.axes` (``tuning_axes(arch, "wmma")`` for the
prefill, ``tuning_axes(arch, "wmma_decode")`` for the split-KV decode). This
module owns what is specific to those kernels: which axes a policy spec reads
and the launch checks a knob can break. :class:`WmmaSpace` states those rules
as the hooks of the shared :class:`rocke.dispatch.tuning.KnobSpace`.

The policy specs (``gfx1151._prefill_spec`` / ``gfx1151._decode_spec``) carry
every knob explicitly -- the prefill's derived ``block_m`` / ``wave_pair``, the
decode's per-family resolution -- so every in-scope knob is ``recorded``: a
``tuning_id`` names the explicit setting of each knob its problem reads, and
the same knobs name the same kernel on every problem that reads them.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields as _dataclass_fields, replace
from functools import lru_cache
from typing import Any, Optional, Tuple

from kernels.gfx1151.wmma_fmha_decode import (
    WmmaFmhaDecodeSpec,
    is_valid_spec as _decode_is_valid,
    wmma_fmha_decode_grid,
    wmma_fmha_decode_pack_groups,
    wmma_fmha_decode_row_tiles,
)
from kernels.gfx1151.wmma_fmha_fwd_v2 import (
    WmmaFmhaFwdV2Spec,
    is_valid_spec as _prefill_is_valid,
    wmma_fmha_fwd_v2_grid,
)
from rocke.dispatch.tuning import KnobSpace

from .axes import KnobAxis, axis_knob_names, tuning_axes
from .common import AttentionTuningSpec

WMMA_PATH = "wmma"  # the multi-wave prefill
WMMA_DECODE_PATH = "wmma_decode"  # the split-KV decode (+ merge kernel)

_SPEC_TYPES = {WMMA_PATH: WmmaFmhaFwdV2Spec, WMMA_DECODE_PATH: WmmaFmhaDecodeSpec}
_BUILDER_KINDS = {WMMA_PATH: "wmma_fmha_fwd_v2", WMMA_DECODE_PATH: "wmma_fmha_decode"}

# Which policy specs read each decode knob: the FP8 conversion choice exists
# only for FP8 K/V on gfx12 (gfx11 always decodes in integer ops), the tanh
# form only under softcap. Every prefill knob is read by every prefill spec.
_DECODE_SCOPE = {
    "fp8_cvt": lambda s: bool(s.kv_dtype) and s.wmma_family == "gfx12",
    "tanh_fast": lambda s: s.softcap,
}


@dataclass(frozen=True)
class WmmaProblem:
    """What a WMMA space builds from for one request: the policy's spec and the
    launch dimensions a knob's legality depends on."""

    spec: Any  # WmmaFmhaFwdV2Spec | WmmaFmhaDecodeSpec
    batch: int
    seqlen_q: int
    seqlen_k: int
    num_query_heads: int
    num_kv_heads: int


def wmma_launch_error(spec, problem: WmmaProblem) -> Optional[str]:
    """Why ``spec`` cannot launch ``problem``, or None."""
    try:
        if isinstance(spec, WmmaFmhaDecodeSpec):
            heads = dict(
                num_query_heads=problem.num_query_heads,
                num_kv_heads=problem.num_kv_heads,
                num_v_heads=problem.num_kv_heads,
            )
            wmma_fmha_decode_grid(
                spec,
                batch=problem.batch,
                pack_groups=wmma_fmha_decode_pack_groups(**heads),
                row_tiles=wmma_fmha_decode_row_tiles(
                    seqlen_q=problem.seqlen_q, **heads
                ),
                num_splits=1,
            )
        else:
            wmma_fmha_fwd_v2_grid(
                spec,
                batch=problem.batch,
                seqlen_q=problem.seqlen_q,
                num_query_heads=problem.num_query_heads,
            )
    except ValueError as e:
        return str(e)
    return None


@lru_cache(maxsize=256)
def _wmma_axes_for(spec, arch: str, path: str) -> Tuple[KnobAxis, ...]:
    return tuple(
        axis
        for axis in tuning_axes(arch, path)
        if _DECODE_SCOPE.get(axis.name, lambda s: True)(spec)
    )


@lru_cache(maxsize=None)
def _wmma_knob_defaults(arch: str, path: str) -> Tuple[Tuple[str, object], ...]:
    """The dataclass defaults of every knob ``arch`` declares on ``path``: what
    a knob-free spec is a delta against, the same for every problem."""
    knobs = axis_knob_names(tuning_axes(arch, path))
    return tuple(
        sorted(
            (f.name, f.default)
            for f in _dataclass_fields(_SPEC_TYPES[path])
            if f.name in knobs
        )
    )


@dataclass(frozen=True)
class WmmaSpace(KnobSpace):
    """One RDNA arch's prefill (``path="wmma"``) or decode
    (``path="wmma_decode"``) space. ``base`` is a :class:`WmmaProblem`."""

    def axes(self, base: WmmaProblem):
        return _wmma_axes_for(base.spec, self.arch, self.path)

    def known_knobs(self, base):
        return axis_knob_names(tuning_axes(self.arch, self.path))

    def recorded(self, base: WmmaProblem):
        return axis_knob_names(self.axes(base))

    def base_value(self, base: WmmaProblem, kernel, name: str):
        return getattr(base.spec, name, MISSING)

    def defaults(self, base, kernel):
        return dict(_wmma_knob_defaults(self.arch, self.path))

    def build(self, base: WmmaProblem, knobs):
        return replace(base.spec, **knobs)

    def validate(self, base: WmmaProblem, kernel):
        is_valid = (
            _decode_is_valid if self.path == WMMA_DECODE_PATH else _prefill_is_valid
        )
        ok, why = is_valid(kernel, arch=self.arch)
        if not ok:
            return False, why
        why = wmma_launch_error(kernel, base)
        return (False, why) if why else (True, "ok")

    def wrap(self, base, kernel, knobs, key, tid):
        return AttentionTuningSpec(
            path=self.path,
            arch=self.arch,
            builder_kind=_BUILDER_KINDS[self.path],
            compile_backend="llvm",
            candidate_name=self.candidate_name,
            tuning_id=tid,
            kernel_spec=kernel,
            variant_id=self.variant_id,
            config_key=key,
            knobs=knobs,
        )


def wmma_variants(space: WmmaSpace, problem: WmmaProblem) -> Tuple[Any, ...]:
    """The production kernel specs of ``space`` for ``problem``, default first."""
    return tuple(s.kernel_spec for s in space.stream(problem, "production"))
