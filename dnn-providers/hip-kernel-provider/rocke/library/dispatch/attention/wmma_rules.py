# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""RDNA WMMA forward-kernel tuning rules and the per-arch WMMA space.

The axes are data in :mod:`.axes` (``tuning_axes(arch, "wmma")``). This module
owns what is specific to the WMMA forward body: which axes a policy spec reads
(the standard body's V staging, causal tile skip, output-column tile and
machine scheduler; the transposed-QK body's key tile and wave count), and the
launch checks a knob can break. :class:`WmmaSpace` states those rules as the
hooks of the shared :class:`rocke.dispatch.tuning.KnobSpace`.

Every knob's default is resolved per problem by the selection policy
(``gfx1151._wmma_fwd_spec``), so every in-scope knob is ``recorded``: a
``tuning_id`` names the explicit setting of each knob its problem reads, and
the same knobs name the same kernel on every problem that reads them.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields as _dataclass_fields, replace
from functools import lru_cache
from itertools import product
from typing import Iterable, Optional, Tuple

from kernels.gfx1151.wmma_fmha_fwd import (
    WmmaFmhaFwdSpec,
    is_valid_spec,
    wmma_fmha_fwd_grid,
)
from rocke.dispatch.tuning import KnobSpace
from rocke.dispatch.tuning.walk import one_knob_at_a_time

from .axes import KnobAxis, axis_knob_names, tuning_axes
from .common import AttentionTuningSpec

WMMA_PATH = "wmma"

# Which policy specs read each knob. The transposed-QK body has its own fixed
# schedule: it never stages V, tiles output columns or skips causal tiles (it
# bounds the loop at the diagonal by construction), and the policy leaves its
# scheduler at the default; the standard body runs one 32-key wave.
_SCOPE = {
    "v_lds_stage": lambda s: not s.transposed_qk,
    "causal_tile_skip": lambda s: s.mask_mode == "causal" and not s.transposed_qk,
    "value_tile_size": lambda s: s.layout == "dense"
    and s.v_dim >= 128
    and not s.transposed_qk,
    "scheduler_strategy": lambda s: not s.transposed_qk,
    "block_n": lambda s: s.transposed_qk,
    "num_waves": lambda s: s.transposed_qk,
}
# The transposed body's key tile and wave count trade against each other, so
# production crosses them instead of offering each on its own.
_PRODUCTION_CROSSED = ("block_n", "num_waves")


@dataclass(frozen=True)
class WmmaProblem:
    """What a WMMA space builds from for one request: the policy's spec and the
    launch dimensions a knob's legality depends on."""

    spec: WmmaFmhaFwdSpec
    seqlen_q: int
    seqlen_k: int
    num_query_heads: int
    batch: int


def wmma_launch_error(spec: WmmaFmhaFwdSpec, problem: WmmaProblem) -> Optional[str]:
    """Why ``spec`` cannot launch ``problem``, or None."""
    if spec.transposed_qk and problem.seqlen_k % spec.block_n:
        return (
            f"transposed QK block_n={spec.block_n} must divide seqlen_k "
            f"{problem.seqlen_k}"
        )
    try:
        wmma_fmha_fwd_grid(
            spec,
            seqlen_q=problem.seqlen_q,
            num_query_heads=problem.num_query_heads,
            batch=problem.batch,
        )
    except ValueError as e:
        return str(e)
    return None


@lru_cache(maxsize=256)
def _wmma_axes_for(spec: WmmaFmhaFwdSpec, arch: str) -> Tuple[KnobAxis, ...]:
    return tuple(
        axis for axis in tuning_axes(arch, WMMA_PATH) if _SCOPE[axis.name](spec)
    )


@lru_cache(maxsize=None)
def _wmma_knob_defaults(arch: str) -> Tuple[Tuple[str, object], ...]:
    """The dataclass defaults of every knob ``arch`` declares: what a knob-free
    WMMA spec is a delta against, the same for every problem."""
    knobs = axis_knob_names(tuning_axes(arch, WMMA_PATH))
    return tuple(
        sorted(
            (f.name, f.default)
            for f in _dataclass_fields(WmmaFmhaFwdSpec)
            if f.name in knobs
        )
    )


@dataclass(frozen=True)
class WmmaSpace(KnobSpace):
    """One RDNA arch's WMMA forward space. ``base`` is a :class:`WmmaProblem`."""

    def axes(self, base: WmmaProblem):
        return _wmma_axes_for(base.spec, self.arch)

    def known_knobs(self, base):
        return axis_knob_names(tuning_axes(self.arch, WMMA_PATH))

    def recorded(self, base: WmmaProblem):
        return axis_knob_names(self.axes(base))

    def base_value(self, base: WmmaProblem, kernel, name: str):
        return getattr(base.spec, name, MISSING)

    def defaults(self, base, kernel):
        return dict(_wmma_knob_defaults(self.arch))

    def build(self, base: WmmaProblem, knobs):
        return replace(base.spec, **knobs)

    def validate(self, base: WmmaProblem, kernel):
        ok, why = is_valid_spec(kernel, arch=self.arch)
        if not ok:
            return False, why
        why = wmma_launch_error(kernel, base)
        return (False, why) if why else (True, "ok")

    def production(self, base, axes, is_valid) -> Iterable[dict]:
        crossed = [a for a in axes if a.name in _PRODUCTION_CROSSED]
        rest = tuple(a for a in axes if a.name not in _PRODUCTION_CROSSED)
        for choices in product(*(a.choices for a in crossed)):
            point = {k: v for choice in choices for k, v in choice}
            if point and not is_valid(point):
                continue
            for knobs in one_knob_at_a_time(
                rest, lambda k, point=point: is_valid({**point, **k})
            ):
                yield {**point, **knobs}

    def wrap(self, base, kernel, knobs, key, tid):
        return AttentionTuningSpec(
            path=WMMA_PATH,
            arch=self.arch,
            builder_kind="wmma_fmha_fwd",
            compile_backend="llvm",
            candidate_name=self.candidate_name,
            tuning_id=tid,
            kernel_spec=kernel,
            variant_id=self.variant_id,
            config_key=key,
            knobs=knobs,
        )


def wmma_variants(
    space: WmmaSpace, problem: WmmaProblem
) -> Tuple[WmmaFmhaFwdSpec, ...]:
    """The production kernel specs of ``space`` for ``problem``, default first."""
    return tuple(s.kernel_spec for s in space.stream(problem, "production"))
