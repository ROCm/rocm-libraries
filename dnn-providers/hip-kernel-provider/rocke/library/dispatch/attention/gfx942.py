# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 attention candidates (CDNA3, wave64).

Two families live here and they do not share an MFMA atom: the unified
``dense_pipe`` flash path runs on the narrow 16x16x16 atom, while the standalone
``attention_dense`` prefill kernel runs on 32x32x8 (CDNA3 has no 32x32x16 fp16/bf16
atom, so it doubles the K loop). See ``builders/gfx942/attention/prefill/README.md``
for why the dense kernel is a per-gfx module rather than an arch branch in the
gfx950 body.
"""

from __future__ import annotations

from typing import Tuple

from kernels.common.attention_unified import supports_native_unified_attention
from rocke.dispatch.core import (
    Capability,
    CandidateRegistry,
    KernelCandidate,
    OperatorRequest,
    ShapeRange,
)

from .common import (
    ATTENTION_ABI_VERSION,
    ATTENTION_FEATURES,
    UNIFIED_BLOCK_SIZES,
    UNIFIED_HEAD_SIZES,
    AttentionRequest,
    AttentionSpec,
    FAMILY,
    _problem,
    _request_errors,
    _selector_matches,
)

# block_n (KV tile) the dense candidate ships; 64 is the resource-efficient peak
# (see AttentionDenseSpec.block_n).
_DENSE_BLOCK_N = 64

# Persistent-grid CTA count for gfx942 (AICK-1664 P4). The shared
# ``AttentionRequest.dense_num_persistent`` default is 256, which is a gfx950 CU
# count; gfx942's largest part has 304. Applied only when the caller left the
# field at that shared default, so an explicit request value is still respected.
_GFX942_NUM_PERSISTENT = 304
_SHARED_NUM_PERSISTENT_DEFAULT = 256


def _make_gfx942_dense_pipe_candidate() -> KernelCandidate:
    """Fast gfx942 fp16 prefill kernel — transposed-x8 flash with ring-sliced K.

    Registered at priority 5 so it outranks the generic unified_2d candidate
    (priority 10) whenever both would match the same gfx942 fp16 2D problem.
    The registry sorts ascending (lower = higher precedence).
    Callers can also force this path explicitly via algorithm="dense_pipe".
    """
    spec_id = "gfx942_dense_pipe"
    name = "attention_gfx942_dense_pipe"

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, AttentionRequest)
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        problem = _problem(req)
        ok, why = supports_native_unified_attention(problem)
        if not ok:
            return False, why
        if problem.select_path() != "2d":
            return False, "problem routes to 3D, not 2D"
        from kernels.common.attention_unified import _enable_gfx942_fp16_flash

        if not _enable_gfx942_fp16_flash(problem):
            return False, "gfx942 fp16 flash not eligible for this shape"
        return True, "ok"

    def select(req: OperatorRequest) -> AttentionSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, AttentionRequest)
        problem = _problem(req)
        return AttentionSpec(
            path="2d",
            head_size=problem.head_size,
            block_size=problem.block_size,
            dtype=problem.dtype,
            num_query_heads=problem.num_query_heads,
            num_kv_heads=problem.num_kv_heads,
            name="rocke_attention_gfx942_dense_pipe",
        )

    candidate = KernelCandidate(
        name=name,
        family=FAMILY,
        algorithm="dense_pipe",
        spec_id=spec_id,
        abi_version=ATTENTION_ABI_VERSION,
        priority=5,
        capability=Capability(
            arches=("gfx942",),
            dtypes=("fp16",),
            shapes=(
                ShapeRange("hdim_q", allowed=UNIFIED_HEAD_SIZES),
                ShapeRange("kv_block_size", allowed=UNIFIED_BLOCK_SIZES),
            ),
            # ``_enable_gfx942_fp16_flash`` is the real narrowing; nothing here
            # claims a feature it turns down, so the full set stays declared.
            supports_features=ATTENTION_FEATURES,
        ),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=lambda spec, req: (0, 0, 0),
        block=lambda spec: (0, 0, 0),
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
    )
    return candidate


def _dense_spec(req: OperatorRequest):
    """Build the gfx942 ``AttentionDenseSpec`` for ``req`` at its best config.

    The gfx942 twin of :func:`dispatch.attention.gfx950._dense_spec`. Same shape
    logic -- persistent ("auto") turns on the grid-stride variant once there is
    enough work to fill the persistent grid (``nqb*Hq*B >= num_persistent``, the
    large-Sq prefill regime), and non-tile-multiple self-attention lengths take the
    on-chip ragged path (no host pad) -- but a different tuning, which is the whole
    reason the two are separate functions rather than one with an arch branch:

    * ``num_persistent`` defaults to :data:`_GFX942_NUM_PERSISTENT` (304 CUs) rather
      than the shared 256. It drives BOTH the auto persistent decision and the grid
      size, so it is resolved before the mode branch.
    * ``waves_per_eu`` and ``d64_kpad`` come from the kernel's own per-config policy
      (``_p0_waves_per_eu`` / ``_p0_d64_kpad``), so the ``kernel_name`` tags and the
      emitted binary cannot disagree. The kernel's module-level ``_P0_D64_KPAD``
      default stays False -- this spec is what flips the shipped D64 path on, which
      keeps a non-dispatch ``build_attention_dense`` byte-identical.

    The spec dataclass itself is REUSED from the gfx950 kernel module (as the gfx942
    kernel body reuses it); it is arch-neutral, only its tuned values differ.
    """
    from kernels.gfx942.attention_dense import _p0_d64_kpad, _p0_waves_per_eu
    from kernels.gfx950.attention_dense import AttentionDenseSpec, _BLOCK_M

    assert isinstance(req, AttentionRequest)
    sq, sk = int(req.seqlen_q), int(req.seqlen_k)
    bn = _DENSE_BLOCK_N
    head_size = int(req.hdim_q)
    dtype = req.dtype.lower()
    # on-chip ragged padding for ragged self-attention lengths (seqlen_q==seqlen_kv,
    # not a 256/block_n multiple). Cross-attention ragged is left to the validator.
    ragged = (sq == sk) and ((sq % _BLOCK_M != 0) or (sk % bn != 0))
    nqb = (sq + _BLOCK_M - 1) // _BLOCK_M
    work = nqb * int(req.nhead_q) * int(req.batch)
    np = int(req.dense_num_persistent)
    if np == _SHARED_NUM_PERSISTENT_DEFAULT:
        np = _GFX942_NUM_PERSISTENT
    mode = req.dense_persistent.strip().lower()
    if mode == "on":
        persistent = True
    elif mode == "off":
        persistent = False
    elif mode == "auto":
        persistent = work >= np  # enough work to fill the persistent grid
    else:
        raise ValueError(
            f"dense_persistent must be 'auto'/'on'/'off', got {req.dense_persistent!r}"
        )
    return AttentionDenseSpec(
        batch=int(req.batch),
        seqlen_q=sq,
        seqlen_kv=sk,
        num_query_heads=int(req.nhead_q),
        num_kv_heads=int(req.nhead_k),
        head_size=head_size,
        causal=(int(req.mask_type) != 0),
        dtype=dtype,
        block_n=bn,
        persistent=persistent,
        num_persistent=np,
        persist_decode=req.dense_persist_decode.strip().lower(),
        ragged=ragged,
        waves_per_eu=_p0_waves_per_eu(head_size, dtype),
        d64_kpad=_p0_d64_kpad(head_size),
    )


def dense_spec_for_request(req: AttentionRequest):
    """Public builder: the launch-ready gfx942 ``AttentionDenseSpec`` for ``req``
    at its best config (see :func:`_dense_spec`). Pair with
    ``run_attention_dense_torch`` to execute the dispatched dense candidate.

    Import it from THIS module, not from the ``dispatch.attention`` package: the
    package-level re-export of that name is gfx950's, and it would silently hand a
    gfx942 request the untuned spec (256 CTAs, no kpad, no waves-per-eu bump).
    """
    return _dense_spec(req)


def _make_gfx942_attention_dense_candidate() -> KernelCandidate:
    """Dense flash-attn prefill on gfx942 (bf16/fp16, causal/full) — AICK-1664.

    OPT-IN ONLY (mirrors the gfx950 sibling): matches solely when the request names
    ``algorithm="attention_dense"`` / ``spec_id="gfx942_attention_dense"``, so it
    never auto-overrides the generic unified_2d / dense_pipe paths. That check
    cannot move into ``capability`` -- it constrains the request's selector, not its
    shape -- and at priority 3 it is the only thing keeping this candidate off the
    default path.

    Carries the AICK-1664 P1-P5 levers: the 32x32x8 atom with K-loop doubling,
    conflict-free V (D128 fp16), exp2_fast + fused softmax rescale, per-config
    waves-per-eu and the D64 K-bank-conflict pad, and the persistent grid-stride
    variant (``dense_persistent='auto'`` turns it on once there is enough work to
    fill the grid). Which config gets which lever, and why, is the table in
    ``builders/gfx942/attention/prefill/README.md``.

    Scope is delegated entirely to ``supports_attention_dense``, which rejects every
    spec the builder cannot emit (varlen / ragged / sliding-window are later
    follow-ups; plus block_n, LDS-budget and 32-bit-extent limits). That keeps
    ``admits`` and ``build`` in agreement, so an out-of-scope request falls through
    to another candidate instead of being selected and then failing to build.
    """
    spec_id = "gfx942_attention_dense"
    name = "attention_gfx942_dense"

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, AttentionRequest)
        # Opt-in: never selected under algorithm/spec_id "auto".
        if req.algorithm.strip().lower() != "attention_dense" and (
            req.spec_id.strip().lower() != spec_id
        ):
            return False, "attention_dense is opt-in (algorithm='attention_dense')"
        from kernels.gfx942.attention_dense import supports_attention_dense

        try:
            spec = _dense_spec(req)
        except ValueError as e:
            return False, str(e)
        ok, why = supports_attention_dense(spec, arch=req.arch)
        if not ok:
            return False, why
        return True, "ok"

    def select(req: OperatorRequest) -> AttentionSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, AttentionRequest)
        from kernels.gfx942.attention_dense import p0_kernel_name

        problem = _problem(req)
        dense_spec = _dense_spec(req)
        return AttentionSpec(
            path="2d",
            head_size=problem.head_size,
            block_size=problem.block_size,
            dtype=problem.dtype,
            num_query_heads=problem.num_query_heads,
            num_kv_heads=problem.num_kv_heads,
            name="rocke_attention_dense_gfx942",
            # batch-unique: the kernel bakes batch into its buffer extents, which
            # the shared AttentionDenseSpec.kernel_name() omits -- two batches would
            # otherwise collide in the name cache. p0_kernel_name adds it, plus the
            # wpe/kpad tags for the tuning folded in above.
            kernel_name_override=p0_kernel_name(dense_spec),
        )

    candidate = KernelCandidate(
        name=name,
        family=FAMILY,
        algorithm="attention_dense",
        spec_id=spec_id,
        abi_version=ATTENTION_ABI_VERSION,
        priority=3,
        capability=Capability(
            arches=("gfx942",),
            dtypes=("bf16", "fp16"),
            # Dense: no sliding-window, no sinks. Causal is a mask, not a feature
            # this path turns down. Head size stays out -- D64/D128 coverage is
            # ``supports_attention_dense``'s call, and it reads the built spec
            # (LDS budget, block_n divisibility), which a ShapeRange cannot.
            supports_features=frozenset({"causal"}),
        ),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=lambda spec, req: (0, 0, 0),
        block=lambda spec: (0, 0, 0),
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
    )
    return candidate


def register(registry: CandidateRegistry) -> None:
    registry.register(_make_gfx942_attention_dense_candidate())
    registry.register(_make_gfx942_dense_pipe_candidate())
