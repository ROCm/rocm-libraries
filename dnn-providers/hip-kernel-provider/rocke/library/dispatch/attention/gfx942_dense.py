# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 dense attention candidate (CDNA3, wave64).

The standalone ``attention_dense`` prefill kernel runs on the 32x32x8 atom (CDNA3
has no 32x32x16 fp16/bf16 atom, so it doubles the K loop), unlike the unified
``dense_pipe`` flash path on the narrow 16x16x16 atom, which lives in
:mod:`.gfx942_unified`. See ``builders/gfx942/attention/prefill/README.md`` for
why the dense kernel is a per-gfx module rather than an arch branch in the
gfx950 body.

One candidate, identified like every other tuned candidate: ``spec_id``
(``gfx942_dense``) names it and ``tuning_id`` names one configuration
from its knob space, geometry included (``auto`` is the default spec). It is
opt-in: only an explicit ``spec_id`` selects it. Admission is the kernel's own
``supports_attention_dense``.
"""

from __future__ import annotations

from rocke.dispatch.core import CandidateRegistry, KernelCandidate

from .candidate import make_dense_candidate
from .common import (
    AttentionMaskType,
    AttentionRequest,
    _parse_attention_mask_type,
)

CANDIDATE_NAME = "attention_gfx942_dense"
SPEC_ID = "gfx942_dense"
_VARIANT_ID = "dense"

# block_n (KV tile) the default spec ships; 64 is the resource-efficient peak
# (see AttentionDenseSpec.block_n). The knob space sweeps block_m / block_n.
_DENSE_BLOCK_N = 64
# Persistent-grid CTA count of the default spec: gfx942's largest part has 304
# CUs. The knob space sweeps it.
_GFX942_NUM_PERSISTENT = 304


def _base_spec(req: AttentionRequest):
    """The default ``Gfx942AttentionDenseSpec`` for ``req``.

    The gfx942 twin of ``gfx950_dense._base_spec``. The persistent grid turns on
    once there is enough work to fill it (``nqb*Hq*B >= num_persistent``), and
    non-tile-multiple self-attention lengths take the on-chip ragged path.
    ``waves_per_eu`` comes from the kernel's own per-config policy
    (``_tuned_waves_per_eu``): any value the kernel bakes into its
    ``kernel_name`` must be resolved by the kernel's policy, or the name tag and
    the compiled binary can disagree. Every gfx942-private codegen knob stays at
    the concrete spec's default; the knob space varies them.
    """
    from kernels.common.attention_dense_spec import DENSE_TILE_GEOMETRIES
    from kernels.gfx942.attention_dense import (
        Gfx942AttentionDenseSpec,
        _tuned_waves_per_eu,
    )

    if req.arch != "gfx942":
        raise ValueError(
            f"gfx942 dense spec factory requires arch='gfx942', got {req.arch!r}"
        )
    sq, sk = int(req.seqlen_q), int(req.seqlen_k)
    mask_type = _parse_attention_mask_type(req.mask_type)
    bm = int(DENSE_TILE_GEOMETRIES["default"]["block_m"])
    bn = _DENSE_BLOCK_N
    head_size = int(req.hdim_q)
    dtype = req.dtype.lower()
    # on-chip ragged padding for ragged self-attention lengths (seqlen_q==seqlen_kv,
    # not a 256/block_n multiple). Cross-attention ragged is left to the validator.
    ragged = (sq == sk) and ((sq % bm != 0) or (sk % bn != 0))
    nqb = (sq + bm - 1) // bm
    work = nqb * int(req.nhead_q) * int(req.batch)
    return Gfx942AttentionDenseSpec(
        batch=int(req.batch),
        seqlen_q=sq,
        seqlen_kv=sk,
        num_query_heads=int(req.nhead_q),
        num_kv_heads=int(req.nhead_k),
        head_size=head_size,
        causal=mask_type != AttentionMaskType.NO_MASK,
        dtype=dtype,
        block_m=bm,
        block_n=bn,
        persistent=work >= _GFX942_NUM_PERSISTENT,
        num_persistent=_GFX942_NUM_PERSISTENT,
        persist_decode="auto",
        ragged=ragged,
        sliding_window=int(req.sliding_window),
        waves_per_eu=_tuned_waves_per_eu(head_size, dtype),
    )


def _supports(spec, *, arch):
    from kernels.gfx942.attention_dense import supports_attention_dense

    return supports_attention_dense(spec, arch=arch)


def _make_gfx942_attention_dense_candidate() -> KernelCandidate:
    """Dense flash-attn prefill on gfx942 (bf16/fp16, causal/full).

    Carries the port's P1-P5 levers: the 32x32x8 atom with K-loop doubling,
    conflict-free V (D128 fp16), exp2_fast + fused softmax rescale, per-config
    waves-per-eu and the D64 K-bank-conflict pad, and the persistent grid-stride
    variant. Which config gets which lever, and why, is the table in
    ``builders/gfx942/attention/prefill/README.md``.

    When ``Gfx942AttentionDenseSpec.runtime_shape`` holds (not persistent and no
    sliding window), the body reads ``batch``, ``seqlen_q``, ``seqlen_kv``,
    ``num_query_heads``, and ``num_kv_heads`` from kernel params
    (``spec.runtime_param_fields``). ``signature`` is ``attention_dense_signature``,
    which appends those five i32s after ``scale``, and ``bind_torch`` launches
    through ``run_attention_dense_torch``, which packs them via
    ``attention_dense_runtime_args``. The persistent and sliding-window bodies
    declare no shape params and keep the shape in the kernel name.
    """

    return make_dense_candidate(
        arch="gfx942",
        name=CANDIDATE_NAME,
        spec_id=SPEC_ID,
        variant_id=_VARIANT_ID,
        base_spec=_base_spec,
        supports=_supports,
        # Dense: causal + sliding-window; no sinks or moving bottom-right
        # diagonal. Head size stays out -- D64/D128 coverage is
        # ``supports_attention_dense``'s call.
        features=frozenset({"causal", "sliding_window"}),
        # _base_spec picks these from the work size and dtype/head size.
        recorded=frozenset({"persistent", "waves_per_eu"}),
    )


def register(route: CandidateRegistry, execution: CandidateRegistry) -> None:
    candidate = _make_gfx942_attention_dense_candidate()
    route.register(candidate)
    execution.register(candidate)
