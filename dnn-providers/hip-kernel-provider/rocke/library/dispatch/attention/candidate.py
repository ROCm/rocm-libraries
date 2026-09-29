# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Attention's tuned candidates, built on :func:`rocke.dispatch.tuning.make_tuned_candidate`.

:func:`make_tuning_candidate` builds one per unified geometry variant and
:func:`make_dense_candidate` one per dense variant. Each supplies its space,
the per-request base the space builds from, and attention's signature, grid,
and Torch binding; admission, pin resolution and sweeping are shared.
"""

from __future__ import annotations

from typing import Callable

from kernels.common.attention_unified import supports_native_unified_attention
from rocke.dispatch.core import Capability, KernelCandidate, ShapeRange
from rocke.dispatch.tuning import make_tuned_candidate

from .common import (
    ATTENTION_ABI_VERSION,
    ATTENTION_FEATURES,
    FAMILY,
    UNIFIED_BLOCK_SIZES,
    UNIFIED_DTYPES,
    UNIFIED_HEAD_SIZES,
    AttentionRequest,
    _dense_kernel_module,
    _problem,
    _request_errors,
)
from .dense_rules import DenseSpace, DenseSupports
from .unified_rules import TUNING_ALGORITHM, AttentionGeometryVariant, unified_space

DENSE_ALGORITHM = "attention_dense"


def make_tuning_candidate(variant: AttentionGeometryVariant) -> KernelCandidate:
    def build(spec, arch):
        from .tuning_specs import validate_explicit_fp8_encoding

        validate_explicit_fp8_encoding(
            arch=arch,
            use_fp8=spec.kernel_spec.kv_storage_dtype == "fp8e4m3",
            fp8_fnuz=spec.fp8_fnuz,
        )
        return spec.build(arch)

    def signature(spec):
        from kernels.common.attention_unified import _3d_signature, _attn_signature

        if spec.path == "3d":
            return _3d_signature(
                spec.kernel_spec.dtype, kv_dtype=spec.kernel_spec.kv_storage_dtype
            )
        return _attn_signature(
            spec.kernel_spec.dtype,
            include_bt_stride=True,
            include_qq_bias_stride=True,
            kv_dtype=spec.kernel_spec.kv_storage_dtype,
        )

    def bind_torch(request, spec, tensors, **kwargs):
        from .bindings import bind_tuning_attention_torch

        payload = dict(tensors)
        if "problem" not in payload:
            payload["problem"] = _problem(request)
        return bind_tuning_attention_torch(request, spec, payload, **kwargs)

    return make_tuned_candidate(
        name=variant.candidate_name,
        family=FAMILY,
        algorithm=TUNING_ALGORITHM,
        spec_id=variant.spec_id,
        abi_version=ATTENTION_ABI_VERSION,
        priority=30,
        capability=Capability(
            arches=(variant.arch,),
            dtypes=UNIFIED_DTYPES,
            shapes=(
                ShapeRange("hdim_q", allowed=UNIFIED_HEAD_SIZES),
                ShapeRange("kv_block_size", allowed=UNIFIED_BLOCK_SIZES),
            ),
            supports_features=ATTENTION_FEATURES,
        ),
        space=unified_space(variant),
        base=_problem,
        request_errors=_request_errors,
        precheck=lambda req: supports_native_unified_attention(
            _problem(req), arch=variant.arch
        ),
        signature=signature,
        build=build,
        bind_torch=bind_torch,
        grid=lambda spec, req: spec.launch_grid(_problem(req)),
    )


def make_dense_candidate(
    *,
    arch: str,
    name: str,
    spec_id: str,
    variant_id: str,
    base_spec: Callable[[AttentionRequest], object],
    supports: DenseSupports,
    features: frozenset,
    recorded: frozenset = frozenset(),
) -> KernelCandidate:
    """One dense variant. ``base_spec(req)`` is its default kernel spec (raise
    ``ValueError`` when the request cannot have one); ``supports`` is the
    kernel's own validator; ``features`` is the capability's feature set;
    ``recorded`` names the fields ``base_spec`` resolves per problem (see
    :meth:`rocke.dispatch.tuning.KnobSpace.recorded`)."""

    def bind_torch(request, spec, tensors, **kwargs):
        from .bindings import bind_dense_attention_torch

        return bind_dense_attention_torch(request, spec, tensors, **kwargs)

    return make_tuned_candidate(
        name=name,
        family=FAMILY,
        algorithm=DENSE_ALGORITHM,
        spec_id=spec_id,
        abi_version=ATTENTION_ABI_VERSION,
        priority=3,
        capability=Capability(
            arches=(arch,), dtypes=("bf16", "fp16"), supports_features=features
        ),
        space=DenseSpace(
            abi=ATTENTION_ABI_VERSION,
            arch=arch,
            path="dense",
            variant_id=variant_id,
            candidate_name=name,
            supports=supports,
            recorded_fields=frozenset(recorded),
        ),
        base=base_spec,
        request_errors=_request_errors,
        signature=lambda spec: _dense_kernel_module(arch).attention_dense_signature(
            spec.kernel_spec
        ),
        build=lambda spec, arch_: spec.build(arch_),
        bind_torch=bind_torch,
        grid=lambda spec, req: spec.launch_grid(),
    )
