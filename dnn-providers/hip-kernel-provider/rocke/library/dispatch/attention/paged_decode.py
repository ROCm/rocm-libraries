# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Executable token-mapped decode; existing page sizes retain default routing.

Binding checks tensor structure without reading device metadata by default.
Callers maintain in-bounds lengths, query offsets and referenced page IDs, and
may mutate valid values in place between launches. Pass
``unsafe_skip_paged_value_validation=False`` for synchronized content validation
at bind time; rebind after mutation when relying on that validation.
"""

from dataclasses import replace

from kernels.common.attention_unified import _3d_signature, _num_segments
from rocke.dispatch.core import Capability, KernelCandidate, ShapeRange

from .bindings import bind_tuning_attention_torch
from .common import (
    FAMILY,
    AttentionMaskType,
    AttentionTuningSpec,
    _parse_attention_mask_type,
    _problem,
    _request_errors,
    _selector_matches,
)
from .tuning_specs import ExplicitAttention3DConfig, make_explicit_attention_3d_specs


def _validate_feature_inputs(values):
    if values.get("softcap", 0.0) or any(
        values.get(name) is not None for name in ("sinks", "alibi_slopes", "qq_bias")
    ):
        raise ValueError("paged decode binding does not support softcap, sinks or bias")


def make_candidate():
    def support(req):
        if errors := _request_errors(req):
            return False, "; ".join(errors)
        if req.kv_layout != "paged":
            return False, "requires paged KV"
        if (
            _parse_attention_mask_type(req.mask_type)
            == AttentionMaskType.TOP_LEFT_CAUSAL
        ):
            return False, "decode uses bottom-right causal alignment"
        if req.nhead_q // req.nhead_k not in (1, 2, 4, 8, 16):
            return False, "GQA ratio must divide 16"
        if req.kv_block_size != 1 and not (
            req.algorithm == "paged_decode_t32" or req.spec_id == "paged_decode_t32"
        ):
            return False, "token-mapped tile 32 is opt-in for existing page sizes"
        if req.tuning_id not in ("auto", "paged_t32"):
            return False, "unknown paged decode tuning_id"
        if req.tuning_knobs:
            return False, "paged decode does not accept tuning_knobs"
        return _selector_matches(req, candidate)

    def select(req):
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(why)
        problem = _problem(req)
        segment, reduce = make_explicit_attention_3d_specs(
            problem,
            ExplicitAttention3DConfig(
                num_segments=_num_segments(problem), tile_policy="32"
            ),
            arch=req.arch.lower(),
        )
        return AttentionTuningSpec(
            path="3d",
            arch=req.arch.lower(),
            builder_kind="tiled_3d",
            compile_backend="llvm",
            candidate_name=candidate.name,
            tuning_id="paged_t32",
            kernel_spec=segment,
            reduce_spec=reduce,
        )

    def bind(request, spec, tensors, **kwargs):
        ok, why = candidate.admits(request)
        if not ok:
            raise ValueError(why)
        if (
            spec.path != "3d"
            or spec.kernel_spec.tile_size != 32
            or spec.kernel_spec.kv_layout != "paged"
        ):
            raise ValueError("paged_decode_t32 requires a paged tile-32 segment spec")
        tensors = dict(tensors)
        tensors.setdefault("problem", _problem(request))
        if tensors["problem"].sliding_window != request.sliding_window:
            raise ValueError("request sliding_window disagrees with problem")
        problem = tensors["problem"]
        if (
            problem.softcap
            or problem.use_sinks
            or problem.use_alibi
            or problem.use_qq_bias
        ):
            raise ValueError(
                "paged decode binding does not support softcap, sinks or bias"
            )
        _validate_feature_inputs(tensors)
        _validate_feature_inputs(kwargs)
        kwargs.setdefault("unsafe_skip_paged_value_validation", True)
        kwargs.setdefault("grid", spec.launch_grid(tensors["problem"]))
        kwargs.setdefault("block", spec.launch_block())
        binding = bind_tuning_attention_torch(request, spec, tensors, **kwargs)

        def launch(**overrides):
            _validate_feature_inputs(overrides)
            return binding.launch(**overrides)

        return replace(binding, launch=launch)

    candidate = KernelCandidate(
        name="attention_paged_decode_t32",
        family=FAMILY,
        algorithm="paged_decode_t32",
        spec_id="paged_decode_t32",
        abi_version="rocke-attention-paged/v1",
        priority=20,
        capability=Capability(
            arches=("gfx942", "gfx950"),
            dtypes=("fp16", "bf16"),
            shapes=(
                ShapeRange("hdim_q", allowed=(64, 128, 256)),
                ShapeRange("seqlen_q", allowed=(1,)),
                ShapeRange("kv_block_size", allowed=(1, 16, 32, 64)),
            ),
            supports_features=frozenset(
                {"causal", "causal_bottom_right", "sliding_window"}
            ),
        ),
        _supports=support,
        select_spec=select,
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
        build=lambda spec, arch: spec.build(arch),
        signature=lambda spec: _3d_signature(spec.kernel_spec.dtype),
        grid=lambda spec, req: spec.launch_grid(_problem(req)),
        block=lambda spec: spec.launch_block(),
        bind_torch=bind,
    )
    return candidate


def register(route, execution) -> None:
    candidate = make_candidate()
    route.register(candidate)
    execution.register(candidate)
