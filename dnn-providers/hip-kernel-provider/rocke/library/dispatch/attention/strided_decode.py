# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Non-paged input admission for the existing tiled split-KV decode kernels."""

from __future__ import annotations

import math

from kernels.common.attention_unified import (
    UnifiedAttentionProblem,
    _3d_signature,
    _strided_3d_specs_from_problem,
    _validate_strided_3d_spec,
)
from rocke.dispatch.core import Capability, KernelCandidate, ShapeRange, TorchBinding

from .common import (
    AttentionMaskType,
    AttentionTuningSpec,
    FAMILY,
    _parse_attention_mask_type,
    _problem,
    _request_errors,
    _selector_matches,
)


def make_candidate() -> KernelCandidate:
    def support(req):
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        if req.kv_layout != "strided":
            return False, "requires non-paged strided KV input"
        if (
            _parse_attention_mask_type(req.mask_type)
            == AttentionMaskType.TOP_LEFT_CAUSAL
        ):
            return False, "strided decode uses bottom-right causal alignment"
        if req.nhead_q // req.nhead_k not in (1, 2, 4, 8, 16):
            return False, "GQA ratio must divide the 16-row query tile"
        if req.tuning_id not in ("auto", "strided"):
            return False, "unknown strided decode tuning_id"
        if req.tuning_knobs:
            return False, "strided decode does not accept tuning_knobs"
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return ok, why
        try:
            _strided_3d_specs_from_problem(_problem(req), arch=req.arch.lower())
        except ValueError as exc:
            return False, str(exc)
        return True, "supported"

    def select(req):
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(why)
        spec, reduce = _strided_3d_specs_from_problem(
            _problem(req), arch=req.arch.lower()
        )
        return AttentionTuningSpec(
            path="3d",
            arch=req.arch.lower(),
            builder_kind="tiled_3d",
            compile_backend="llvm",
            candidate_name=candidate.name,
            tuning_id="strided",
            kernel_spec=spec,
            reduce_spec=reduce,
        )

    def bind(request, spec, tensors, *, softmax_scale=None, stream=0):
        from kernels.common.attention_unified import (
            Attention3DExecution,
            run_unified_attention_torch,
        )
        from .bindings import validate_tuning_attention_contract

        ok, why = candidate.admits(request)
        if not ok:
            raise ValueError(why)
        problem = tensors.get("problem")
        if problem is None:
            problem = _problem(request)
        if not isinstance(problem, UnifiedAttentionProblem):
            raise TypeError(
                "tensors['problem'] must be a UnifiedAttentionProblem, got "
                f"{type(problem).__name__}"
            )
        validate_tuning_attention_contract(request, problem, spec)
        if problem.sliding_window != request.sliding_window:
            raise ValueError("request sliding_window disagrees with problem")
        if (
            problem.softcap
            or problem.use_sinks
            or problem.use_alibi
            or problem.use_qq_bias
        ):
            raise ValueError(
                "strided decode binding does not support softcap, sinks or bias"
            )
        _validate_strided_3d_spec(problem, spec)
        required = {"q", "k", "v", "out", "cu_seqlens_q", "seqused_k"}
        if missing := required - tensors.keys():
            raise ValueError("missing attention tensors: " + ", ".join(sorted(missing)))
        if unknown := tensors.keys() - required - {"problem", "block_table"}:
            raise ValueError(
                "unsupported strided decode tensors: " + ", ".join(sorted(unknown))
            )
        if tensors.get("block_table") is not None:
            raise ValueError("strided decode does not take a block table")
        if softmax_scale is None:
            softmax_scale = 1 / math.sqrt(problem.head_size)

        execution = Attention3DExecution()

        def launch(*, softmax_scale=softmax_scale, stream=stream):
            return run_unified_attention_torch(
                problem=problem,
                q=tensors["q"],
                k=tensors["k"],
                v=tensors["v"],
                out=tensors["out"],
                cu_seqlens_q=tensors["cu_seqlens_q"],
                seqused_k=tensors["seqused_k"],
                block_table=None,
                softmax_scale=float(softmax_scale),
                softcap=0.0,
                stream=int(stream),
                backend="3d",
                kv_layout="strided",
                tuning_spec=spec,
                execution=execution,
            )

        # TorchBinding's platform ABI stays unchanged. Explicit lifecycle is
        # exposed on the callable for long-lived library bindings.
        launch.close = execution.close
        launch.execution = execution
        return TorchBinding(
            launch=launch, grid=spec.launch_grid(problem), block=spec.launch_block()
        )

    candidate = KernelCandidate(
        name="attention_strided_decode",
        family=FAMILY,
        algorithm="strided_decode",
        spec_id="strided_decode",
        abi_version="rocke-attention-strided-kv/v1",
        priority=20,
        capability=Capability(
            arches=("gfx942", "gfx950"),
            dtypes=("fp16", "bf16"),
            shapes=(
                ShapeRange("hdim_q", allowed=(64, 128, 256)),
                ShapeRange("seqlen_q", allowed=(1,)),
                ShapeRange("kv_block_size", allowed=(16, 32, 64)),
            ),
            supports_features=frozenset(
                {"strided_kv", "causal", "causal_bottom_right", "sliding_window"}
            ),
        ),
        _supports=support,
        select_spec=select,
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
        build=lambda spec, arch: spec.build(arch),
        signature=lambda spec: _3d_signature(spec.kernel_spec.dtype, strided_kv=True),
        grid=lambda spec, req: spec.launch_grid(_problem(req)),
        block=lambda spec: spec.launch_block(),
        bind_torch=bind,
    )
    return candidate


def register(route, execution) -> None:
    candidate = make_candidate()
    route.register(candidate)
    execution.register(candidate)
