# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU contracts for non-paged KV metadata and the tiled decode ABI."""

from dataclasses import fields, make_dataclass, replace
from importlib import import_module
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from kernels.common.attention_kv_cache import KvTensorLayout, StridedKvCacheLayout
from kernels.common.attention_unified import _3d_signature
from rocke.core.verify import verify


@pytest.mark.parametrize("flavor", ["llvm20", "llvm22", "llvm23"])
def test_attention_strided_kv_golden(flavor):
    tests = Path(__file__).resolve().parent
    golden = json.loads(
        (tests / "golden" / "attention_strided_kv_ir_sha256.json").read_text()
    )
    from rocke.assets import platform_root

    env = dict(os.environ, ROCKE_LLVM_FLAVOR=flavor, ROCKE_BACKEND="python")
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(platform_root() / "tests" / "instances" / "parity"),
            str(platform_root() / "python"),
            str(tests.parent),
        ]
    )
    assert set(golden[flavor]) == {str(i) for i in range(12)}
    for idx, expected in golden[flavor].items():
        data = subprocess.check_output(
            [
                sys.executable,
                str(tests / "parity" / "attention_strided_kv_emit.py"),
                idx,
            ],
            env=env,
        )
        assert hashlib.sha256(data).hexdigest() == expected, (flavor, idx)


@pytest.mark.parametrize("shape", [(3, 2, 65, 64), (2, 4, 17, 128), (1, 1, 1, 256)])
def test_contiguous_layout_arguments(shape):
    b, h, c, d = shape
    bshd = KvTensorLayout(shape, (c * h * d, d, h * d, 1))
    bhsd = KvTensorLayout(shape, (h * c * d, c * d, d, 1))
    args = StridedKvCacheLayout(bshd, bhsd).arguments()
    assert args["k_stride_token_bytes"] == h * d * 2
    assert args["v_stride_head_bytes"] == c * d * 2
    assert args["k_span_bytes"] == ((c - 1) * h * d + d) * 2
    assert args["v_span_bytes"] == c * d * 2
    assert bshd.storage_span_bytes == bhsd.storage_span_bytes == b * h * c * d * 2


def test_outer_padding_and_large_batch_offsets():
    layout = KvTensorLayout((2, 3, 65, 64), (1 << 32, 65 * 80, 80, 1))
    assert layout.arguments("k")["k_stride_batch_bytes"] == 1 << 33
    assert layout.head_span_bytes == (64 * 80 + 64) * 2


@pytest.mark.parametrize(
    "shape,strides,reason",
    [
        ((2, 2, 65, 64), (2 * 65 * 128, 65 * 128, 128, 2), "unit D"),
        ((2, 2, 65, 64), (2 * 65 * 64, 65 * 64, 63, 1), "alignment"),
        ((2, 2, 65, 64), (2 * 65 * 64, 64, 64, 1), "overlap"),
        ((2, 2, 65, 64), (-2 * 65 * 64, 65 * 64, 64, 1), "positive"),
        ((2, 2, 65, 64), (0, 65 * 64, 64, 1), "positive"),
        ((1, 1, 1 << 25, 64), (1 << 32, 1 << 31, 64, 1), "buffer-offset"),
        ((2, 2, 65, 64), (1 << 63, 65 * 64, 64, 1), "64-bit"),
    ],
)
def test_unsupported_layouts_decline(shape, strides, reason):
    with pytest.raises(ValueError, match=reason):
        KvTensorLayout(shape, strides)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_strided_segment_signature_and_identity(arch, dtype):
    module = import_module(f"kernels.{arch}.attention_tiled_3d")
    fields = dict(
        head_size=64,
        block_size=16,
        num_query_heads=4,
        num_kv_heads=2,
        dtype=dtype,
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_segments=4,
    )
    paged = module.UnifiedAttention3DTiledSpec(**fields)
    strided = module.UnifiedAttention3DTiledSpec(**fields, kv_layout="strided")
    assert strided.kernel_name() == paged.kernel_name() + "_stridedkv"
    kernel = module.build_unified_attention_3d_tiled(strided, arch=arch)
    assert not verify(kernel)
    signature = _3d_signature(dtype, strided_kv=True)
    assert [p.name for p in kernel.params] == [p["name"] for p in signature]
    assert [p.type.name.replace(" ", "") for p in kernel.params] == [
        p["type"].replace(" ", "") for p in signature
    ]
    for kw in (
        dict(kv_storage_dtype="fp8e4m3"),
        dict(use_wide_kv_load=True),
        dict(use_i64_kv_addr=True),
    ):
        with pytest.raises(ValueError, match="strided KV requires"):
            module.UnifiedAttention3DTiledSpec(**fields, kv_layout="strided", **kw)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_strided_dispatch_is_exclusive(arch):
    from dataclasses import replace
    from dispatch.attention import (
        AttentionRequest,
        dispatch_attention,
        attention_candidates,
        attention_execution_candidates,
    )

    req = AttentionRequest(
        batch=3,
        nhead_q=8,
        nhead_k=2,
        seqlen_q=1,
        seqlen_k=65,
        hdim_q=128,
        hdim_v=128,
        arch=arch,
        dtype="bf16",
        kv_layout="strided",
        target_ctas=8,
    )
    result = dispatch_attention(req)
    assert result.candidate.name == "attention_strided_decode"
    assert result.spec.kernel_spec.kv_layout == "strided"
    for candidates in (attention_candidates(), attention_execution_candidates()):
        assert [c.name for c in candidates if c.admits(req)[0]] == [
            "attention_strided_decode"
        ]
    for changes in (
        dict(seqlen_q=2),
        dict(mask_type=1),
        dict(use_fp8=True),
        dict(use_sinks=True),
        dict(arch="gfx1250"),
    ):
        assert not result.candidate.admits(replace(req, **changes))[0]
    assert not result.candidate.admits(replace(req, kv_layout="paged"))[0]


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_strided_dispatch_pin_roundtrip(arch):
    from dispatch.attention import attention_tuning_spec, dispatch_attention

    req = _strided_request(arch)
    result = dispatch_attention(req)
    replay = dispatch_attention(result.request)
    assert replay.candidate is result.candidate
    assert replay.spec == result.spec
    assert replay.kernel_id == result.kernel_id
    assert result.request.tuning_id == "strided"
    assert result.request.tuning_knobs == ()
    assert attention_tuning_spec(req, "strided_decode", "strided") == result.spec


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize(
    "changes",
    [{"tuning_id": "missing"}, {"tuning_knobs": {"waves_per_eu": 3}}],
)
def test_strided_dispatch_rejects_unresolved_pins(arch, changes):
    from dispatch.attention import dispatch_attention
    from rocke.dispatch.core import PinRefused

    req = replace(
        _strided_request(arch),
        algorithm="strided_decode",
        spec_id="strided_decode",
        **changes,
    )
    with pytest.raises(PinRefused, match="tuning_id|tuning_knobs"):
        dispatch_attention(req)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("flavor", ["llvm20", "llvm22", "llvm23"])
@pytest.mark.parametrize("waves_per_eu", [None, 3])
def test_native_strided_builder_matches_python(arch, flavor, waves_per_eu, monkeypatch):
    from dataclasses import asdict

    engine = pytest.importorskip("rocke_engine")
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", flavor)
    module = import_module(f"kernels.{arch}.attention_tiled_3d")
    spec = module.UnifiedAttention3DTiledSpec(
        head_size=64,
        block_size=16,
        num_query_heads=4,
        num_kv_heads=2,
        dtype="fp16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_segments=4,
        num_seqs=3,
        kv_layout="strided",
        waves_per_eu=waves_per_eu,
    )
    kernel = module.build_unified_attention_3d_tiled(spec, arch=arch)
    expected = _lower_kernel_to_llvm_python(kernel, arch=arch)
    if arch == "gfx950":
        reduce = module.UnifiedAttentionReduceTiledSpec(
            head_size=64,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="fp16",
            num_segments=4,
            waves_per_eu=waves_per_eu,
        )
        expected += _lower_kernel_to_llvm_python(
            module.build_unified_attention_reduce_tiled(reduce, arch=arch), arch=arch
        )
    actual = getattr(engine, f"{arch}_attention_tiled_3d_lower_llvm")(
        asdict(spec), arch
    )
    assert actual == expected


class _IntegerLikeMask:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


def _strided_request(arch):
    from dispatch.attention import AttentionRequest

    return AttentionRequest(
        batch=3,
        nhead_q=8,
        nhead_k=2,
        seqlen_q=1,
        seqlen_k=65,
        hdim_q=128,
        hdim_v=128,
        arch=arch,
        dtype="bf16",
        kv_layout="strided",
        kv_block_size=32,
        target_ctas=8,
    )


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("mask", [0, 1, 2])
def test_strided_mask_uses_integer_protocol(arch, mask):
    from dispatch.attention.strided_decode import make_candidate

    candidate = make_candidate()
    req = _strided_request(arch)
    ordinary = candidate.admits(replace(req, mask_type=mask))
    integer_like = candidate.admits(replace(req, mask_type=_IntegerLikeMask(mask)))
    assert integer_like == ordinary
    assert integer_like[0] == (mask != 1)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("block_size", [16, 32, 64])
def test_strided_selection_uses_runtime_policy(arch, block_size, monkeypatch):
    from dispatch.attention import dispatch_attention
    from dispatch.attention.common import _problem
    from kernels.common import attention_unified as au

    req = replace(_strided_request(arch), kv_block_size=block_size)
    problem = _problem(req)
    # Off-device selection must not use the host's architecture policy.
    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: "gfx1250")
    selected = dispatch_attention(req).spec
    segment, reduce = au._strided_3d_specs_from_problem(problem, arch=arch)
    assert (selected.kernel_spec, selected.reduce_spec) == (segment, reduce)
    assert segment.waves_per_eu is None
    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: arch)
    paged = au._tiled_3d_spec_from_problem(problem)
    assert segment == replace(
        paged, kv_layout="strided", use_wide_kv_load=False, use_i64_kv_addr=False
    )
    explicit, explicit_reduce = au._strided_3d_specs_from_problem(
        replace(problem, waves_per_eu=3), arch=arch
    )
    assert explicit.waves_per_eu == explicit_reduce.waves_per_eu == 3


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_strided_selection_ignores_hoist_environment(arch, monkeypatch):
    from dispatch.attention import dispatch_attention
    from dispatch.attention.common import _problem
    from kernels.common import attention_unified as au

    req = _strided_request(arch)
    problem = _problem(req)
    monkeypatch.delenv("HIPDNN_GFX942_3D_HOIST", raising=False)
    expected = dispatch_attention(req).spec
    for value in ("0", "1"):
        monkeypatch.setenv("HIPDNN_GFX942_3D_HOIST", value)
        selected = dispatch_attention(req).spec
        assert selected == expected
        assert not selected.kernel_spec.use_invariant_hoist
        assert au._strided_3d_specs_from_problem(problem, arch=arch) == (
            expected.kernel_spec,
            expected.reduce_spec,
        )
    # Explicit tuning retains the existing knob.
    explicit = replace(
        expected, kernel_spec=replace(expected.kernel_spec, use_invariant_hoist=True)
    )
    au._validate_strided_3d_spec(problem, explicit)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_strided_spec_policy_rejects_conflicting_clamp_arch(arch):
    from dispatch.attention.common import _problem
    from kernels.common import attention_unified as au

    problem = _problem(_strided_request(arch))
    expected = au._strided_3d_specs_from_problem(problem, arch=arch)
    assert (
        au._strided_3d_specs_from_problem(replace(problem, clamp_arch=None), arch=arch)
        == expected
    )
    other = "gfx950" if arch == "gfx942" else "gfx942"
    with pytest.raises(ValueError, match="clamp_arch.*conflicts"):
        au._strided_3d_specs_from_problem(replace(problem, clamp_arch=other), arch=arch)


@pytest.fixture
def strided_binding_case():
    from dispatch.attention import dispatch_attention
    from dispatch.attention.common import _problem

    req = _strided_request("gfx942")
    result = dispatch_attention(req)
    tensors = {
        name: object() for name in ("q", "k", "v", "out", "cu_seqlens_q", "seqused_k")
    }
    tensors["problem"] = _problem(req)
    return req, result, tensors


def test_strided_binding_scale_stream_and_unknown_options(
    strided_binding_case, monkeypatch
):
    from kernels.common import attention_unified as au

    req, result, tensors = strided_binding_case
    calls = []
    monkeypatch.setattr(
        au, "run_unified_attention_torch", lambda **kw: calls.append(kw)
    )
    bind = result.candidate.bind_torch
    binding = bind(req, result.spec, tensors, softmax_scale=0.25, stream=17)
    binding.launch()
    binding.launch(softmax_scale=0.5, stream=19)
    assert [(c["softmax_scale"], c["stream"]) for c in calls] == [(0.25, 17), (0.5, 19)]
    for kw in ({"softcap": 1.0}, {"softmax_sclae": 0.5}, {"backend": "auto"}):
        with pytest.raises(TypeError):
            bind(req, result.spec, tensors, **kw)
        with pytest.raises(TypeError):
            binding.launch(**kw)
    assert len(calls) == 2


@pytest.mark.parametrize(
    "problem", [False, 0, {}, "", [], True, 1, "invalid", object()]
)
def test_strided_binding_rejects_invalid_problem_type(strided_binding_case, problem):
    req, result, tensors = strided_binding_case
    with pytest.raises(TypeError, match="must be a UnifiedAttentionProblem"):
        result.candidate.bind_torch(req, result.spec, {**tensors, "problem": problem})


@pytest.mark.parametrize("mode", ["omitted", "none", "explicit"])
def test_strided_binding_problem_default_and_identity(
    strided_binding_case, monkeypatch, mode
):
    from kernels.common import attention_unified as au

    req, result, tensors = strided_binding_case
    expected = tensors.pop("problem")
    if mode == "none":
        tensors["problem"] = None
    elif mode == "explicit":
        tensors["problem"] = expected
    calls = []
    monkeypatch.setattr(
        au, "run_unified_attention_torch", lambda **kw: calls.append(kw)
    )
    binding = result.candidate.bind_torch(req, result.spec, tensors)
    binding.launch()
    assert len(calls) == 1
    assert calls[0]["problem"] == expected
    if mode == "explicit":
        assert calls[0]["problem"] is expected


@pytest.mark.parametrize("extra", ["sinks", "alibi_slopes", "qq_bias", "bias"])
def test_strided_binding_rejects_unconsumed_tensors(strided_binding_case, extra):
    req, result, tensors = strided_binding_case
    with pytest.raises(ValueError, match="unsupported strided decode tensors"):
        result.candidate.bind_torch(req, result.spec, {**tensors, extra: object()})


@pytest.mark.parametrize(
    "changes",
    [
        {"sliding_window": 17},
        {"softcap": 1.0},
        {"use_sinks": True},
        {"use_alibi": True},
        {"use_qq_bias": True},
    ],
)
def test_strided_binding_rejects_problem_semantic_mismatch(
    strided_binding_case, changes
):
    req, result, tensors = strided_binding_case
    tensors["problem"] = replace(tensors["problem"], **changes)
    with pytest.raises(ValueError):
        result.candidate.bind_torch(req, result.spec, tensors)


@pytest.mark.parametrize(
    "component,changes",
    [
        ("kernel_spec", {"sliding_window": 17}),
        ("kernel_spec", {"has_softcap": True}),
        ("kernel_spec", {"head_size": 64}),
        ("kernel_spec", {"num_seqs": 2}),
        ("kernel_spec", {"kv_layout": "paged"}),
        ("kernel_spec", {"use_sinks": True}),
        ("reduce_spec", {"num_segments": 32}),
        ("reduce_spec", {"dtype": "fp16"}),
        ("reduce_spec", {"num_query_heads": 4}),
    ],
)
def test_strided_runtime_rejects_incoherent_specs(
    strided_binding_case, monkeypatch, component, changes
):
    from kernels.common import attention_unified as au

    _, result, tensors = strided_binding_case
    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: "gfx942")
    spec = replace(
        result.spec, **{component: replace(getattr(result.spec, component), **changes)}
    )
    # Validate before inspecting device tensors or compiling/launching a kernel.
    with pytest.raises(ValueError, match="disagrees"):
        au.run_unified_attention_torch(
            **tensors,
            block_table=None,
            softmax_scale=0.125,
            softcap=0,
            tuning_spec=spec,
            kv_layout="strided",
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"k_scale": 2.0},
        {"v_scale": 0.5},
        {"out_scale": 0.5},
        {"softcap": 1.0},
    ],
)
def test_strided_runtime_rejects_ignored_scaling(
    strided_binding_case, monkeypatch, overrides
):
    from kernels.common import attention_unified as au

    _, result, tensors = strided_binding_case
    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: "gfx942")
    with pytest.raises(ValueError, match="scaling|softcap"):
        au.run_unified_attention_torch(
            **tensors,
            block_table=None,
            softmax_scale=0.125,
            tuning_spec=result.spec,
            kv_layout="strided",
            **{"softcap": 0, **overrides},
        )


@pytest.mark.parametrize("missing", ["kv_layout", "kv_storage_dtype"])
def test_strided_runtime_rejects_missing_semantic_field(
    strided_binding_case, monkeypatch, missing
):
    from kernels.common import attention_unified as au

    _, result, tensors = strided_binding_case
    segment = result.spec.kernel_spec
    values = {
        f.name: getattr(segment, f.name) for f in fields(segment) if f.name != missing
    }
    malformed_type = make_dataclass(
        "MissingSemanticField", [(name, object) for name in values], frozen=True
    )
    spec = replace(result.spec, kernel_spec=malformed_type(**values))
    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: "gfx942")
    with pytest.raises(ValueError, match=f"missing {missing}"):
        au.run_unified_attention_torch(
            **tensors,
            block_table=None,
            softmax_scale=0.125,
            softcap=0,
            tuning_spec=spec,
            kv_layout="strided",
        )


@pytest.mark.parametrize(
    "arch,path", [("gfx942", "2d"), ("gfx950", "2d"), ("gfx1250", "3d")]
)
def test_paged_runtime_accepts_specs_without_layout(arch, path, monkeypatch):
    from dispatch.attention.common import AttentionTuningSpec, _problem
    from kernels.common import attention_unified as au

    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: arch)
    problem = _problem(
        replace(
            _strided_request(arch), kv_layout="paged", hdim_q=64, hdim_v=64, nhead_q=16
        )
    )
    factory = (
        au._tiled_3d_spec_from_problem if path == "3d" else au._tiled_spec_from_problem
    )
    kernel = factory(problem)
    assert not hasattr(kernel, "kv_layout")
    spec = AttentionTuningSpec(
        path=path,
        arch=arch,
        builder_kind=f"tiled_{path}",
        compile_backend="llvm",
        candidate_name="paged_compatibility",
        tuning_id="paged_compatibility",
        kernel_spec=kernel,
        num_kv_blocks=1,
    )
    calls = []

    def stop_before_launch(problem, tuning_spec, kind):
        calls.append((tuning_spec.kernel_spec, kind))
        return False, "paged contract reached"

    monkeypatch.setattr(au, "_explicit_path_supported", stop_before_launch)
    # Exercise the public runtime boundary with real paged-only spec classes;
    # stop at support checking before any device access or compilation.
    with pytest.raises(NotImplementedError, match="paged contract reached"):
        au.run_unified_attention_torch(
            problem=problem,
            q=object(),
            k=SimpleNamespace(shape=(1,)),
            v=object(),
            out=object(),
            cu_seqlens_q=object(),
            seqused_k=object(),
            block_table=SimpleNamespace(shape=(3, 1)),
            softmax_scale=0.125,
            softcap=0,
            tuning_spec=spec,
        )
    assert calls == [(kernel, path)]


@pytest.mark.parametrize("explicit", [False, True])
def test_strided_runtime_rejects_conflicting_clamp_arch(
    strided_binding_case, monkeypatch, explicit
):
    from kernels.common import attention_unified as au

    _, result, tensors = strided_binding_case
    tensors = dict(tensors, problem=replace(tensors["problem"], clamp_arch="gfx950"))
    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: "gfx942")
    with pytest.raises(ValueError, match="clamp_arch.*conflicts"):
        au.run_unified_attention_torch(
            **tensors,
            block_table=None,
            softmax_scale=0.125,
            softcap=0,
            tuning_spec=result.spec if explicit else None,
            kv_layout="strided",
        )
