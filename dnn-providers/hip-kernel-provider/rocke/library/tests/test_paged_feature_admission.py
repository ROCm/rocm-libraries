# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Final paged feature admission through direct, bound and prepared calls."""

from dataclasses import replace
from importlib import import_module

import pytest

from tests import test_paged_kv_decode as paged
from kernels.common import attention_unified as au
from dispatch.attention import dispatch_attention
from dispatch.attention.common import _problem

paged_binding_case = paged.paged_binding_case


class FeatureTensor(paged.Tensor):
    def __init__(self, shape, dtype, strides=None, pointer=4096, device="cuda:0"):
        super().__init__(shape, dtype)
        self.strides, self.pointer, self.device = strides, pointer, device

    def stride(self, dim=None):
        strides = self.strides if self.strides is not None else super().stride()
        return strides if dim is None else strides[dim]

    def data_ptr(self):
        return self.pointer

    def cpu(self):
        pytest.fail("feature validation read device contents")


def feature_tensor(feature, compute_dtype="fp16", **kwargs):
    shape = (1, 1) if feature == "qq_bias" else (8,)
    kind = (
        ("torch.float16" if compute_dtype == "fp16" else "torch.bfloat16")
        if feature == "sinks"
        else "torch.float32"
    )
    return FeatureTensor(
        kwargs.pop("shape", shape), kwargs.pop("dtype", kind), **kwargs
    )


def enabled_call(arch, page, dtype, feature):
    arguments = paged._paged_runtime_kwargs(arch, page)
    flag = {
        "sinks": "use_sinks",
        "alibi_slopes": "use_alibi",
        "qq_bias": "use_qq_bias",
    }[feature]
    arguments["problem"] = replace(arguments["problem"], dtype=dtype, **{flag: True})
    spec = arguments["tuning_spec"]
    arguments["tuning_spec"] = replace(
        spec,
        kernel_spec=replace(spec.kernel_spec, dtype=dtype, **{flag: True}),
        reduce_spec=replace(spec.reduce_spec, dtype=dtype),
    )
    for name in ("q", "k", "v", "out"):
        arguments[name] = paged.Tensor(
            arguments[name].shape,
            "torch.bfloat16" if dtype == "bf16" else "torch.float16",
        )
    arguments[feature] = feature_tensor(feature, dtype)
    arguments["qq_bias_stride_0"] = 1 if feature == "qq_bias" else 0
    return arguments


def intercept(monkeypatch, arch):
    calls = []
    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: arch)
    monkeypatch.setattr(au, "_run_3d_tiled", lambda **kw: calls.append(kw))
    return calls


def malformed(feature, dtype, defect):
    if defect == "missing_metadata":
        return object()
    if defect == "short":
        return feature_tensor(
            feature, dtype, shape=(0, 1) if feature == "qq_bias" else (1,)
        )
    if defect == "rank":
        return feature_tensor(
            feature, dtype, shape=(1,) if feature == "qq_bias" else (1, 8)
        )
    if defect == "dtype":
        return feature_tensor(feature, dtype, dtype="torch.int32")
    if defect == "layout":
        return feature_tensor(
            feature, dtype, strides=(0, 1) if feature == "qq_bias" else (2,)
        )
    if defect == "alignment":
        return feature_tensor(feature, dtype, pointer=4097)
    if defect == "pointer":
        return feature_tensor(feature, dtype, pointer=0)
    return feature_tensor(feature, dtype, device="cpu" if defect == "cpu" else "cuda:1")


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("feature", ["sinks", "alibi_slopes", "qq_bias"])
@pytest.mark.parametrize(
    "defect",
    [
        "missing_metadata",
        "short",
        "rank",
        "dtype",
        "layout",
        "alignment",
        "pointer",
        "cpu",
        "device",
    ],
)
def test_direct_rejects_feature_metadata_before_retarget(
    monkeypatch, arch, page, dtype, feature, defect
):
    calls = intercept(monkeypatch, arch)
    args = enabled_call(arch, page, dtype, feature)
    args[feature] = malformed(feature, dtype, defect)
    args["tuning_spec"] = paged._structural_paged_spec(
        replace(args["tuning_spec"], allow_unsupported=True),
        retarget=lambda count: pytest.fail("invalid input reached retarget"),
    )
    with pytest.raises(ValueError, match=feature):
        au.run_unified_attention_torch(**args)
    assert not calls


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("feature", ["sinks", "alibi_slopes", "qq_bias"])
@pytest.mark.parametrize("entry", ["auto", "prepared"])
def test_default_and_prepared_reject_replaced_features(
    monkeypatch, arch, dtype, feature, entry
):
    calls = intercept(monkeypatch, arch)
    args = enabled_call(arch, 1, dtype, feature)
    bad = malformed(feature, dtype, "dtype")
    if entry == "auto":
        with pytest.raises(ValueError, match=feature):
            au.run_unified_attention_torch(
                **dict(args, tuning_spec=None, **{feature: bad})
            )
    else:
        owner = au.Attention3DExecution()
        owner._arguments = args
        with pytest.raises(ValueError, match=feature):
            owner.launch(**{feature: bad})
    assert not calls


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize("feature", ["sinks", "alibi_slopes", "qq_bias"])
@pytest.mark.parametrize("stage", ["tensor", "bind", "launch"])
def test_binding_validates_effective_feature_replacements(
    paged_binding_case, monkeypatch, arch, page, feature, stage
):
    req, _, tensors = paged._retarget_binding_case(paged_binding_case, arch, page)
    flag = {
        "sinks": "use_sinks",
        "alibi_slopes": "use_alibi",
        "qq_bias": "use_qq_bias",
    }[feature]
    req = replace(req, **{flag: True})
    result = dispatch_attention(req)
    tensors["problem"] = _problem(req)
    calls = intercept(monkeypatch, arch)
    good, bad = feature_tensor(feature), malformed(feature, "fp16", "dtype")
    base = {feature: good, "qq_bias_stride_0": 1 if feature == "qq_bias" else 0}
    with pytest.raises(ValueError, match=feature):
        if stage == "tensor":
            result.candidate.bind_torch(
                req, result.spec, dict(tensors, **dict(base, **{feature: bad}))
            )
        elif stage == "bind":
            result.candidate.bind_torch(
                req, result.spec, tensors, **dict(base, **{feature: bad})
            )
        else:
            result.candidate.bind_torch(req, result.spec, tensors, **base).launch(
                **{feature: bad}
            )
    assert not calls
    # A valid bind override supersedes a malformed tensor-map input.
    binding = result.candidate.bind_torch(
        req, result.spec, dict(tensors, **{feature: bad}), **base
    )
    replacement = feature_tensor(feature)
    binding.launch(**{feature: replacement})
    assert calls[-1][feature] is replacement
    assert tensors["seqused_k"].reads == []


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize("window,compiled", [(0, 17), (17, 0)])
@pytest.mark.parametrize("retargeted", [False, True])
def test_all_paged_loaders_reject_window_mismatch(
    monkeypatch, arch, page, window, compiled, retargeted
):
    calls = intercept(monkeypatch, arch)
    args = paged._paged_runtime_kwargs(arch, page)
    args["problem"] = replace(args["problem"], sliding_window=window)
    original = replace(
        args["tuning_spec"],
        allow_unsupported=True,
        kernel_spec=replace(args["tuning_spec"].kernel_spec, sliding_window=window),
    )
    bad = replace(
        original, kernel_spec=replace(original.kernel_spec, sliding_window=compiled)
    )
    args["tuning_spec"] = (
        paged._structural_paged_spec(original, retarget=lambda count: bad)
        if retargeted
        else bad
    )
    with pytest.raises(ValueError, match="sliding_window"):
        au.run_unified_attention_torch(**args)
    assert not calls
    au.run_unified_attention_torch(**dict(args, tuning_spec=original))
    assert len(calls) == 1


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize("stride", [0, -1, True, 1.5, None, "1", 2**31, 2])
def test_direct_rejects_invalid_enabled_bias_stride(monkeypatch, arch, page, stride):
    calls = intercept(monkeypatch, arch)
    args = enabled_call(arch, page, "fp16", "qq_bias")
    with pytest.raises(ValueError, match="qq_bias_stride_0"):
        au.run_unified_attention_torch(**dict(args, qq_bias_stride_0=stride))
    assert not calls


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_automatic_bias_requires_positive_stride(monkeypatch, arch):
    calls = intercept(monkeypatch, arch)
    args = enabled_call(arch, 1, "fp16", "qq_bias")
    args["tuning_spec"] = None
    with pytest.raises(ValueError, match="qq_bias_stride_0"):
        au.run_unified_attention_torch(**dict(args, qq_bias_stride_0=0))
    au.run_unified_attention_torch(**args)
    assert len(calls) == 1


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("page", [1, 16, 32, 64])
def test_binding_defers_stride_only_until_launch(
    paged_binding_case, monkeypatch, arch, page
):
    calls = intercept(monkeypatch, arch)
    req, _, tensors = paged._retarget_binding_case(paged_binding_case, arch, page)
    req = replace(req, use_qq_bias=True)
    result = dispatch_attention(req)
    tensors["problem"] = _problem(req)
    bias = feature_tensor("qq_bias", shape=(2, 2), strides=(8, 1))
    binding = result.candidate.bind_torch(req, result.spec, tensors, qq_bias=bias)
    with pytest.raises(ValueError, match="qq_bias_stride_0"):
        binding.launch()
    assert not calls
    binding.launch(qq_bias_stride_0=8)
    assert calls[-1]["qq_bias"] is bias
    replacement = feature_tensor("qq_bias")
    binding.launch(qq_bias=replacement, qq_bias_stride_0=1)
    assert calls[-1]["qq_bias"] is replacement
    with pytest.raises(ValueError, match="qq_bias_stride_0"):
        result.candidate.bind_torch(
            req, result.spec, tensors, qq_bias=bias, qq_bias_stride_0=2
        )


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("fp8", [False, True])
def test_valid_padded_bias_and_feature_dtypes(monkeypatch, arch, dtype, fp8):
    calls = intercept(monkeypatch, arch)
    args = enabled_call(arch, 32, dtype, "qq_bias")
    args["problem"] = replace(
        args["problem"],
        use_sinks=True,
        use_alibi=True,
        use_fp8=fp8,
        fp8_fnuz=fp8 and arch == "gfx942",
    )
    spec = args["tuning_spec"]
    args["tuning_spec"] = replace(
        spec,
        kernel_spec=replace(
            spec.kernel_spec,
            use_sinks=True,
            use_alibi=True,
            kv_storage_dtype="fp8e4m3" if fp8 else None,
        ),
    )
    args.update(
        sinks=feature_tensor("sinks", dtype),
        alibi_slopes=feature_tensor("alibi_slopes"),
        qq_bias=feature_tensor("qq_bias", shape=(4, 3), strides=(8, 1)),
        qq_bias_stride_0=8,
    )
    if fp8:
        args.update(k_scale=0.5, v_scale=2.0)
    au.run_unified_attention_torch(**args)
    assert len(calls) == 1
    assert all(
        calls[0][name] is args[name] for name in ("sinks", "alibi_slopes", "qq_bias")
    )


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize(
    "shape,strides", [((1, 1), (2**31, 1)), ((2, 2), (2**31 - 1, 1))]
)
def test_bias_index_bounds(monkeypatch, arch, shape, strides):
    calls = intercept(monkeypatch, arch)
    args = enabled_call(arch, 32, "fp16", "qq_bias")
    args["qq_bias"] = feature_tensor("qq_bias", shape=shape, strides=strides)
    args["qq_bias_stride_0"] = strides[0]
    if shape[0] == 2:
        args["problem"] = replace(args["problem"], max_seqlen_q=2, total_q=6)
    with pytest.raises(ValueError, match="qq_bias"):
        au.run_unified_attention_torch(**args)
    assert not calls


def test_feature_numeric_guard_skips_before_allocations():
    module = import_module("tests.test_paged_kv_decode_numeric")
    with pytest.raises(pytest.skip.Exception):
        module.test_paged_features_numeric((object(), "gfx1250"), 1, "fp16", "combined")


@pytest.mark.parametrize("storage_bytes", [4, 32])
def test_single_row_bias_padding_requires_accessible_tail(monkeypatch, storage_bytes):
    from types import SimpleNamespace

    calls = intercept(monkeypatch, "gfx950")
    args = enabled_call("gfx950", 1, "fp16", "qq_bias")
    bias = feature_tensor("qq_bias", strides=(8, 1))
    bias.untyped_storage = lambda: SimpleNamespace(nbytes=lambda: storage_bytes)
    bias.storage_offset = lambda: 0
    args.update(qq_bias=bias, qq_bias_stride_0=8)
    if storage_bytes == 4:
        with pytest.raises(ValueError, match="qq_bias backing storage"):
            au.run_unified_attention_torch(**args)
        assert not calls
    else:
        au.run_unified_attention_torch(**args)
        assert calls[0]["qq_bias"] is bias


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("row_stride", [1, 4])
def test_partial_width_bias_preserves_zero_fill(monkeypatch, arch, row_stride):
    calls = intercept(monkeypatch, arch)
    args = enabled_call(arch, 32, "fp16", "qq_bias")
    args["problem"] = replace(args["problem"], max_seqlen_q=2, total_q=6)
    args["tuning_spec"] = replace(args["tuning_spec"], allow_unsupported=True)
    args.update(
        qq_bias=feature_tensor("qq_bias", shape=(2, 1), strides=(row_stride, 1)),
        qq_bias_stride_0=row_stride,
    )
    if row_stride == 1:
        au.run_unified_attention_torch(**args)
        assert len(calls) == 1
    else:
        with pytest.raises(ValueError, match="partial-width"):
            au.run_unified_attention_torch(**args)
        assert not calls
