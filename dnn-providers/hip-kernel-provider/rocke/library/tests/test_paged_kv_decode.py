# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Admission, identity and native parity for the paged tile loader."""

import hashlib
import json
from dataclasses import asdict, replace
from importlib import import_module
from pathlib import Path

import pytest
from rocke.core.lower_llvm import LLVM_FLAVORS
from kernels.common.attention_paged_decode import BUFFER_LIMIT, validate_paged_decode
from tests.parity.attention_paged_kv_emit import CONFIG_COUNT, build, spec_for_index


@pytest.fixture(params=range(CONFIG_COUNT))
def paged_emission(request, flavor, monkeypatch):
    idx = request.param
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", flavor)
    spec, arch = spec_for_index(idx)
    expected = _lower_kernel_to_llvm_python(build(spec, arch), arch=arch)
    if arch == "gfx942" and spec.head_size == 256:
        import re

        sizes = re.findall(r"global \[(\d+) x i8\]", expected)
        assert sizes and max(map(int, sizes)) <= 65536
    return idx, spec, arch, expected


@pytest.fixture(params=LLVM_FLAVORS)
def flavor(request):
    return request.param


def test_paged_python_golden(paged_emission, flavor):
    idx, _spec, _arch, expected = paged_emission
    golden = json.loads(
        (
            Path(__file__).parent / "golden" / "attention_paged_kv_ir_sha256.json"
        ).read_text()
    )
    assert set(golden) == set(LLVM_FLAVORS)
    assert set(golden[flavor]) == {str(i) for i in range(CONFIG_COUNT)}
    assert hashlib.sha256(expected.encode()).hexdigest() == golden[flavor][str(idx)]


def test_paged_native_parity(paged_emission):
    engine = pytest.importorskip("rocke_engine")
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    _idx, spec, arch, expected = paged_emission
    if not hasattr(spec, "block_size"):
        assert engine.paged_decode_reduce_lower_llvm(asdict(spec), arch) == expected
        return
    if arch == "gfx950":
        m = import_module(f"kernels.{arch}.attention_tiled_3d")
        reduce = m.UnifiedAttentionReduceTiledSpec(
            head_size=spec.head_size,
            num_query_heads=8,
            num_kv_heads=2,
            dtype=spec.dtype,
            num_segments=8,
            waves_per_eu=3,
            metadata_containment=spec.metadata_containment,
        )
        expected += _lower_kernel_to_llvm_python(
            m.build_unified_attention_reduce_tiled(reduce, arch=arch), arch=arch
        )
    assert (
        getattr(engine, f"{arch}_attention_tiled_3d_lower_llvm")(asdict(spec), arch)
        == expected
    )


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_paged_selection_and_identity(arch):
    from dispatch.attention import AttentionRequest, dispatch_attention

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
        kv_block_size=1,
        target_ctas=8,
    )
    result = dispatch_attention(req)
    assert result.candidate.name == "attention_paged_decode_t32"
    assert result.spec.kernel_spec.tile_size == 32
    for page in (1, 16, 32, 64):
        selected = dispatch_attention(
            replace(req, kv_block_size=page, algorithm="paged_decode_t32")
        )
        spec = selected.spec.kernel_spec
        assert spec.block_size == page and spec.tile_size == 32
        assert selected.spec.with_num_kv_blocks(100).kernel_spec == spec
    spec = result.spec.kernel_spec
    other = replace(
        spec, metadata_containment=False, block_size=64, tile_size_override=64
    )
    assert other.kernel_name() != replace(other, tile_size_override=32).kernel_name()
    assert asdict(spec)["tile_size_override"] == 32
    for kwargs in (
        {"tile_size_override": 16},
        {"use_i64_kv_addr": True},
        {"kv_storage_dtype": "fp8e4m3"},
        {"use_wide_kv_load": True},
    ):
        with pytest.raises(ValueError, match="paged gather|bounded decode"):
            replace(spec, **kwargs)


class Tensor:
    next_pointer = 4096

    def __init__(self, shape, dtype="torch.float16"):
        self.pointer = Tensor.next_pointer
        Tensor.next_pointer += 1 << 48
        self.shape, self.dtype = shape, dtype
        self.device, self.is_cuda = "cuda:0", True

    def stride(self, dim=None):
        from math import prod

        strides = tuple(prod(self.shape[i + 1 :]) for i in range(len(self.shape)))
        return strides if dim is None else strides[dim]

    def numel(self):
        import math

        return math.prod(self.shape)

    def is_contiguous(self):
        return True

    def data_ptr(self):
        return self.pointer


def inputs(page=1):
    from kernels.common.attention_unified import UnifiedAttentionProblem

    p = UnifiedAttentionProblem(
        total_q=3,
        num_seqs=3,
        num_query_heads=8,
        num_kv_heads=2,
        head_size=64,
        block_size=page,
        max_seqlen_q=1,
        max_seqlen_k=33,
        dtype="fp16",
        clamp_arch="gfx950",
    )
    ts = [
        Tensor((3, 8, 64)),
        Tensor((99, page, 2, 64)),
        Tensor((99, page, 2, 64)),
        Tensor((3, 8, 64)),
        Tensor((4,), "torch.int32"),
        Tensor((3,), "torch.int32"),
        Tensor((3, (33 + page - 1) // page), "torch.int32"),
    ]
    return p, ts


def test_paged_runtime_bounds():
    p, ts = inputs()
    validate_paged_decode(p, ts)
    for index in (1, 2):
        changed = list(ts)
        changed[index] = Tensor((BUFFER_LIMIT // 256 + 1, 1, 2, 64))
        changed[3 - index] = Tensor(changed[index].shape)
        with pytest.raises(ValueError, match="buffer-offset"):
            validate_paged_decode(p, changed)
    ts[-1] = Tensor((3, 32), "torch.int32")
    with pytest.raises(ValueError, match="block table"):
        validate_paged_decode(p, ts)
    p, ts = inputs()
    with pytest.raises(ValueError, match="token-offset"):
        validate_paged_decode(replace(p, max_seqlen_k=2**31), ts)


class AddressInterpreter:
    """Evaluate the emitted scalar address operations against a finite table."""

    const_i32 = staticmethod(int)
    add = staticmethod(lambda x, y: x + y)
    mul = staticmethod(lambda x, y: x * y)
    div = staticmethod(lambda x, y: x // y)
    mod = staticmethod(lambda x, y: x % y)
    cmp_lt = staticmethod(lambda x, y: x < y)
    select = staticmethod(lambda pred, x, y: x if pred else y)

    @staticmethod
    def global_load(table, index, dtype, align):
        assert 0 <= index < len(table), "speculative page-table read escaped allocation"
        return table[index]


@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize("hd", [64, 128, 256])
def test_logical_tile_address_mapping(page, hd):
    from kernels.common._attention_paged_kv import paged_kv_offset

    length, heads, tile_size = 65, 2, 32
    n = (length + page - 1) // page
    # Independent physical pages for two batches, including gaps in the pool.
    table = [3 * i + 7 for i in reversed(range(2 * n))]
    for seq in range(2):
        for head in range(heads):
            for token in range(160):
                for dim in (0, hd - 8):
                    tile, row = divmod(token, tile_size)
                    got = paged_kv_offset(
                        AddressInterpreter(),
                        table,
                        seq * n,
                        head,
                        tile,
                        row * hd + dim,
                        length,
                        hd,
                        page,
                        tile_size,
                        heads,
                    )
                    if token >= length:
                        assert got == 0x7FFFFFFF
                    else:
                        physical = table[seq * n + token // page]
                        expected = 2 * (
                            ((physical * page + token % page) * heads + head) * hd + dim
                        )
                        assert got == expected


def _paged_runtime_kwargs(arch, page):
    from dispatch.attention import AttentionRequest, dispatch_attention

    p, ts = inputs(page)
    p = replace(p, clamp_arch=arch)
    req = AttentionRequest(
        batch=3,
        nhead_q=8,
        nhead_k=2,
        seqlen_q=1,
        seqlen_k=33,
        hdim_q=64,
        hdim_v=64,
        arch=arch,
        dtype="fp16",
        kv_block_size=page,
        target_ctas=8,
        algorithm="paged_decode_t32",
    )
    spec = dispatch_attention(req).spec
    # This fixture exercises legacy direct/structural loader contracts. The
    # dedicated containment suite covers the candidate's marked pair.
    spec = replace(
        spec,
        kernel_spec=replace(spec.kernel_spec, metadata_containment=False),
        reduce_spec=replace(spec.reduce_spec, metadata_containment=False),
    )
    return {
        "problem": p,
        "q": ts[0],
        "k": ts[1],
        "v": ts[2],
        "out": ts[3],
        "cu_seqlens_q": ts[4],
        "seqused_k": ts[5],
        "block_table": ts[6],
        "softmax_scale": 1 / 8,
        "softcap": 0,
        "backend": "3d",
        "tuning_spec": spec,
    }


def _structural_paged_spec(concrete, *, kernel_spec=None, retarget=None):
    """Expose only the runtime protocol, with no candidate identity metadata."""
    from types import SimpleNamespace

    segment = concrete.kernel_spec
    snapshot = asdict(segment)
    snapshot.update(
        tile_size=segment.tile_size, uses_paged_gather=segment.uses_paged_gather
    )
    spec = SimpleNamespace(
        arch=concrete.arch,
        path=concrete.path,
        kernel_spec=(
            kernel_spec if kernel_spec is not None else SimpleNamespace(**snapshot)
        ),
        reduce_spec=concrete.reduce_spec,
        compile_backend=concrete.compile_backend,
        allow_unsupported=concrete.allow_unsupported,
        cache_key=concrete.cache_key,
        build=concrete.build,
        launch_grid=concrete.launch_grid,
        launch_block=concrete.launch_block,
    )
    spec.with_num_kv_blocks = retarget if retarget is not None else lambda count: spec
    assert not hasattr(spec, "candidate_name") and not hasattr(spec, "tuning_id")
    return spec


def _forbid_paged_launch(arch, monkeypatch):
    from kernels.common import attention_unified as au

    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: arch)

    def forbidden(**kwargs):
        pytest.fail("invalid paged features reached launch")

    monkeypatch.setattr(au, "_run_3d_tiled", forbidden)
    return au


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize("structural", [False, True])
@pytest.mark.parametrize("allow_unsupported", [False, True])
def test_paged_direct_rejects_problem_feature_mismatch(
    arch, page, structural, allow_unsupported, monkeypatch
):
    au = _forbid_paged_launch(arch, monkeypatch)
    kwargs = _paged_runtime_kwargs(arch, page)
    concrete = replace(kwargs["tuning_spec"], allow_unsupported=allow_unsupported)
    kwargs["tuning_spec"] = _structural_paged_spec(concrete) if structural else concrete
    for name in ("softcap", "use_sinks", "use_alibi", "use_qq_bias", "use_fp8"):
        problem = replace(
            kwargs["problem"], **{name: 2.0 if name == "softcap" else True}
        )
        with pytest.raises(ValueError, match="kernel_spec"):
            au.run_unified_attention_torch(**dict(kwargs, problem=problem))


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize("structural", [False, True])
@pytest.mark.parametrize("allow_unsupported", [False, True])
def test_paged_direct_rejects_ignored_feature_inputs(
    arch, page, structural, allow_unsupported, monkeypatch
):
    au = _forbid_paged_launch(arch, monkeypatch)
    kwargs = _paged_runtime_kwargs(arch, page)
    concrete = replace(kwargs["tuning_spec"], allow_unsupported=allow_unsupported)
    kwargs["tuning_spec"] = _structural_paged_spec(concrete) if structural else concrete
    for name in ("softcap", "sinks", "alibi_slopes", "qq_bias"):
        value = 2.0 if name == "softcap" else object()
        with pytest.raises(ValueError, match=name):
            au.run_unified_attention_torch(**dict(kwargs, **{name: value}))
    for name in ("k_scale", "v_scale", "out_scale"):
        for value in (
            2.0,
            None,
            True,
            False,
            "1.0",
            [],
            object(),
            float("nan"),
            float("inf"),
        ):
            with pytest.raises(ValueError, match=name):
                au.run_unified_attention_torch(**dict(kwargs, **{name: value}))


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize("structural", [False, True])
def test_paged_direct_accepts_numeric_unit_scales(arch, page, structural, monkeypatch):
    from numbers import Real

    from kernels.common import attention_unified as au

    kwargs = _paged_runtime_kwargs(arch, page)
    if structural:
        kwargs["tuning_spec"] = _structural_paged_spec(kwargs["tuning_spec"])
    calls = []
    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: arch)
    monkeypatch.setattr(au, "_run_3d_tiled", lambda **kw: calls.append(kw))
    from fractions import Fraction

    for value in (1, 1.0, Fraction(1, 1)):
        assert isinstance(value, Real)
        scales = dict.fromkeys(("k_scale", "v_scale", "out_scale"), value)
        au.run_unified_attention_torch(**dict(kwargs, softmax_scale=0.25, **scales))
    assert len(calls) == 3 and all(c["softmax_scale"] == 0.25 for c in calls)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_paged_default_rejects_boolean_scales(arch, monkeypatch):
    au = _forbid_paged_launch(arch, monkeypatch)
    kwargs = dict(_paged_runtime_kwargs(arch, 1), tuning_spec=None)
    for name in ("k_scale", "v_scale", "out_scale"):
        with pytest.raises(ValueError, match="scaling"):
            au.run_unified_attention_torch(**dict(kwargs, **{name: True}))
    calls = []
    monkeypatch.setattr(au, "_run_3d_tiled", lambda **kw: calls.append(kw))
    au.run_unified_attention_torch(**kwargs)
    assert len(calls) == 1


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("backend", ["auto", "3d"])
@pytest.mark.parametrize(
    "value",
    [
        False,
        True,
        None,
        [],
        {},
        "",
        "0.0",
        float("nan"),
        float("inf"),
        float("-inf"),
        -1,
        1,
    ],
)
def test_paged_default_rejects_invalid_softcap(arch, backend, value, monkeypatch):
    au = _forbid_paged_launch(arch, monkeypatch)
    kwargs = dict(_paged_runtime_kwargs(arch, 1), tuning_spec=None, backend=backend)
    with pytest.raises(ValueError, match="softcap"):
        au.run_unified_attention_torch(**dict(kwargs, softcap=value))


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("backend", ["auto", "3d"])
def test_paged_default_accepts_numeric_zero_softcap(arch, backend, monkeypatch):
    from fractions import Fraction

    from kernels.common import attention_unified as au

    kwargs = dict(_paged_runtime_kwargs(arch, 1), tuning_spec=None, backend=backend)
    calls = []
    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: arch)
    monkeypatch.setattr(au, "_run_3d_tiled", lambda **kw: calls.append(kw))
    for value in (0, 0.0, Fraction(0, 1)):
        au.run_unified_attention_torch(**dict(kwargs, softcap=value))
        assert calls[-1]["softcap"] is value
        assert calls[-1]["tuning_spec"] is None
    assert len(calls) == 3


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("page", [16, 32])
@pytest.mark.parametrize("retargeted", [False, True])
@pytest.mark.parametrize("allow_unsupported", [False, True])
def test_paged_direct_rejects_malformed_feature_contract(
    arch, page, retargeted, allow_unsupported, monkeypatch
):
    au = _forbid_paged_launch(arch, monkeypatch)
    kwargs = _paged_runtime_kwargs(arch, page)
    concrete = replace(kwargs["tuning_spec"], allow_unsupported=allow_unsupported)

    def forbidden_retarget(count):
        pytest.fail("invalid initial feature contract reached retargeting")

    for field in (
        "has_softcap",
        "use_sinks",
        "use_alibi",
        "use_qq_bias",
        "kv_storage_dtype",
    ):
        for value in ("missing", True, 1, None, "false"):
            if field == "kv_storage_dtype" and value is None:
                continue
            bad = _structural_paged_spec(concrete)
            if value == "missing":
                delattr(bad.kernel_spec, field)
            else:
                setattr(bad.kernel_spec, field, value)
            if retargeted:
                initial = _structural_paged_spec(concrete, retarget=lambda count: bad)
            else:
                bad.with_num_kv_blocks = forbidden_retarget
                initial = bad
            with pytest.raises(ValueError, match=field):
                au.run_unified_attention_torch(**dict(kwargs, tuning_spec=initial))


@pytest.mark.parametrize(
    "arch,page,policy",
    [
        ("gfx942", 1, "32"),
        ("gfx942", 16, "32"),
        ("gfx950", 1, "32"),
        ("gfx950", 16, "32"),
        ("gfx950", 64, "32"),
        ("gfx942", 32, "1x"),
        ("gfx942", 64, "1x"),
        ("gfx942", 64, "32"),
        ("gfx942", 32, "half"),
        ("gfx950", 32, "1x"),
        ("gfx950", 64, "1x"),
    ],
)
@pytest.mark.parametrize("structural", [False, True])
def test_paged_direct_preserves_generic_features(
    arch, page, policy, structural, monkeypatch, subtests
):
    from dispatch.attention.tuning_specs import (
        ExplicitAttention3DConfig,
        make_explicit_attention_3d_specs,
    )
    from kernels.common import attention_unified as au

    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: arch)
    calls = []
    monkeypatch.setattr(au, "_run_3d_tiled", lambda **kw: calls.append(kw))
    for feature in ("plain", "softcap", "sinks", "alibi", "qq_bias", "window", "fp8"):
        kwargs = _paged_runtime_kwargs(arch, page)
        updates, launch = {}, {}
        if feature == "softcap":
            updates["softcap"], launch["softcap"] = 2.0, 3.0
        elif feature in ("sinks", "alibi", "qq_bias"):
            updates[
                {"sinks": "use_sinks", "alibi": "use_alibi", "qq_bias": "use_qq_bias"}[
                    feature
                ]
            ] = True
            launch[
                {"sinks": "sinks", "alibi": "alibi_slopes", "qq_bias": "qq_bias"}[
                    feature
                ]
            ] = Tensor(
                (1, 1) if feature == "qq_bias" else (8,),
                "torch.float16" if feature == "sinks" else "torch.float32",
            )
            if feature == "qq_bias":
                launch["qq_bias_stride_0"] = 1
        elif feature == "window":
            updates["sliding_window"] = 16
        elif feature == "fp8":
            if page < 32 or (arch == "gfx950" and page == 64 and policy == "32"):
                continue
            updates.update(use_fp8=True, fp8_fnuz=arch == "gfx942")
            launch.update(k_scale=0.5, v_scale=2.0)
        problem = replace(kwargs["problem"], **updates)
        segment, reducer = make_explicit_attention_3d_specs(
            problem,
            ExplicitAttention3DConfig(num_segments=8, tile_policy=policy),
            arch=arch,
        )
        if feature == "fp8" and segment.uses_paged_gather:
            continue
        concrete = replace(
            kwargs["tuning_spec"], kernel_spec=segment, reduce_spec=reducer
        )
        spec = _structural_paged_spec(concrete) if structural else concrete
        valid = dict(kwargs, problem=problem, tuning_spec=spec, **launch)
        au.run_unified_attention_torch(**valid)
        assert (
            calls[-1]["tuning_spec"].kernel_spec is spec.kernel_spec or not structural
        )
        if feature in ("softcap", "sinks", "alibi", "qq_bias"):
            input_name = {
                "softcap": "softcap",
                "sinks": "sinks",
                "alibi": "alibi_slopes",
                "qq_bias": "qq_bias",
            }[feature]
            before = len(calls)
            for missing in (
                (0, -1, float("nan"), float("inf"), None, True)
                if feature == "softcap"
                else (None,)
            ):
                with pytest.raises(ValueError, match=input_name):
                    au.run_unified_attention_torch(
                        **dict(valid, **{input_name: missing})
                    )
            # A compiled enabled flag must also reject a disabled problem.
            with pytest.raises(ValueError, match="kernel_spec"):
                au.run_unified_attention_torch(**dict(valid, problem=kwargs["problem"]))
            field = {
                "softcap": "has_softcap",
                "sinks": "use_sinks",
                "alibi": "use_alibi",
                "qq_bias": "use_qq_bias",
            }[feature]
            returned = replace(concrete, kernel_spec=replace(segment, **{field: False}))
            initial = _structural_paged_spec(concrete, retarget=lambda count: returned)
            with pytest.raises(ValueError, match=field):
                au.run_unified_attention_torch(**dict(valid, tuning_spec=initial))
            assert len(calls) == before
        if feature == "fp8":
            before = len(calls)
            for name in ("k_scale", "v_scale"):
                for value in (
                    True,
                    False,
                    None,
                    [],
                    {},
                    "1.0",
                    float("nan"),
                    float("inf"),
                    float("-inf"),
                ):
                    with subtests.test(name=name, value=value):
                        with pytest.MonkeyPatch.context() as patch:

                            def forbidden_retarget(*args):
                                pytest.fail("malformed FP8 scale reached retargeting")

                            if structural:
                                patch.setattr(
                                    spec, "with_num_kv_blocks", forbidden_retarget
                                )
                            else:
                                patch.setattr(
                                    type(spec), "with_num_kv_blocks", forbidden_retarget
                                )
                            with pytest.raises(ValueError, match=name):
                                au.run_unified_attention_torch(
                                    **dict(valid, **{name: value})
                                )
                        assert len(calls) == before
            au.run_unified_attention_torch(**dict(valid, k_scale=1, v_scale=1.0))
            assert calls[-1]["k_scale"] == 1 and calls[-1]["v_scale"] == 1.0
            with pytest.raises(ValueError, match="out_scale"):
                au.run_unified_attention_torch(**dict(valid, out_scale=0.5))
            returned = replace(
                concrete, kernel_spec=replace(segment, kv_storage_dtype=None)
            )
            initial = _structural_paged_spec(concrete, retarget=lambda count: returned)
            with pytest.raises(ValueError, match="kv_storage_dtype"):
                au.run_unified_attention_torch(**dict(valid, tuning_spec=initial))
    assert len(calls) == (6 if segment.uses_paged_gather else 8)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_paged_direct_preserves_legacy_wide_addressing(arch, monkeypatch):
    from kernels.common import attention_unified as au

    kwargs = _paged_runtime_kwargs(arch, 32)
    pages = 0x80000000 // (32 * 2 * 64 * 2) + 1
    for name in ("k", "v"):
        kwargs[name] = Tensor((pages, 32, 2, 64))
    assert not kwargs["tuning_spec"].kernel_spec.uses_paged_gather
    calls = []
    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: arch)
    monkeypatch.setattr(au, "_run_3d_tiled", lambda **kw: calls.append(kw))
    if arch == "gfx942":
        with pytest.raises(NotImplementedError, match="use_i64_kv_addr"):
            au.run_unified_attention_torch(**kwargs)
        assert calls == []
        return
    au.run_unified_attention_torch(**kwargs)
    assert len(calls) == 1
    assert calls[0]["tuning_spec"].kernel_spec.use_i64_kv_addr


@pytest.mark.parametrize(
    "arch,page",
    [("gfx942", 1), ("gfx942", 16), ("gfx950", 1), ("gfx950", 16), ("gfx950", 64)],
)
def test_runtime_rejects_invalid_gather_before_launch(arch, page, monkeypatch):
    from kernels.common import attention_unified as au

    kwargs = _paged_runtime_kwargs(arch, page)
    spec, p = kwargs["tuning_spec"], kwargs["problem"]
    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: arch)

    def forbidden(**kwargs):
        pytest.fail("invalid gather reached launch")

    monkeypatch.setattr(au, "_run_3d_tiled", forbidden)
    with pytest.raises(ValueError, match="softcap"):
        au.run_unified_attention_torch(**dict(kwargs, softcap=1.0))
    bad = replace(
        spec,
        allow_unsupported=True,
        kernel_spec=replace(spec.kernel_spec, head_size=128),
    )
    with pytest.raises(ValueError, match="head_size"):
        au.run_unified_attention_torch(**dict(kwargs, tuning_spec=bad))
    with pytest.raises(ValueError, match="architecture"):
        au.run_unified_attention_torch(
            **dict(kwargs, problem=replace(p, clamp_arch="gfx1250"))
        )
    from types import SimpleNamespace

    incomplete = asdict(spec.kernel_spec)
    del incomplete["kv_storage_dtype"]
    incomplete["tile_size"] = spec.kernel_spec.tile_size
    incomplete["uses_paged_gather"] = True
    with pytest.raises(ValueError, match="kv_storage_dtype"):
        au.run_unified_attention_torch(
            **dict(
                kwargs,
                tuning_spec=replace(spec, kernel_spec=SimpleNamespace(**incomplete)),
            )
        )


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize("flag", ["missing", None, 0, "false"])
@pytest.mark.parametrize("allow_unsupported", [False, True])
def test_runtime_rejects_incomplete_loader_contract(
    arch, page, flag, allow_unsupported, monkeypatch
):
    from types import SimpleNamespace

    from kernels.common import attention_unified as au

    kwargs = _paged_runtime_kwargs(arch, page)
    concrete = replace(kwargs["tuning_spec"], allow_unsupported=allow_unsupported)
    snapshot = asdict(concrete.kernel_spec)
    if flag != "missing":
        snapshot["uses_paged_gather"] = flag

    class SnapshotAdapter:
        kernel_spec = SimpleNamespace(**snapshot)

        def __getattr__(self, name):
            return getattr(concrete, name)

        def with_num_kv_blocks(self, count):
            pytest.fail("incomplete loader contract reached address retargeting")

    def forbidden(**kwargs):
        pytest.fail("incomplete loader contract reached launch")

    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: arch)
    monkeypatch.setattr(au, "_run_3d_tiled", forbidden)
    with pytest.raises(ValueError, match="uses_paged_gather"):
        au.run_unified_attention_torch(**dict(kwargs, tuning_spec=SnapshotAdapter()))


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize("allow_unsupported", [False, True])
def test_runtime_rejects_geometry_inconsistent_loader_flag(
    arch, page, allow_unsupported, monkeypatch
):
    from types import SimpleNamespace

    from kernels.common import attention_unified as au

    kwargs = _paged_runtime_kwargs(arch, page)
    concrete = replace(kwargs["tuning_spec"], allow_unsupported=allow_unsupported)
    snapshot = asdict(concrete.kernel_spec)
    snapshot.update(
        tile_size=concrete.kernel_spec.tile_size,
        uses_paged_gather=not concrete.kernel_spec.uses_paged_gather,
    )

    class SnapshotAdapter:
        kernel_spec = SimpleNamespace(**snapshot)

        def __getattr__(self, name):
            return getattr(concrete, name)

        def with_num_kv_blocks(self, count):
            pytest.fail("inconsistent geometry reached retargeting")

    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: arch)
    with pytest.raises(ValueError, match="uses_paged_gather.*geometry"):
        au.run_unified_attention_torch(**dict(kwargs, tuning_spec=SnapshotAdapter()))


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("tile_size", "missing"),
        ("block_size", "missing"),
        ("tile_size_override", "missing"),
        ("kv_layout", "missing"),
        ("tile_size", 0),
        ("tile_size", True),
        ("tile_size", 32.0),
        ("block_size", 0),
        ("block_size", True),
        ("block_size", 16.0),
        ("block_size", 64),
        ("tile_size_override", 16),
        ("tile_size_override", False),
    ],
)
def test_runtime_rejects_malformed_paged_geometry(arch, field, value, monkeypatch):
    from types import SimpleNamespace

    from kernels.common import attention_unified as au

    kwargs = _paged_runtime_kwargs(arch, 16)
    concrete = kwargs["tuning_spec"]
    snapshot = asdict(concrete.kernel_spec)
    snapshot.update(tile_size=32, uses_paged_gather=True)
    if value == "missing":
        del snapshot[field]
    else:
        snapshot[field] = value

    class SnapshotAdapter:
        kernel_spec = SimpleNamespace(**snapshot)

        def __getattr__(self, name):
            return getattr(concrete, name)

        def with_num_kv_blocks(self, count):
            pytest.fail("malformed geometry reached retargeting")

    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: arch)
    with pytest.raises(ValueError, match=field):
        au.run_unified_attention_torch(**dict(kwargs, tuning_spec=SnapshotAdapter()))


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("allow_unsupported", [False, True])
@pytest.mark.parametrize(
    "change",
    [
        "arch",
        "path",
        "to_legacy",
        "to_gather",
        "dtype",
        "reduce",
        "segments",
        "workspace",
    ],
)
def test_runtime_revalidates_retargeted_paged_spec(
    arch, allow_unsupported, change, monkeypatch
):
    from kernels.common import attention_unified as au

    kwargs = _paged_runtime_kwargs(arch, 16)
    concrete = replace(kwargs["tuning_spec"], allow_unsupported=allow_unsupported)
    retargeted = concrete
    if change == "arch":
        retargeted = replace(concrete, arch="gfx1250")
    elif change == "path":
        retargeted = replace(concrete, path="2d")
    elif change in ("to_legacy", "to_gather"):
        legacy = replace(
            concrete, kernel_spec=replace(concrete.kernel_spec, tile_size_override=16)
        )
        if change == "to_legacy":
            retargeted = legacy
        else:
            concrete = legacy
    elif change == "dtype":
        retargeted = replace(
            concrete, kernel_spec=replace(concrete.kernel_spec, dtype="bf16")
        )
    elif change == "reduce":
        retargeted = replace(
            concrete,
            reduce_spec=replace(
                concrete.reduce_spec, num_segments=concrete.reduce_spec.num_segments + 1
            ),
        )
    elif change == "segments":
        retargeted = replace(
            concrete,
            kernel_spec=replace(concrete.kernel_spec, num_segments=256),
            reduce_spec=replace(concrete.reduce_spec, num_segments=256),
        )
    elif change == "workspace":
        batch = 32768
        kwargs["problem"] = replace(kwargs["problem"], num_seqs=batch, total_q=batch)
        for name in ("q", "out"):
            kwargs[name].shape = (batch, 8, 64)
        kwargs["cu_seqlens_q"].shape = (batch + 1,)
        kwargs["seqused_k"].shape = (batch,)
        kwargs["block_table"].shape = (batch, 3)
        concrete = replace(
            concrete,
            kernel_spec=replace(concrete.kernel_spec, num_seqs=batch, num_segments=16),
            reduce_spec=replace(concrete.reduce_spec, num_segments=16),
        )
        retargeted = replace(
            concrete,
            kernel_spec=replace(concrete.kernel_spec, num_segments=32),
            reduce_spec=replace(concrete.reduce_spec, num_segments=32),
        )

    class RetargetAdapter:
        def __getattr__(self, name):
            return getattr(concrete, name)

        def with_num_kv_blocks(self, count):
            return retargeted

    def forbidden(**kw):
        pytest.fail("changed retargeted spec reached launch")

    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: arch)
    monkeypatch.setattr(au, "_run_3d_tiled", forbidden)
    with pytest.raises(ValueError):
        au.run_unified_attention_torch(**dict(kwargs, tuning_spec=RetargetAdapter()))


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize("default_tile", [False, True])
@pytest.mark.parametrize("structural", [False, True])
def test_runtime_accepts_concrete_paged_loader_contract(
    arch, page, default_tile, structural, monkeypatch
):
    from types import SimpleNamespace

    from kernels.common import attention_unified as au

    kwargs = _paged_runtime_kwargs(arch, page)
    concrete = kwargs["tuning_spec"]
    if default_tile:
        concrete = replace(
            concrete, kernel_spec=replace(concrete.kernel_spec, tile_size_override=None)
        )
    snapshot = asdict(concrete.kernel_spec)
    snapshot.update(
        tile_size=concrete.kernel_spec.tile_size,
        uses_paged_gather=concrete.kernel_spec.uses_paged_gather,
    )

    class SnapshotAdapter:
        kernel_spec = SimpleNamespace(**snapshot)

        def __getattr__(self, name):
            return getattr(concrete, name)

    kwargs["tuning_spec"] = SnapshotAdapter() if structural else concrete
    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: arch)
    marker = object()
    monkeypatch.setattr(au, "_run_3d_tiled", lambda **kw: marker)
    assert au.run_unified_attention_torch(**kwargs) is marker


@pytest.mark.parametrize("dim", [64, 128, 256])
def test_gfx942_page_one_default_uses_async_gather(dim, monkeypatch):
    from kernels.common import attention_unified as au

    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: "gfx942")
    p, _ = inputs()
    p = replace(p, head_size=dim, clamp_arch="gfx942")
    spec = au._tiled_3d_spec_from_problem(p, "gfx942")
    assert spec.tile_size == 32 and spec.uses_paged_gather
    assert not spec.use_wide_kv_load


@pytest.mark.parametrize(
    "arch,path", [("gfx942", "2d"), ("gfx950", "2d"), ("gfx1250", "3d")]
)
def test_runtime_preserves_specs_without_gather_property(arch, path, monkeypatch):
    from types import SimpleNamespace

    from kernels.common import attention_unified as au

    kwargs = _paged_runtime_kwargs("gfx950", 32)
    p = replace(kwargs["problem"], clamp_arch=arch, dtype="bf16", num_query_heads=16)
    for name in ("q", "out"):
        kwargs[name].shape = (3, 16, 64)
    for name in ("q", "k", "v", "out"):
        kwargs[name].dtype = "torch.bfloat16"
    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: arch)
    kernel = (
        au._tiled_spec_from_problem(p, arch)
        if path == "2d"
        else au._tiled_3d_spec_from_problem(p, arch)
    )
    assert not hasattr(kernel, "uses_paged_gather")

    class ReachedLegacyPath(Exception):
        pass

    def reached(*args, **kwargs):
        raise ReachedLegacyPath

    spec = SimpleNamespace(
        arch=arch,
        path=path,
        kernel_spec=kernel,
        compile_backend="llvm",
        allow_unsupported=True,
        cache_key=lambda: (),
        build=reached,
        launch_grid=reached,
        launch_block=reached,
        with_num_kv_blocks=lambda count: spec,
    )
    monkeypatch.setattr(au, "_run_3d_tiled", reached)
    monkeypatch.setattr(au, "_get_2d_launcher", reached)
    with pytest.raises(ReachedLegacyPath):
        au.run_unified_attention_torch(
            **dict(kwargs, problem=p, tuning_spec=spec, backend="auto")
        )


def test_gather_workspace_bound():
    p, ts = inputs()
    with pytest.raises(ValueError, match="workspace"):
        validate_paged_decode(replace(p, num_seqs=65536, total_q=65536), ts)


_GATHER_GEOMETRIES = [
    ("gfx942", 1),
    ("gfx942", 16),
    ("gfx950", 1),
    ("gfx950", 16),
    ("gfx950", 64),
]


@pytest.mark.parametrize("arch,page", _GATHER_GEOMETRIES)
@pytest.mark.parametrize("value", ["missing", True, None, 0, 1, "false"])
@pytest.mark.parametrize("retarget", [False, True])
@pytest.mark.parametrize("allow_unsupported", [False, True])
def test_gather_requires_exact_false_wide_load(
    arch, page, value, retarget, allow_unsupported, monkeypatch
):
    from types import SimpleNamespace

    from kernels.common import attention_unified as au

    kwargs = _paged_runtime_kwargs(arch, page)
    concrete = replace(kwargs["tuning_spec"], allow_unsupported=allow_unsupported)
    snapshot = asdict(concrete.kernel_spec)
    snapshot.update(tile_size=32, uses_paged_gather=True)
    if value == "missing":
        del snapshot["use_wide_kv_load"]
    else:
        snapshot["use_wide_kv_load"] = value
    malformed = replace(concrete, kernel_spec=SimpleNamespace(**snapshot))

    class Adapter:
        def __getattr__(self, name):
            return getattr(concrete if retarget else malformed, name)

        def with_num_kv_blocks(self, count):
            if not retarget:
                pytest.fail("invalid wide-load metadata reached retargeting")
            return malformed

    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: arch)
    monkeypatch.setattr(
        au, "_run_3d_tiled", lambda **kw: pytest.fail("invalid spec reached launch")
    )
    with pytest.raises(ValueError, match="use_wide_kv_load=False"):
        au.run_unified_attention_torch(**dict(kwargs, tuning_spec=Adapter()))
    tensors = [
        kwargs[name]
        for name in ("q", "k", "v", "out", "cu_seqlens_q", "seqused_k", "block_table")
    ]
    with pytest.raises(ValueError, match="use_wide_kv_load=False"):
        validate_paged_decode(kwargs["problem"], tensors, malformed)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_paged_dispatch_replay_identity(arch):
    from dispatch.attention import (
        ATTENTION_EXECUTION_REGISTRY,
        attention_tuning_spec,
        dispatch_attention,
    )
    from rocke.dispatch.core import spec_identity

    kwargs = _paged_runtime_kwargs(arch, 1)
    from dispatch.attention.common import AttentionRequest

    p = kwargs["problem"]
    req = AttentionRequest(
        batch=p.num_seqs,
        nhead_q=p.num_query_heads,
        nhead_k=p.num_kv_heads,
        seqlen_q=1,
        seqlen_k=p.max_seqlen_k,
        hdim_q=64,
        hdim_v=64,
        arch=arch,
        dtype="fp16",
        kv_block_size=1,
        target_ctas=8,
    )
    result = dispatch_attention(req)
    replay = dispatch_attention(result.request)
    assert replay.candidate is result.candidate
    assert replay.spec == result.spec and replay.kernel_id == result.kernel_id
    assert result.request.tuning_id == "paged_t32" and result.request.tuning_knobs == ()
    assert attention_tuning_spec(req, "paged_decode_t32", "paged_t32") == result.spec
    assert result.candidate in ATTENTION_EXECUTION_REGISTRY.candidates()
    assert json.dumps(spec_identity(result.spec), sort_keys=True) == json.dumps(
        spec_identity(replay.spec), sort_keys=True
    )
    changed = dispatch_attention(replace(req, seqlen_k=129))
    assert changed.spec.tuning_id == result.spec.tuning_id
    assert changed.kernel_id != result.kernel_id
    for changes in ({"tuning_id": "unknown"}, {"tuning_knobs": {"waves_per_eu": 3}}):
        with pytest.raises(ValueError):
            dispatch_attention(replace(req, **changes))


class MetadataTensor(Tensor):
    def __init__(self, shape, values):
        super().__init__(shape, "torch.int32")
        self.values, self.reads, self.allow_read = values, [], False

    def stride(self, dim):
        return self.shape[1] if dim == 0 else 1

    def detach(self):
        return self

    def cpu(self):
        self.reads.append("cpu")
        assert self.allow_read, "unexpected device metadata read"
        return self

    def tolist(self):
        self.reads.append("tolist")
        return self.values


@pytest.fixture
def paged_binding_case():
    from dispatch.attention import AttentionRequest, dispatch_attention

    kwargs = _paged_runtime_kwargs("gfx950", 1)
    req = AttentionRequest(
        batch=3,
        nhead_q=8,
        nhead_k=2,
        seqlen_q=1,
        seqlen_k=33,
        hdim_q=64,
        hdim_v=64,
        arch="gfx950",
        dtype="fp16",
        kv_block_size=1,
        target_ctas=8,
    )
    result = dispatch_attention(req)
    tensors = {name: kwargs[name] for name in ("problem", "q", "k", "v", "out")}
    tensors.update(
        cu_seqlens_q=MetadataTensor((4,), [0, 1, 2, 3]),
        seqused_k=MetadataTensor((3,), [33, 17, 0]),
        block_table=MetadataTensor((3, 33), [list(range(33)) for _ in range(3)]),
    )
    return req, result, tensors


def test_paged_binding_default_keeps_live_metadata(paged_binding_case, monkeypatch):
    from kernels.common import attention_unified as au

    req, result, tensors = paged_binding_case
    calls = []
    monkeypatch.setattr(
        au, "run_unified_attention_torch", lambda **kw: calls.append(kw)
    )
    binding = result.candidate.bind_torch(req, result.spec, tensors)
    tensors["seqused_k"].values[:] = [1, 2, 3]
    tensors["block_table"].values[0][0] = 42
    binding.launch()
    for name in ("cu_seqlens_q", "seqused_k", "block_table"):
        assert calls[0][name] is tensors[name] and tensors[name].reads == []
    assert calls[0]["seqused_k"].values == [1, 2, 3]
    assert calls[0]["block_table"].values[0][0] == 42


@pytest.mark.parametrize("invalid", [None, "length", "page"])
def test_paged_binding_explicit_content_validation(paged_binding_case, invalid):
    req, result, tensors = paged_binding_case
    for name in ("cu_seqlens_q", "seqused_k", "block_table"):
        tensors[name].allow_read = True
    if invalid == "length":
        tensors["seqused_k"].values[0] = 34
    if invalid == "page":
        tensors["block_table"].values[0][0] = 99
    if invalid:
        with pytest.raises(ValueError, match="outside"):
            result.candidate.bind_torch(
                req, result.spec, tensors, unsafe_skip_paged_value_validation=False
            )
    else:
        result.candidate.bind_torch(
            req, result.spec, tensors, unsafe_skip_paged_value_validation=False
        )
    for name in ("cu_seqlens_q", "seqused_k", "block_table"):
        assert tensors[name].reads == ["cpu", "tolist"]


@pytest.mark.parametrize("invalid", ["shape", "dtype", "device"])
def test_paged_binding_skip_still_validates_structure(paged_binding_case, invalid):
    req, result, tensors = paged_binding_case
    if invalid == "shape":
        tensors["seqused_k"].shape = (2,)
    elif invalid == "dtype":
        tensors["seqused_k"].dtype = "torch.float32"
    else:
        tensors["seqused_k"].device = "cuda:1"
    with pytest.raises(ValueError):
        result.candidate.bind_torch(req, result.spec, tensors)


def _retarget_binding_case(case, arch, page):
    from dispatch.attention import dispatch_attention

    req, _, tensors = case
    req = replace(req, arch=arch, kv_block_size=page, algorithm="paged_decode_t32")
    tensors = dict(tensors)
    tensors["problem"] = replace(tensors["problem"], clamp_arch=arch, block_size=page)
    tensors["k"] = Tensor((99, page, 2, 64))
    tensors["v"] = Tensor((99, page, 2, 64))
    return req, dispatch_attention(req), tensors


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize(
    "feature", ["softcap", "use_sinks", "use_alibi", "use_qq_bias"]
)
def test_paged_binding_rejects_problem_feature_disagreement(
    paged_binding_case, arch, page, feature
):
    req, result, tensors = _retarget_binding_case(paged_binding_case, arch, page)
    tensors["problem"] = replace(
        tensors["problem"], **{feature: 1.0 if feature == "softcap" else True}
    )
    with pytest.raises(ValueError, match="disagrees"):
        result.candidate.bind_torch(req, result.spec, tensors)
    assert tensors["seqused_k"].reads == []


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize("feature", ["softcap", "sinks", "alibi_slopes", "qq_bias"])
@pytest.mark.parametrize("stage", ["tensor", "bind", "launch"])
def test_paged_binding_rejects_input_feature_disagreement(
    paged_binding_case, monkeypatch, arch, page, feature, stage
):
    from kernels.common import attention_unified as au

    req, result, tensors = _retarget_binding_case(paged_binding_case, arch, page)
    calls = []
    monkeypatch.setattr(
        au, "run_unified_attention_torch", lambda **kw: calls.append(kw)
    )
    override = {feature: 1.0 if feature == "softcap" else object()}
    with pytest.raises(ValueError, match="disagrees"):
        if stage == "tensor":
            result.candidate.bind_torch(req, result.spec, dict(tensors, **override))
        elif stage == "bind":
            result.candidate.bind_torch(req, result.spec, tensors, **override)
        else:
            binding = result.candidate.bind_torch(req, result.spec, tensors)
            binding.launch(**override)
    assert calls == []


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize("stage", ["tensor", "bind", "launch"])
@pytest.mark.parametrize(
    "value",
    [False, True, None, [], {}, "0.0", float("nan"), float("inf"), float("-inf")],
)
def test_paged_binding_rejects_malformed_softcap(
    paged_binding_case, monkeypatch, arch, page, stage, value
):
    from kernels.common import attention_unified as au

    req, result, tensors = _retarget_binding_case(paged_binding_case, arch, page)
    calls = []
    monkeypatch.setattr(
        au, "run_unified_attention_torch", lambda **kw: calls.append(kw)
    )
    override = {"softcap": value}
    with pytest.raises(ValueError, match="softcap"):
        if stage == "tensor":
            result.candidate.bind_torch(req, result.spec, dict(tensors, **override))
        elif stage == "bind":
            result.candidate.bind_torch(req, result.spec, tensors, **override)
        else:
            binding = result.candidate.bind_torch(req, result.spec, tensors)
            binding.launch(**override)
    assert calls == []
    for name in ("cu_seqlens_q", "seqused_k", "block_table"):
        assert tensors[name].reads == []


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize("stage", ["tensor", "bind", "launch"])
@pytest.mark.parametrize("value", [0, 0.0])
def test_paged_binding_accepts_numeric_zero_softcap(
    paged_binding_case, monkeypatch, arch, page, stage, value
):
    from kernels.common import attention_unified as au

    req, result, tensors = _retarget_binding_case(paged_binding_case, arch, page)
    calls = []
    monkeypatch.setattr(
        au, "run_unified_attention_torch", lambda **kw: calls.append(kw)
    )
    override = {"softcap": value}
    bound_tensors = dict(tensors, **override) if stage == "tensor" else tensors
    binding = result.candidate.bind_torch(
        req, result.spec, bound_tensors, **(override if stage == "bind" else {})
    )
    binding.launch(**(override if stage == "launch" else {}))
    assert len(calls) == 1 and calls[0]["softcap"] == 0.0
    for name in ("cu_seqlens_q", "seqused_k", "block_table"):
        assert tensors[name].reads == []


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("page", [1, 16, 32, 64])
def test_paged_binding_preserves_window_and_launch_overrides(
    paged_binding_case, monkeypatch, arch, page
):
    from dispatch.attention import dispatch_attention
    from kernels.common import attention_unified as au

    req, _, tensors = _retarget_binding_case(paged_binding_case, arch, page)
    from dispatch.attention.common import AttentionMaskType

    req = replace(
        req, sliding_window=17, mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL
    )
    result = dispatch_attention(req)
    tensors["problem"] = replace(tensors["problem"], sliding_window=17)
    calls = []
    monkeypatch.setattr(
        au, "run_unified_attention_torch", lambda **kw: calls.append(kw)
    )
    binding = result.candidate.bind_torch(req, result.spec, tensors)
    binding.launch(softmax_scale=0.25, stream=29, softcap=0.0, sinks=None)
    assert calls[0]["softmax_scale"] == 0.25 and calls[0]["stream"] == 29
    assert calls[0]["problem"].sliding_window == 17
    assert binding.grid == result.spec.launch_grid(tensors["problem"])
    assert binding.block == result.spec.launch_block()


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize("name", ["k_scale", "v_scale", "out_scale"])
@pytest.mark.parametrize("stage", ["tensor", "bind", "launch"])
def test_paged_binding_rejects_nonunit_scaling(
    paged_binding_case, monkeypatch, arch, page, name, stage
):
    from kernels.common import attention_unified as au

    req, result, tensors = _retarget_binding_case(paged_binding_case, arch, page)
    calls = []
    monkeypatch.setattr(
        au, "run_unified_attention_torch", lambda **kw: calls.append(kw)
    )
    override = {name: 2.0}
    with pytest.raises(ValueError, match=name):
        if stage == "tensor":
            result.candidate.bind_torch(req, result.spec, dict(tensors, **override))
        elif stage == "bind":
            result.candidate.bind_torch(req, result.spec, tensors, **override)
        else:
            binding = result.candidate.bind_torch(req, result.spec, tensors)
            binding.launch(**override)
    assert calls == []
    assert tensors["seqused_k"].reads == []


@pytest.mark.parametrize("name", ["k_scale", "v_scale", "out_scale"])
@pytest.mark.parametrize(
    "value", [None, True, False, "1.0", object(), float("nan"), float("inf")]
)
@pytest.mark.parametrize("stage", ["tensor", "bind", "launch"])
def test_paged_binding_rejects_malformed_scaling(
    paged_binding_case, name, value, stage, monkeypatch
):
    from kernels.common import attention_unified as au

    req, result, tensors = paged_binding_case
    calls = []
    monkeypatch.setattr(
        au, "run_unified_attention_torch", lambda **kw: calls.append(kw)
    )
    override = {name: value}
    with pytest.raises(ValueError, match=name):
        if stage == "tensor":
            result.candidate.bind_torch(req, result.spec, dict(tensors, **override))
        elif stage == "bind":
            result.candidate.bind_torch(req, result.spec, tensors, **override)
        else:
            binding = result.candidate.bind_torch(req, result.spec, tensors)
            binding.launch(**override)
    assert calls == []


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize("value", [1, 1.0])
def test_paged_binding_accepts_unit_scaling(
    paged_binding_case, monkeypatch, arch, page, value
):
    from kernels.common import attention_unified as au

    req, result, tensors = _retarget_binding_case(paged_binding_case, arch, page)
    calls = []
    monkeypatch.setattr(
        au, "run_unified_attention_torch", lambda **kw: calls.append(kw)
    )
    scales = dict.fromkeys(("k_scale", "v_scale", "out_scale"), value)
    binding = result.candidate.bind_torch(
        req, result.spec, dict(tensors, **scales), **scales, softmax_scale=0.125
    )
    binding.launch(**scales, softmax_scale=0.25, stream=29)
    assert len(calls) == 1
    assert calls[0]["softmax_scale"] == 0.25 and calls[0]["stream"] == 29
    assert tensors["seqused_k"].reads == []


@pytest.mark.parametrize("value", [None, "False", "true", 0, 1, object()])
def test_paged_binding_rejects_malformed_skip(paged_binding_case, value, monkeypatch):
    from dispatch.attention import bindings

    req, result, tensors = paged_binding_case

    def forbidden(*args, **kwargs):
        pytest.fail("malformed skip reached tensor validation")

    monkeypatch.setattr(bindings, "validate_tuning_attention_tensors", forbidden)
    with pytest.raises(TypeError, match="unsafe_skip_paged_value_validation"):
        result.candidate.bind_torch(
            req, result.spec, tensors, unsafe_skip_paged_value_validation=value
        )


@pytest.mark.parametrize("skip", [False, True])
def test_paged_binding_boolean_skip_policy(paged_binding_case, skip):
    req, result, tensors = paged_binding_case
    tensors["seqused_k"].values[0] = 34
    for name in ("cu_seqlens_q", "seqused_k", "block_table"):
        tensors[name].allow_read = True
    if skip:
        result.candidate.bind_torch(
            req, result.spec, tensors, unsafe_skip_paged_value_validation=True
        )
        assert tensors["seqused_k"].reads == []
    else:
        with pytest.raises(ValueError, match="outside"):
            result.candidate.bind_torch(
                req, result.spec, tensors, unsafe_skip_paged_value_validation=False
            )
        assert tensors["seqused_k"].reads == ["cpu", "tolist"]


_TYPED_SEGMENT_INTS = (
    "head_size",
    "block_size",
    "num_query_heads",
    "num_kv_heads",
    "sliding_window",
    "num_segments",
    "num_seqs",
    "waves_per_eu",
    "tile_size_override",
)
_TYPED_SEGMENT_BOOLS = (
    "use_sinks",
    "has_softcap",
    "use_alibi",
    "use_qq_bias",
    "use_invariant_hoist",
    "use_wide_kv_load",
    "use_i64_kv_addr",
)


class _RawGeometry:
    def __int__(self):
        pytest.fail("malformed geometry was coerced")

    def __index__(self):
        pytest.fail("malformed geometry was coerced")

    def __bool__(self):
        pytest.fail("malformed flag used truthiness")


def _typed_bad_values(boolean=False):
    from fractions import Fraction

    if boolean:
        return (0, 1, 0.0, 1.0, "false", None, _RawGeometry())
    return (
        True,
        False,
        32.0,
        32.9,
        "32",
        Fraction(32),
        float("nan"),
        float("inf"),
        None,
        _RawGeometry(),
    )


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("reducer", [False, True])
def test_tiled_spec_constructors_and_builders_reject_raw_types(
    arch, reducer, monkeypatch, subtests
):
    m = import_module(f"kernels.{arch}.attention_tiled_3d")
    kwargs = _paged_runtime_kwargs(arch, 32)
    concrete = (
        kwargs["tuning_spec"].reduce_spec
        if reducer
        else kwargs["tuning_spec"].kernel_spec
    )
    build = (
        m.build_unified_attention_reduce_tiled
        if reducer
        else m.build_unified_attention_3d_tiled
    )
    integer_fields = (
        ("head_size", "num_query_heads", "num_kv_heads", "num_segments", "waves_per_eu")
        if reducer
        else _TYPED_SEGMENT_INTS
    )
    monkeypatch.setattr(
        m, "IRBuilder", lambda *a, **k: pytest.fail("invalid spec reached IR creation")
    )
    for field in (*integer_fields, *(() if reducer else _TYPED_SEGMENT_BOOLS)):
        boolean = field in _TYPED_SEGMENT_BOOLS
        for value in _typed_bad_values(boolean):
            if field in ("waves_per_eu", "tile_size_override") and value is None:
                continue
            with subtests.test(field=field, value=repr(value)):
                with pytest.raises(ValueError, match=field):
                    replace(concrete, **{field: value})
                # Recheck builders, independent of dataclass construction.
                bad = replace(concrete)
                object.__setattr__(bad, field, value)
                with pytest.raises(ValueError, match=field):
                    build(bad, arch=arch)
    for field in ("dtype",) if reducer else ("dtype", "kv_layout", "kv_storage_dtype"):
        with subtests.test(field=field), pytest.raises(ValueError, match=field):
            replace(concrete, **{field: 1})


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("page", [1, 32])
@pytest.mark.parametrize("retargeted", [False, True])
@pytest.mark.parametrize("reducer", [False, True])
def test_runtime_rejects_typed_segment_and_reducer_before_use(
    arch, page, retargeted, reducer, monkeypatch, subtests
):
    from types import SimpleNamespace
    from kernels.common import attention_unified as au

    kwargs = _paged_runtime_kwargs(arch, page)
    concrete = kwargs["tuning_spec"]
    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: arch)
    monkeypatch.setattr(
        au, "_run_3d_tiled", lambda **k: pytest.fail("invalid spec reached launch")
    )
    fields = (
        "head_size",
        "num_query_heads",
        "num_kv_heads",
        "num_segments",
        "waves_per_eu",
    )
    if not reducer:
        fields += ("num_seqs", "use_i64_kv_addr", "use_invariant_hoist")
    for field in fields:
        value = 0 if field.startswith("use_") else 8.0
        bad = _structural_paged_spec(concrete)
        if reducer:
            bad.reduce_spec = SimpleNamespace(**asdict(concrete.reduce_spec))
            setattr(bad.reduce_spec, field, value)
        else:
            setattr(bad.kernel_spec, field, value)
        if retargeted:
            initial = _structural_paged_spec(concrete, retarget=lambda count: bad)
        else:
            initial = bad
            bad.with_num_kv_blocks = lambda count: pytest.fail(
                "invalid initial spec reached retarget"
            )
        initial.allow_unsupported = True
        with subtests.test(field=field), pytest.raises(ValueError, match=field):
            au.run_unified_attention_torch(**dict(kwargs, tuning_spec=initial))


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_strided_structural_types_before_lds_and_launch(arch, monkeypatch, subtests):
    from types import SimpleNamespace
    from kernels.common import attention_unified as au

    kwargs = _paged_runtime_kwargs(arch, 32)
    concrete = kwargs["tuning_spec"]
    segment = replace(concrete.kernel_spec, kv_layout="strided")
    monkeypatch.setattr(
        au,
        "_validate_strided_3d_lds",
        lambda *a, **k: pytest.fail("bad spec reached LDS arithmetic"),
    )
    for reducer in (False, True):
        for field, value in (("num_segments", 8.0), ("head_size", 64.0)):
            bad = SimpleNamespace(
                arch=arch,
                path="3d",
                kernel_spec=replace(segment),
                reduce_spec=replace(concrete.reduce_spec),
            )
            target = bad.reduce_spec if reducer else bad.kernel_spec
            object.__setattr__(target, field, value)
            with (
                subtests.test(reducer=reducer, field=field),
                pytest.raises(ValueError, match=field),
            ):
                au._validate_strided_3d_spec(kwargs["problem"], bad)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("operation", ["lower_llvm", "serialize_ir", "verify"])
def test_native_tiled_dictionary_types_before_casts(
    arch, operation, monkeypatch, subtests
):
    engine = pytest.importorskip("rocke_engine")
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", "llvm23")
    fn = getattr(engine, f"{arch}_attention_tiled_3d_{operation}")
    for field in (*_TYPED_SEGMENT_INTS, *_TYPED_SEGMENT_BOOLS):
        for value in _typed_bad_values(field in _TYPED_SEGMENT_BOOLS):
            if value is None:  # The native partial dictionary declares None as default.
                continue
            with (
                subtests.test(field=field, value=repr(value)),
                pytest.raises(ValueError, match=field),
            ):
                fn({field: value}, arch)
    for field in ("dtype", "kv_storage_dtype", "kv_layout"):
        with subtests.test(field=field), pytest.raises(ValueError, match=field):
            fn({field: 1}, arch)
    with pytest.raises((RuntimeError, TypeError, OverflowError, ValueError)):
        fn({"block_size": 1 << 100}, arch)
    # Existing partial dictionary defaults must remain byte-for-byte stable.
    required = dict(
        head_size=64,
        block_size=32,
        num_query_heads=8,
        num_kv_heads=2,
        num_segments=8,
        dtype="fp16",
        use_sinks=False,
        has_softcap=False,
        sliding_window=0,
    )
    expected = fn(required, arch)
    defaults = {
        name: None
        for name in (
            "waves_per_eu",
            "tile_size_override",
            "use_alibi",
            "use_qq_bias",
            "num_seqs",
            "use_invariant_hoist",
            "use_wide_kv_load",
            "use_i64_kv_addr",
            "kv_storage_dtype",
            "kv_layout",
        )
    }
    assert fn(dict(required, **defaults), arch) == expected
    # Omission/None on required numeric fields retains zero and still rejects.
    with pytest.raises(RuntimeError):
        fn({}, arch)
    with pytest.raises(RuntimeError):
        fn(dict(required, block_size=None), arch)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("reducer", [False, True])
def test_builder_rejects_missing_declared_spec_fields(
    arch, reducer, monkeypatch, subtests
):
    from types import SimpleNamespace

    m = import_module(f"kernels.{arch}.attention_tiled_3d")
    concrete = _paged_runtime_kwargs(arch, 32)["tuning_spec"]
    spec = concrete.reduce_spec if reducer else concrete.kernel_spec
    build = (
        m.build_unified_attention_reduce_tiled
        if reducer
        else m.build_unified_attention_3d_tiled
    )
    monkeypatch.setattr(
        m, "IRBuilder", lambda *a, **k: pytest.fail("missing field reached build")
    )
    for field in asdict(spec):
        raw = SimpleNamespace(**asdict(spec))
        delattr(raw, field)
        with subtests.test(field=field), pytest.raises(ValueError, match=field):
            build(raw, arch=arch)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize("stage", ["tensor", "bind", "launch"])
def test_paged_binding_preserves_enabled_features(
    paged_binding_case, monkeypatch, arch, page, stage
):
    from dispatch.attention import dispatch_attention
    from dispatch.attention.common import _problem
    from kernels.common import attention_unified as au

    req, _, tensors = _retarget_binding_case(paged_binding_case, arch, page)
    req = replace(req, softcap=2.0, use_sinks=True, use_alibi=True, use_qq_bias=True)
    result = dispatch_attention(req)
    tensors["problem"] = _problem(req)
    features = dict(
        softcap=3.0,
        sinks=Tensor((8,)),
        alibi_slopes=Tensor((8,), "torch.float32"),
        qq_bias=Tensor((1, 1), "torch.float32"),
    )
    calls = []
    monkeypatch.setattr(
        au, "run_unified_attention_torch", lambda **kw: calls.append(kw)
    )
    if stage == "tensor":
        binding = result.candidate.bind_torch(
            req, result.spec, dict(tensors, **features)
        )
        binding.launch(qq_bias_stride_0=1)
    elif stage == "bind":
        binding = result.candidate.bind_torch(
            req, result.spec, tensors, **features, qq_bias_stride_0=1
        )
        binding.launch()
    else:
        binding = result.candidate.bind_torch(
            req, result.spec, dict(tensors, **features)
        )
        features = dict(
            softcap=4.0,
            sinks=Tensor((8,)),
            alibi_slopes=Tensor((8,), "torch.float32"),
            qq_bias=Tensor((1, 1), "torch.float32"),
        )
        binding.launch(**features, qq_bias_stride_0=1)
    assert len(calls) == 1
    assert all(calls[0][name] is value for name, value in features.items())
    assert calls[0]["qq_bias_stride_0"] == 1
    assert tensors["seqused_k"].reads == []


@pytest.mark.parametrize("stage", ["tensor", "bind", "launch"])
@pytest.mark.parametrize("value", [True, False, 1.5, "1", None, -1])
def test_paged_binding_rejects_raw_bias_stride(paged_binding_case, stage, value):
    req, result, tensors = paged_binding_case
    override = dict(qq_bias_stride_0=value)
    with pytest.raises(ValueError, match="qq_bias_stride_0"):
        if stage == "tensor":
            result.candidate.bind_torch(req, result.spec, dict(tensors, **override))
        elif stage == "bind":
            result.candidate.bind_torch(req, result.spec, tensors, **override)
        else:
            result.candidate.bind_torch(req, result.spec, tensors).launch(**override)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "feature", ["softcap", "sinks", "alibi", "qq_bias", "combined"]
)
@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
def test_paged_feature_native_parity(arch, page, dtype, feature, flavor, monkeypatch):
    engine = pytest.importorskip("rocke_engine")
    from dispatch.attention import AttentionRequest, dispatch_attention
    from dispatch.attention.common import AttentionMaskType
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", flavor)
    enabled = lambda name: feature in (name, "combined")
    request = AttentionRequest(
        batch=3,
        nhead_q=8,
        nhead_k=2,
        seqlen_q=1,
        seqlen_k=97,
        hdim_q=128,
        hdim_v=128,
        arch=arch,
        dtype=dtype,
        kv_block_size=page,
        algorithm="paged_decode_t32",
        target_ctas=8,
        sliding_window=33,
        mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
        softcap=1.5 if enabled("softcap") else 0.0,
        use_sinks=enabled("sinks"),
        use_alibi=enabled("alibi"),
        use_qq_bias=enabled("qq_bias"),
    )
    spec = dispatch_attention(request).spec
    module = import_module(f"kernels.{arch}.attention_tiled_3d")
    expected = _lower_kernel_to_llvm_python(
        module.build_unified_attention_3d_tiled(spec.kernel_spec, arch=arch), arch=arch
    )
    if arch == "gfx950":
        expected += _lower_kernel_to_llvm_python(
            module.build_unified_attention_reduce_tiled(spec.reduce_spec, arch=arch),
            arch=arch,
        )
    assert (
        getattr(engine, f"{arch}_attention_tiled_3d_lower_llvm")(
            asdict(spec.kernel_spec), arch
        )
        == expected
    )


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("tile", [16, 32])
def test_structural_subpage_loader_contract(arch, tile):
    from types import SimpleNamespace

    from kernels.common.attention_paged_decode import validate_paged_loader_geometry

    kwargs = _paged_runtime_kwargs(arch, 64)
    concrete = kwargs["tuning_spec"]
    snapshot = asdict(concrete.kernel_spec)
    snapshot.update(
        tile_size=tile, tile_size_override=tile, uses_paged_gather=arch == "gfx950"
    )
    structural = _structural_paged_spec(
        concrete, kernel_spec=SimpleNamespace(**snapshot)
    )
    assert not hasattr(structural.kernel_spec, "kv_loader_kind")
    if arch == "gfx950" and tile == 16:
        with pytest.raises(ValueError, match="paged gather requires tile_size=32"):
            validate_paged_loader_geometry(kwargs["problem"], structural)
    else:
        assert validate_paged_loader_geometry(kwargs["problem"], structural) == (
            64,
            tile,
            arch == "gfx950",
        )
        structural.kernel_spec.uses_paged_gather = arch != "gfx950"
        with pytest.raises(ValueError, match="disagrees with loader geometry"):
            validate_paged_loader_geometry(kwargs["problem"], structural)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("field", ["block_size", "tile_size", "tile_size_override"])
@pytest.mark.parametrize("value", [True, "32", 32.0])
def test_structural_raw_geometry_checked_before_loader_resolution(
    arch, field, value, monkeypatch
):
    from kernels.common import attention_paged_decode as admission

    kwargs = _paged_runtime_kwargs(arch, 16)
    structural = _structural_paged_spec(kwargs["tuning_spec"])
    setattr(structural.kernel_spec, field, value)

    def forbidden(*args):
        pytest.fail("malformed raw geometry reached loader resolution")

    monkeypatch.setattr(admission, "resolve_attention_kv_loader", forbidden)
    with pytest.raises((TypeError, ValueError)):
        admission.validate_paged_loader_geometry(kwargs["problem"], structural)
