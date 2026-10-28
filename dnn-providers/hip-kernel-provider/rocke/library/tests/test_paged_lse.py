# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Combined containment/LSE admission, argument ABI and strict native lowering."""

from dataclasses import asdict, replace
from importlib import import_module
from types import SimpleNamespace

import pytest

from dispatch.attention import AttentionRequest, dispatch_attention
from dispatch.attention import bindings
from dispatch.attention.common import _problem
from kernels.common import attention_unified as au
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python, lower_kernel_to_llvm
from tests._paged_decode_support import Tensor, inputs


def request(arch, page=1, **kwargs):
    return AttentionRequest(
        batch=3,
        nhead_q=8,
        nhead_k=2,
        seqlen_q=1,
        seqlen_k=33,
        hdim_q=64,
        hdim_v=64,
        dtype="fp16",
        arch=arch,
        kv_block_size=page,
        algorithm="paged_decode_t32",
        **kwargs,
    )


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("page", [1, 16, 32, 64])
def test_real_selection_default_and_retarget_keep_lse(arch, page):
    req = request(arch, page, emit_lse=True)
    selected = dispatch_attention(req)
    assert selected.candidate.name == "attention_paged_decode_t32"
    assert selected.spec.kernel_spec.metadata_containment
    assert selected.spec.reduce_spec.metadata_containment
    assert selected.spec.reduce_spec.emit_lse
    retargeted = selected.spec.with_num_kv_blocks(99)
    assert retargeted.reduce_spec == selected.spec.reduce_spec
    assert retargeted.kernel_spec.metadata_containment
    off = dispatch_attention(replace(req, emit_lse=False)).spec
    assert off.cache_key() != selected.spec.cache_key()
    if page == 1:
        auto = dispatch_attention(replace(req, algorithm="auto"))
        assert auto.candidate.name == selected.candidate.name
        segment, reducer = au._default_3d_specs_from_problem(_problem(req), arch=arch)
        assert segment.metadata_containment and reducer.metadata_containment
        assert reducer.emit_lse


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("lse", [False, True])
def test_all_four_reducer_abis(arch, bounded, lse):
    mod = import_module(f"kernels.{arch}.attention_tiled_3d")
    spec = mod.UnifiedAttentionReduceTiledSpec(
        head_size=64,
        num_query_heads=8,
        num_kv_heads=2,
        dtype="fp16",
        num_segments=4,
        metadata_containment=bounded,
        emit_lse=lse,
    )
    kernel = mod.build_unified_attention_reduce_tiled(spec, arch=arch)
    expected = [
        "output_ptr",
        "segm_output_ptr",
        "segm_max_ptr",
        "segm_expsum_ptr",
        "seq_lens_ptr",
    ]
    if bounded:
        expected += ["metadata_status_ptr", "num_seqs"]
    if lse:
        expected += ["lse_ptr"]
    assert [p.name for p in kernel.params] == expected
    assert [
        p["name"]
        for p in au._reduce_signature(
            "fp16", metadata_containment=bounded, emit_lse=lse
        )
    ] == expected
    assert ("bounds_v1" in spec.kernel_name()) == bounded
    assert spec.kernel_name().endswith("_lse") == lse


@pytest.mark.parametrize("page,policy,bounded", [(32, "1x", False), (16, "2x", True)])
@pytest.mark.parametrize("lse", [False, True])
def test_tuning_dispatch_signature_matches_segment(page, policy, bounded, lse):
    from dispatch.attention.gfx950_unified import GFX950_TUNING_VARIANTS

    variant = next(
        v
        for v in GFX950_TUNING_VARIANTS
        if v.path == "3d" and v.tile_policy == policy and v.num_segments == 4
    )
    req = replace(
        request("gfx950", page, emit_lse=lse),
        algorithm="unified_tuning",
        spec_id=variant.spec_id,
    )
    selected = dispatch_attention(req)
    assert selected.spec.kernel_spec.metadata_containment == bounded
    segment, _ = selected.spec.build("gfx950")
    assert [p["name"] for p in selected.signature] == [p.name for p in segment.params]


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("bad", [None, 0, 1, "false"])
def test_structural_reducer_rejects_malformed_lse(arch, bad):
    req = request(arch)
    spec = dispatch_attention(req).spec
    reducer = SimpleNamespace(**asdict(spec.reduce_spec))
    reducer.emit_lse = bad
    with pytest.raises(ValueError, match="emit_lse must be a bool"):
        bindings.validate_tuning_attention_contract(
            req, _problem(req), replace(spec, reduce_spec=reducer)
        )


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_lse_off_structural_compatibility_and_on_omission(arch):
    req = request(arch)
    spec = dispatch_attention(req).spec
    fields = asdict(spec.reduce_spec)
    del fields["emit_lse"]
    structural = replace(spec, reduce_spec=SimpleNamespace(**fields))
    bindings.validate_tuning_attention_contract(req, _problem(req), structural)
    on = replace(req, emit_lse=True)
    with pytest.raises(ValueError, match="LSE output disagrees"):
        bindings.validate_tuning_attention_contract(on, _problem(on), structural)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_binding_forwards_and_requires_exact_lse(arch, monkeypatch):
    req = request(arch, emit_lse=True)
    selected = dispatch_attention(req)
    problem, ts = inputs()
    problem = replace(problem, clamp_arch=arch, emit_lse=True)
    tensors = dict(
        zip(("q", "k", "v", "out", "cu_seqlens_q", "seqused_k", "block_table"), ts)
    )
    tensors["problem"] = problem
    with pytest.raises(ValueError, match="requires tensors"):
        selected.candidate.bind_torch(req, selected.spec, tensors)
    tensors["lse"] = Tensor((3, 8, 1), "torch.float32")
    monkeypatch.setattr(
        bindings, "validate_tuning_attention_tensors", lambda *a, **kw: 99
    )
    calls = []
    monkeypatch.setattr(
        au, "run_unified_attention_torch", lambda **kw: calls.append(kw)
    )
    bound = selected.candidate.bind_torch(req, selected.spec, tensors)
    bound.launch()
    assert calls[-1]["lse"] is tensors["lse"]
    assert calls[-1]["tuning_spec"].reduce_spec.emit_lse
    assert calls[-1]["tuning_spec"].kernel_spec.metadata_containment
    off = dispatch_attention(replace(req, emit_lse=False))
    tensors["problem"] = replace(problem, emit_lse=False)
    with pytest.raises(ValueError, match="emit_lse is off"):
        off.candidate.bind_torch(replace(req, emit_lse=False), off.spec, tensors)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_lse_with_sinks_refused(arch):
    with pytest.raises(ValueError, match="LSE|lse"):
        dispatch_attention(request(arch, emit_lse=True, use_sinks=True))


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("dtype,dim", [("fp16", 64), ("bf16", 128), ("bf16", 256)])
@pytest.mark.parametrize("flavor", ["llvm20", "llvm22", "llvm23"])
def test_strict_cpp_lowering_python_authored_bounded_lse(
    arch, dtype, dim, flavor, monkeypatch
):
    engine = pytest.importorskip("rocke_engine")
    monkeypatch.setenv("ROCKE_CPP_STRICT", "1")
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", flavor)
    mod = import_module(f"kernels.{arch}.attention_tiled_3d")
    spec = mod.UnifiedAttentionReduceTiledSpec(
        head_size=dim,
        num_query_heads=8,
        num_kv_heads=2,
        dtype=dtype,
        num_segments=4,
        metadata_containment=True,
        emit_lse=True,
    )
    kernel = mod.build_unified_attention_reduce_tiled(spec, arch=arch)
    expected = _lower_kernel_to_llvm_python(kernel, arch=arch)
    monkeypatch.setenv("ROCKE_BACKEND", "cpp")
    assert lower_kernel_to_llvm(kernel, arch=arch) == expected
    assert engine.build_id


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("lse", [True, None, 0, 1, "false"])
def test_native_builders_reject_enabled_or_malformed_lse(arch, lse):
    engine = pytest.importorskip("rocke_engine")
    mod = import_module(f"kernels.{arch}.attention_tiled_3d")
    reducer = mod.UnifiedAttentionReduceTiledSpec(
        head_size=64,
        num_query_heads=8,
        num_kv_heads=2,
        dtype="bf16",
        num_segments=4,
        metadata_containment=True,
    )
    fields = asdict(reducer)
    fields["emit_lse"] = lse
    with pytest.raises(ValueError, match="emit_lse"):
        engine.paged_decode_reduce_lower_llvm(fields, arch)
    selected = dispatch_attention(request(arch)).spec
    fields = asdict(selected.kernel_spec)
    fields["emit_lse"] = lse
    with pytest.raises(ValueError, match="emit_lse"):
        getattr(engine, f"{arch}_attention_tiled_3d_lower_llvm")(fields, arch)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_windowed_causal_lse_selection(arch):
    selected = dispatch_attention(
        request(arch, emit_lse=True, mask_type=2, sliding_window=33)
    )
    assert selected.spec.reduce_spec.emit_lse
    assert selected.spec.kernel_spec.sliding_window == 33
    assert not getattr(selected.spec.kernel_spec, "causal_top_left", False)
    assert getattr(selected.spec.kernel_spec, "right_bound", 0) == 0


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_malformed_lse_cannot_enter_cache_identity(arch):
    spec = dispatch_attention(request(arch)).spec
    reducer = replace(spec.reduce_spec)
    object.__setattr__(reducer, "emit_lse", 1)
    with pytest.raises(ValueError, match="emit_lse must be a bool"):
        replace(spec, reduce_spec=reducer).cache_key()
