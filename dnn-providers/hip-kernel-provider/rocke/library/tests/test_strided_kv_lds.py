# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Static LDS admission agrees with the emitted tiled split-KV pools."""

from dataclasses import replace
from importlib import import_module
import re
from types import SimpleNamespace

import pytest

from kernels.common import attention_unified as au
from dispatch.attention import AttentionRequest, dispatch_attention
from dispatch.attention.common import _problem
from dispatch.attention.strided_decode import make_candidate
from rocke.core.arch import ArchTarget
from rocke.core.ir import F32, FP4E2M1
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python


def _request(arch="gfx942", dtype="bf16", head=256, block=64):
    return AttentionRequest(
        batch=3,
        nhead_q=8,
        nhead_k=2,
        seqlen_q=1,
        seqlen_k=65,
        hdim_q=head,
        hdim_v=head,
        arch=arch,
        dtype=dtype,
        kv_layout="strided",
        kv_block_size=block,
        target_ctas=8,
    )


def _explicit(req, tile):
    supported = dispatch_attention(replace(req, kv_block_size=32)).spec
    return replace(
        supported,
        kernel_spec=replace(
            supported.kernel_spec, block_size=req.kv_block_size, tile_size_override=tile
        ),
    )


def _run(problem, spec=None):
    return au.run_unified_attention_torch(
        problem=problem,
        q=object(),
        k=object(),
        v=object(),
        out=object(),
        cu_seqlens_q=object(),
        seqused_k=object(),
        block_table=None,
        softmax_scale=0.125,
        softcap=0,
        kv_layout="strided",
        tuning_spec=spec,
    )


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_oversized_default_rejected_at_every_strided_entry(dtype, monkeypatch):
    req = _request(dtype=dtype)
    problem = _problem(req)
    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: "gfx942")
    monkeypatch.setattr(
        au, "compile_kernel", lambda *a, **kw: pytest.fail("must reject before compile")
    )
    ok, reason = make_candidate().admits(req)
    assert not ok
    assert "74752 B LDS" in reason and "65536 B" in reason
    assert "tile_size=32" in reason
    for pinned in (False, True):
        request = replace(req, spec_id="strided_decode") if pinned else req
        with pytest.raises(ValueError, match="74752 B LDS"):
            dispatch_attention(request)
    with pytest.raises(ValueError, match="74752 B LDS"):
        au._strided_3d_specs_from_problem(problem, arch="gfx942")
    with pytest.raises(ValueError, match="74752 B LDS"):
        _run(problem)
    for allow_unsupported in (False, True):
        spec = replace(_explicit(req, 32), allow_unsupported=allow_unsupported)
        with pytest.raises(ValueError, match="74752 B LDS"):
            au._validate_strided_3d_spec(problem, spec)
        with pytest.raises(ValueError, match="74752 B LDS"):
            _run(problem, spec)
        with pytest.raises(ValueError, match="74752 B LDS"):
            make_candidate().bind_torch(req, spec, {})


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_explicit_smaller_tile_uses_actual_footprint(dtype, monkeypatch):
    from kernels.common.attention_kv_cache import StridedKvCacheLayout

    req = _request(dtype=dtype)
    spec = _explicit(req, 16)
    problem = _problem(req)
    au._validate_strided_3d_spec(problem, spec)
    assert au._tiled_3d_segment_lds_bytes(spec.kernel_spec) == 41472
    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: "gfx942")

    def stop_at_tensor_validation(*args):
        raise RuntimeError("passed LDS admission")

    monkeypatch.setattr(StridedKvCacheLayout, "from_tensors", stop_at_tensor_validation)
    with pytest.raises(RuntimeError, match="passed LDS admission"):
        _run(problem, spec)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("head", [64, 128, 256])
@pytest.mark.parametrize("block", [16, 32, 64])
def test_lds_estimate_matches_lowered_pools(arch, dtype, head, block, monkeypatch):
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", "llvm20")
    req = _request(arch, dtype, head, block)
    if (arch, head, block) == ("gfx942", 256, 64):
        # Raw emission remains available as a negative control, including goldens.
        spec = _explicit(req, 32)
    else:
        assert make_candidate().admits(req)[0]
        spec = dispatch_attention(req).spec
    module = import_module(f"kernels.{arch}.attention_tiled_3d")
    for kernel, estimated in (
        (
            module.build_unified_attention_3d_tiled(spec.kernel_spec, arch=arch),
            au._tiled_3d_segment_lds_bytes(spec.kernel_spec),
        ),
        (
            module.build_unified_attention_reduce_tiled(spec.reduce_spec, arch=arch),
            au._tiled_3d_reduce_lds_bytes(spec.reduce_spec),
        ),
    ):
        llvm = _lower_kernel_to_llvm_python(kernel, arch=arch)
        pools = re.findall(r"addrspace\(3\) global \[(\d+) x i8\]", llvm)
        assert pools == [str(estimated)]


def test_lds_width_comes_from_allocation_dtype():
    spec = SimpleNamespace(block_m=16, head_size=256, tile_size=32, dtype_ir=F32)
    assert au._tiled_3d_segment_lds_bytes(spec) == 149504
    spec.dtype_ir = FP4E2M1
    with pytest.raises(ValueError, match="byte-addressable"):
        au._tiled_3d_segment_lds_bytes(spec)


def test_reducer_budget_is_independent_and_allows_exact_capacity():
    req = _request(block=32)
    spec = dispatch_attention(req).spec
    target = ArchTarget.from_gfx(req.arch)
    from rocke.core.dtypes import dtype_info

    segments = target.lds_capacity_bytes // (dtype_info(F32.name).encoded_bits // 8)
    for count, fits in ((segments, True), (segments + 1, False)):
        candidate = replace(
            spec,
            kernel_spec=replace(spec.kernel_spec, num_segments=count),
            reduce_spec=replace(spec.reduce_spec, num_segments=count),
        )
        if fits:
            au._validate_strided_3d_spec(_problem(req), candidate)
        else:
            with pytest.raises(ValueError, match="reducer requires.*capacity"):
                au._validate_strided_3d_spec(_problem(req), candidate)
