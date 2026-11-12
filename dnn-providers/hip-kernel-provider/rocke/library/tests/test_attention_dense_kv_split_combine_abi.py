# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The dense segment-combine launch packs every argument the tiled reduce declares.

``launch_segment_combine`` launches the tiled 3D reduce kernel. That kernel reads
``num_query_heads`` as a kernarg (``RUNTIME_KERNARG_FIELDS``), so a launcher built
from the pointer-only reduce signature leaves the head count, and with it the
reduce's token strides, undefined. This captures the launcher's signature and
values on the host and checks them against the reduce kernel's params.

``test_combine_abi_matches_reduce_params`` needs neither torch nor a GPU, so it
runs in every CPU lane; the launch test runs where torch is installed.
"""

from __future__ import annotations

import pytest
from kernels.common import attention_dense_kv_split as kv_split
from kernels.gfx942.attention_dense import Gfx942AttentionDenseSpec
from kernels.gfx950.attention_dense import Gfx950AttentionDenseSpec

_ARCHES = [("gfx942", Gfx942AttentionDenseSpec), ("gfx950", Gfx950AttentionDenseSpec)]


def _dense_spec(spec_cls):
    return spec_cls(
        batch=2,
        seqlen_q=256,
        seqlen_kv=2048,
        num_query_heads=16,
        num_kv_heads=2,
        head_size=128,
        causal=True,
        dtype="bf16",
        kv_segments=4,
    )


@pytest.mark.parametrize("arch, spec_cls", _ARCHES)
def test_combine_abi_matches_reduce_params(arch, spec_cls):
    """The signature the launcher is built from names the reduce kernel's params
    in order, and every runtime scalar among them gets the combine's value."""
    spec = _dense_spec(spec_cls)
    red, signature, scalars = kv_split.combine_launch_abi(spec, arch)
    _, build = kv_split._combine_impl(arch)
    params = [p.name for p in build(red, arch=arch).params]
    assert params == [a["name"] for a in signature]
    assert scalars == {"num_query_heads": spec.num_query_heads}
    pointers = {"output_ptr", "seq_lens_ptr"} | set(kv_split.SEGMENT_PARAMS)
    assert set(params) == pointers | set(scalars)


class _Art:
    hsaco = b""

    def __init__(self, kernel):
        self.kernel_name = kernel.name


@pytest.mark.parametrize("arch, spec_cls", _ARCHES)
def test_combine_launch_matches_reduce_params(monkeypatch, arch, spec_cls):
    torch = pytest.importorskip("torch")
    import rocke.helpers.compile as compile_mod
    import rocke.runtime as runtime_mod

    built = {}

    def fake_compile(kernel, **_):
        built["kernel"] = kernel
        return _Art(kernel)

    class FakeLauncher:
        def __init__(self, *, hsaco, kernel_name, signature):
            built["signature"] = signature

        def __call__(self, vals, *, config):
            built["vals"] = vals

    monkeypatch.setattr(compile_mod, "compile_kernel", fake_compile)
    monkeypatch.setattr(runtime_mod, "KernelLauncher", FakeLauncher)
    monkeypatch.setattr(kv_split, "_COMBINE_LAUNCHER_CACHE", {})

    spec = _dense_spec(spec_cls)
    workspace = kv_split.allocate_segment_workspace(spec, "cpu")
    out = torch.empty(
        (spec.batch, spec.seqlen_q, spec.num_query_heads, spec.head_size),
        dtype=torch.bfloat16,
    )
    kv_split.launch_segment_combine(spec, arch, out, workspace, stream=0)

    params = [p.name for p in built["kernel"].params]
    assert params == [a["name"] for a in built["signature"]]
    assert "num_query_heads" in params
    assert set(params) <= set(built["vals"])
    assert built["vals"]["num_query_heads"] == spec.num_query_heads
