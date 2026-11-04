# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The dense segment-combine launch packs every argument the tiled reduce declares.

``launch_segment_combine`` launches the tiled 3D reduce kernel. That kernel reads
``num_query_heads`` as a kernarg (``RUNTIME_KERNARG_FIELDS``), so a launcher built
from the pointer-only reduce signature leaves the head count, and with it the
reduce's token strides, undefined. This captures the launcher's signature and
values on the host and checks them against the reduce kernel's params.
"""

from __future__ import annotations

import pytest
from kernels.common import attention_dense_kv_split as kv_split
from kernels.gfx942.attention_dense import Gfx942AttentionDenseSpec
from kernels.gfx950.attention_dense import Gfx950AttentionDenseSpec

torch = pytest.importorskip("torch")


class _Art:
    hsaco = b""

    def __init__(self, kernel):
        self.kernel_name = kernel.name


@pytest.mark.parametrize(
    "arch, spec_cls",
    [("gfx942", Gfx942AttentionDenseSpec), ("gfx950", Gfx950AttentionDenseSpec)],
)
def test_combine_launch_matches_reduce_params(monkeypatch, arch, spec_cls):
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

    spec = spec_cls(
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
