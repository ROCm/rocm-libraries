# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only tests for ``launch_attention_dense``, the torch-free pointer entry of
the gfx942 dense prefill kernel.

The build, compile and launcher are replaced with recorders, so the host-side
launch path runs without a GPU or comgr. These tests pin that
``run_attention_dense_torch`` delegates to the pointer entry (same compile, same
kernargs, same launch geometry) and that the pointer entry refuses an incomplete
ABI before compiling. The oracle-free gfx942 suite launches through this entry,
so it is enrolled in the attention CPU lane rather than left in
``test_attention_dense_gfx942.py``, which no CI entry collects.
"""

import types

import pytest

from kernels.gfx942.attention_dense import (
    Gfx942AttentionDenseSpec,
    attention_dense_block,
    attention_dense_grid,
    attention_dense_runtime_args,
    launch_attention_dense,
    run_attention_dense_torch,
)


def _spec(**kw) -> Gfx942AttentionDenseSpec:
    base = dict(
        batch=1,
        seqlen_q=2048,
        seqlen_kv=2048,
        num_query_heads=128,
        num_kv_heads=8,
        head_size=128,
        causal=True,
        dtype="bf16",
        block_n=64,
    )
    base.update(kw)
    return Gfx942AttentionDenseSpec(**base)


@pytest.fixture
def recorded_dense_launch(monkeypatch):
    """Replace the build, compile and launcher with recorders and empty the cache, so
    the host-side launch path runs on CPU and every launch's kernargs and config
    can be compared."""
    import rocke.helpers.compile as compile_module
    import rocke.runtime as runtime_module
    from kernels.gfx942 import attention_dense as dense_module

    record = types.SimpleNamespace(compiled=[], launches=[])

    def fake_compile(kernel, *, arch, backend, capture_ir_text):
        record.compiled.append((kernel, arch, backend))
        return types.SimpleNamespace(hsaco=b"", kernel_name=kernel.kernel_name())

    class FakeLauncher:
        def __init__(self, *, hsaco, kernel_name, signature):
            self.kernel_name = kernel_name

        def __call__(self, values, *, config):
            record.launches.append((self.kernel_name, dict(values), config))

    monkeypatch.setattr(
        dense_module, "build_attention_dense", lambda spec, *, arch: spec
    )
    monkeypatch.setattr(compile_module, "compile_kernel", fake_compile)
    monkeypatch.setattr(runtime_module, "KernelLauncher", FakeLauncher)
    monkeypatch.setattr(dense_module, "_DENSE_LAUNCHER_CACHE", {})
    return record


def test_torch_entry_and_pointer_entry_issue_the_same_launch(recorded_dense_launch):
    """The torch entry delegates: integer pointers through either entry produce one
    compile, then identical kernargs and launch geometry."""
    spec = _spec(head_size=128, dtype="fp16")
    pointers = dict(q=0x1000, k=0x2000, v=0x3000, out=0x4000)

    returned = run_attention_dense_torch(spec=spec, **pointers, scale=0.125, stream=7)
    launch_attention_dense(spec=spec, **pointers, scale=0.125, stream=7)

    assert returned == pointers["out"]
    assert len(recorded_dense_launch.compiled) == 1
    torch_launch, pointer_launch = recorded_dense_launch.launches
    assert torch_launch == pointer_launch
    name, values, config = pointer_launch
    assert name == spec.kernel_name()
    assert values == {
        "q_ptr": 0x1000,
        "k_ptr": 0x2000,
        "v_ptr": 0x3000,
        "o_ptr": 0x4000,
        "scale": 0.125,
        **attention_dense_runtime_args(spec),
    }
    assert config.grid == attention_dense_grid(spec)
    assert config.block == attention_dense_block(spec)
    assert config.stream == 7
    assert config.fence


def test_pointer_entry_passes_spec_required_buffers(recorded_dense_launch):
    """Sinks and the split-KV workspace reach the kernargs under their ABI names."""
    launch_attention_dense(
        spec=_spec(use_sinks=True), q=1, k=2, v=3, out=4, scale=1.0, sinks=5
    )
    workspace = {"segm_output_ptr": 6, "segm_max_ptr": 7, "segm_expsum_ptr": 8}
    launch_attention_dense(
        spec=_spec(kv_segments=4),
        q=1,
        k=2,
        v=3,
        out=4,
        scale=1.0,
        segment_workspace=workspace,
    )
    (_, sink_values, _), (_, split_values, _) = recorded_dense_launch.launches
    assert sink_values["sink_ptr"] == 5
    assert {name: split_values[name] for name in workspace} == workspace


@pytest.mark.parametrize(
    "spec_kw,launch_kw,error,match",
    [
        (dict(varlen=True), {}, NotImplementedError, "unsupported"),
        ({}, dict(sinks=5), ValueError, "sinks provided"),
        ({}, dict(lse=5), ValueError, "lse provided"),
        ({}, dict(segment_workspace={}), ValueError, "segment_workspace provided"),
        (dict(use_sinks=True), {}, ValueError, "requires sinks"),
        (dict(kv_segments=4), {}, ValueError, "requires segment_workspace"),
        (
            dict(kv_segments=4),
            dict(segment_workspace={"segm_output_ptr": 6}),
            ValueError,
            "segment_workspace keys",
        ),
    ],
)
def test_pointer_entry_rejects_before_compiling(
    recorded_dense_launch, spec_kw, launch_kw, error, match
):
    """Spec legality and buffer presence are checked before any compile or launch,
    so a raw-pointer caller cannot launch a kernel whose ABI it did not fill."""
    with pytest.raises(error, match=match):
        launch_attention_dense(
            spec=_spec(**spec_kw), q=1, k=2, v=3, out=4, scale=1.0, **launch_kw
        )
    assert recorded_dense_launch.compiled == []
    assert recorded_dense_launch.launches == []
