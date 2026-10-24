# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Saved paged decode binding preparation, without generic runtime changes."""

from __future__ import annotations

import sys
import gc
import struct
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Thread
from types import SimpleNamespace

import pytest
from dispatch.attention import AttentionRequest, attention_execution_candidates
from kernels.common import attention_unified as au
from rocke.runtime.launcher import LaunchConfig, LaunchSummary


@dataclass(frozen=True)
class _Device:
    index: int = 0
    type: str = "cuda"

    def __str__(self):
        return f"cuda:{self.index}"


class _Tensor:
    def __init__(self, shape, dtype, values=None, *, strides=None):
        self.shape, self.dtype, self.values = tuple(shape), dtype, values
        self.device, self.strides = _Device(), strides
        self.pointer = id(self)

    def data_ptr(self):
        return self.pointer

    def stride(self, dim=None):
        strides = self.strides
        if strides is None:
            size, result = 1, []
            for dimension in reversed(self.shape):
                result.append(size)
                size *= dimension
            strides = tuple(reversed(result))
        return strides if dim is None else strides[dim]

    def is_contiguous(self):
        return self.strides is None

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


@pytest.fixture
def runtime(monkeypatch):
    state = SimpleNamespace(
        stream=17,
        device=0,
        capturing=False,
        graph=False,
        allocations=[],
        contexts=[],
        wrappers=[],
        calls=[],
        public_calls=[],
        requests=[],
        callback=None,
    )

    @contextmanager
    def device_context(device):
        previous, state.device = state.device, device.index
        state.contexts.append(("device", device))
        try:
            yield
        finally:
            state.device = previous

    @contextmanager
    def stream_context(stream):
        previous, state.stream = state.stream, stream.cuda_stream
        state.contexts.append(("stream", state.stream))
        try:
            yield
        finally:
            state.stream = previous

    def external_stream(stream, *, device):
        obj = SimpleNamespace(cuda_stream=stream, device=device)
        state.wrappers.append(obj)
        return obj

    def empty(shape, *, dtype, device):
        tensor = _Tensor(shape, dtype)
        tensor.device = device
        state.allocations.append(tensor)
        return tensor

    torch = SimpleNamespace(
        float32="torch.float32",
        empty=empty,
        cuda=SimpleNamespace(
            device=device_context,
            stream=stream_context,
            ExternalStream=external_stream,
            current_device=lambda: state.device,
            current_stream=lambda device=None: SimpleNamespace(
                cuda_stream=state.stream
            ),
            is_current_stream_capturing=lambda: state.capturing,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: "gfx942")
    monkeypatch.setattr(au, "_enable_3d_graph_replay", lambda *a: state.graph)
    monkeypatch.setattr(au, "release_retained_for_stream", lambda stream: None)

    def pipeline(values, configs, *, stream):
        state.calls.append((values, stream, state.stream))
        if state.callback is not None:
            state.callback(values)
        return LaunchSummary(launches=2)

    prepared = SimpleNamespace(
        pipeline=pipeline,
        seg_config=LaunchConfig(grid=(1, 4, 128), block=(64, 1, 1)),
        red_config=LaunchConfig(grid=(1, 32, 1), block=(64, 1, 1)),
    )
    monkeypatch.setattr(au, "_get_3d_pipeline", lambda *a, **kw: prepared)
    public_run = au.run_unified_attention_torch

    def run(**kw):
        state.public_calls.append(kw)
        return public_run(**kw)

    monkeypatch.setattr(au, "run_unified_attention_torch", run)
    from rocke.runtime import torch_interop

    resolve = torch_interop.resolve_stream

    def resolve_stream(stream, device=None):
        state.requests.append(stream)
        return resolve(stream, device)

    monkeypatch.setattr(torch_interop, "resolve_stream", resolve_stream)
    return state


def _binding(*, candidate_entry=False, **kwargs):
    candidate = next(
        c
        for c in attention_execution_candidates()
        if c.spec_id == "gfx942_u3d_splitkv_seg128_t1xb"
    )
    request = AttentionRequest(
        batch=1,
        nhead_q=32,
        nhead_k=4,
        seqlen_q=1,
        seqlen_k=2048,
        hdim_q=128,
        hdim_v=128,
        arch="gfx942",
        dtype="bf16",
        kv_block_size=16,
        num_cus=304,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
    )
    spec = candidate.select_spec(request)
    tensors = {
        "q": _Tensor((1, 32, 128), "torch.bfloat16"),
        "out": _Tensor((1, 32, 128), "torch.bfloat16"),
        "k": _Tensor((128, 16, 4, 128), "torch.bfloat16"),
        "v": _Tensor((128, 16, 4, 128), "torch.bfloat16"),
        "cu_seqlens_q": _Tensor((2,), "torch.int32", [0, 1]),
        "seqused_k": _Tensor((1,), "torch.int32", [2048]),
        "block_table": _Tensor((1, 128), "torch.int32", [list(range(128))]),
    }
    from dispatch.attention.bindings import bind_tuning_attention_torch
    from dispatch.attention.common import _problem

    if candidate_entry:
        return candidate.bound_torch(request, spec, tensors, **kwargs), tensors
    tensors["problem"] = _problem(request)
    return bind_tuning_attention_torch(request, spec, tensors, **kwargs), tensors


@pytest.mark.parametrize("candidate_entry", [False, True])
def test_warm_binding_reuses_private_scratch_and_context(runtime, candidate_entry):
    binding, _ = _binding(candidate_entry=candidate_entry)
    assert not runtime.allocations and not runtime.contexts and not runtime.requests
    assert binding.launch().launches == 2
    scratch = runtime.calls[-1][0][0]["segm_output_ptr"]
    contexts, wrappers = len(runtime.contexts), len(runtime.wrappers)
    for _ in range(50):
        binding.launch()
    assert len(runtime.allocations) == 3 and len(runtime.public_calls) == 1
    assert len(runtime.contexts) == contexts and len(runtime.wrappers) == wrappers
    assert runtime.requests[:2] == [None, 17]  # Cold public runner resolves again.
    assert runtime.requests[2:] == [None] * 50
    assert all(call[0][0]["segm_output_ptr"] is scratch for call in runtime.calls)
    other, _ = _binding()
    other.launch()
    assert len(runtime.allocations) == 6
    assert runtime.calls[-1][0][0]["segm_output_ptr"] is not scratch


@pytest.mark.parametrize("stream", [None, 0, 29])
def test_stream_overrides_restore_caller_context(runtime, stream):
    binding, _ = _binding(stream=stream)
    binding.launch()
    binding.launch()
    expected = 17 if stream in (None, 0) else stream
    assert runtime.calls[-1][1:] == (expected, expected)
    assert runtime.stream == 17
    scratch = runtime.calls[-1][0][0]["segm_output_ptr"]
    binding.launch(stream=41)
    assert runtime.calls[-1][1:] == (41, 41)
    assert runtime.calls[-1][0][0]["segm_output_ptr"] is not scratch
    assert runtime.stream == 17
    runtime.stream = 53
    binding.launch(stream=None)
    assert runtime.calls[-1][1:] == (53, 53)
    assert runtime.calls[-1][0][0]["segm_output_ptr"] is not scratch


def test_live_pointers_scalars_and_layout_change(runtime):
    binding, tensors = _binding()
    binding.launch()
    for name in ("q", "k", "v", "out", "seqused_k"):
        original = tensors[name]
        tensors[name] = _Tensor(original.shape, original.dtype, original.values)
    binding.launch(softmax_scale=0.3, softcap=2.0)
    segment, reduce = runtime.calls[-1][0]
    assert segment["query_ptr"] is tensors["q"]
    assert segment["key_cache_ptr"] is tensors["k"]
    assert segment["value_cache_ptr"] is tensors["v"]
    assert segment["seq_lens_ptr"] is reduce["seq_lens_ptr"] is tensors["seqused_k"]
    assert reduce["output_ptr"] is tensors["out"]
    assert segment["scale"] == 0.3 and segment["softcap"] == 2.0
    assert len(runtime.public_calls) == 1
    scratch = segment["segm_output_ptr"]
    tensors["block_table"] = _Tensor(
        (1, 128), "torch.int32", [list(range(128))], strides=(256, 1)
    )
    binding.launch()
    assert len(runtime.public_calls) == 2
    assert runtime.calls[-1][0][0]["block_table_stride"] == 256
    assert runtime.calls[-1][0][0]["segm_output_ptr"] is not scratch


def test_foreign_thread_and_reentrant_pair_get_private_scratch(runtime):
    binding, _ = _binding()
    binding.launch()
    owned = runtime.calls[-1][0][0]["segm_output_ptr"]
    errors = []

    def launch():
        try:
            binding.launch()
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=launch)
    thread.start()
    thread.join()
    assert not errors
    assert runtime.calls[-1][0][0]["segm_output_ptr"] is not owned

    def callback(values):
        runtime.callback = None
        binding.launch()
        assert runtime.calls[-1][0][0]["segm_output_ptr"] is not owned
        assert values[0]["segm_output_ptr"] is owned

    runtime.callback = callback
    binding.launch()
    binding.launch()
    assert runtime.calls[-1][0][0]["segm_output_ptr"] is owned


def test_first_capture_and_later_capture_use_general_runner(runtime):
    binding, _ = _binding()
    runtime.capturing = True
    binding.launch()
    captured = runtime.calls[-1][0][0]["segm_output_ptr"]
    runtime.capturing = False
    binding.launch()
    owned = runtime.calls[-1][0][0]["segm_output_ptr"]
    assert owned is not captured
    runtime.capturing = True
    runtime.stream = 41
    binding.launch()
    assert runtime.calls[-1][0][0]["segm_output_ptr"] is not owned
    assert runtime.calls[-1][1:] == (41, 41)
    assert len(runtime.public_calls) == 3


def test_graph_mode_changes_keep_general_runner(runtime, monkeypatch):
    binding, _ = _binding()
    binding.launch()
    runtime.graph = True
    monkeypatch.setattr(au, "_run_3d_graph", lambda **kw: "graph")
    assert binding.launch() == "graph"
    assert len(runtime.public_calls) == 2
    runtime.graph = False
    binding.launch()
    assert len(runtime.public_calls) == 2


def test_enqueue_failure_retries_without_hiding_errors(runtime):
    binding, _ = _binding()

    def fail(values):
        raise RuntimeError("enqueue failed")

    runtime.callback = fail
    with pytest.raises(RuntimeError, match="enqueue failed"):
        binding.launch()
    runtime.callback = None
    binding.launch()
    runtime.callback = fail
    with pytest.raises(RuntimeError, match="enqueue failed"):
        binding.launch()
    runtime.callback = None
    binding.launch()
    with pytest.raises(ValueError, match="additive_bias"):
        binding.launch(additive_bias=object())
    with pytest.raises(ValueError, match="could not convert string"):
        binding.launch(softmax_scale="invalid")
    assert len(runtime.public_calls) == 2


def test_ordinary_pipeline_repackages_and_retains_discarded_binding(
    runtime, monkeypatch
):
    from rocke.runtime import hip_module, launcher

    rt = hip_module.Runtime.__new__(hip_module.Runtime)
    rt._pending_args = {}
    monkeypatch.setattr(rt, "_enqueue", lambda *args: ())
    monkeypatch.setattr(
        rt,
        "load_module",
        lambda blob: SimpleNamespace(get_function=lambda name: name),
    )
    monkeypatch.setattr(launcher, "_runtime", lambda: rt)
    stages = [
        launcher.KernelLauncher(
            hsaco=b"segment", kernel_name="segment", signature=au._3d_signature("bf16")
        ),
        launcher.KernelLauncher(
            hsaco=b"reduce",
            kernel_name="reduce",
            signature=au._reduce_signature("bf16"),
        ),
    ]
    prepared = SimpleNamespace(
        pipeline=launcher.PipelineLauncher(stages),
        seg_config=LaunchConfig(grid=(1, 4, 128), block=(64, 1, 1)),
        red_config=LaunchConfig(grid=(1, 32, 1), block=(64, 1, 1)),
    )
    monkeypatch.setattr(au, "_get_3d_pipeline", lambda *a, **kw: prepared)
    binding, tensors = _binding()
    with launcher.no_fence():
        binding.launch()
        old_pointer = tensors["q"].data_ptr()
        tensors["q"].pointer += 4096  # Same layout/object, new live address.
        binding.launch()
    bucket = rt._pending_args[17]
    assert len(bucket) == 4  # Ordinary per-call segment and reducer entries.
    assert struct.unpack_from("<Q", bytes(bucket[0][0][0]), 24)[0] == old_pointer
    assert (
        struct.unpack_from("<Q", bytes(bucket[2][0][0]), 24)[0]
        == tensors["q"].data_ptr()
    )
    scratch = weakref.ref(
        next(
            ref
            for ref in bucket[2][0]
            if isinstance(ref, _Tensor) and ref.dtype == "torch.float32"
        )
    )
    output = weakref.ref(tensors["out"])
    runtime.public_calls.clear()
    runtime.allocations.clear()
    del binding, tensors, bucket
    gc.collect()
    assert scratch() is not None and output() is not None
    completed = []
    monkeypatch.setattr(
        rt,
        "stream_sync",
        lambda stream: completed.append((stream, scratch() is not None)),
    )
    rt.wait_stream(17)
    gc.collect()
    assert completed == [(17, True)]
    assert scratch() is None and output() is None
    assert not rt._pending_args
