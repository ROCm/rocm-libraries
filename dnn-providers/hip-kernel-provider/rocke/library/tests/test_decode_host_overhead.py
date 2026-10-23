# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Independent oracle and host timer boundaries for saved-spec decode."""

import numpy as np
import pytest
from benchmarks.common import benchmark_decode_host_overhead as bench


def test_reference_obeys_gqa_and_actual_device_lengths():
    q = np.zeros((2, 4, 2), dtype=np.float32)
    k = np.ones((2, 3, 2, 2), dtype=np.float32)
    v = np.array(
        [[[[2, 4], [6, 8]], [[4, 6], [8, 10]], [[900, 900], [900, 900]]]] * 2,
        dtype=np.float32,
    )
    got = bench.reference(q, k, v, [2, 1])
    np.testing.assert_array_equal(got[0], [[3, 5], [3, 5], [7, 9], [7, 9]])
    np.testing.assert_array_equal(got[1], [[2, 4], [2, 4], [6, 8], [6, 8]])
    assert bench.check(got, got)["passed"]
    with pytest.raises(AssertionError, match="independent FP64"):
        bench.check(np.full_like(got, np.nan), got)
    with pytest.raises(AssertionError, match="independent FP64"):
        bench.check(got + 1, got)


def test_host_timer_excludes_drains_and_event_timer_is_separate(monkeypatch):
    from rocke import runtime

    events = []
    monkeypatch.setattr(
        bench.time, "perf_counter_ns", lambda: events.append("clock") or len(events)
    )
    monkeypatch.setattr(
        runtime,
        "wait_stream_and_release",
        lambda stream: events.append(("drain", stream)),
    )
    monkeypatch.setattr(
        runtime, "time_launches", lambda *a, **kw: events.append("hip_event") or 1
    )
    bench.measure(
        lambda: events.append("call"), stream=17, warmup=1, iters=2, samples=2
    )
    assert events == [
        "call",
        ("drain", 17),
        "clock",
        "call",
        "call",
        "clock",
        ("drain", 17),
        "clock",
        "call",
        "call",
        "clock",
        ("drain", 17),
        "hip_event",
        "hip_event",
    ]


def test_failed_host_batch_drains_before_propagating(monkeypatch):
    from rocke import runtime

    drained = []
    monkeypatch.setattr(runtime, "wait_stream_and_release", drained.append)

    def fail():
        raise RuntimeError("partial submission")

    with pytest.raises(RuntimeError, match="partial submission"):
        bench.measure(fail, stream=17, warmup=0, iters=1, samples=2)
    assert drained == [17, 17]


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_saved_spec_provenance_lowers_with_explicit_target(arch, monkeypatch):
    from dispatch.attention import AttentionRequest, attention_tuning_spec

    monkeypatch.setenv("ROCKE_BACKEND", "python")
    request = AttentionRequest(
        batch=1,
        nhead_q=8,
        nhead_k=1,
        seqlen_q=1,
        seqlen_k=65,
        hdim_q=64,
        hdim_v=64,
        arch=arch,
        dtype="bf16",
        kv_block_size=16,
    )
    spec = attention_tuning_spec(request, f"{arch}_u3d_splitkv_seg128_t1xb")
    hashes = bench.kernel_llvm_hashes(spec, arch=arch)
    assert len(hashes) == 2 and hashes[0] != hashes[1]
    assert all(len(value) == 64 for value in hashes)
