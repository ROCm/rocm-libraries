# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-only baseline planning, equivalent final C, and unfenced FIFO enqueue tests."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
from rocke.instances.common import peer_gemm_pipeline as pipeline
from rocke.instances.common.peer_gemm import PeerGemmAllocation, PeerGemmSpec
from rocke.instances.common.peer_gemm_pipeline import (
    PeerCopyTask,
    enqueue_peer_gemm_pipeline,
    make_peer_gemm_pipeline,
)


def allocation(device, slot, words):
    return PeerGemmAllocation(
        device, (device + 1) * 0x1000000 + slot * 0x10000, words * 2
    )


def panel(pointer, regions, memory, rows, cols):
    region = next(
        region
        for region in regions
        if region.pointer <= pointer < region.pointer + region.nbytes
    )
    offset = (pointer - region.pointer) // 2
    return memory[(region.device, region.pointer)][
        offset : offset + rows * cols
    ].reshape(rows, cols)


@pytest.mark.parametrize("sources", [1, 2, 4, 8])
@pytest.mark.parametrize("inputs", ["gather", "replicated"])
@pytest.mark.parametrize("outputs", ["local", "scatter"])
def test_unfused_and_fused_plans_have_identical_logical_output(
    sources, inputs, outputs
) -> None:
    spec = PeerGemmSpec(16, 16, 16, sources=sources)
    devices = tuple(range(sources))
    a = (np.arange(sources * 16 * 16).reshape(sources * 16, 16) % 3 - 1).astype(
        np.float32
    )
    b = (np.arange(16 * sources * 16).reshape(16, sources * 16) % 5 - 2).astype(
        np.float32
    )
    expected = a @ b
    memory = {}
    all_regions = []

    def create(device, slot, values):
        region = allocation(device, slot, values.size)
        memory[(region.device, region.pointer)] = values.copy().ravel()
        all_regions.append(region)
        return region

    a_shards = tuple(create(s, 1, a[s * 16 : (s + 1) * 16]) for s in devices)
    a_rep = tuple(create(r, 2, a) for r in devices)
    local_b = tuple(create(r, 3, b[:, r * 16 : (r + 1) * 16]) for r in devices)
    local_c = tuple(
        create(r, 4, np.zeros((sources * 16, 16), dtype=np.float32)) for r in devices
    )
    scattered = tuple(
        create(s, 5, np.zeros((sources, 16, 16), dtype=np.float32)) for s in devices
    )
    local_astage = tuple(
        create(r, 6, np.zeros((sources * 16, 16), dtype=np.float32)) for r in devices
    )
    local_cstage = tuple(
        create(r, 7, np.zeros((sources * 16, 16), dtype=np.float32)) for r in devices
    )
    plans = []
    for rank in devices:
        plan = make_peer_gemm_pipeline(
            spec,
            rank=rank,
            rank_devices=devices,
            a_buffers=a_shards if inputs == "gather" else (a_rep[rank],),
            b_buffer=local_b[rank],
            c_buffers=(local_c[rank],) if outputs == "local" else scattered,
            a_staging=local_astage[rank] if inputs == "gather" else None,
            c_staging=local_cstage[rank] if outputs == "scatter" else None,
            input_placement=inputs,
            output_placement=outputs,
        )
        plans.append(plan)
        assert len(plan.gather) == (sources if inputs == "gather" else 0)
        assert len(plan.scatter) == (sources if outputs == "scatter" else 0)
        assert not plan.local_tables.required_peer_paths
        byte_counts = plan.logical_bytes()
        assert byte_counts["gather_copy_bytes"] == (
            sources * 512 if inputs == "gather" else 0
        )
        assert byte_counts["gather_peer_payload_bytes"] == (
            (sources - 1) * 512 if inputs == "gather" else 0
        )
        assert byte_counts["scatter_copy_bytes"] == (
            sources * 512 if outputs == "scatter" else 0
        )
        assert byte_counts["scatter_peer_payload_bytes"] == (
            (sources - 1) * 512 if outputs == "scatter" else 0
        )

    def evaluate(fused):
        for rank, plan in enumerate(plans):
            if not fused:
                for task in plan.gather:
                    panel(
                        task.destination.pointer,
                        all_regions,
                        memory,
                        task.rows,
                        task.cols,
                    )[:] = panel(
                        task.source.pointer, all_regions, memory, task.rows, task.cols
                    )
            tables = plan.fused_tables if fused else plan.local_tables
            bp = panel(local_b[rank].pointer, all_regions, memory, 16, 16)
            for source in devices:
                panel(tables.c_addresses[source], all_regions, memory, 16, 16)[:] = (
                    panel(tables.a_addresses[source], all_regions, memory, 16, 16) @ bp
                )
            if not fused:
                for task in plan.scatter:
                    panel(
                        task.destination.pointer,
                        all_regions,
                        memory,
                        task.rows,
                        task.cols,
                    )[:] = panel(
                        task.source.pointer, all_regions, memory, task.rows, task.cols
                    )
        if outputs == "local":
            return tuple(
                memory[(c.device, c.pointer)].reshape(sources * 16, 16).copy()
                for c in local_c
            )
        return tuple(
            memory[(c.device, c.pointer)]
            .reshape(sources, 16, 16)
            .transpose(1, 0, 2)
            .reshape(16, sources * 16)
            .copy()
            for c in scattered
        )

    baseline = evaluate(False)
    direct = evaluate(True)
    for rank, (left, right) in enumerate(zip(baseline, direct)):
        np.testing.assert_array_equal(left, right)
        reference = (
            expected[:, rank * 16 : (rank + 1) * 16]
            if outputs == "local"
            else expected[rank * 16 : (rank + 1) * 16]
        )
        np.testing.assert_array_equal(left, reference)


def plan_fixture():
    spec = PeerGemmSpec(16, 16, 16)
    inputs = (allocation(0, 1, 256), allocation(1, 1, 256))
    b = allocation(0, 2, 256)
    c = (allocation(0, 3, 512), allocation(1, 3, 512))
    astage = allocation(0, 4, 512)
    cstage = allocation(0, 5, 512)
    return make_peer_gemm_pipeline(
        spec,
        rank=0,
        rank_devices=(0, 1),
        a_buffers=inputs,
        b_buffer=b,
        c_buffers=c,
        a_staging=astage,
        c_staging=cstage,
    )


@pytest.mark.parametrize("fused", [False, True])
def test_enqueue_is_one_rank_fifo_without_interphase_fences(monkeypatch, fused) -> None:
    plan = plan_fixture()
    calls = []

    @contextmanager
    def scope(device):
        calls.append(("enter", device))
        yield
        calls.append(("leave", device))

    monkeypatch.setattr(pipeline, "device_scope", scope)
    rt = Mock()
    rt.launch.side_effect = lambda fn, *args, **kwargs: calls.append(
        ("launch", fn, kwargs)
    )
    a_table = allocation(0, 6, 8)
    c_table = allocation(0, 7, 8)
    summary = enqueue_peer_gemm_pipeline(
        rt,
        plan,
        gemm_function="gemm",
        copy_function="copy",
        a_table=a_table,
        c_table=c_table,
        fused=fused,
        stream=9,
    )
    sequence = [call[1] for call in calls if call[0] == "launch"]
    assert sequence == (["gemm"] if fused else ["copy", "copy", "gemm", "copy", "copy"])
    assert summary.launches == len(sequence)
    assert calls[0] == ("enter", 0) and calls[-1] == ("leave", 0)
    assert all(
        call[2]["stream"] == 9 and call[2]["record_event"] is False
        for call in calls
        if call[0] == "launch"
    )
    assert rt.retain_for_stream.call_count == len(sequence)
    rt.wait_stream.assert_not_called()
    rt.sync.assert_not_called()


def test_partial_enqueue_failure_retains_earlier_work(monkeypatch) -> None:
    plan = plan_fixture()

    @contextmanager
    def scope(device):
        yield

    monkeypatch.setattr(pipeline, "device_scope", scope)
    rt = Mock()
    rt.launch.side_effect = [None, RuntimeError("copy failed")]
    with pytest.raises(RuntimeError, match="copy failed"):
        enqueue_peer_gemm_pipeline(
            rt,
            plan,
            gemm_function="gemm",
            copy_function="copy",
            a_table=allocation(0, 6, 8),
            c_table=allocation(0, 7, 8),
            fused=False,
        )
    assert rt.retain_for_stream.call_count == 1
    rt.wait_stream.assert_not_called()


def test_staging_and_copy_alias_validation() -> None:
    plan = plan_fixture()
    with pytest.raises(ValueError, match="overlap"):
        PeerCopyTask(plan.gather[0].source, plan.gather[0].source, 16, 16)
    with pytest.raises(ValueError, match="exceeds"):
        PeerCopyTask(
            plan.gather[0].source,
            replace(plan.gather[0].destination, nbytes=16),
            16,
            16,
        )
    args = {
        "spec": plan.spec,
        "rank": 0,
        "rank_devices": (0, 1),
        "a_buffers": tuple(task.source for task in plan.gather),
        "b_buffer": plan.fused_tables.b,
        "c_buffers": tuple(allocation(s, 3, 512) for s in range(2)),
        "c_staging": allocation(0, 5, 512),
    }
    with pytest.raises(ValueError, match="requires local A staging"):
        make_peer_gemm_pipeline(**args)
    with pytest.raises(ValueError, match="staging must not overlap"):
        make_peer_gemm_pipeline(
            **args,
            a_staging=replace(allocation(0, 4, 512), pointer=args["b_buffer"].pointer),
        )
