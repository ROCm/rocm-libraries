# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-free multi-rank correctness runner and in-memory timing methodology.

Correctness reports intentionally omit measured durations. Timing is opt-in and
returned in memory for agent-chat inspection or protected AMD Confluence records;
never persist achieved software-performance values in the repository.
"""

from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass
from typing import Self

import numpy as np

from ...helpers.compile import compile_kernel
from ...instances.common.peer_gemm import (
    PeerGemmAllocation,
    PeerGemmSpec,
    build_peer_gemm,
)
from ...instances.common.peer_gemm_pipeline import (
    PeerGemmPipelinePlan,
    enqueue_peer_gemm_pipeline,
    make_peer_gemm_pipeline,
)
from ...instances.common.peer_tile_copy import PeerTileCopySpec, build_peer_tile_copy
from ...runtime import Runtime, device_scope, peer_access_scope
from ...runtime.hip_module import get_device_arch, get_device_count

__all__ = ["PeerGemmMeasurement", "PeerGemmRunner"]

_GUARD_WORDS = 16


def _encode(values: np.ndarray, dtype: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if dtype == "f16":
        return values.astype(np.float16).view(np.uint16).copy()
    words = values.view(np.uint32)
    rounded = words + np.uint32(0x7FFF) + ((words >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def _decode(words: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "f16":
        return words.view(np.float16).astype(np.float32)
    return (words.astype(np.uint32) << np.uint32(16)).view(np.float32)


@dataclass
class _Buffer:
    runtime: Runtime
    device: int
    pointer: int
    words: int
    initial: np.ndarray

    def region(self) -> PeerGemmAllocation:
        return PeerGemmAllocation(
            self.device, self.pointer + _GUARD_WORDS * 2, self.words * 2, owner=self
        )

    def upload(self, full: np.ndarray) -> None:
        host = (ctypes.c_uint16 * full.size).from_buffer_copy(full.tobytes())
        with device_scope(self.device):
            self.runtime.memcpy_h2d(self.pointer, host, int(full.nbytes))

    def download(self) -> np.ndarray:
        total = self.words + 2 * _GUARD_WORDS
        host = (ctypes.c_uint16 * total)()
        with device_scope(self.device):
            self.runtime.memcpy_d2h(host, self.pointer, total * 2)
        return np.ctypeslib.as_array(host).copy()

    def reset(self) -> None:
        self.upload(self.initial)


@dataclass(frozen=True)
class PeerGemmMeasurement:
    """Measured values for protected/session use, never part of correctness JSON.

    Host makespan includes Python submission, event recording, and all-rank
    drains. Device durations bracket each rank's stream work on that device's
    own clock. They must not be subtracted across different GPUs.
    """

    fused: bool
    host_makespan_ms: float
    device_stream_ms: tuple[float, ...]
    launches_per_rank: tuple[int, ...]


class PeerGemmRunner:
    """Own a fixed multi-rank workload and equivalent fused/unfused resources.

    Use a context manager. Inputs, pointer tables, and all source/output owners
    remain alive through every consumer. On a failed drain, do not free memory
    or disable mappings while device work may still be live.
    """

    def __init__(
        self,
        spec: PeerGemmSpec,
        *,
        devices: tuple[int, ...],
        arch: str,
        input_placement: str = "gather",
        output_placement: str = "scatter",
        backend: str = "python",
        seed: int = 12345,
    ) -> None:
        if len(devices) != spec.sources or len(set(devices)) != len(devices):
            raise ValueError("one unique device per source/rank is required")
        if any(
            type(device) is not int or not 0 <= device < get_device_count()
            for device in devices
        ):
            raise ValueError("rank devices must be allocated visible HIP devices")
        if any(get_device_arch(device) != arch for device in devices):
            raise ValueError("every execution device must match the explicit target")
        self.spec, self.devices, self.arch = spec, devices, arch
        self.input_placement, self.output_placement = input_placement, output_placement
        self.runtime = Runtime()
        self.buffers: list[_Buffer] = []
        self.modules = []
        self.plans: list[PeerGemmPipelinePlan] = []
        self.tables = []
        self.functions = []
        self._events = []
        self._mapping = None
        self._closed = False
        self._active = False
        self._work_may_be_live = False
        self._validated_paths: set[bool] = set()
        m, n, k, p = spec.rows_per_source, spec.cols_per_rank, spec.k, spec.sources
        rng = np.random.default_rng(seed)
        # Dyadic inputs avoid reference accumulation-order differences in focused checks.
        a = rng.integers(-4, 5, size=(p * m, k)).astype(np.float32) / 8
        b = rng.integers(-5, 6, size=(k, p * n)).astype(np.float32) / 8
        a_words, b_words = _encode(a, spec.dtype), _encode(b, spec.dtype)
        self.reference = _decode(a_words, spec.dtype) @ _decode(b_words, spec.dtype)
        self.immutable: list[_Buffer] = []
        try:
            gemm = compile_kernel(
                build_peer_gemm(spec, arch), arch=arch, backend=backend
            )
            copy_spec = PeerTileCopySpec(dtype=spec.dtype)
            copy = compile_kernel(
                build_peer_tile_copy(copy_spec, arch), arch=arch, backend=backend
            )
            sources = [
                self._allocate(device, m * k, a_words[index * m : (index + 1) * m])
                for index, device in enumerate(devices)
            ]
            replicas = [
                self._allocate(device, p * m * k, a_words) for device in devices
            ]
            b_panels = [
                self._allocate(
                    device, k * n, b_words[:, rank * n : (rank + 1) * n].copy()
                )
                for rank, device in enumerate(devices)
            ]
            self.outputs = [self._allocate(device, p * m * n) for device in devices]
            a_stages = (
                [self._allocate(device, p * m * k) for device in devices]
                if input_placement == "gather"
                else [None] * p
            )
            c_stages = (
                [self._allocate(device, p * m * n) for device in devices]
                if output_placement == "scatter"
                else [None] * p
            )
            self.immutable.extend((*sources, *replicas, *b_panels))
            for rank, device in enumerate(devices):
                plan = make_peer_gemm_pipeline(
                    spec,
                    rank=rank,
                    rank_devices=devices,
                    a_buffers=(
                        tuple(buffer.region() for buffer in sources)
                        if input_placement == "gather"
                        else (replicas[rank].region(),)
                    ),
                    b_buffer=b_panels[rank].region(),
                    c_buffers=(
                        (self.outputs[rank].region(),)
                        if output_placement == "local"
                        else tuple(buffer.region() for buffer in self.outputs)
                    ),
                    a_staging=a_stages[rank].region() if a_stages[rank] else None,
                    c_staging=c_stages[rank].region() if c_stages[rank] else None,
                    input_placement=input_placement,
                    output_placement=output_placement,
                )
                self.plans.append(plan)
                rank_tables = []
                for tables in (plan.fused_tables, plan.local_tables):
                    at = self._allocate(
                        device,
                        len(tables.a_table_bytes()) // 2,
                        np.frombuffer(tables.a_table_bytes(), dtype=np.uint16),
                    )
                    ct = self._allocate(
                        device,
                        len(tables.c_table_bytes()) // 2,
                        np.frombuffer(tables.c_table_bytes(), dtype=np.uint16),
                    )
                    self.immutable.extend((at, ct))
                    rank_tables.append((at, ct))
                self.tables.append(rank_tables)
                with device_scope(device):
                    gm = self.runtime.load_module(gemm.hsaco)
                    self.modules.append(gm)
                    cm = self.runtime.load_module(copy.hsaco)
                    self.modules.append(cm)
                    self.functions.append(
                        (
                            gm.get_function(spec.kernel_name()),
                            cm.get_function(copy_spec.kernel_name()),
                        )
                    )
            paths = tuple(
                sorted(
                    {path for plan in self.plans for path in plan.required_peer_paths}
                )
            )
            self._mapping = peer_access_scope(paths)
            self._mapping.__enter__()
            self._active = True
        except BaseException:
            self.close()
            raise

    def _allocate(
        self, device: int, words: int, payload: np.ndarray | None = None
    ) -> _Buffer:
        full = np.full(words + 2 * _GUARD_WORDS, 0xABCD, dtype=np.uint16)
        if payload is not None:
            data = np.asarray(payload, dtype=np.uint16).ravel()
            if data.size != words:
                raise ValueError("payload size does not match its region")
            full[_GUARD_WORDS : _GUARD_WORDS + words] = data
        with device_scope(device):
            pointer = self.runtime.alloc(int(full.nbytes))
        buffer = _Buffer(self.runtime, device, pointer, words, full)
        self.buffers.append(buffer)
        buffer.reset()
        return buffer

    def _ensure_active(self) -> None:
        if self._closed or not self._active:
            raise RuntimeError("peer GEMM runner is not active")

    def enqueue(self, *, fused: bool) -> tuple[int, ...]:
        """Enqueue all ranks; no output consumption or reset before a full drain."""
        self._ensure_active()
        if type(fused) is not bool:
            raise ValueError("fused must be a boolean")
        counts = []
        self._work_may_be_live = True
        for rank, plan in enumerate(self.plans):
            at, ct = self.tables[rank][0 if fused else 1]
            gemm, copy = self.functions[rank]
            summary = enqueue_peer_gemm_pipeline(
                self.runtime,
                plan,
                gemm_function=gemm,
                copy_function=copy,
                a_table=at.region(),
                c_table=ct.region(),
                fused=fused,
            )
            counts.append(summary.launches)
        return tuple(counts)

    def drain(self) -> None:
        """Wait for every execution device before releasing any shared resource."""
        errors = []
        for device in self.devices:
            try:
                with device_scope(device):
                    self.runtime.wait_stream(0)
            except (RuntimeError, OSError, AttributeError) as error:
                errors.append(error)
        if errors:
            raise RuntimeError(
                "one or more execution streams failed to drain"
            ) from errors[0]
        self._work_may_be_live = False

    def validate(self, *, fused: bool, repeats: int = 1) -> dict:
        """Run exact reference/guard/immutability checks with no measured durations."""
        self._ensure_active()
        if type(fused) is not bool:
            raise ValueError("fused must be a boolean")
        if type(repeats) is not int or repeats < 1:
            raise ValueError("repeats must be a positive integer")
        self._validated_paths.discard(fused)
        self.drain()
        for buffer in self.outputs:
            buffer.reset()
        counts = ()
        for _ in range(repeats):
            counts = self.enqueue(fused=fused)
        self.drain()
        m, n, p = self.spec.rows_per_source, self.spec.cols_per_rank, self.spec.sources
        for rank, buffer in enumerate(self.outputs):
            actual = buffer.download()
            if not (
                np.all(actual[:_GUARD_WORDS] == 0xABCD)
                and np.all(actual[-_GUARD_WORDS:] == 0xABCD)
            ):
                raise AssertionError("output guards were modified")
            if self.output_placement == "local":
                reference = self.reference[:, rank * n : (rank + 1) * n]
            else:
                reference = (
                    self.reference[rank * m : (rank + 1) * m]
                    .reshape(m, p, n)
                    .transpose(1, 0, 2)
                )
            np.testing.assert_array_equal(
                actual[_GUARD_WORDS : _GUARD_WORDS + buffer.words],
                _encode(reference, self.spec.dtype).ravel(),
            )
        for buffer in self.immutable:
            np.testing.assert_array_equal(buffer.download(), buffer.initial)
        for buffer in self.buffers:
            full = buffer.download()
            if not (
                np.all(full[:_GUARD_WORDS] == 0xABCD)
                and np.all(full[-_GUARD_WORDS:] == 0xABCD)
            ):
                raise AssertionError("input/table/staging guards were modified")
        self._validated_paths.add(fused)
        return {
            "status": "passed",
            "fused": fused,
            "ranks": p,
            "dtype": self.spec.dtype,
            "input_placement": self.input_placement,
            "output_placement": self.output_placement,
            "repeats": repeats,
            "launches_per_rank_per_iteration": counts,
            "logical_bytes_per_rank": tuple(
                plan.logical_bytes() for plan in self.plans
            ),
            "numeric_bits": "pass",
            "guards": "pass",
            "inputs_and_tables": "immutable",
        }

    def measure(self, *, fused: bool) -> PeerGemmMeasurement:
        """Return opt-in durations in memory after correctness has been established."""
        self._ensure_active()
        if type(fused) is not bool:
            raise ValueError("fused must be a boolean")
        if fused not in self._validated_paths:
            raise RuntimeError("validate this path before requesting measurements")
        self.drain()
        events = []
        try:
            for device in self.devices:
                with device_scope(device):
                    first = self.runtime.event()
                    self._events.append(first)
                    last = self.runtime.event()
                    self._events.append(last)
                    events.append((first, last))
            self._work_may_be_live = True
            start = time.perf_counter()
            counts = []
            for rank, plan in enumerate(self.plans):
                at, ct = self.tables[rank][0 if fused else 1]
                gemm, copy = self.functions[rank]
                with device_scope(plan.execution_device):
                    events[rank][0].record()
                    summary = enqueue_peer_gemm_pipeline(
                        self.runtime,
                        plan,
                        gemm_function=gemm,
                        copy_function=copy,
                        a_table=at.region(),
                        c_table=ct.region(),
                        fused=fused,
                    )
                    events[rank][1].record()
                counts.append(summary.launches)
            self.drain()
            makespan = (time.perf_counter() - start) * 1000
            durations = tuple(first.elapsed_to(last) for first, last in events)
            return PeerGemmMeasurement(fused, makespan, durations, tuple(counts))
        finally:
            if not self._work_may_be_live:
                self._destroy_events()

    def _destroy_events(self) -> None:
        while self._events:
            self._events[-1].destroy()
            self._events.pop()

    def close(self) -> None:
        """Drain before teardown; on drain failure preserve potentially live owners."""
        if self._closed:
            return
        self.drain()
        self._destroy_events()
        if self._mapping is not None and self._active:
            self._mapping.__exit__(None, None, None)
        self._active = False
        while self.modules:
            self.modules[-1].unload()
            self.modules.pop()
        while self.buffers:
            buffer = self.buffers[-1]
            with device_scope(buffer.device):
                self.runtime.free(buffer.pointer)
            self.buffers.pop()
        self._closed = True

    def __enter__(self) -> Self:
        self._ensure_active()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
