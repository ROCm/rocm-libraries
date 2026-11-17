# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Deterministic HIP function-table double; never resolves a ROCm library."""

from __future__ import annotations

import ctypes
from collections.abc import Iterator
from unittest.mock import Mock

import pytest
from rocke.runtime import hip_module as hip


class FakeHip:
    """Model device-local handles and directed peer mappings without hardware."""

    def __init__(self) -> None:
        self.device = 0
        self.calls: list[tuple] = []
        self.capabilities: dict[tuple[int, int], bool] = {}
        self.mappings: set[tuple[int, int]] = set()
        self.errors: dict[tuple, int] = {}
        self.event_devices: dict[int, int] = {}
        self.ready: set[int] = set()
        self.next_event = 1000

    def status(self, *call: object) -> int:
        key = tuple(call)
        self.calls.append(key)
        return self.errors.get(key, hip.HIP_SUCCESS)

    def get_device(self, pointer: object) -> int:
        status = self.status("get", self.device)
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int)).contents.value = self.device
        return status

    def set_device(self, device: int) -> int:
        status = self.status("set", device)
        if status == hip.HIP_SUCCESS:
            self.device = device
        return status

    def can_access(self, pointer: object, device: int, peer: int) -> int:
        status = self.status("can", device, peer)
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int)).contents.value = int(
            self.capabilities.get((device, peer), device != peer)
        )
        return status

    def enable(self, peer: int, flags: int) -> int:
        assert flags == 0
        status = self.status("enable", self.device, peer)
        if status != hip.HIP_SUCCESS:
            return status
        pair = (self.device, peer)
        if pair in self.mappings:
            return hip.HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED
        self.mappings.add(pair)
        return hip.HIP_SUCCESS

    def disable(self, peer: int) -> int:
        status = self.status("disable", self.device, peer)
        if status != hip.HIP_SUCCESS:
            return status
        pair = (self.device, peer)
        if pair not in self.mappings:
            return hip.HIP_ERROR_PEER_ACCESS_NOT_ENABLED
        self.mappings.remove(pair)
        return hip.HIP_SUCCESS

    def stream_sync(self, stream: ctypes.c_void_p) -> int:
        return self.status("stream_sync", self.device, stream.value or 0)

    def module_load(self, pointer: object, _blob: object) -> int:
        status = self.status("module_load", self.device)
        ctypes.cast(pointer, ctypes.POINTER(hip._HipModuleHandle)).contents.p = (
            100 + self.device
        )
        return status

    def module_function(self, pointer: object, _module: object, name: bytes) -> int:
        status = self.status("module_function", self.device, name)
        ctypes.cast(pointer, ctypes.POINTER(hip._HipFunctionHandle)).contents.p = (
            200 + self.device
        )
        return status

    def event_create(self, pointer: object) -> int:
        status = self.status("event_create", self.device)
        handle = self.next_event
        self.next_event += 1
        ctypes.cast(pointer, ctypes.POINTER(hip._HipEventHandle)).contents.p = handle
        self.event_devices[handle] = self.device
        return status

    def event_record(self, handle: hip._HipEventHandle, stream: ctypes.c_void_p) -> int:
        return self.status("event_record", self.device, handle.p, stream.value or 0)

    def event_query(self, handle: hip._HipEventHandle) -> int:
        status = self.status("event_query", self.device, handle.p)
        if status != hip.HIP_SUCCESS:
            return status
        return hip.HIP_SUCCESS if handle.p in self.ready else hip.HIP_ERROR_NOT_READY

    def event_elapsed(self, pointer: object, _start: object, _end: object) -> int:
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_float)).contents.value = 1.25
        return self.status("event_elapsed", self.device)


@pytest.fixture
def fake_hip(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeHip]:
    """Patch every HIP entry point touched by the peer/multi-device tests."""
    fake = FakeHip()
    prior = dict(hip.Runtime._pending_args)
    hip.Runtime._pending_args.clear()
    monkeypatch.setattr(hip, "_hip_inited", True)
    functions = {
        "_hipInit": lambda flags: fake.status("init", flags),
        "_hipGetDevice": fake.get_device,
        "_hipSetDevice": fake.set_device,
        "_hipDeviceCanAccessPeer": fake.can_access,
        "_hipDeviceEnablePeerAccess": fake.enable,
        "_hipDeviceDisablePeerAccess": fake.disable,
        "_hipDeviceSynchronize": lambda: fake.status("sync", fake.device),
        "_hipStreamSynchronize": fake.stream_sync,
        "_hipModuleLoadData": fake.module_load,
        "_hipModuleGetFunction": fake.module_function,
        "_hipModuleUnload": lambda handle: fake.status("module_unload", fake.device),
        "_hipModuleLaunchKernel": lambda *args: fake.status("launch", fake.device),
        "_hipEventCreate": fake.event_create,
        "_hipEventRecord": fake.event_record,
        "_hipEventQuery": fake.event_query,
        "_hipEventSynchronize": lambda handle: fake.status(
            "event_sync", fake.device, handle.p
        ),
        "_hipEventDestroy": lambda handle: fake.status(
            "event_destroy", fake.device, handle.p
        ),
        "_hipEventElapsedTime": fake.event_elapsed,
        "_hipGetErrorString": lambda status: b"synthetic HIP error",
    }
    for name, function in functions.items():
        monkeypatch.setattr(hip, name, Mock(side_effect=function))
    try:
        yield fake
    finally:
        hip.Runtime._pending_args.clear()
        hip.Runtime._pending_args.update(prior)
