# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Optional hardware admission and scoped decode launch/replay support."""

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def gpu():
    required = os.environ.get("ROCKE_REQUIRE_DECODE_GPU")
    if required:
        import torch
    else:
        torch = pytest.importorskip("torch")
    from rocke.runtime.hip_module import get_device_arch

    if not torch.cuda.is_available():
        if required:
            pytest.fail("required decode GPU is unavailable")
        pytest.skip("requires a ROCm GPU")
    arch = get_device_arch()
    if required and arch != required:
        pytest.fail(f"requested {required}, found {arch}")
    if arch not in ("gfx942", "gfx950", "gfx1250"):
        if required:
            pytest.fail(f"unsupported required decode target {arch}")
        pytest.skip("decode validation targets gfx942/gfx950/gfx1250")
    return torch, arch


@contextmanager
def decode_launch_session(
    torch: ModuleType, au: ModuleType, args: dict[str, object], mode: str
) -> Iterator[Callable[[], None]]:
    """Retain the original buffers and close prepared owners after every test scope.

    Internal graphs still use the runtime entry and its pointer-sensitive cache.
    External graphs capture once and replay the prepared owner on the same buffers.
    """
    if mode not in ("eager", "prepared", "graph", "external"):
        raise ValueError(f"unknown decode launch mode: {mode}")
    owner = graph = None
    try:
        if mode in ("prepared", "external"):
            owner = au.prepare_unified_attention_torch(**args)
        if mode == "external":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                owner.launch()

        def launch():
            if graph is not None:
                graph.replay()
            elif owner is not None:
                owner.launch()
            else:
                au.run_unified_attention_torch(**args)
            torch.cuda.synchronize()

        yield launch
    finally:
        try:
            if owner is not None:
                owner.close()
        finally:
            au.clear_attention_3d_graph_cache()
