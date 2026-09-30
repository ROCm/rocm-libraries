# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Cluster launch through the ctypes HIP runtime, with HIP mocked out.

These pin what reaches ``hipDrvLaunchKernelEx`` and when a launch is refused
before HIP sees it. They need no GPU and no HIP library. The ctypes layouts are
checked against the HIP headers in ``test_hip_device_properties_layout.py``.
"""

from __future__ import annotations

import ctypes
from unittest import mock

import pytest

from rocke import run_manifest
from rocke.runtime import hip_module, launcher
from rocke.runtime.hip_module import HipError, Runtime

# A stream nothing else uses, so the class-level bucket can be inspected.
_STREAM = 0x5EED


class _Hip:
    """Stands in for the two launch entry points and records their calls."""

    def __init__(self):
        self.plain = []
        self.ex = []

    def module_launch(self, fn, gx, gy, gz, bx, by, bz, shared, stream, params, extra):
        self.plain.append(
            {
                "grid": (gx.value, gy.value, gz.value),
                "block": (bx.value, by.value, bz.value),
                "shared": shared.value,
                "stream": stream.value,
                "params": params,
                "extra": extra,
            }
        )
        return 0

    def drv_launch_ex(self, cfg_ref, fn, params, extra):
        cfg = cfg_ref._obj
        attr = cfg.attrs[0]
        self.ex.append(
            {
                "cfg": cfg,
                "grid": (cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ),
                "block": (cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ),
                "shared": cfg.sharedMemBytes,
                "stream": cfg.hStream,
                "num_attrs": cfg.numAttrs,
                "attr_id": attr.id,
                "cluster": (
                    attr.val.clusterDim.x,
                    attr.val.clusterDim.y,
                    attr.val.clusterDim.z,
                ),
                "params": params,
                "extra": extra,
            }
        )
        return 0


@pytest.fixture
def hip():
    fake = _Hip()
    with mock.patch.object(
        hip_module, "_hipModuleLaunchKernel", fake.module_launch
    ), mock.patch.object(
        hip_module, "_hipDrvLaunchKernelEx", fake.drv_launch_ex
    ), mock.patch.object(
        hip_module, "_hipStreamSynchronize", return_value=0
    ), mock.patch.object(
        hip_module, "_current_device", return_value=0
    ), mock.patch.object(
        hip_module, "device_supports_cluster_launch", return_value=True
    ):
        yield fake
    Runtime._pending_args.pop(_STREAM, None)


def test_struct_layout_matches_hip():
    cfg = hip_module._HipLaunchConfig
    assert ctypes.sizeof(cfg) == 56
    assert (cfg.hStream.offset, cfg.attrs.offset, cfg.numAttrs.offset) == (32, 40, 48)
    attr = hip_module._HipLaunchAttribute
    assert ctypes.sizeof(attr) == 72
    assert attr.val.offset == 8
    assert ctypes.sizeof(hip_module._HipLaunchAttributeValue) == 64
    assert hip_module._HIP_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION == 4
    assert hip_module.HipDevicePropR0600.clusterLaunch.offset == 772


def test_plain_launch_is_unchanged(hip):
    Runtime().launch(object(), (4, 1, 1), (64, 1, 1), b"\0" * 8, stream=_STREAM)
    assert not hip.ex
    (call,) = hip.plain
    assert call["grid"] == (4, 1, 1) and call["stream"] == _STREAM
    (entry,) = Runtime._pending_args[_STREAM]
    assert len(entry[0]) == 3  # args, size, extra: no cluster config


def test_cluster_launch_goes_through_launch_kernel_ex(hip):
    Runtime().launch(
        object(),
        (8, 2, 1),
        (64, 1, 1),
        b"\0" * 8,
        shared_bytes=512,
        stream=_STREAM,
        cluster=[4, 2, 1],
    )
    assert not hip.plain
    (call,) = hip.ex
    assert call["grid"] == (8, 2, 1)
    assert call["block"] == (64, 1, 1)
    assert call["shared"] == 512
    assert call["stream"] == _STREAM
    assert call["num_attrs"] == 1
    assert call["attr_id"] == 4
    assert call["cluster"] == (4, 2, 1)
    assert call["params"] is None and call["extra"] is not None
    # HIP was handed pointers into the config and attribute, so both must
    # live as long as the args buffer.
    (entry,) = Runtime._pending_args[_STREAM]
    assert any(ref is call["cfg"] for ref in entry[0])
    assert any(isinstance(ref, hip_module._HipLaunchAttribute) for ref in entry[0])


def test_blocking_launch_with_cluster(hip):
    Runtime().launch_blocking(
        object(), (2, 1, 1), (64, 1, 1), b"\0" * 8, stream=_STREAM, cluster=(2, 1, 1)
    )
    (call,) = hip.ex
    assert call["cluster"] == (2, 1, 1)
    assert _STREAM not in Runtime._pending_args


def test_kernelparams_launch_with_cluster(hip):
    Runtime().launch_kernelparams(
        object(),
        (2, 2, 1),
        (64, 1, 1),
        [ctypes.c_int(7)],
        stream=_STREAM,
        record_event=False,
        cluster=(1, 2, 1),
    )
    (call,) = hip.ex
    assert call["cluster"] == (1, 2, 1)
    assert call["params"] is not None and call["extra"] is None
    (entry,) = Runtime._pending_args[_STREAM]
    assert any(ref is call["cfg"] for ref in entry[0])


@pytest.mark.parametrize(
    "grid, cluster, msg",
    [
        ((4, 1, 1), (2, 1), "three integers"),
        ((4, 1, 1), (16, 1, 1), "1..15"),
        ((8, 4, 1), (4, 4, 2), "limit of 16"),
        ((3, 1, 1), (2, 1, 1), r"grid \(3, 1, 1\) is not a multiple of cluster \(2, 1, 1\) in x"),
        ((4, 3, 1), (1, 2, 1), "in y"),
    ],
)
def test_bad_cluster_is_refused_before_hip(hip, grid, cluster, msg):
    with pytest.raises(ValueError, match=msg):
        Runtime().launch(
            object(), grid, (64, 1, 1), b"\0" * 8, stream=_STREAM, cluster=cluster
        )
    assert not hip.ex and not hip.plain
    assert _STREAM not in Runtime._pending_args


def test_unsupported_device_is_refused(hip):
    with mock.patch.object(
        hip_module, "device_supports_cluster_launch", return_value=False
    ), mock.patch.object(hip_module, "_current_device", return_value=3):
        with pytest.raises(HipError, match="device 3 does not support cluster launch"):
            Runtime().launch(
                object(), (2, 1, 1), (64, 1, 1), b"", stream=_STREAM, cluster=(2, 1, 1)
            )
    assert not hip.ex


def test_unknown_support_still_launches(hip):
    """An old runtime that cannot report ``clusterLaunch`` gets to try; HIP
    itself refuses the launch if the device cannot do it."""
    with mock.patch.object(
        hip_module, "device_supports_cluster_launch", return_value=None
    ):
        Runtime().launch(
            object(), (2, 1, 1), (64, 1, 1), b"", stream=_STREAM, cluster=(2, 1, 1)
        )
    assert len(hip.ex) == 1


def test_support_query_is_cached_per_device():
    props = mock.Mock(clusterLaunch=1)
    with mock.patch.dict(hip_module._cluster_launch_support, clear=True):
        with mock.patch.object(hip_module, "_device_props", return_value=props) as q:
            assert hip_module.device_supports_cluster_launch(0) is True
            assert hip_module.device_supports_cluster_launch(0) is True
            q.assert_called_once_with(0)
        with mock.patch.object(hip_module, "_device_props", return_value=None):
            assert hip_module.device_supports_cluster_launch(1) is None
        props.clusterLaunch = 0
        with mock.patch.object(hip_module, "_device_props", return_value=props):
            assert hip_module.device_supports_cluster_launch(2) is False


class _FakeRuntime:
    def __init__(self):
        self.calls = []

    def load_module(self, blob):
        return mock.Mock(get_function=lambda name: name)

    def launch_blocking(self, fn, grid, block, args, **kw):
        self.calls.append(("blocking", kw.get("cluster")))

    def launch(self, fn, grid, block, args, **kw):
        self.calls.append(("async", kw.get("cluster")))

    def retain_for_stream(self, *a):
        pass


def _launcher(rt, cluster_dims):
    with mock.patch.object(launcher, "_runtime", return_value=rt):
        return launcher.KernelLauncher(
            hsaco=b"", kernel_name="k", signature=[], cluster_dims=cluster_dims
        )


def _call(rt, kl, **cfg):
    with mock.patch.object(launcher, "_runtime", return_value=rt):
        kl({}, config=launcher.LaunchConfig(stream=_STREAM, **cfg))


def test_launcher_uses_the_compiled_cluster():
    rt = _FakeRuntime()
    kl = _launcher(rt, [2, 1, 1])
    assert kl.cluster_dims == (2, 1, 1)
    _call(rt, kl, grid=(4, 1, 1), block=(64, 1, 1))
    _call(rt, kl, grid=(4, 1, 1), block=(64, 1, 1), cluster=(2, 1, 1), fence=False)
    assert rt.calls == [("blocking", (2, 1, 1)), ("async", (2, 1, 1))]


def test_launcher_refuses_a_different_cluster():
    rt = _FakeRuntime()
    kl = _launcher(rt, (2, 1, 1))
    with pytest.raises(ValueError, match=r"launch cluster \(4, 1, 1\) does not match"):
        _call(rt, kl, grid=(4, 1, 1), block=(64, 1, 1), cluster=(4, 1, 1))
    assert not rt.calls


def test_launcher_without_cluster_dims():
    rt = _FakeRuntime()
    kl = _launcher(rt, None)
    assert kl.cluster_dims is None
    _call(rt, kl, grid=(4, 1, 1), block=(64, 1, 1))
    _call(rt, kl, grid=(4, 1, 1), block=(64, 1, 1), cluster=(4, 1, 1))
    assert rt.calls == [("blocking", None), ("blocking", (4, 1, 1))]


def test_launcher_validates_cluster_dims_at_construction():
    with pytest.raises(ValueError, match="limit of 16"):
        _launcher(_FakeRuntime(), (4, 4, 4))


def test_manifest_timing_passes_the_cluster_through_time_launches():
    rt = mock.Mock()
    with mock.patch.object(run_manifest, "HAS_TORCH_LAUNCHER", True), mock.patch.object(
        run_manifest, "time_launches", side_effect=lambda fn, **kw: fn() or 1.0
    ):
        run_manifest._launch_timed(
            rt, "fn", (4, 1, 1), (64, 1, 1), b"", 1, 1, cluster=(2, 1, 1)
        )
    rt.launch.assert_called_once_with("fn", (4, 1, 1), (64, 1, 1), b"", cluster=(2, 1, 1))


def test_manifest_timing_passes_the_cluster_without_torch():
    rt = mock.Mock()
    rt.event.return_value.elapsed_to.return_value = 3.0
    with mock.patch.object(run_manifest, "HAS_TORCH_LAUNCHER", False):
        run_manifest._launch_timed(
            rt, "fn", (4, 1, 1), (64, 1, 1), b"", 2, 3, cluster=(2, 1, 1)
        )
    assert rt.launch.call_count == 5
    for call in rt.launch.call_args_list:
        assert call.kwargs == {"cluster": (2, 1, 1)}
