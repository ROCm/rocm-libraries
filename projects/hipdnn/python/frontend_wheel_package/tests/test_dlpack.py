# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""DLPack interoperability, matching cuDNN frontend's tensor_like and variant-pack rules."""

import ctypes
import os

import numpy as np
import pytest

import hipdnn_frontend as hipdnn
from hipdnn_frontend.hipdnn_frontend_python import _get_data_ptr

from . import helpers
from .graph_builders import build_pointwise_add_graph
from .helpers import build_all_plans


def _rocm_torch():
    """Return torch when it is a ROCm build with a visible device, else skip."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires a ROCm build of torch with a visible device")
    return torch


class _NoneDlpack:
    def __dlpack__(self, *args, **kwargs):
        return None


class _DataPtr:
    def data_ptr(self):
        return 4096


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_DLTENSOR = b"dltensor"


class _OffsetProducer:
    """Exports one float32 element at ``base + byte_offset`` with a nonzero
    DLPack byte_offset. Real producers (NumPy, torch) advance the data pointer
    instead, so they never exercise the offset field."""

    def __init__(self, base, byte_offset, device_type, device_id=0):
        self._args = (base, byte_offset, device_type, device_id)
        self._keep = []  # structs and shapes must outlive the consumer

    def __dlpack__(self, **kwargs):
        base, byte_offset, device_type, device_id = self._args
        shape = (ctypes.c_int64 * 1)(1)
        mt = _DLManagedTensor(
            data=base,
            device=_DLDevice(device_type, device_id),
            ndim=1,
            dtype=_DLDataType(2, 32, 1),  # kDLFloat, 32 bits
            shape=shape,
            byte_offset=byte_offset,
        )
        self._keep += [shape, mt]
        return _PyCapsule_New(ctypes.addressof(mt), _DLTENSOR, None)

    def __dlpack_device__(self):
        return (self._args[2], self._args[3])


class _DlpackOnly:
    """Hides data_ptr() so the bindings must import the DLPack capsule."""

    def __init__(self, tensor):
        self._tensor = tensor

    def __dlpack__(self, **kwargs):
        return self._tensor.__dlpack__(**kwargs)

    def __dlpack_device__(self):
        return self._tensor.__dlpack_device__()


class TestTensorLikeHost:
    """tensor_like() metadata inference from host (NumPy) DLPack producers."""

    def test_shape_strides_dtype_and_runtime_scalar(self):
        t = hipdnn.Graph.tensor_like(np.zeros((2, 3, 4), np.float32), "x")
        assert t.get_dim() == [2, 3, 4]
        assert t.get_stride() == [12, 4, 1]
        assert t.get_data_type() == hipdnn.DataType.FLOAT
        assert t.get_name() == "x"
        assert t.has_uid() is False
        # cuDNN: every host tensor is a runtime pass-by-value tensor.
        assert t.get_is_pass_by_value() is True

    def test_non_contiguous_strides(self):
        t = hipdnn.Graph.tensor_like(np.zeros((4, 6), np.float32)[:, ::2])
        assert t.get_dim() == [4, 3]
        assert t.get_stride() == [6, 2]

    def test_zero_dim_keeps_empty_shape(self):
        t = hipdnn.Graph.tensor_like(np.array(2.5, np.float32))
        assert t.get_dim() == []
        assert t.get_stride() == []
        assert t.get_is_pass_by_value() is True

    @pytest.mark.parametrize(
        "np_dtype, data_type",
        [
            (np.float16, hipdnn.DataType.HALF),
            (np.float64, hipdnn.DataType.DOUBLE),
            (np.int8, hipdnn.DataType.INT8),
            (np.int32, hipdnn.DataType.INT32),
            (np.int64, hipdnn.DataType.INT64),
            (np.uint8, hipdnn.DataType.UINT8),
            (np.bool_, hipdnn.DataType.BOOLEAN),
        ],
    )
    def test_dtype_mapping(self, np_dtype, data_type):
        t = hipdnn.Graph.tensor_like(np.zeros((1,), np_dtype))
        assert t.get_data_type() == data_type

    def test_unsupported_dtype(self):
        with pytest.raises(ValueError, match="unsupported DLPack dtype"):
            hipdnn.Graph.tensor_like(np.zeros((2, 2), np.uint16))

    def test_object_without_dlpack(self):
        with pytest.raises(TypeError, match="__dlpack__"):
            hipdnn.Graph.tensor_like(object())

    def test_malformed_capsule(self):
        with pytest.raises(ValueError, match="valid DLPack capsule"):
            hipdnn.Graph.tensor_like(_NoneDlpack())

    def test_tensor_overload_still_works(self):
        src = hipdnn.Tensor.create([2, 2], hipdnn.DataType.FLOAT)
        t = hipdnn.Graph.tensor_like(src, "copy")
        assert t.get_dim() == [2, 2]
        assert t.get_name() == "copy"
        assert t.get_is_pass_by_value() is False

    def test_set_is_pass_by_value_round_trips(self):
        t = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        assert t.set_is_pass_by_value(True).get_is_pass_by_value() is True

    def test_read_only_producer(self):
        host = np.zeros((2, 3), np.float32)
        host.setflags(write=False)
        t = hipdnn.Graph.tensor_like(host)
        assert t.get_dim() == [2, 3]
        assert _get_data_ptr(host) == host.ctypes.data


class TestDataPointerConversion:
    """Variant-pack value conversion, checked through the private hook."""

    def test_host_dlpack_honors_byte_offset(self):
        host = np.zeros(4, np.float32)
        producer = _OffsetProducer(host.ctypes.data, 8, device_type=1)
        assert _get_data_ptr(producer) == host.ctypes.data + 8

    def test_integer_like_values(self):
        assert _get_data_ptr(np.uint64(4096)) == 4096
        assert _get_data_ptr(np.int64(4096)) == 4096
        # A 0-d integer array implements __index__ too, but it is data, not a
        # pointer: DLPack takes precedence.
        scalar = np.array(7, np.int64)
        assert _get_data_ptr(scalar) == scalar.ctypes.data

    def test_data_ptr_is_used_before_dlpack(self):
        assert _get_data_ptr(_DataPtr()) == 4096

    def test_int_passthrough(self):
        assert _get_data_ptr(1234) == 1234

    def test_bool_rejected(self):
        with pytest.raises(TypeError):
            _get_data_ptr(True)

    def test_unsupported_type(self):
        with pytest.raises(TypeError, match="__dlpack__"):
            _get_data_ptr("0x10")

    def test_malformed_capsule(self):
        with pytest.raises(ValueError, match="valid DLPack capsule"):
            _get_data_ptr(_NoneDlpack())


@pytest.mark.gpu
class TestDlpackDevice:
    """Zero-copy device pointers and execution with DLPack producers."""

    def test_device_buffer(self):
        buf = hipdnn.DeviceBuffer(64)
        assert _get_data_ptr(buf) == buf.ptr()

    def test_torch_pointer_and_byte_offset(self):
        torch = _rocm_torch()
        x = torch.empty(16, dtype=torch.float32, device="cuda")
        assert _get_data_ptr(x) == x.data_ptr()
        assert _get_data_ptr(x[3:]) == x.data_ptr() + 12
        assert _get_data_ptr(_DlpackOnly(x[3:])) == x.data_ptr() + 12
        device = torch.cuda.current_device()
        producer = _OffsetProducer(x.data_ptr(), 12, device_type=10, device_id=device)
        assert _get_data_ptr(producer) == x.data_ptr() + 12

    def test_torch_tensor_like(self):
        torch = _rocm_torch()
        x = torch.empty(16, dtype=torch.float32, device="cuda")
        t = hipdnn.Graph.tensor_like(x.view(4, 4).t())
        assert t.get_dim() == [4, 4]
        assert t.get_stride() == [1, 4]
        assert t.get_is_pass_by_value() is False
        assert hipdnn.Graph.tensor_like(x.cpu()).get_is_pass_by_value() is True

    def test_execute_with_torch_tensors_and_tensor_keys(self):
        torch = _rocm_torch()
        graph, a, b, out = build_pointwise_add_graph()
        handle = build_all_plans(graph)
        tensors = {
            t: torch.empty(t.get_dim(), dtype=torch.float32, device="cuda")
            for t in (a, b, out)
        }
        uid_pack = {t.get_uid(): x for t, x in tensors.items()}
        ws_size = graph.get_workspace_size()
        workspace = (
            torch.empty(ws_size, dtype=torch.uint8, device="cuda") if ws_size else None
        )

        assert graph.execute(handle, tensors, workspace).is_good()
        assert graph.execute(handle, uid_pack, workspace).is_good()
        assert graph.execute_plan_at_index(handle, tensors, workspace, 0).is_good()

        buf = hipdnn.DeviceBuffer(tensors[b].nbytes)
        mixed = {a: tensors[a], b: buf, out.get_uid(): tensors[out].data_ptr()}
        assert graph.execute(handle, mixed, workspace).is_good()

        with pytest.raises(ValueError, match="has no uid"):
            graph.execute(handle, {hipdnn.Tensor(): tensors[a]}, workspace)

    def test_execute_with_dlpack_only_producers(self):
        """Values and workspace that expose only __dlpack__, on the device."""
        torch = _rocm_torch()
        graph, a, b, out = build_pointwise_add_graph()
        handle = build_all_plans(graph)
        pack = {
            t: _DlpackOnly(torch.empty(t.get_dim(), dtype=torch.float32, device="cuda"))
            for t in (a, b, out)
        }
        workspace = _DlpackOnly(
            torch.empty(
                max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda"
            )
        )
        assert graph.execute(handle, pack, workspace).is_good()
        assert graph.execute_plan_at_index(handle, pack, workspace, 0).is_good()

    def test_execute_with_numpy_integer_keys_and_pointers(self):
        """Integer-pointer packs built with NumPy scalars keep working."""
        graph, a, b, out = build_pointwise_add_graph()
        handle = build_all_plans(graph)
        buffers = {t: hipdnn.DeviceBuffer(4 * t.get_volume()) for t in (a, b, out)}
        pack = {
            np.int64(t.get_uid()): np.uint64(buf.ptr()) for t, buf in buffers.items()
        }
        assert graph.execute(handle, pack, np.uint64(0)).is_good()

    def test_host_runtime_scalar_reaches_the_engine(self):
        """cuDNN pattern: tensor_like(host) declares the scalar, execute passes it.

        Two values go through one compiled plan, so the engine must read the
        host tensor at execute time rather than a value baked in at build time.
        """
        plugin = (
            "test_pass_by_value_recorder_plugin.dll"
            if os.name == "nt"
            else "libtest_pass_by_value_recorder_plugin.so"
        )
        report = helpers.run_plugin_probe(
            "pass_by_value.py", plugin, "no engine records runtime scalars"
        )
        uid = report["scale_uid"]
        assert report["is_pass_by_value"] is True
        assert report["received"] == [[uid, 2.5], [uid, -7.0]]

    def test_autotune_with_torch_tensors(self):
        torch = _rocm_torch()
        graph, a, b, out = build_pointwise_add_graph()
        handle = hipdnn.create_handle()
        assert graph.validate().is_good()
        assert graph.build_operation_graph(handle).is_good()
        assert graph.create_execution_plans().is_good()
        assert graph.check_support().is_good()
        assert graph.build_plans(hipdnn.BuildPlanPolicy.ALL).is_good()
        tensors = {
            t: torch.empty(t.get_dim(), dtype=torch.float32, device="cuda")
            for t in (a, b, out)
        }
        ws_size = graph.get_autotune_workspace_size()
        workspace = (
            torch.empty(ws_size, dtype=torch.uint8, device="cuda") if ws_size else None
        )
        cfg = hipdnn.AutotuneConfig()
        cfg.strategy = hipdnn.AutotuneStrategy.FIXED_AVERAGE
        cfg.warmup_iterations = 1
        cfg.timed_iterations = 1

        results = graph.autotune(handle, tensors, workspace, config=cfg)
        assert any(result.succeeded for result in results)
