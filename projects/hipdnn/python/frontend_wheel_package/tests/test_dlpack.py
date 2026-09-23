# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for DLPack interoperability in Graph.tensor_like and Graph.execute."""

import numpy as np
import pytest

import hipdnn_frontend as hipdnn
from hipdnn_frontend.hipdnn_frontend_python import _dlpack_device_ptr

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


class TestTensorLikeHost:
    """tensor_like() metadata inference from host (NumPy) DLPack producers."""

    def test_contiguous_shape_strides_dtype(self):
        t = hipdnn.Graph.tensor_like(np.zeros((2, 3, 4), np.float32), "x")
        assert t.get_dim() == [2, 3, 4]
        assert t.get_stride() == [12, 4, 1]
        assert t.get_data_type() == hipdnn.DataType.FLOAT
        assert t.get_name() == "x"
        assert t.has_uid() is False
        assert t.get_is_pass_by_value() is False

    def test_non_contiguous_strides(self):
        t = hipdnn.Graph.tensor_like(np.zeros((4, 6), np.float32)[:, ::2])
        assert t.get_dim() == [4, 3]
        assert t.get_stride() == [6, 2]

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
        t = hipdnn.Graph.tensor_like(np.zeros((2, 2), np_dtype))
        assert t.get_data_type() == data_type

    def test_unsupported_dtype(self):
        with pytest.raises(ValueError, match="unsupported DLPack dtype"):
            hipdnn.Graph.tensor_like(np.zeros((2, 2), np.uint16))

    def test_host_scalar_is_pass_by_value(self):
        t = hipdnn.Graph.tensor_like(np.array([2.5], np.float32))
        assert t.get_is_pass_by_value() is True
        assert t.get_dim() == [1]
        assert t.get_data_type() == hipdnn.DataType.FLOAT

    def test_host_scalar_unsupported_dtype(self):
        with pytest.raises(ValueError, match="not supported for pass-by-value"):
            hipdnn.Graph.tensor_like(np.array([1], np.int8))

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


class TestDevicePointerConversion:
    """Variant-pack value conversion, checked through the private hook."""

    def test_host_dlpack_rejected(self):
        with pytest.raises(ValueError, match="ROCm device memory"):
            _dlpack_device_ptr(np.zeros(4, np.float32))

    def test_bool_rejected(self):
        with pytest.raises(TypeError):
            _dlpack_device_ptr(True)

    def test_int_passthrough(self):
        assert _dlpack_device_ptr(1234) == 1234

    def test_unsupported_type(self):
        with pytest.raises(TypeError, match="__dlpack__"):
            _dlpack_device_ptr("0x10")


@pytest.mark.gpu
class TestDlpackDevice:
    """Zero-copy device pointers and execution with DLPack producers."""

    def test_device_buffer(self):
        buf = hipdnn.DeviceBuffer(64)
        assert _dlpack_device_ptr(buf) == buf.ptr()

    def test_torch_pointer_and_byte_offset(self):
        torch = _rocm_torch()
        x = torch.empty(16, dtype=torch.float32, device="cuda")
        assert _dlpack_device_ptr(x) == x.data_ptr()
        assert _dlpack_device_ptr(x[3:]) == x.data_ptr() + 12

    def test_torch_tensor_like_strides(self):
        torch = _rocm_torch()
        x = torch.empty(16, dtype=torch.float32, device="cuda")
        t = hipdnn.Graph.tensor_like(x.view(4, 4).t())
        assert t.get_dim() == [4, 4]
        assert t.get_stride() == [1, 4]
        assert t.get_is_pass_by_value() is False

    def test_execute_with_torch_tensors(self):
        torch = _rocm_torch()
        graph, a, b, out = build_pointwise_add_graph()
        handle = build_all_plans(graph)
        tensors = {
            t.get_uid(): torch.empty(t.get_dim(), dtype=torch.float32, device="cuda")
            for t in (a, b, out)
        }
        ws_size = graph.get_workspace_size()
        workspace = (
            torch.empty(ws_size, dtype=torch.uint8, device="cuda") if ws_size else 0
        )

        assert graph.execute(handle, tensors, workspace).is_good()
        assert graph.execute_plan_at_index(handle, tensors, workspace, 0).is_good()

        buf = hipdnn.DeviceBuffer(tensors[b.get_uid()].nbytes)
        mixed = {
            a.get_uid(): tensors[a.get_uid()],
            b.get_uid(): buf,
            out.get_uid(): tensors[out.get_uid()].data_ptr(),
        }
        assert graph.execute(handle, mixed, workspace).is_good()

        host = dict(tensors)
        host[a.get_uid()] = np.zeros(a.get_dim(), np.float32)
        with pytest.raises(ValueError, match=rf"variant_pack\[{a.get_uid()}\]"):
            graph.execute(handle, host, workspace)
