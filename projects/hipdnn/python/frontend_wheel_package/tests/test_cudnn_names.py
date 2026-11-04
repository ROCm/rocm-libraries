# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""cuDNN module names, Graph.tensor(dim=..., ...), and dtype-object data types."""

import numpy as np
import pytest

import hipdnn_frontend as hipdnn


@pytest.mark.parametrize(
    "cudnn_name, hipdnn_name",
    [
        ("tensor", "Tensor"),
        ("data_type", "DataType"),
        ("heur_mode", "HeuristicMode"),
        ("norm_forward_phase", "NormFwdPhase"),
        ("reduction_mode", "ReductionMode"),
        ("convolution_mode", "ConvolutionMode"),
        ("build_plan_policy", "BuildPlanPolicy"),
        ("behavior_note", "BehaviorNote"),
        ("diagonal_alignment", "DiagonalAlignment"),
        ("attention_implementation", "AttentionImplementation"),
        ("moe_grouped_matmul_mode", "MoeGroupedMatmulMode"),
        ("scalar_type", "ScalarType"),
        ("cudnnGraphNotSupportedError", "GraphNotSupportedError"),
    ],
)
def test_cudnn_module_name_is_the_hipdnn_object(cudnn_name, hipdnn_name):
    assert getattr(hipdnn, cudnn_name) is getattr(hipdnn, hipdnn_name)


def test_pygraph_and_error_class_are_exported():
    assert issubclass(hipdnn.pygraph, hipdnn.Graph)
    assert issubclass(hipdnn.GraphNotSupportedError, RuntimeError)


def test_set_data_type_takes_cudnn_keyword_and_default():
    tensor = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
    assert tensor.set_data_type(data_type=hipdnn.data_type.HALF) is tensor
    assert tensor.get_data_type() == hipdnn.DataType.HALF
    assert tensor.set_data_type() is tensor
    assert tensor.get_data_type() == hipdnn.DataType.NOT_SET


class TestCudnnTensor:
    def test_keywords_round_trip(self):
        tensor = hipdnn.Graph().tensor(
            name="X", dim=[2, 3, 4], stride=[12, 4, 1], data_type=hipdnn.data_type.HALF
        )
        assert tensor.get_name() == "X"
        assert tensor.get_dim() == [2, 3, 4]
        assert tensor.get_stride() == [12, 4, 1]
        assert tensor.get_data_type() == hipdnn.DataType.HALF
        assert not tensor.has_uid()

    def test_positional_dim_and_stride_and_flags(self):
        tensor = hipdnn.Graph.tensor(
            (4, 1),
            (1, 1),
            hipdnn.DataType.FLOAT,
            is_virtual=True,
            is_pass_by_value=True,
            uid=7,
        )
        assert tensor.get_dim() == [4, 1]
        assert tensor.get_is_virtual() is True
        assert tensor.get_is_pass_by_value() is True
        assert tensor.get_uid() == 7

    def test_default_data_type_is_not_set(self):
        tensor = hipdnn.Graph.tensor(dim=[1], stride=[1])
        assert tensor.get_data_type() == hipdnn.DataType.NOT_SET

    def test_reordering_type_raises(self):
        with pytest.raises(NotImplementedError, match="reordering_type"):
            hipdnn.Graph.tensor(dim=[1], stride=[1], reordering_type=0)

    def test_ragged_offset_is_set_on_the_tensor(self):
        offset = hipdnn.Tensor.create([2, 1, 1, 1], hipdnn.DataType.INT32)
        tensor = hipdnn.Graph.tensor(dim=[1], stride=[1], ragged_offset=offset)
        assert tensor.has_ragged_offset()
        assert tensor.get_ragged_offset() is offset

    def test_ragged_offset_rejects_a_non_tensor(self):
        with pytest.raises(TypeError):
            hipdnn.Graph.tensor(dim=[1], stride=[1], ragged_offset=0)

    def test_tensor_like_input_keyword(self):
        source = hipdnn.Tensor.create([3], hipdnn.DataType.FLOAT)
        copy = hipdnn.Graph.tensor_like(input=source, name="copy")
        assert copy.get_dim() == [3]
        assert copy.get_name() == "copy"


class TestDtypeObjects:
    @pytest.mark.parametrize(
        "dtype, expected",
        [
            (np.float16, hipdnn.DataType.HALF),
            (np.dtype("float32"), hipdnn.DataType.FLOAT),
            (np.float64, hipdnn.DataType.DOUBLE),
            (np.dtype("int32"), hipdnn.DataType.INT32),
            (np.int64, hipdnn.DataType.INT64),
            (np.uint8, hipdnn.DataType.UINT8),
            (np.int8, hipdnn.DataType.INT8),
            (np.bool_, hipdnn.DataType.BOOLEAN),
        ],
    )
    def test_numpy_dtypes(self, dtype, expected):
        assert (
            hipdnn.Graph.tensor(dim=[1], stride=[1], data_type=dtype).get_data_type()
            == expected
        )
        tensor = hipdnn.Tensor.create([1], hipdnn.DataType.NOT_SET)
        assert tensor.set_data_type(dtype) is tensor
        assert tensor.get_data_type() == expected

    @pytest.mark.parametrize("scalar_type", [np.intc, np.longlong, np.int_, np.half])
    def test_numpy_scalar_type_aliases_map_like_their_dtype(self, scalar_type):
        # Aliases such as numpy.intc have a class name that is not the canonical
        # dtype name, so they must resolve through numpy.dtype().
        expected = hipdnn.Graph.tensor(
            dim=[1], stride=[1], data_type=np.dtype(scalar_type)
        ).get_data_type()
        assert (
            hipdnn.Graph.tensor(
                dim=[1], stride=[1], data_type=scalar_type
            ).get_data_type()
            == expected
        )
        assert hipdnn.Tensor().set_data_type(scalar_type).get_data_type() == expected

    def test_numpy_dtype_without_hipdnn_type_raises_value_error(self):
        with pytest.raises(ValueError, match="complex64"):
            hipdnn.Graph.tensor(dim=[1], stride=[1], data_type=np.complex64)
        with pytest.raises(ValueError, match="complex64"):
            hipdnn.Tensor().set_data_type(np.dtype("complex64"))

    def test_torch_dtypes(self):
        torch = pytest.importorskip("torch")
        cases = {
            torch.float16: hipdnn.DataType.HALF,
            torch.half: hipdnn.DataType.HALF,
            torch.bfloat16: hipdnn.DataType.BFLOAT16,
            torch.float32: hipdnn.DataType.FLOAT,
            torch.int64: hipdnn.DataType.INT64,
            torch.bool: hipdnn.DataType.BOOLEAN,
        }
        for dtype, expected in cases.items():
            assert (
                hipdnn.Graph.tensor(
                    dim=[1], stride=[1], data_type=dtype
                ).get_data_type()
                == expected
            )
