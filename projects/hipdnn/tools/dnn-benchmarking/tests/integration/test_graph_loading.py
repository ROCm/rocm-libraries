# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for graph loading."""

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from dnn_benchmarking.common.exceptions import GraphLoadError
from dnn_benchmarking.graph import GraphLoader, TensorInfo


class TestGraphLoading:
    """Integration tests for GraphLoader."""

    def test_load_sample_conv_fwd_json(self) -> None:
        """Test loading the sample conv fwd JSON file."""
        sample_path = (
            Path(__file__).parent.parent.parent / "graphs" / "sample_conv_fwd.json"
        )

        if not sample_path.exists():
            pytest.skip(f"Sample graph not found: {sample_path}")

        loader = GraphLoader()
        graph_json = loader.load_json(sample_path)

        assert graph_json["name"] == "sample_conv_fwd_16x16x16x16_k16_3x3"
        assert len(graph_json["tensors"]) == 3
        assert len(graph_json["nodes"]) == 1

    def test_validate_sample_conv_fwd(self) -> None:
        """Test validating the sample conv fwd graph."""
        sample_path = (
            Path(__file__).parent.parent.parent / "graphs" / "sample_conv_fwd.json"
        )

        if not sample_path.exists():
            pytest.skip(f"Sample graph not found: {sample_path}")

        loader = GraphLoader()
        graph_json = loader.load_json(sample_path)

        # Should not raise
        loader.validate(graph_json)

    def test_extract_tensor_info(self, sample_conv_fwd_json: Dict[str, Any]) -> None:
        """Test extracting tensor info from graph JSON."""
        loader = GraphLoader()
        tensor_infos = loader.extract_tensor_info(sample_conv_fwd_json)

        # Should have 3 non-virtual tensors
        assert len(tensor_infos) == 3

        # Check UIDs (0=output, 1=input_x, 2=weight)
        uids = {ti.uid for ti in tensor_infos}
        assert uids == {0, 1, 2}

        # Check output tensor is marked correctly
        output_tensors = [ti for ti in tensor_infos if ti.is_output]
        assert len(output_tensors) == 1
        assert output_tensors[0].uid == 0

    def test_tensor_info_size_calculation(
        self, sample_conv_fwd_json: Dict[str, Any]
    ) -> None:
        """Test tensor size calculation."""
        loader = GraphLoader()
        tensor_infos = loader.extract_tensor_info(sample_conv_fwd_json)

        # Input tensor: [16, 16, 16, 16] float32 = 16*16*16*16*4 bytes
        input_tensor = next(ti for ti in tensor_infos if ti.uid == 1)
        assert input_tensor.dims == [16, 16, 16, 16]
        assert input_tensor.num_elements == 16 * 16 * 16 * 16
        assert input_tensor.size_bytes == 16 * 16 * 16 * 16 * 4  # float32

        # Weight tensor: [16, 16, 3, 3] float32
        weight_tensor = next(ti for ti in tensor_infos if ti.uid == 2)
        assert weight_tensor.dims == [16, 16, 3, 3]
        assert weight_tensor.num_elements == 16 * 16 * 3 * 3
        assert weight_tensor.size_bytes == 16 * 16 * 3 * 3 * 4

    def test_load_from_temp_file(
        self, temp_json_file: Path, sample_conv_fwd_json: Dict[str, Any]
    ) -> None:
        """Test loading from a temporary JSON file."""
        loader = GraphLoader()
        graph_json = loader.load_json(temp_json_file)

        assert graph_json["name"] == sample_conv_fwd_json["name"]
        assert len(graph_json["tensors"]) == len(sample_conv_fwd_json["tensors"])

    def test_load_nonexistent_file_raises(self) -> None:
        """Test that loading a nonexistent file raises GraphLoadError."""
        loader = GraphLoader()

        with pytest.raises(GraphLoadError, match="not found"):
            loader.load_json(Path("/nonexistent/path/graph.json"))

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        """Test that loading invalid JSON raises GraphLoadError."""
        invalid_json = tmp_path / "invalid.json"
        invalid_json.write_text("{ invalid json }")

        loader = GraphLoader()

        with pytest.raises(GraphLoadError, match="Invalid JSON"):
            loader.load_json(invalid_json)

    def test_get_graph_name(self, sample_conv_fwd_json: Dict[str, Any]) -> None:
        """Test getting graph name."""
        loader = GraphLoader()
        name = loader.get_graph_name(sample_conv_fwd_json)

        assert name == "sample_conv_fwd_16x16x16x16_k16_3x3"

    def test_get_graph_name_default(self) -> None:
        """Test default graph name when not specified."""
        loader = GraphLoader()
        name = loader.get_graph_name({})

        assert name == "unnamed_graph"

    def test_get_engine_id(self, sample_conv_fwd_json: Dict[str, Any]) -> None:
        """Test getting engine ID from graph."""
        loader = GraphLoader()
        engine_id = loader.get_engine_id(sample_conv_fwd_json)

        # Current fixture doesn't include preferred_engine_id (hipDNN format)
        assert engine_id is None

    def test_get_engine_id_none(self) -> None:
        """Test getting engine ID when not specified."""
        loader = GraphLoader()
        engine_id = loader.get_engine_id({})

        assert engine_id is None

    def test_load_sample_matmul_json(self) -> None:
        """Test loading the sample matmul JSON file."""
        sample_path = (
            Path(__file__).parent.parent.parent / "graphs" / "sample_matmul.json"
        )

        if not sample_path.exists():
            pytest.skip(f"Sample graph not found: {sample_path}")

        loader = GraphLoader()
        graph_json = loader.load_json(sample_path)

        assert graph_json["name"] == "sample_matmul_256x512x1024"
        assert len(graph_json["tensors"]) == 3  # A, B, C
        assert len(graph_json["nodes"]) == 1
        assert graph_json["nodes"][0]["type"] == "MatmulAttributes"

    def test_validate_sample_matmul(self) -> None:
        """Test validating the sample matmul graph."""
        sample_path = (
            Path(__file__).parent.parent.parent / "graphs" / "sample_matmul.json"
        )

        if not sample_path.exists():
            pytest.skip(f"Sample graph not found: {sample_path}")

        loader = GraphLoader()
        graph_json = loader.load_json(sample_path)

        # Should not raise
        loader.validate(graph_json)

    def test_matmul_tensor_info(self) -> None:
        """Test tensor info extraction for matmul graph."""
        sample_path = (
            Path(__file__).parent.parent.parent / "graphs" / "sample_matmul.json"
        )

        if not sample_path.exists():
            pytest.skip(f"Sample graph not found: {sample_path}")

        loader = GraphLoader()
        graph_json = loader.load_json(sample_path)
        tensor_infos = loader.extract_tensor_info(graph_json)

        assert len(tensor_infos) == 3

        # A: [256, 512], B: [512, 1024], C: [256, 1024]
        a_tensor = next(ti for ti in tensor_infos if ti.uid == 1)
        assert a_tensor.dims == [256, 512]
        assert a_tensor.size_bytes == 256 * 512 * 4  # float32

        b_tensor = next(ti for ti in tensor_infos if ti.uid == 2)
        assert b_tensor.dims == [512, 1024]
        assert b_tensor.size_bytes == 512 * 1024 * 4

        c_tensor = next(ti for ti in tensor_infos if ti.uid == 3)
        assert c_tensor.dims == [256, 1024]
        assert c_tensor.size_bytes == 256 * 1024 * 4
        assert c_tensor.is_output

    def test_load_sample_relu_json(self) -> None:
        """Test loading the sample ReLU JSON file."""
        sample_path = (
            Path(__file__).parent.parent.parent / "graphs" / "sample_relu.json"
        )

        if not sample_path.exists():
            pytest.skip(f"Sample graph not found: {sample_path}")

        loader = GraphLoader()
        graph_json = loader.load_json(sample_path)

        assert graph_json["name"] == "sample_relu_activation_64x128x56x56"
        assert len(graph_json["tensors"]) == 2  # input, output
        assert len(graph_json["nodes"]) == 1
        assert graph_json["nodes"][0]["type"] == "PointwiseAttributes"

    def test_validate_sample_relu(self) -> None:
        """Test validating the sample ReLU graph."""
        sample_path = (
            Path(__file__).parent.parent.parent / "graphs" / "sample_relu.json"
        )

        if not sample_path.exists():
            pytest.skip(f"Sample graph not found: {sample_path}")

        loader = GraphLoader()
        graph_json = loader.load_json(sample_path)

        # Should not raise
        loader.validate(graph_json)

    def test_relu_tensor_info(self) -> None:
        """Test tensor info extraction for ReLU graph."""
        sample_path = (
            Path(__file__).parent.parent.parent / "graphs" / "sample_relu.json"
        )

        if not sample_path.exists():
            pytest.skip(f"Sample graph not found: {sample_path}")

        loader = GraphLoader()
        graph_json = loader.load_json(sample_path)
        tensor_infos = loader.extract_tensor_info(graph_json)

        assert len(tensor_infos) == 2

        # Input and output: [64, 128, 56, 56]
        input_tensor = next(ti for ti in tensor_infos if ti.uid == 1)
        assert input_tensor.dims == [64, 128, 56, 56]
        assert input_tensor.size_bytes == 64 * 128 * 56 * 56 * 4

        output_tensor = next(ti for ti in tensor_infos if ti.uid == 2)
        assert output_tensor.dims == [64, 128, 56, 56]
        assert output_tensor.is_output

    def test_load_sample_add_json(self) -> None:
        """Test loading the sample element-wise add JSON file."""
        sample_path = Path(__file__).parent.parent.parent / "graphs" / "sample_add.json"

        if not sample_path.exists():
            pytest.skip(f"Sample graph not found: {sample_path}")

        loader = GraphLoader()
        graph_json = loader.load_json(sample_path)

        assert graph_json["name"] == "sample_pointwise_add_128x256x14x14"
        assert len(graph_json["tensors"]) == 3  # x, y, z
        assert len(graph_json["nodes"]) == 1
        assert graph_json["nodes"][0]["type"] == "PointwiseAttributes"

    def test_validate_sample_add(self) -> None:
        """Test validating the sample add graph."""
        sample_path = Path(__file__).parent.parent.parent / "graphs" / "sample_add.json"

        if not sample_path.exists():
            pytest.skip(f"Sample graph not found: {sample_path}")

        loader = GraphLoader()
        graph_json = loader.load_json(sample_path)

        # Should not raise
        loader.validate(graph_json)

    def test_add_tensor_info(self) -> None:
        """Test tensor info extraction for add graph."""
        sample_path = Path(__file__).parent.parent.parent / "graphs" / "sample_add.json"

        if not sample_path.exists():
            pytest.skip(f"Sample graph not found: {sample_path}")

        loader = GraphLoader()
        graph_json = loader.load_json(sample_path)
        tensor_infos = loader.extract_tensor_info(graph_json)

        assert len(tensor_infos) == 3

        # All tensors: [128, 256, 14, 14]
        for ti in tensor_infos:
            assert ti.dims == [128, 256, 14, 14]
            assert ti.size_bytes == 128 * 256 * 14 * 14 * 4

        # Check output
        output_tensor = next(ti for ti in tensor_infos if ti.uid == 3)
        assert output_tensor.is_output

    def test_load_sample_batchnorm_json(self) -> None:
        """Test loading the sample batchnorm JSON file."""
        sample_path = (
            Path(__file__).parent.parent.parent / "graphs" / "sample_batchnorm.json"
        )

        if not sample_path.exists():
            pytest.skip(f"Sample graph not found: {sample_path}")

        loader = GraphLoader()
        graph_json = loader.load_json(sample_path)

        assert graph_json["name"] == "sample_batchnorm_inference_32x64x28x28"
        assert len(graph_json["tensors"]) == 6  # x, mean, inv_var, scale, bias, y
        assert len(graph_json["nodes"]) == 1
        assert graph_json["nodes"][0]["type"] == "BatchnormInferenceAttributes"

    def test_validate_sample_batchnorm(self) -> None:
        """Test validating the sample batchnorm graph."""
        sample_path = (
            Path(__file__).parent.parent.parent / "graphs" / "sample_batchnorm.json"
        )

        if not sample_path.exists():
            pytest.skip(f"Sample graph not found: {sample_path}")

        loader = GraphLoader()
        graph_json = loader.load_json(sample_path)

        # Should not raise
        loader.validate(graph_json)

    def test_batchnorm_tensor_info(self) -> None:
        """Test tensor info extraction for batchnorm graph."""
        sample_path = (
            Path(__file__).parent.parent.parent / "graphs" / "sample_batchnorm.json"
        )

        if not sample_path.exists():
            pytest.skip(f"Sample graph not found: {sample_path}")

        loader = GraphLoader()
        graph_json = loader.load_json(sample_path)
        tensor_infos = loader.extract_tensor_info(graph_json)

        assert len(tensor_infos) == 6

        # Input/output: [32, 64, 28, 28]
        x_tensor = next(ti for ti in tensor_infos if ti.uid == 1)
        assert x_tensor.dims == [32, 64, 28, 28]
        assert x_tensor.size_bytes == 32 * 64 * 28 * 28 * 4

        # Statistics tensors: [1, 64, 1, 1]
        mean_tensor = next(ti for ti in tensor_infos if ti.uid == 2)
        assert mean_tensor.dims == [1, 64, 1, 1]
        assert mean_tensor.size_bytes == 64 * 4

        # Output
        y_tensor = next(ti for ti in tensor_infos if ti.uid == 6)
        assert y_tensor.dims == [32, 64, 28, 28]
        assert y_tensor.is_output
