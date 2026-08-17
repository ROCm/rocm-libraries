# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for error and failure paths across the graph/handle/buffer APIs."""

from pathlib import Path

import numpy as np
import pytest

import hipdnn_frontend as hipdnn

from .graph_builders import build_conv_fprop_graph
from .helpers import build_all_plans, create_float_graph, stub_engine_active


def _hipdnn_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _error_code_names_from_header() -> list[str]:
    error_header = _hipdnn_project_root() / "frontend/include/hipdnn_frontend/Error.hpp"
    in_error_code_enum = False
    names: list[str] = []

    for line in error_header.read_text(encoding="utf-8").splitlines():
        if not in_error_code_enum:
            in_error_code_enum = "enum class ErrorCode" in line
            continue

        entry = line.split("//", maxsplit=1)[0].strip()
        if entry == "};":
            break
        if not entry or entry in {"{", "}"} or entry.startswith(("/", "*")):
            continue

        names.append(entry.split(",", maxsplit=1)[0].split(maxsplit=1)[0])

    return names


class TestErrorCodeBindings:
    """ErrorCode binding parity with the C++ frontend enum."""

    def test_error_codes_match_frontend_header(self):
        """Every C++ ErrorCode is exposed to Python and usable by Error."""
        for name in _error_code_names_from_header():
            code = getattr(hipdnn.ErrorCode, name)
            assert hipdnn.Error(code, "sentinel").get_code() == code


class TestValidationErrors:
    """Graph validation failures (no GPU required)."""

    def test_missing_tensor_dims_fail_validation(self):
        """A pointwise input tensor with unset dims fails validation."""
        graph = create_float_graph()

        a = hipdnn.Tensor()  # no dims/data type configured
        a.set_name("a")
        b = hipdnn.Tensor.create([8, 16], hipdnn.DataType.FLOAT)
        b.set_name("b")

        attrs = hipdnn.PointwiseAttributes()
        attrs.set_name("add")
        attrs.set_mode(hipdnn.PointwiseMode.ADD)

        out = graph.pointwise(a, b, attrs)
        out.set_name("out")
        out.set_output(True)

        assert not graph.validate().is_good()

    def test_duplicate_uids_fail_validation(self):
        """Two tensors sharing a UID fail validation."""
        graph = create_float_graph()

        a = hipdnn.Tensor.create([8, 16], hipdnn.DataType.FLOAT)
        a.set_name("a")
        a.set_uid(1)
        b = hipdnn.Tensor.create([8, 16], hipdnn.DataType.FLOAT)
        b.set_name("b")
        b.set_uid(1)

        attrs = hipdnn.PointwiseAttributes()
        attrs.set_name("add")
        attrs.set_mode(hipdnn.PointwiseMode.ADD)

        out = graph.pointwise(a, b, attrs)
        out.set_name("out")
        out.set_output(True)

        assert not graph.validate().is_good()


@pytest.mark.gpu
class TestExecutionErrors:
    """Execution and resource failures (require GPU)."""

    def test_missing_variant_pack_entry_fails_execute(self):
        """Executing with an incomplete variant pack returns a bad result."""
        if stub_engine_active():
            pytest.skip(
                "the ABSOLUTE-mode test stub's execute() is a no-op that never "
                "validates variant-pack completeness; only a real engine does"
            )

        graph, x, weight, y = build_conv_fprop_graph(
            n=1, c=2, h=8, w=8, k=4, r=3, s=3, stride=1, pad=1
        )
        handle = build_all_plans(graph)

        x_data = np.random.uniform(0.0, 1.0, x.get_dim()).astype(np.float32)
        x_buf = hipdnn.DeviceBuffer(x_data.nbytes)
        x_buf.copy_from_host(x_data.tobytes())

        # Variant pack only provides x; weight and y are missing.
        variant_pack = {x.get_uid(): x_buf.ptr()}

        result = graph.execute(handle, variant_pack, 0)
        assert result.is_bad()

    def test_destroyed_handle_get_stream_raises(self):
        """Reusing a destroyed handle raises RuntimeError."""
        handle = hipdnn.create_handle()
        hipdnn.destroy_handle(handle)
        with pytest.raises(RuntimeError):
            handle.get_stream()


@pytest.mark.gpu
class TestBufferErrors:
    """DeviceBuffer copy-size mismatches (require GPU)."""

    def test_copy_from_host_size_mismatch_raises(self):
        """copy_from_host() with the wrong byte count raises RuntimeError."""
        buf = hipdnn.DeviceBuffer(16)
        too_big = np.zeros(8, dtype=np.float32)  # 32 bytes
        with pytest.raises(RuntimeError):
            buf.copy_from_host(too_big.tobytes())
