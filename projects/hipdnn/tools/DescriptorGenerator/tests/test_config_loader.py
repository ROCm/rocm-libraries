# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for codegen/config_loader.py.

Tests cover load_config() happy path, field parsing verification,
parser functions, validation rules, and error paths.
"""

import pytest

from codegen.config_loader import (
    ConfigError,
    _parse_enum_def,
    _parse_frontend_config,
    _parse_frontend_tensors,
    _parse_infer_properties,
    _parse_validation,
    _reject_deprecated_keys,
    _validate_config,
    load_config,
    validate_for_mode,
)
from codegen.models import (
    EnumDef,
    FrontendConfig,
    GraphMethodParam,
    InferPropertiesConfig,
    ValidationConfig,
)
from tests.helpers import (
    make_data_field,
    make_frontend_config,
    make_frontend_tensor_config,
    make_minimal_config,
    make_tensor_field,
    make_test_data,
)


# ---------------------------------------------------------------------------
# Task 2B.1: Happy path -- load all 10 configs without error
# ---------------------------------------------------------------------------

from tests.helpers import ALL_CONFIG_NAMES


@pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
def test_load_config_all_configs_succeed(load_test_config, config_name):
    """Every YAML config file loads without error."""
    config = load_test_config(config_name)
    assert config.name
    assert config.class_name
    assert config.fbs_table
    assert config.fbs_generated_header


class TestLoadConvolutionFwd:
    """Verify key fields from the reference convolution_fwd config."""

    def test_class_name(self, convolution_fwd_config):
        assert convolution_fwd_config.class_name == "ConvolutionFwdOperationDescriptor"

    def test_tensor_fields_count(self, convolution_fwd_config):
        assert len(convolution_fwd_config.tensor_fields) == 3

    def test_tensor_field_names(self, convolution_fwd_config):
        names = [tf.name for tf in convolution_fwd_config.tensor_fields]
        assert names == ["x", "w", "y"]

    def test_data_fields_count(self, convolution_fwd_config):
        assert len(convolution_fwd_config.data_fields) == 5

    def test_data_field_names(self, convolution_fwd_config):
        names = [df.name for df in convolution_fwd_config.data_fields]
        assert "pre_padding" in names
        assert "conv_mode" in names

    def test_frontend_node_class(self, convolution_fwd_config):
        assert convolution_fwd_config.frontend.node_class == "ConvolutionFpropNode"

    def test_frontend_attributes_class(self, convolution_fwd_config):
        assert convolution_fwd_config.frontend.attributes_class == "ConvFpropAttributes"

    def test_frontend_inputs(self, convolution_fwd_config):
        input_names = [t.name for t in convolution_fwd_config.frontend.inputs]
        assert input_names == ["x", "w"]

    def test_frontend_outputs(self, convolution_fwd_config):
        output_names = [t.name for t in convolution_fwd_config.frontend.outputs]
        assert output_names == ["y"]

    def test_has_compute_data_type(self, convolution_fwd_config):
        assert convolution_fwd_config.has_compute_data_type is True
        assert (
            convolution_fwd_config.compute_data_type_attr
            == "HIPDNN_ATTR_CONVOLUTION_COMP_TYPE"
        )

    def test_error_label(self, convolution_fwd_config):
        assert convolution_fwd_config.error_label == "conv"

    def test_operation_type_enum(self, convolution_fwd_config):
        assert (
            convolution_fwd_config.operation_type_enum
            == "HIPDNN_OPERATION_TYPE_CONVOLUTION_FORWARD_EXT"
        )

    def test_infer_properties(self, convolution_fwd_config):
        assert convolution_fwd_config.infer_properties is not None
        assert convolution_fwd_config.infer_properties.strategy == "stub"

    def test_validation(self, convolution_fwd_config):
        assert convolution_fwd_config.validation is not None
        assert "x" in convolution_fwd_config.validation.required_input_tensors


class TestLoadMatmul:
    """Verify matmul config (minimal: no data fields)."""

    def test_name(self, matmul_config):
        assert matmul_config.name == "Matmul"

    def test_no_data_fields(self, matmul_config):
        assert len(matmul_config.data_fields) == 0

    def test_three_tensor_fields(self, matmul_config):
        assert len(matmul_config.tensor_fields) == 3

    def test_no_mode_fields(self, matmul_config):
        assert matmul_config.has_mode_fields is False

    def test_no_tensor_array_fields(self, matmul_config):
        assert len(matmul_config.tensor_array_fields) == 0


class TestLoadPointwise:
    """Verify pointwise config (optional tensors, mode field, optional scalars)."""

    def test_optional_tensors(self, pointwise_config):
        optional = pointwise_config.optional_tensor_fields
        assert len(optional) == 2
        optional_names = {tf.name for tf in optional}
        assert optional_names == {"in_1", "in_2"}

    def test_mode_field(self, pointwise_config):
        assert pointwise_config.has_mode_fields is True
        mode_names = [df.name for df in pointwise_config.mode_fields]
        assert "operation" in mode_names

    def test_optional_scalars(self, pointwise_config):
        scalars = pointwise_config.optional_scalar_fields
        assert len(scalars) > 0
        scalar_names = {df.name for df in scalars}
        assert "relu_lower_clip" in scalar_names

    def test_has_enum_def(self, pointwise_config):
        mode_field = next(
            df for df in pointwise_config.data_fields if df.name == "operation"
        )
        assert mode_field.has_enum_def is True
        assert mode_field.enum_def is not None
        assert len(mode_field.enum_def.values) > 0


class TestLoadBatchnorm:
    """Verify batchnorm config (tensor arrays, many optional tensors)."""

    def test_tensor_array_fields(self, batchnorm_config):
        assert len(batchnorm_config.tensor_array_fields) == 1
        taf = batchnorm_config.tensor_array_fields[0]
        assert taf.name == "peer_stats"
        assert taf.test_uids == [515, 516]
        assert taf.test_label == "PeerStats"

    def test_many_optional_tensors(self, batchnorm_config):
        optional = batchnorm_config.optional_tensor_fields
        assert len(optional) >= 5

    def test_has_tensor_array_fields(self, batchnorm_config):
        assert batchnorm_config.has_tensor_array_fields is True


# ---------------------------------------------------------------------------
# Task 2B.2: Field parsing verification
# ---------------------------------------------------------------------------


class TestTensorFieldParsing:
    """Verify tensor fields are parsed with correct attributes."""

    def test_tensor_field_name(self, convolution_fwd_config):
        x = convolution_fwd_config.tensor_fields[0]
        assert x.name == "x"

    def test_tensor_field_fbs_field(self, convolution_fwd_config):
        x = convolution_fwd_config.tensor_fields[0]
        assert x.fbs_field == "x_tensor_uid"

    def test_tensor_field_attr_suffix(self, convolution_fwd_config):
        x = convolution_fwd_config.tensor_fields[0]
        assert x.attr_suffix == "X"

    def test_tensor_field_required_default(self, convolution_fwd_config):
        x = convolution_fwd_config.tensor_fields[0]
        assert x.required is True

    def test_tensor_field_frontend_getter(self, convolution_fwd_config):
        x = convolution_fwd_config.tensor_fields[0]
        # Phase 2 cleanup removed tensor_fields[].frontend_getter from
        # convolution_fwd.yaml; the parser must tolerate its absence and
        # leave the field at its default empty string.
        assert x.frontend_getter == ""

    def test_optional_tensor_field(self, pointwise_config):
        in_1 = next(tf for tf in pointwise_config.tensor_fields if tf.name == "in_1")
        assert in_1.required is False


class TestDataFieldParsing:
    """Verify data fields are parsed with correct types and attributes."""

    def test_vector_int64_type(self, convolution_fwd_config):
        padding = next(
            df for df in convolution_fwd_config.data_fields if df.name == "pre_padding"
        )
        assert padding.type == "vector_int64"
        assert padding.is_vector is True

    def test_mode_type(self, convolution_fwd_config):
        mode = next(
            df for df in convolution_fwd_config.data_fields if df.name == "conv_mode"
        )
        assert mode.type == "mode"
        assert mode.is_mode is True
        assert mode.cpp_enum == "hipdnn_flatbuffers_sdk::data_objects::ConvMode"
        assert mode.frontend_type == "ConvolutionMode"
        assert mode.backend_setter == "setConvMode"
        assert mode.backend_getter == "getConvMode"
        assert mode.backend_type_name == "HIPDNN_TYPE_CONVOLUTION_MODE"

    def test_scalar_float_type(self, pointwise_config):
        clip = next(
            df for df in pointwise_config.data_fields if df.name == "relu_lower_clip"
        )
        assert clip.type == "scalar_float"
        assert clip.is_scalar is True
        assert clip.required is False

    def test_bool_type(self, sdpa_config):
        gen_stats = next(
            df for df in sdpa_config.data_fields if df.name == "generate_stats"
        )
        assert gen_stats.type == "bool"
        assert gen_stats.is_scalar is True

    def test_scalar_int64_type(self, pointwise_config):
        axis = next(df for df in pointwise_config.data_fields if df.name == "axis")
        assert axis.type == "scalar_int64"
        assert axis.is_scalar is True

    def test_data_field_shared_flag(self, sdpa_config):
        diag = next(
            df for df in sdpa_config.data_fields if df.name == "diagonal_alignment"
        )
        assert diag.shared is True

    def test_data_field_test_values(self, convolution_fwd_config):
        padding = next(
            df for df in convolution_fwd_config.data_fields if df.name == "pre_padding"
        )
        assert padding.test_value == [2, 3]
        assert padding.test_label == "Convolution"
        assert padding.test_constant_name == "K_FPROP_CONV_PADDING"


class TestTensorArrayFieldParsing:
    """Verify tensor array fields are parsed correctly."""

    def test_tensor_array_name(self, batchnorm_config):
        taf = batchnorm_config.tensor_array_fields[0]
        assert taf.name == "peer_stats"

    def test_tensor_array_fbs_field(self, batchnorm_config):
        taf = batchnorm_config.tensor_array_fields[0]
        assert taf.fbs_field == "peer_stats_tensor_uid"

    def test_tensor_array_attr_name(self, batchnorm_config):
        taf = batchnorm_config.tensor_array_fields[0]
        assert taf.attr_name == "HIPDNN_ATTR_OPERATION_BATCHNORM_PEER_STATS_EXT"

    def test_tensor_array_test_uids(self, batchnorm_config):
        taf = batchnorm_config.tensor_array_fields[0]
        assert taf.test_uids == [515, 516]

    def test_tensor_array_test_label(self, batchnorm_config):
        taf = batchnorm_config.tensor_array_fields[0]
        assert taf.test_label == "PeerStats"

    def test_tensor_array_required_default(self, batchnorm_config):
        taf = batchnorm_config.tensor_array_fields[0]
        assert taf.required is False


class TestExtraDataTypeFieldParsing:
    """Verify extra_data_type_fields are parsed correctly (mma_core_mode on SDPA)."""

    def test_sdpa_has_extra_data_type_field(self, sdpa_config):
        assert len(sdpa_config.extra_data_type_fields) == 1

    def test_sdpa_extra_data_type_name(self, sdpa_config):
        edt = sdpa_config.extra_data_type_fields[0]
        assert edt.name == "mma_core_mode"

    def test_sdpa_extra_data_type_attr(self, sdpa_config):
        edt = sdpa_config.extra_data_type_fields[0]
        assert edt.attr_name == "HIPDNN_ATTR_SDPA_FWD_MMA_CORE_MODE_EXT"

    def test_sdpa_extra_data_type_sentinel(self, sdpa_config):
        edt = sdpa_config.extra_data_type_fields[0]
        assert edt.sentinel == "DataType::NOT_SET"

    def test_sdpa_extra_data_type_member_access(self, sdpa_config):
        """SdpaAttributes exposes mma_core_mode as a public member."""
        edt = sdpa_config.extra_data_type_fields[0]
        assert edt.frontend_uses_member_access is True
        assert edt.frontend_member_name == "mma_core_mode"

    def test_other_configs_have_no_extra_data_type_fields(
        self, convolution_fwd_config, matmul_config
    ):
        assert convolution_fwd_config.extra_data_type_fields == []
        assert matmul_config.extra_data_type_fields == []


class TestTestDataParsing:
    """Verify test_data section is parsed correctly."""

    def test_tensor_uids(self, convolution_fwd_config):
        uids = convolution_fwd_config.test_data.tensor_uids
        assert uids == {"x": 1000, "w": 1001, "y": 1002}

    def test_tensor_configs(self, convolution_fwd_config):
        configs = convolution_fwd_config.test_data.tensor_configs
        assert "x" in configs
        assert configs["x"].dims == [1, 3, 32, 32]
        assert configs["x"].strides == [3072, 1024, 32, 1]

    def test_field_values(self, convolution_fwd_config):
        values = convolution_fwd_config.test_data.field_values
        assert values["pre_padding"] == [1, 1]
        assert values["dilation"] == [1, 1]

    def test_constants_include(self, convolution_fwd_config):
        assert (
            convolution_fwd_config.test_data.constants_include == "ConvFpropConstants"
        )

    def test_tensor_const_prefix(self, convolution_fwd_config):
        assert convolution_fwd_config.test_data.tensor_const_prefix == "K_FPROP_"


# ---------------------------------------------------------------------------
# Task 2B.3: _parse_frontend_config() and _parse_frontend_tensors()
# ---------------------------------------------------------------------------


class TestParseFrontendConfig:
    """Test _parse_frontend_config() edge cases."""

    def test_empty_frontend_returns_default(self):
        result = _parse_frontend_config({}, "test_op")
        assert isinstance(result, FrontendConfig)
        assert result.packer_function == ""
        assert result.inputs == []
        assert result.outputs == []

    def test_none_frontend_returns_default(self):
        result = _parse_frontend_config(None, "test_op")
        assert isinstance(result, FrontendConfig)
        assert result.inputs == []

    def test_parses_all_frontend_fields(self):
        raw = {
            "packer_function": "createTestOp",
            "node_class": "TestNode",
            "attributes_class": "TestAttributes",
            "attributes_include": "TestInc",
            "unpacker_function": "unpackTest",
            "unpacker_include": "TestUnpackInc",
            "graph_method_name": "test_method",
            "graph_return_type": "array",
            "graph_return_outputs": ["out"],
            "node_type_enum": "NodeType::TEST",
            "node_attributes_union_type": "TestAttributes",
            "compatibility_typedef": "Test_attributes",
            "inputs": [{"name": "x"}],
            "outputs": [{"name": "y"}],
        }
        result = _parse_frontend_config(raw, "test_op")
        assert result.packer_function == "createTestOp"
        assert result.node_class == "TestNode"
        assert result.graph_method_name == "test_method"
        assert result.graph_return_type == "array"
        assert len(result.inputs) == 1
        assert len(result.outputs) == 1

    def test_graph_method_params_parsing(self):
        raw = {
            "inputs": [{"name": "x"}],
            "graph_method_params": [
                {"name": "x", "tensor_name": "x"},
                {"name": "scale", "type": "float", "optional": True},
            ],
        }
        result = _parse_frontend_config(raw, "test_op")
        assert len(result.graph_method_params) == 2
        assert result.graph_method_params[0].tensor_name == "x"
        assert result.graph_method_params[1].type == "float"
        assert result.graph_method_params[1].optional is True


class TestParseFrontendTensors:
    """Test _parse_frontend_tensors() edge cases."""

    def test_string_short_form(self):
        """String tensors are expanded to dict form."""
        result = _parse_frontend_tensors(["x", "y"], "input", "test_op")
        assert len(result) == 2
        assert result[0].name == "x"
        assert result[1].name == "y"

    def test_auto_assigned_enum_value(self):
        """Tensors without enum_value get sequential values starting at 0."""
        result = _parse_frontend_tensors(
            [{"name": "a"}, {"name": "b"}, {"name": "c"}], "input", "test_op"
        )
        assert result[0].enum_value == 0
        assert result[1].enum_value == 1
        assert result[2].enum_value == 2

    def test_explicit_enum_value(self):
        """Explicit enum_value is respected and subsequent auto-assignment continues."""
        result = _parse_frontend_tensors(
            [{"name": "a", "enum_value": 5}, {"name": "b"}], "input", "test_op"
        )
        assert result[0].enum_value == 5
        assert result[1].enum_value == 6

    def test_explicit_enum_value_with_gap(self):
        """Auto-assignment after an explicit value produces sequential values."""
        result = _parse_frontend_tensors(
            [{"name": "a"}, {"name": "b", "enum_value": 10}, {"name": "c"}],
            "input",
            "test_op",
        )
        assert result[0].enum_value == 0
        assert result[1].enum_value == 10
        assert result[2].enum_value == 11

    def test_mixed_short_and_dict_form(self):
        """String and dict forms can be mixed in the same list."""
        result = _parse_frontend_tensors(
            ["x", {"name": "y", "required": False}], "input", "test_op"
        )
        assert result[0].name == "x"
        assert result[0].required is True
        assert result[1].name == "y"
        assert result[1].required is False

    def test_empty_list_returns_empty(self):
        result = _parse_frontend_tensors([], "input", "test_op")
        assert result == []


# ---------------------------------------------------------------------------
# Task 2B.4: _parse_enum_def()
# ---------------------------------------------------------------------------


class TestParseEnumDef:
    """Test _parse_enum_def() behavior."""

    def test_none_returns_none(self):
        assert _parse_enum_def(None) is None

    def test_valid_enum_def(self):
        raw = {
            "backend_header": "HipdnnTestMode.h",
            "backend_prefix": "HIPDNN_TEST_",
            "values": [
                {
                    "name": "UNSET",
                    "value": 0,
                    "sentinel": True,
                    "description": "Not set",
                },
                {
                    "name": "ADD",
                    "value": 1,
                    "sdk_name": "ADD_OP",
                    "frontend_name": "Add",
                    "frontend_value": 10,
                },
            ],
        }
        result = _parse_enum_def(raw)
        assert isinstance(result, EnumDef)
        assert result.backend_header == "HipdnnTestMode.h"
        assert result.backend_prefix == "HIPDNN_TEST_"
        assert len(result.values) == 2
        # Sentinel value
        assert result.values[0].sentinel is True
        assert result.values[0].name == "UNSET"
        # Non-sentinel value
        assert result.values[1].name == "ADD"
        assert result.values[1].sdk_name == "ADD_OP"
        assert result.values[1].frontend_name == "Add"
        assert result.values[1].frontend_value == 10

    def test_empty_values_prints_warning(self, capsys):
        raw = {
            "backend_header": "HipdnnTestMode.h",
            "backend_prefix": "HIPDNN_TEST_",
            "values": [],
        }
        result = _parse_enum_def(raw)
        assert isinstance(result, EnumDef)
        assert len(result.values) == 0
        captured = capsys.readouterr()
        assert "Warning" in captured.err
        assert "no values" in captured.err

    def test_enum_def_without_optional_fields(self):
        raw = {
            "values": [{"name": "FOO", "value": 1}],
        }
        result = _parse_enum_def(raw)
        assert result.backend_header == ""
        assert result.backend_prefix == ""
        assert result.values[0].sdk_name == ""
        assert result.values[0].frontend_name == ""
        assert result.values[0].frontend_value is None


# ---------------------------------------------------------------------------
# Task 2B.5: _parse_infer_properties() and _parse_validation()
# ---------------------------------------------------------------------------


class TestParseInferProperties:
    """Test _parse_infer_properties() behavior."""

    def test_none_returns_none(self):
        assert _parse_infer_properties(None) is None

    def test_valid_infer_properties(self):
        raw = {
            "strategy": "match_input",
            "reference_input": "x",
            "dimension_formula": "same as input",
        }
        result = _parse_infer_properties(raw)
        assert isinstance(result, InferPropertiesConfig)
        assert result.strategy == "match_input"
        assert result.reference_input == "x"
        assert result.dimension_formula == "same as input"

    def test_defaults_applied(self):
        result = _parse_infer_properties({})
        assert result.strategy == "stub"
        assert result.reference_input == ""
        assert result.dimension_formula == ""

    def test_strategy_broadcast_is_silently_accepted(self):
        """Documents the intentional silent fall-through for non-stub,
        non-match_input strategy values.

        ``broadcast`` and ``custom`` were proposed strategy values that
        never got implementation in the templates. Rather than break
        configs that still set them, the loader stores the raw string and
        the templates fall through to the ``stub`` branch. This test pins
        that behavior so a future contributor doesn't accidentally tighten
        it without updating the docs.
        """
        result = _parse_infer_properties({"strategy": "broadcast"})
        assert result.strategy == "broadcast"  # stored verbatim, not rejected
        result = _parse_infer_properties({"strategy": "custom"})
        assert result.strategy == "custom"

    @pytest.mark.parametrize("garbage", ["", "not-a-strategy", "STUB", "stub  "])
    def test_strategy_garbage_value_is_silently_accepted(self, garbage):
        """Pins the same silent-pass behavior for arbitrary garbage values.

        The loader does NOT validate the strategy string against a known
        allowlist — anything that isn't ``stub`` or ``match_input`` falls
        through to the ``stub`` template branch. This is the same contract
        as broadcast/custom; we pin garbage values explicitly so a future
        contributor who tightens validation has to update both this and
        the broadcast pin (and the docs in CLAUDE.md's "Removed" table).
        """
        result = _parse_infer_properties({"strategy": garbage})
        assert result.strategy == garbage  # stored verbatim, not rejected


class TestParseValidation:
    """Test _parse_validation() behavior."""

    def test_none_returns_none(self):
        assert _parse_validation(None) is None

    def test_valid_validation(self):
        raw = {
            "required_input_tensors": ["x", "w"],
            "required_input_dims": ["x"],
            "custom_checks": ["check something"],
        }
        result = _parse_validation(raw)
        assert isinstance(result, ValidationConfig)
        assert result.required_input_tensors == ["x", "w"]
        assert result.required_input_dims == ["x"]
        assert result.custom_checks == ["check something"]

    def test_defaults_applied(self):
        result = _parse_validation({})
        assert result.required_input_tensors == []
        assert result.required_input_dims == []
        assert result.custom_checks == []

    def test_dim_consistency_checks_is_rejected(self):
        """validation.dim_consistency_checks is rejected by the centralized
        deprecated-key driver alongside the data_fields and tensor_fields
        keys.
        """
        op = {
            "name": "TestOp",
            "validation": {
                "required_input_tensors": ["x"],
                "dim_consistency_checks": ["x.dim == y.dim"],
            },
        }
        with pytest.raises(ConfigError) as excinfo:
            _reject_deprecated_keys(op)
        msg = str(excinfo.value)
        assert "dim_consistency_checks" in msg
        assert "custom_checks" in msg
        assert "TestOp" in msg


# ---------------------------------------------------------------------------
# Task 2B.6: _validate_config() error paths (6 rules)
# ---------------------------------------------------------------------------


class TestValidateConfigErrors:
    """Test _validate_config() raises ConfigError for invalid configs."""

    def test_missing_compute_data_type_attr(self):
        """has_compute_data_type=True but compute_data_type_attr is empty."""
        config = make_minimal_config(
            has_compute_data_type=True, compute_data_type_attr=""
        )
        with pytest.raises(ConfigError, match="compute_data_type_attr"):
            _validate_config(config)

    def test_enum_field_without_test_enum_value(self):
        """Enum field missing test_enum_value."""
        config = make_minimal_config(
            data_fields=[
                make_data_field(
                    name="my_enum",
                    type="enum",
                    test_enum_value="",
                )
            ]
        )
        with pytest.raises(ConfigError, match="test_enum_value"):
            _validate_config(config)

    def test_mode_field_without_test_backend_value(self):
        """Mode field missing test_backend_value."""
        config = make_minimal_config(
            data_fields=[
                make_data_field(
                    name="my_mode",
                    type="mode",
                    test_enum_value="FOO",
                    test_backend_value="",
                    backend_setter="setFoo",
                    backend_getter="getFoo",
                    backend_type_name="HIPDNN_TYPE_FOO",
                )
            ]
        )
        with pytest.raises(ConfigError, match="test_backend_value"):
            _validate_config(config)

    def test_mode_field_without_backend_setter_getter(self):
        """Mode field missing backend_setter."""
        config = make_minimal_config(
            data_fields=[
                make_data_field(
                    name="my_mode",
                    type="mode",
                    test_enum_value="FOO",
                    test_backend_value="HIPDNN_FOO",
                    backend_setter="",
                    backend_getter="",
                    backend_type_name="HIPDNN_TYPE_FOO",
                )
            ]
        )
        with pytest.raises(ConfigError, match="backend_setter"):
            _validate_config(config)

    def test_mode_field_without_backend_type_name(self):
        """Mode field missing backend_type_name."""
        config = make_minimal_config(
            data_fields=[
                make_data_field(
                    name="my_mode",
                    type="mode",
                    test_enum_value="FOO",
                    test_backend_value="HIPDNN_FOO",
                    backend_setter="setFoo",
                    backend_getter="getFoo",
                    backend_type_name="",
                )
            ]
        )
        with pytest.raises(ConfigError, match="backend_type_name"):
            _validate_config(config)

    def test_missing_tensor_in_test_data_uids(self):
        """Tensor field not present in test_data.tensor_uids."""
        config = make_minimal_config(
            tensor_fields=[
                make_tensor_field(name="x"),
                make_tensor_field(name="y", fbs_field="y_tensor_uid", attr_suffix="Y"),
                make_tensor_field(name="z", fbs_field="z_tensor_uid", attr_suffix="Z"),
            ],
            test_data=make_test_data(tensor_uids={"x": 1, "y": 2}),
        )
        with pytest.raises(ConfigError, match="tensor_uids"):
            _validate_config(config)


class TestValidateConfigWarnings:
    """Test _validate_config() prints warnings for soft requirements."""

    def test_mode_field_without_frontend_inverse_converter(self, capsys):
        """Mode field without frontend_inverse_converter produces a warning."""
        config = make_minimal_config(
            data_fields=[
                make_data_field(
                    name="my_mode",
                    type="mode",
                    test_enum_value="FOO",
                    test_backend_value="HIPDNN_FOO",
                    backend_setter="setFoo",
                    backend_getter="getFoo",
                    backend_type_name="HIPDNN_TYPE_FOO",
                    frontend_inverse_converter="",
                    test_alt_enum_value="BAR",
                )
            ]
        )
        _validate_config(config)
        captured = capsys.readouterr()
        assert "frontend_inverse_converter" in captured.err

    def test_mode_field_without_test_alt_enum_value(self, capsys):
        """Mode field without test_alt_enum_value produces a warning."""
        config = make_minimal_config(
            data_fields=[
                make_data_field(
                    name="my_mode",
                    type="mode",
                    test_enum_value="FOO",
                    test_backend_value="HIPDNN_FOO",
                    backend_setter="setFoo",
                    backend_getter="getFoo",
                    backend_type_name="HIPDNN_TYPE_FOO",
                    frontend_inverse_converter="fromFoo",
                    test_alt_enum_value="",
                )
            ]
        )
        _validate_config(config)
        captured = capsys.readouterr()
        assert "test_alt_enum_value" in captured.err

    @pytest.mark.parametrize("value", [True, False, None, ""])
    def test_data_field_frontend_getter_returns_optional_is_rejected(self, value):
        """data_fields[].frontend_getter_returns_optional is rejected on
        key-presence, not value-truthiness. Setting it to ``False`` or
        ``None`` still raises — the contract is that the key must not
        appear at all.
        """
        op = {
            "name": "TestOp",
            "data_fields": [
                {
                    "name": "legacy_optional_field",
                    "frontend_getter_returns_optional": value,
                }
            ],
        }
        with pytest.raises(ConfigError) as excinfo:
            _reject_deprecated_keys(op)
        msg = str(excinfo.value)
        assert "frontend_getter_returns_optional" in msg
        assert "fbs_optional" in msg
        assert "legacy_optional_field" in msg
        assert "TestOp" in msg

    def test_data_field_without_frontend_getter_returns_optional_passes(self):
        """No error when frontend_getter_returns_optional is absent."""
        op = {"name": "TestOp", "data_fields": [{"name": "plain_field"}]}
        # Should not raise.
        _reject_deprecated_keys(op)

    # --- Phase 0.3: mode_sentinel auto-detect warning ---

    def test_mode_field_no_sentinel_emits_auto_default_warning(self, capsys):
        """A mode field whose enum_def has no sentinel and no mode_sentinel set
        triggers a one-line auto-default warning so the author can opt into
        ``mode_sentinel: none`` explicitly."""
        from codegen.models import EnumDef, EnumValue

        ed = EnumDef(
            values=[EnumValue(name="ADD", value=0), EnumValue(name="MUL", value=1)]
        )
        config = make_minimal_config(
            data_fields=[
                make_data_field(
                    name="my_unsentineled_mode",
                    type="mode",
                    enum_def=ed,
                    mode_sentinel=None,
                    backend_setter="setMyMode",
                    backend_getter="getMyMode",
                    backend_type_name="HIPDNN_TYPE_MY_MODE",
                    test_enum_value="ADD",
                    test_backend_value="HIPDNN_ADD",
                    test_alt_enum_value="MUL",
                    frontend_inverse_converter="fromMyMode",
                )
            ]
        )
        _validate_config(config)
        captured = capsys.readouterr()
        assert "my_unsentineled_mode" in captured.err
        assert "mode_sentinel" in captured.err

    def test_mode_field_with_explicit_mode_sentinel_no_auto_default_warning(
        self, capsys
    ):
        """Setting mode_sentinel explicitly silences the auto-default warning."""
        from codegen.models import EnumDef, EnumValue

        ed = EnumDef(values=[EnumValue(name="ADD", value=0)])
        config = make_minimal_config(
            data_fields=[
                make_data_field(
                    name="silent_mode",
                    type="mode",
                    enum_def=ed,
                    mode_sentinel="none",
                    backend_setter="setSilent",
                    backend_getter="getSilent",
                    backend_type_name="HIPDNN_TYPE_SILENT_MODE",
                    test_enum_value="ADD",
                    test_backend_value="HIPDNN_ADD",
                    test_alt_enum_value="ADD",
                    frontend_inverse_converter="fromSilent",
                )
            ]
        )
        _validate_config(config)
        captured = capsys.readouterr()
        assert "auto-defaulted to mode_sentinel" not in captured.err

    # --- C5 restoration: tensor_fields[].frontend_getter is honored again ---

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("get_legacy()", "get_legacy"),
            ("get_legacy", "get_legacy"),
            ("", ""),
            (None, ""),
        ],
    )
    def test_tensor_fields_frontend_getter_accepted(self, tmp_path, raw, expected):
        """C5: tensor_fields[].frontend_getter is honored as an accessor
        override and stored on the model with the trailing ``()`` stripped.
        Also accepts bare names, empty strings, and missing keys.
        """
        yaml_content = f"""
operation:
  name: "TestOp"
  class_name: "TestOpDescriptor"
  fbs_table: "TestOpAttributes"
  fbs_generated_header: "test_op_attributes_generated.h"
  has_compute_data_type: false
  tensor_fields:
    - name: "x"
      fbs_field: "x_tensor_uid"
      attr_suffix: "X"
      required: true
"""
        if raw is not None:
            yaml_content += f'      frontend_getter: "{raw}"\n'
        yaml_content += """  test_data:
    tensor_uids: { x: 1 }
    tensor_configs:
      x: { dims: [1], strides: [1] }
"""
        config_path = tmp_path / "test_op.yaml"
        config_path.write_text(yaml_content)
        config = load_config(str(config_path))
        assert config.tensor_fields[0].frontend_getter == expected

    def test_tensor_fields_frontend_getter_rejects_internal_parens(self, tmp_path):
        """A frontend_getter expression with internal parens (e.g.,
        ``get_x().nested()``) is malformed — the loader rejects it instead
        of silently accepting a value that won't render correctly."""
        yaml_content = """
operation:
  name: "TestOp"
  class_name: "TestOpDescriptor"
  fbs_table: "TestOpAttributes"
  fbs_generated_header: "test_op_attributes_generated.h"
  has_compute_data_type: false
  tensor_fields:
    - name: "x"
      fbs_field: "x_tensor_uid"
      attr_suffix: "X"
      required: true
      frontend_getter: "get_x().nested()"
  test_data:
    tensor_uids: { x: 1 }
    tensor_configs:
      x: { dims: [1], strides: [1] }
"""
        config_path = tmp_path / "bad.yaml"
        config_path.write_text(yaml_content)
        with pytest.raises(ConfigError) as excinfo:
            load_config(str(config_path))
        assert "frontend_getter" in str(excinfo.value)

    def test_tensor_fields_without_frontend_getter_passes(self):
        """No error when tensor_fields[].frontend_getter is absent."""
        op = {"name": "TestOp", "tensor_fields": [{"name": "x"}]}
        # Should not raise — only data_fields[].frontend_getter_returns_optional
        # and validation.dim_consistency_checks remain hard-rejected.
        _reject_deprecated_keys(op)

    # --- C5: tensor_field → frontend resolution hard-rejection ---

    def test_unresolved_tensor_field_raises(self):
        """A tensor_field that doesn't match any frontend input/output and
        carries no ``frontend_getter`` override raises a ``ConfigError``
        naming the operation, the unresolved tensor_field, and the
        available frontend tensors. Promotes the historical Risk-R4
        soft-warn to a hard reject so the silent ``attributes.()`` packer
        bug can't ship."""
        config = make_minimal_config(
            tensor_fields=[
                make_tensor_field(
                    name="orphan", fbs_field="orphan_uid", attr_suffix="O"
                ),
            ],
            frontend=make_frontend_config(
                inputs=[make_frontend_tensor_config(name="input_0")],
                outputs=[make_frontend_tensor_config(name="output_0", enum_value=1)],
            ),
            test_data=make_test_data(tensor_uids={"orphan": 1}),
        )
        with pytest.raises(ConfigError) as excinfo:
            _validate_config(config)
        msg = str(excinfo.value)
        assert "TestOp" in msg
        assert "orphan" in msg
        assert "input_0" in msg
        assert "output_0" in msg
        assert "frontend_getter" in msg

    def test_unresolved_tensor_field_with_override_passes(self):
        """A tensor_field that does not match any frontend input/output is
        accepted when it carries a ``frontend_getter`` override. The
        synthetic FrontendTensorConfig populates the resolution map."""
        config = make_minimal_config(
            tensor_fields=[
                make_tensor_field(
                    name="orphan",
                    fbs_field="orphan_uid",
                    attr_suffix="O",
                    frontend_getter="get_orphan",
                ),
            ],
            frontend=make_frontend_config(
                inputs=[make_frontend_tensor_config(name="input_0")],
                outputs=[make_frontend_tensor_config(name="output_0", enum_value=1)],
            ),
            test_data=make_test_data(tensor_uids={"orphan": 1}),
        )
        # Should not raise.
        _validate_config(config)

    def test_backend_only_config_emits_no_warning(self, capsys):
        """Configs without any frontend wiring (no inputs and no outputs)
        skip the resolution check entirely — backend-only configs are
        intentional and must stay quiet."""
        config = make_minimal_config(
            tensor_fields=[
                make_tensor_field(
                    name="orphan", fbs_field="orphan_uid", attr_suffix="O"
                ),
            ],
            frontend=make_frontend_config(inputs=[], outputs=[]),
            test_data=make_test_data(tensor_uids={"orphan": 1}),
        )
        _validate_config(config)
        err = capsys.readouterr().err
        assert "no matching frontend input/output" not in err


# ---------------------------------------------------------------------------
# Task 2B.7: validate_for_mode()
# ---------------------------------------------------------------------------


class TestValidateForMode:
    """Test validate_for_mode() frontend validation."""

    def test_non_frontend_mode_is_noop(self):
        """Non-frontend modes return without validation."""
        config = make_minimal_config()
        # Backend: no error even with empty frontend
        validate_for_mode(config, "backend")

    def test_frontend_mode_empty_inputs_raises(self):
        """Frontend mode with empty inputs raises ConfigError."""
        config = make_minimal_config(
            frontend=make_frontend_config(
                inputs=[],
                outputs=[make_frontend_tensor_config(name="y")],
            )
        )
        with pytest.raises(ConfigError, match="inputs must be non-empty"):
            validate_for_mode(config, "frontend")

    def test_frontend_mode_empty_outputs_raises(self):
        """Frontend mode with empty outputs raises ConfigError."""
        config = make_minimal_config(
            frontend=make_frontend_config(
                inputs=[make_frontend_tensor_config(name="x")],
                outputs=[],
            )
        )
        with pytest.raises(ConfigError, match="outputs must be non-empty"):
            validate_for_mode(config, "frontend")

    def test_full_mode_empty_inputs_raises(self):
        """Full mode also validates frontend requirements."""
        config = make_minimal_config(
            frontend=make_frontend_config(
                inputs=[],
                outputs=[make_frontend_tensor_config(name="y")],
            )
        )
        with pytest.raises(ConfigError, match="inputs must be non-empty"):
            validate_for_mode(config, "full")

    def test_missing_node_type_enum_warning(self, capsys):
        """Missing node_type_enum prints a warning."""
        config = make_minimal_config(
            frontend=make_frontend_config(
                inputs=[make_frontend_tensor_config(name="x")],
                outputs=[make_frontend_tensor_config(name="y")],
                node_type_enum="",
                node_attributes_union_type="TestAttrs",
            )
        )
        validate_for_mode(config, "frontend")
        captured = capsys.readouterr()
        assert "node_type_enum" in captured.err

    def test_missing_node_attributes_union_type_warning(self, capsys):
        """Missing node_attributes_union_type prints a warning."""
        config = make_minimal_config(
            frontend=make_frontend_config(
                inputs=[make_frontend_tensor_config(name="x")],
                outputs=[make_frontend_tensor_config(name="y")],
                node_type_enum="NodeType::TEST",
                node_attributes_union_type="",
            )
        )
        validate_for_mode(config, "frontend")
        captured = capsys.readouterr()
        assert "node_attributes_union_type" in captured.err

    def test_invalid_graph_method_params_tensor_name_warning(self, capsys):
        """graph_method_params referencing non-existent tensor prints warning."""
        config = make_minimal_config(
            frontend=FrontendConfig(
                packer_function="createTest",
                node_class="TestNode",
                attributes_class="TestAttributes",
                inputs=[make_frontend_tensor_config(name="x")],
                outputs=[make_frontend_tensor_config(name="y")],
                node_type_enum="NodeType::TEST",
                node_attributes_union_type="TestAttrs",
                graph_method_params=[
                    GraphMethodParam(name="bad", tensor_name="nonexistent")
                ],
            )
        )
        validate_for_mode(config, "frontend")
        captured = capsys.readouterr()
        assert "nonexistent" in captured.err

    def test_valid_frontend_config_no_error(self):
        """Valid frontend config passes without error."""
        config = make_minimal_config(
            frontend=make_frontend_config(
                inputs=[make_frontend_tensor_config(name="x")],
                outputs=[make_frontend_tensor_config(name="y")],
                node_type_enum="NodeType::TEST",
                node_attributes_union_type="TestAttrs",
            )
        )
        validate_for_mode(config, "frontend")


# ---------------------------------------------------------------------------
# Task 2B.8: Error paths -- load_config()
# ---------------------------------------------------------------------------


class TestLoadConfigErrors:
    """Test load_config() error handling."""

    def test_missing_operation_key(self, tmp_path):
        """YAML without top-level 'operation' key raises ConfigError."""
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("not_operation:\n  name: test\n")
        with pytest.raises(ConfigError, match="operation"):
            load_config(config_file)

    @pytest.mark.parametrize(
        "missing_field",
        ["name", "class_name", "fbs_table", "fbs_generated_header"],
    )
    def test_missing_required_field(self, tmp_path, missing_field):
        """Missing required field in operation raises ConfigError."""
        fields = {
            "name": "Test",
            "class_name": "TestDescriptor",
            "fbs_table": "TestTable",
            "fbs_generated_header": "test_generated.h",
        }
        del fields[missing_field]
        yaml_content = "operation:\n"
        for k, v in fields.items():
            yaml_content += f'  {k}: "{v}"\n'
        # Add minimal required data to avoid other errors
        yaml_content += "  has_compute_data_type: false\n"
        config_file = tmp_path / "bad.yaml"
        config_file.write_text(yaml_content)
        with pytest.raises(ConfigError, match=missing_field):
            load_config(config_file)

    def test_empty_operation_key(self, tmp_path):
        """Empty 'operation' key (evaluates to None) raises ConfigError."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("operation:\n")
        with pytest.raises(ConfigError, match="operation"):
            load_config(config_file)

    def test_invalid_mode_sentinel_value_is_rejected(self, tmp_path):
        """A mode_sentinel value outside {'required', 'optional', 'none'} is
        rejected at load time. Catches typos like ``"sometimes"`` before
        they reach template rendering, where they would silently fall
        through the finalize-check branch.
        """
        config_file = tmp_path / "bad_sentinel.yaml"
        config_file.write_text(
            "operation:\n"
            '  name: "Test"\n'
            '  class_name: "TestDescriptor"\n'
            '  fbs_table: "TestTable"\n'
            '  fbs_generated_header: "test_generated.h"\n'
            "  has_compute_data_type: false\n"
            "  data_fields:\n"
            '    - name: "my_mode"\n'
            '      fbs_field: "my_mode"\n'
            '      attr_name: "HIPDNN_ATTR_TEST_MODE"\n'
            '      type: "mode"\n'
            '      mode_sentinel: "sometimes"\n'
        )
        with pytest.raises(ConfigError) as excinfo:
            load_config(config_file)
        msg = str(excinfo.value)
        assert "mode_sentinel" in msg
        assert "sometimes" in msg
