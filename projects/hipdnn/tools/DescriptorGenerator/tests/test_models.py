# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for codegen/models.py computed properties."""

import pytest

from codegen.models import (
    DataField,
    EnumDef,
    EnumValue,
    FrontendConfig,
    FrontendTensorConfig,
    OperationConfig,
    TensorArrayField,
    TensorField,
    _to_camel_case,
    _to_snake_case,
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
# Task 2A.1: Helper functions _to_camel_case and _to_snake_case
# ---------------------------------------------------------------------------


class TestToCamelCase:
    """Tests for the _to_camel_case helper function."""

    @pytest.mark.parametrize(
        "input_str, expected",
        [
            ("hello_world", "helloWorld"),
            ("hello", "hello"),
            ("inv_variance_tensor", "invVarianceTensor"),
            ("x", "x"),
            ("pre_padding", "prePadding"),
            ("conv_mode", "convMode"),
            ("a_b_c_d", "aBCD"),
            ("relu_lower_clip", "reluLowerClip"),
            ("in_0", "in0"),
            ("peer_stats", "peerStats"),
        ],
    )
    def test_conversion(self, input_str, expected):
        assert _to_camel_case(input_str) == expected

    def test_single_char(self):
        assert _to_camel_case("x") == "x"

    def test_already_no_underscores(self):
        assert _to_camel_case("padding") == "padding"


class TestToSnakeCase:
    """Tests for the _to_snake_case helper function."""

    @pytest.mark.parametrize(
        "input_str, expected",
        [
            ("ConvolutionFwd", "convolution_fwd"),
            ("BatchNormBwd", "batch_norm_bwd"),
            ("Matmul", "matmul"),
            ("ConvolutionWrw", "convolution_wrw"),
            ("TestOp", "test_op"),
            ("SDPA", "sdpa"),
            ("ConvFprop", "conv_fprop"),
            ("ABCDef", "abc_def"),
        ],
    )
    def test_conversion(self, input_str, expected):
        assert _to_snake_case(input_str) == expected

    def test_single_lowercase(self):
        assert _to_snake_case("matmul") == "matmul"

    def test_all_uppercase(self):
        # Consecutive uppercase letters treated as a group
        assert _to_snake_case("SDPA") == "sdpa"

    def test_mixed_consecutive_capitals(self):
        # "SDPAForward" -> "sdpa_forward"
        assert _to_snake_case("SDPAForward") == "sdpa_forward"


# ---------------------------------------------------------------------------
# Task 2A.2: EnumValue computed properties
# ---------------------------------------------------------------------------


class TestEnumValue:
    """Tests for EnumValue computed properties."""

    def test_effective_sdk_name_uses_sdk_name_first(self):
        ev = EnumValue(name="MAX", value=1, sdk_name="MAX_OP")
        assert ev.effective_sdk_name == "MAX_OP"

    def test_effective_sdk_name_falls_back_to_frontend_name(self):
        ev = EnumValue(name="TOP_LEFT_EXT", value=1, frontend_name="TOP_LEFT")
        assert ev.effective_sdk_name == "TOP_LEFT"

    def test_effective_sdk_name_falls_back_to_name(self):
        ev = EnumValue(name="CROSS_CORRELATION", value=1)
        assert ev.effective_sdk_name == "CROSS_CORRELATION"

    def test_effective_sdk_name_sdk_takes_priority_over_frontend(self):
        ev = EnumValue(
            name="BACKEND", value=1, sdk_name="SDK_NAME", frontend_name="FRONTEND_NAME"
        )
        assert ev.effective_sdk_name == "SDK_NAME"

    def test_effective_frontend_name_uses_frontend_name(self):
        ev = EnumValue(name="TOP_LEFT_EXT", value=1, frontend_name="TOP_LEFT")
        assert ev.effective_frontend_name == "TOP_LEFT"

    def test_effective_frontend_name_falls_back_to_name(self):
        ev = EnumValue(name="CROSS_CORRELATION", value=1)
        assert ev.effective_frontend_name == "CROSS_CORRELATION"

    def test_effective_frontend_value_uses_frontend_value(self):
        ev = EnumValue(name="CONVOLUTION", value=0, frontend_value=2)
        assert ev.effective_frontend_value == 2

    def test_effective_frontend_value_falls_back_to_value(self):
        ev = EnumValue(name="CONVOLUTION", value=5)
        assert ev.effective_frontend_value == 5

    def test_effective_frontend_value_zero_is_valid(self):
        """frontend_value=0 should be used, not treated as falsy."""
        ev = EnumValue(name="FOO", value=5, frontend_value=0)
        assert ev.effective_frontend_value == 0


# ---------------------------------------------------------------------------
# Task 2A.3: EnumDef.non_sentinel_values
# ---------------------------------------------------------------------------


class TestEnumDef:
    """Tests for EnumDef computed properties."""

    def test_non_sentinel_values_excludes_sentinels(self):
        values = [
            EnumValue(name="NOT_SET", value=0, sentinel=True),
            EnumValue(name="ADD", value=1),
            EnumValue(name="MUL", value=2),
        ]
        ed = EnumDef(values=values)
        result = ed.non_sentinel_values
        assert len(result) == 2
        assert all(not v.sentinel for v in result)

    def test_non_sentinel_values_sorted_by_value(self):
        values = [
            EnumValue(name="MUL", value=3),
            EnumValue(name="ADD", value=1),
            EnumValue(name="SUB", value=2),
        ]
        ed = EnumDef(values=values)
        result = ed.non_sentinel_values
        assert [v.value for v in result] == [1, 2, 3]

    def test_non_sentinel_values_empty_when_all_sentinels(self):
        values = [
            EnumValue(name="NOT_SET", value=0, sentinel=True),
        ]
        ed = EnumDef(values=values)
        assert ed.non_sentinel_values == []

    def test_non_sentinel_values_empty_list(self):
        ed = EnumDef(values=[])
        assert ed.non_sentinel_values == []

    def test_non_sentinel_values_preserves_all_fields(self):
        ev = EnumValue(
            name="ADD",
            value=1,
            description="Addition",
            sdk_name="ADD_OP",
            frontend_name="ADD_FE",
            frontend_value=10,
        )
        ed = EnumDef(values=[ev])
        result = ed.non_sentinel_values
        assert len(result) == 1
        assert result[0].description == "Addition"
        assert result[0].sdk_name == "ADD_OP"


# ---------------------------------------------------------------------------
# Task 2A.4: TensorField computed properties
# ---------------------------------------------------------------------------


class TestTensorField:
    """Tests for TensorField computed properties."""

    def test_camel_name_multi_word(self):
        tf = make_tensor_field(name="inv_variance")
        assert tf.camel_name == "invVariance"

    def test_camel_name_single_word(self):
        tf = make_tensor_field(name="x")
        assert tf.camel_name == "x"

    def test_camel_name_three_parts(self):
        tf = make_tensor_field(name="peer_stats_tensor")
        assert tf.camel_name == "peerStatsTensor"

    def test_pascal_name_multi_word(self):
        tf = make_tensor_field(name="inv_variance")
        assert tf.pascal_name == "InvVariance"

    def test_pascal_name_single_word(self):
        tf = make_tensor_field(name="x")
        assert tf.pascal_name == "X"

    def test_member_name(self):
        tf = make_tensor_field(name="inv_variance")
        assert tf.member_name == "_invVarianceDesc"

    def test_member_name_single_word(self):
        tf = make_tensor_field(name="x")
        assert tf.member_name == "_xDesc"

    def test_local_desc_name(self):
        tf = make_tensor_field(name="inv_variance")
        assert tf.local_desc_name == "invVarianceDesc"

    def test_local_desc_name_single_word(self):
        tf = make_tensor_field(name="x")
        assert tf.local_desc_name == "xDesc"

    def test_err_name(self):
        tf = make_tensor_field(name="inv_variance")
        assert tf.err_name == "errInvVariance"

    def test_err_name_single_word(self):
        tf = make_tensor_field(name="x")
        assert tf.err_name == "errX"

    def test_uid_field(self):
        tf = make_tensor_field(fbs_field="x_tensor_uid")
        assert tf.uid_field == "x_tensor_uid"

    def test_getter_name(self):
        tf = make_tensor_field(name="inv_variance")
        assert tf.getter_name == "getInvVarianceDesc"

    def test_getter_name_single_word(self):
        tf = make_tensor_field(name="x")
        assert tf.getter_name == "getXDesc"

    def test_frontend_setter_from_getter(self):
        tf = make_tensor_field(name="x", frontend_getter="get_x()")
        assert tf.frontend_setter == "set_x"

    def test_frontend_setter_from_getter_multi_word(self):
        tf = make_tensor_field(
            name="inv_variance", frontend_getter="get_inv_variance()"
        )
        assert tf.frontend_setter == "set_inv_variance"

    def test_frontend_setter_falls_back_to_name(self):
        tf = make_tensor_field(name="x", frontend_getter="")
        assert tf.frontend_setter == "set_x"

    def test_frontend_setter_falls_back_to_name_multi_word(self):
        tf = make_tensor_field(name="inv_variance", frontend_getter="")
        assert tf.frontend_setter == "set_inv_variance"

    def test_frontend_setter_non_get_prefix(self):
        """When frontend_getter does not start with 'get_', return base as-is."""
        tf = make_tensor_field(name="x", frontend_getter="fetch_x()")
        assert tf.frontend_setter == "fetch_x"


# ---------------------------------------------------------------------------
# Task 2A.5: DataField computed properties
# ---------------------------------------------------------------------------


class TestDataField:
    """Tests for DataField computed properties."""

    # --- Type predicates ---

    @pytest.mark.parametrize(
        "field_type, expected",
        [
            ("vector_int64", True),
            ("enum", False),
            ("mode", False),
            ("scalar_float", False),
            ("scalar_int64", False),
            ("bool", False),
        ],
    )
    def test_is_vector(self, field_type, expected):
        df = make_data_field(type=field_type)
        assert df.is_vector is expected

    @pytest.mark.parametrize(
        "field_type, expected",
        [
            ("enum", True),
            ("vector_int64", False),
            ("mode", False),
            ("scalar_float", False),
        ],
    )
    def test_is_enum(self, field_type, expected):
        df = make_data_field(type=field_type)
        assert df.is_enum is expected

    @pytest.mark.parametrize(
        "field_type, expected",
        [
            ("mode", True),
            ("vector_int64", False),
            ("enum", False),
            ("scalar_float", False),
        ],
    )
    def test_is_mode(self, field_type, expected):
        df = make_data_field(type=field_type)
        assert df.is_mode is expected

    @pytest.mark.parametrize(
        "field_type, expected",
        [
            ("scalar_float", True),
            ("scalar_int64", True),
            ("bool", True),
            ("vector_int64", False),
            ("enum", False),
            ("mode", False),
        ],
    )
    def test_is_scalar(self, field_type, expected):
        df = make_data_field(type=field_type)
        assert df.is_scalar is expected

    def test_is_optional_scalar_true(self):
        df = make_data_field(type="scalar_float", required=False)
        assert df.is_optional_scalar is True

    def test_is_optional_scalar_false_when_required(self):
        df = make_data_field(type="scalar_float", required=True)
        assert df.is_optional_scalar is False

    def test_is_optional_scalar_false_when_non_scalar(self):
        df = make_data_field(type="vector_int64", required=False)
        assert df.is_optional_scalar is False

    def test_is_optional_scalar_bool_default_is_false(self):
        """Plain bool (FBS bool field = false) is NOT optional scalar."""
        df = make_data_field(type="bool", required=False, fbs_optional=False)
        assert df.is_optional_scalar is False

    def test_is_optional_scalar_bool_fbs_optional_is_true(self):
        """FBS optional bool (bool field (optional)) IS optional scalar."""
        df = make_data_field(type="bool", required=False, fbs_optional=True)
        assert df.is_optional_scalar is True

    def test_is_optional_scalar_bool_required_is_false(self):
        """Required bool is never optional scalar regardless of fbs_optional."""
        df = make_data_field(type="bool", required=True, fbs_optional=False)
        assert df.is_optional_scalar is False

    def test_is_optional_scalar_bool_required_fbs_optional(self):
        """Required bool with fbs_optional=True: fbs_optional takes priority for bool."""
        df = make_data_field(type="bool", required=True, fbs_optional=True)
        assert df.is_optional_scalar is True

    # --- frontend_getter_returns_optional ---

    def test_frontend_getter_returns_optional_default_false(self):
        df = make_data_field()
        assert df.frontend_getter_returns_optional is False

    def test_frontend_getter_returns_optional_set_true(self):
        df = make_data_field(frontend_getter_returns_optional=True)
        assert df.frontend_getter_returns_optional is True

    # --- effective_sentinel_value ---

    def test_effective_sentinel_value_with_sdk_name(self):
        """Sentinel with sdk_name uses effective_sdk_name."""
        enum_def = EnumDef(
            values=[
                EnumValue(name="UNSET", value=0, sentinel=True, sdk_name="UNSET"),
                EnumValue(name="ADD", value=1),
            ]
        )
        df = make_data_field(type="mode", enum_def=enum_def)
        assert df.effective_sentinel_value == "UNSET"

    def test_effective_sentinel_value_without_sdk_name(self):
        """Sentinel without sdk_name falls back to name."""
        enum_def = EnumDef(
            values=[
                EnumValue(name="NOT_SET", value=0, sentinel=True),
                EnumValue(name="ADD", value=1),
            ]
        )
        df = make_data_field(type="mode", enum_def=enum_def)
        assert df.effective_sentinel_value == "NOT_SET"

    def test_effective_sentinel_value_no_enum_def(self):
        """No enum_def returns default 'NOT_SET'."""
        df = make_data_field(type="mode")
        assert df.effective_sentinel_value == "NOT_SET"

    def test_effective_sentinel_value_no_sentinel_in_enum_def(self):
        """enum_def with no sentinel value returns default 'NOT_SET'."""
        enum_def = EnumDef(
            values=[
                EnumValue(name="ADD", value=1),
                EnumValue(name="MUL", value=2),
            ]
        )
        df = make_data_field(type="mode", enum_def=enum_def)
        assert df.effective_sentinel_value == "NOT_SET"

    # --- cpp_type ---

    @pytest.mark.parametrize(
        "field_type, expected",
        [
            ("scalar_float", "float"),
            ("scalar_int64", "int64_t"),
            ("bool", "bool"),
        ],
    )
    def test_cpp_type_scalar_types(self, field_type, expected):
        df = make_data_field(type=field_type)
        assert df.cpp_type == expected

    def test_cpp_type_default_for_non_scalar(self):
        df = make_data_field(type="vector_int64")
        assert df.cpp_type == "int64_t"

    def test_cpp_type_default_for_unknown(self):
        df = make_data_field(type="mode")
        assert df.cpp_type == "int64_t"

    # --- backend_type ---

    @pytest.mark.parametrize(
        "field_type, expected",
        [
            ("vector_int64", "HIPDNN_TYPE_INT64"),
            ("enum", "HIPDNN_TYPE_INT64"),
            ("scalar_float", "HIPDNN_TYPE_FLOAT"),
            ("scalar_int64", "HIPDNN_TYPE_INT64"),
            ("bool", "HIPDNN_TYPE_BOOLEAN"),
        ],
    )
    def test_backend_type_standard_types(self, field_type, expected):
        df = make_data_field(type=field_type)
        assert df.backend_type == expected

    def test_backend_type_mode_with_backend_type_name(self):
        df = make_data_field(
            type="mode", backend_type_name="HIPDNN_TYPE_CONVOLUTION_MODE"
        )
        assert df.backend_type == "HIPDNN_TYPE_CONVOLUTION_MODE"

    def test_backend_type_mode_without_backend_type_name(self):
        df = make_data_field(type="mode", backend_type_name="")
        assert df.backend_type == "HIPDNN_TYPE_INT64"

    # --- camel_name and pascal_name ---

    def test_camel_name(self):
        df = make_data_field(name="pre_padding")
        assert df.camel_name == "prePadding"

    def test_pascal_name(self):
        df = make_data_field(name="relu_lower_clip")
        assert df.pascal_name == "ReluLowerClip"

    def test_pascal_name_single_word(self):
        df = make_data_field(name="padding")
        assert df.pascal_name == "Padding"

    # --- setter/getter helper names ---

    def test_setter_helper_name(self):
        df = make_data_field(name="conv_mode")
        assert df.setter_helper_name == "setConvMode"

    def test_getter_helper_name(self):
        df = make_data_field(name="conv_mode")
        assert df.getter_helper_name == "getConvMode"

    def test_setter_helper_name_single_word(self):
        df = make_data_field(name="mode")
        assert df.setter_helper_name == "setMode"

    def test_getter_helper_name_single_word(self):
        df = make_data_field(name="mode")
        assert df.getter_helper_name == "getMode"

    # --- enum_short_type ---

    def test_enum_short_type_with_namespace(self):
        df = make_data_field(cpp_enum="hipdnn_flatbuffers_sdk::data_objects::ConvMode")
        assert df.enum_short_type == "ConvMode"

    def test_enum_short_type_without_namespace(self):
        df = make_data_field(cpp_enum="ConvMode")
        assert df.enum_short_type == "ConvMode"

    def test_enum_short_type_empty(self):
        df = make_data_field(cpp_enum="")
        assert df.enum_short_type == ""

    def test_enum_short_type_deep_namespace(self):
        df = make_data_field(cpp_enum="a::b::c::PointwiseMode")
        assert df.enum_short_type == "PointwiseMode"

    # --- effective_frontend_type ---

    def test_effective_frontend_type_uses_frontend_type(self):
        df = make_data_field(
            frontend_type="ConvolutionMode",
            cpp_enum="hipdnn_flatbuffers_sdk::data_objects::ConvMode",
        )
        assert df.effective_frontend_type == "ConvolutionMode"

    def test_effective_frontend_type_falls_back_to_cpp_enum_short(self):
        df = make_data_field(
            frontend_type="",
            cpp_enum="hipdnn_flatbuffers_sdk::data_objects::ConvMode",
        )
        assert df.effective_frontend_type == "ConvMode"

    def test_effective_frontend_type_empty_when_both_empty(self):
        df = make_data_field(frontend_type="", cpp_enum="")
        assert df.effective_frontend_type == ""

    # --- frontend_setter_name ---

    def test_frontend_setter_name_from_getter(self):
        df = make_data_field(frontend_getter="get_convolution_mode()")
        assert df.frontend_setter_name == "set_convolution_mode"

    def test_frontend_setter_name_falls_back_to_name(self):
        df = make_data_field(name="conv_mode", frontend_getter="")
        assert df.frontend_setter_name == "set_conv_mode"

    def test_frontend_setter_name_non_get_prefix(self):
        df = make_data_field(frontend_getter="fetch_mode()")
        assert df.frontend_setter_name == "fetch_mode"

    # --- has_enum_def ---

    def test_has_enum_def_true(self):
        ed = EnumDef(values=[EnumValue(name="ADD", value=1)])
        df = make_data_field(enum_def=ed)
        assert df.has_enum_def is True

    def test_has_enum_def_false_none(self):
        df = make_data_field(enum_def=None)
        assert df.has_enum_def is False

    def test_has_enum_def_false_empty_values(self):
        ed = EnumDef(values=[])
        df = make_data_field(enum_def=ed)
        assert df.has_enum_def is False


# ---------------------------------------------------------------------------
# Task 2A.6: TensorArrayField computed properties
# ---------------------------------------------------------------------------


class TestTensorArrayField:
    """Tests for TensorArrayField computed properties."""

    def test_member_name(self):
        taf = TensorArrayField(
            name="peer_stats", fbs_field="peer_stats_tensor_uid", attr_name="ATTR"
        )
        assert taf.member_name == "_peer_statsDescs"

    def test_member_name_single_word(self):
        taf = TensorArrayField(name="stats", fbs_field="stats_uid", attr_name="ATTR")
        assert taf.member_name == "_statsDescs"

    def test_member_name_multi_word(self):
        taf = TensorArrayField(
            name="multi_word_field", fbs_field="mwf_uid", attr_name="ATTR"
        )
        assert taf.member_name == "_multi_word_fieldDescs"


# ---------------------------------------------------------------------------
# Task 2A.7: FrontendTensorConfig computed properties
# ---------------------------------------------------------------------------


class TestFrontendTensorConfig:
    """Tests for FrontendTensorConfig computed properties."""

    def test_camel_name_with_underscore(self):
        ftc = make_frontend_tensor_config(name="input_0")
        assert ftc.camel_name == "input0"

    def test_camel_name_single_word(self):
        ftc = make_frontend_tensor_config(name="x")
        assert ftc.camel_name == "x"

    def test_camel_name_multi_word(self):
        ftc = make_frontend_tensor_config(name="residual_input")
        assert ftc.camel_name == "residualInput"

    def test_effective_enum_name_uses_override(self):
        ftc = make_frontend_tensor_config(name="x", enum_name="INPUT_X")
        assert ftc.effective_enum_name == "INPUT_X"

    def test_effective_enum_name_defaults_to_uppercase(self):
        ftc = make_frontend_tensor_config(name="input_0", enum_name="")
        assert ftc.effective_enum_name == "INPUT_0"

    def test_effective_getter_name_uses_override(self):
        ftc = make_frontend_tensor_config(name="x", getter_name="get_input_x")
        assert ftc.effective_getter_name == "get_input_x"

    def test_effective_getter_name_defaults(self):
        ftc = make_frontend_tensor_config(name="input_0", getter_name="")
        assert ftc.effective_getter_name == "get_input_0"

    def test_effective_setter_name_uses_override(self):
        ftc = make_frontend_tensor_config(name="x", setter_name="set_input_x")
        assert ftc.effective_setter_name == "set_input_x"

    def test_effective_setter_name_defaults(self):
        ftc = make_frontend_tensor_config(name="input_0", setter_name="")
        assert ftc.effective_setter_name == "set_input_0"


# ---------------------------------------------------------------------------
# Task 2A.8: FrontendConfig computed properties
# ---------------------------------------------------------------------------


class TestFrontendConfig:
    """Tests for FrontendConfig computed properties."""

    # --- effective_attributes_include ---

    def test_effective_attributes_include_uses_override(self):
        fc = make_frontend_config(attributes_include="CustomInclude")
        assert fc.effective_attributes_include == "CustomInclude"

    def test_effective_attributes_include_derived_from_node_class(self):
        fc = make_frontend_config(
            attributes_include="", node_class="ConvolutionFpropNode"
        )
        assert fc.effective_attributes_include == "ConvolutionFpropAttributes"

    def test_effective_attributes_include_node_class_without_node_suffix(self):
        fc = make_frontend_config(attributes_include="", node_class="ConvolutionFprop")
        assert fc.effective_attributes_include == "ConvolutionFpropAttributes"

    def test_effective_attributes_include_falls_back_to_attributes_class(self):
        fc = make_frontend_config(
            attributes_include="", node_class="", attributes_class="TestAttributes"
        )
        assert fc.effective_attributes_include == "TestAttributes"

    def test_effective_attributes_include_all_empty(self):
        fc = FrontendConfig()
        assert fc.effective_attributes_include == ""

    # --- Input/output filtering ---

    def test_required_inputs(self):
        inputs = [
            make_frontend_tensor_config(name="x", required=True),
            make_frontend_tensor_config(name="b", required=False),
            make_frontend_tensor_config(name="w", required=True),
        ]
        fc = make_frontend_config(inputs=inputs)
        result = fc.required_inputs
        assert len(result) == 2
        assert result[0].name == "x"
        assert result[1].name == "w"

    def test_optional_inputs(self):
        inputs = [
            make_frontend_tensor_config(name="x", required=True),
            make_frontend_tensor_config(name="b", required=False),
        ]
        fc = make_frontend_config(inputs=inputs)
        result = fc.optional_inputs
        assert len(result) == 1
        assert result[0].name == "b"

    def test_required_outputs(self):
        outputs = [
            make_frontend_tensor_config(name="out", required=True),
            make_frontend_tensor_config(name="extra", required=False),
        ]
        fc = make_frontend_config(outputs=outputs)
        assert len(fc.required_outputs) == 1
        assert fc.required_outputs[0].name == "out"

    def test_optional_outputs(self):
        outputs = [
            make_frontend_tensor_config(name="out", required=True),
            make_frontend_tensor_config(name="extra", required=False),
        ]
        fc = make_frontend_config(outputs=outputs)
        assert len(fc.optional_outputs) == 1
        assert fc.optional_outputs[0].name == "extra"

    def test_all_tensors(self):
        inputs = [make_frontend_tensor_config(name="x")]
        outputs = [make_frontend_tensor_config(name="y")]
        fc = make_frontend_config(inputs=inputs, outputs=outputs)
        result = fc.all_tensors
        assert len(result) == 2
        assert result[0].name == "x"
        assert result[1].name == "y"

    def test_all_tensors_empty(self):
        fc = FrontendConfig()
        assert fc.all_tensors == []

    # --- deserialization_accessor ---

    def test_deserialization_accessor_strips_prefix(self):
        fc = make_frontend_config(
            node_attributes_union_type="NodeAttributes_ConvolutionFwdAttributes"
        )
        assert fc.deserialization_accessor == "attributes_as_ConvolutionFwdAttributes"

    def test_deserialization_accessor_no_prefix(self):
        fc = make_frontend_config(node_attributes_union_type="ConvolutionFwdAttributes")
        assert fc.deserialization_accessor == "attributes_as_ConvolutionFwdAttributes"

    def test_deserialization_accessor_empty(self):
        fc = make_frontend_config(node_attributes_union_type="")
        assert fc.deserialization_accessor == ""


# ---------------------------------------------------------------------------
# Task 2A.9: OperationConfig effective label properties
# ---------------------------------------------------------------------------


class TestOperationConfigLabels:
    """Tests for OperationConfig effective label and simple derivation properties."""

    # --- effective_error_label ---

    def test_effective_error_label_uses_override(self):
        cfg = make_minimal_config(error_label="conv")
        assert cfg.effective_error_label == "conv"

    def test_effective_error_label_defaults_to_lowercase_name(self):
        cfg = make_minimal_config(name="ConvolutionFwd", error_label="")
        assert cfg.effective_error_label == "convolutionfwd"

    # --- effective_packer_operation_label ---

    def test_effective_packer_operation_label_uses_override(self):
        cfg = make_minimal_config(packer_operation_label="convolution forward")
        assert cfg.effective_packer_operation_label == "convolution forward"

    def test_effective_packer_operation_label_default(self):
        cfg = make_minimal_config(name="TestOp", packer_operation_label="")
        assert cfg.effective_packer_operation_label == "testop"

    # --- effective_packer_finalize_label ---

    def test_effective_packer_finalize_label_uses_override(self):
        cfg = make_minimal_config(packer_finalize_label="conv forward")
        assert cfg.effective_packer_finalize_label == "conv forward"

    def test_effective_packer_finalize_label_default(self):
        cfg = make_minimal_config(name="TestOp", packer_finalize_label="")
        assert cfg.effective_packer_finalize_label == "testop"

    # --- packer_params_label ---

    def test_packer_params_label_uses_packer_finalize_label(self):
        cfg = make_minimal_config(packer_finalize_label="conv")
        assert cfg.packer_params_label == "conv"

    def test_packer_params_label_default(self):
        cfg = make_minimal_config(name="TestOp", packer_finalize_label="")
        assert cfg.packer_params_label == "testop"

    # --- effective_test_params_method_name ---

    def test_effective_test_params_method_name_uses_override(self):
        cfg = make_minimal_config(test_params_method_name="setCustomParams")
        assert cfg.effective_test_params_method_name == "setCustomParams"

    def test_effective_test_params_method_name_default(self):
        cfg = make_minimal_config(name="ConvFwd", test_params_method_name="")
        assert cfg.effective_test_params_method_name == "setConvFwdParams"

    # --- effective_data_fields_section_label ---

    def test_effective_data_fields_section_label_uses_override(self):
        cfg = make_minimal_config(data_fields_section_label="Conv Params")
        assert cfg.effective_data_fields_section_label == "Conv Params"

    def test_effective_data_fields_section_label_default(self):
        cfg = make_minimal_config(data_fields_section_label="")
        assert cfg.effective_data_fields_section_label == "Data Fields"

    # --- effective_build_node_attrs_var ---

    def test_effective_build_node_attrs_var_uses_override(self):
        cfg = make_minimal_config(build_node_attrs_var="convAttrs")
        assert cfg.effective_build_node_attrs_var == "convAttrs"

    def test_effective_build_node_attrs_var_default(self):
        cfg = make_minimal_config(build_node_attrs_var="")
        assert cfg.effective_build_node_attrs_var == "attrs"

    # --- Simple derivation properties ---

    def test_error_prefix(self):
        cfg = make_minimal_config(class_name="ConvFwdOperationDescriptor")
        assert cfg.error_prefix == "ConvFwdOperationDescriptor::"

    def test_fbs_namespace(self):
        cfg = make_minimal_config()
        assert cfg.fbs_namespace == "hipdnn_flatbuffers_sdk::data_objects"

    def test_fbs_t_type(self):
        cfg = make_minimal_config(fbs_table="ConvolutionFwdAttributes")
        assert cfg.fbs_t_type == "ConvolutionFwdAttributesT"

    def test_node_attributes_union_member(self):
        cfg = make_minimal_config(fbs_table="ConvolutionFwdAttributes")
        assert cfg.node_attributes_union_member == "ConvolutionFwdAttributes"

    def test_descriptor_type_enum(self):
        from codegen.models import DescriptorTypeConfig

        dt = DescriptorTypeConfig(enum_name="HIPDNN_BACKEND_OPERATION_TEST_DESCRIPTOR")
        cfg = make_minimal_config(descriptor_type=dt)
        assert cfg.descriptor_type_enum == "HIPDNN_BACKEND_OPERATION_TEST_DESCRIPTOR"


# ---------------------------------------------------------------------------
# Task 2A.10: OperationConfig filename properties
# ---------------------------------------------------------------------------


class TestOperationConfigFilenames:
    """Tests for OperationConfig filename computed properties."""

    def test_header_filename(self):
        cfg = make_minimal_config(class_name="ConvFwdOperationDescriptor")
        assert cfg.header_filename == "ConvFwdOperationDescriptor.hpp"

    def test_source_filename(self):
        cfg = make_minimal_config(class_name="ConvFwdOperationDescriptor")
        assert cfg.source_filename == "ConvFwdOperationDescriptor.cpp"

    # --- packer_filename ---

    def test_packer_filename_from_node_class(self):
        frontend = make_frontend_config(
            packer_function="createConvFpropOperation",
            node_class="ConvolutionFpropNode",
        )
        cfg = make_minimal_config(frontend=frontend)
        assert cfg.packer_filename == "ConvolutionFpropPacker.hpp"

    def test_packer_filename_from_node_class_without_node_suffix(self):
        frontend = make_frontend_config(
            packer_function="createConvFpropOperation",
            node_class="ConvolutionFprop",
        )
        cfg = make_minimal_config(frontend=frontend)
        assert cfg.packer_filename == "ConvolutionFpropPacker.hpp"

    def test_packer_filename_fallback_to_name(self):
        frontend = make_frontend_config(packer_function="", node_class="")
        cfg = make_minimal_config(name="ConvFwd", frontend=frontend)
        assert cfg.packer_filename == "ConvFwdPacker.hpp"

    def test_packer_filename_packer_fn_without_node_class(self):
        frontend = make_frontend_config(
            packer_function="createSomething", node_class=""
        )
        cfg = make_minimal_config(name="TestOp", frontend=frontend)
        assert cfg.packer_filename == "TestOpPacker.hpp"

    # --- unpacker_filename ---

    def test_unpacker_filename_from_node_class(self):
        frontend = make_frontend_config(node_class="ConvolutionFpropNode")
        cfg = make_minimal_config(frontend=frontend)
        assert cfg.unpacker_filename == "ConvolutionFpropUnpacker.hpp"

    def test_unpacker_filename_from_node_class_without_suffix(self):
        frontend = make_frontend_config(node_class="ConvolutionFprop")
        cfg = make_minimal_config(frontend=frontend)
        assert cfg.unpacker_filename == "ConvolutionFpropUnpacker.hpp"

    def test_unpacker_filename_fallback_to_name(self):
        frontend = make_frontend_config(node_class="")
        cfg = make_minimal_config(name="TestOp", frontend=frontend)
        assert cfg.unpacker_filename == "TestOpUnpacker.hpp"

    # --- Test filenames ---

    def test_test_from_node_filename(self):
        cfg = make_minimal_config(name="ConvolutionFwd")
        assert cfg.test_from_node_filename == "TestConvolutionFwdOperationFromNode.cpp"

    def test_test_descriptor_filename(self):
        cfg = make_minimal_config(class_name="ConvFwdOperationDescriptor")
        assert cfg.test_descriptor_filename == "TestConvFwdOperationDescriptor.cpp"

    def test_test_graph_filename(self):
        cfg = make_minimal_config(name="ConvolutionFwd")
        assert cfg.test_graph_filename == "TestGraphDescriptorConvolutionFwd.cpp"

    # --- Integration test filenames ---

    def test_test_integration_filename_from_node_class(self):
        frontend = make_frontend_config(node_class="ConvolutionFpropNode")
        cfg = make_minimal_config(frontend=frontend)
        assert (
            cfg.test_integration_filename
            == "IntegrationConvolutionFpropDescriptorLowering.cpp"
        )

    def test_test_integration_filename_fallback_to_name(self):
        frontend = make_frontend_config(node_class="")
        cfg = make_minimal_config(name="ConvFwd", frontend=frontend)
        assert (
            cfg.test_integration_filename == "IntegrationConvFwdDescriptorLowering.cpp"
        )

    def test_test_integration_lifting_filename_from_node_class(self):
        frontend = make_frontend_config(node_class="ConvolutionFpropNode")
        cfg = make_minimal_config(frontend=frontend)
        assert (
            cfg.test_integration_lifting_filename
            == "IntegrationConvolutionFpropDescriptorLifting.cpp"
        )

    def test_test_integration_lifting_filename_fallback_to_name(self):
        frontend = make_frontend_config(node_class="")
        cfg = make_minimal_config(name="ConvFwd", frontend=frontend)
        assert (
            cfg.test_integration_lifting_filename
            == "IntegrationConvFwdDescriptorLifting.cpp"
        )

    # --- _frontend_base_name ---

    def test_frontend_base_name_from_attributes_class(self):
        frontend = make_frontend_config(attributes_class="ConvFpropAttributes")
        cfg = make_minimal_config(frontend=frontend)
        assert cfg._frontend_base_name == "ConvFprop"

    def test_frontend_base_name_attributes_class_without_suffix(self):
        frontend = make_frontend_config(attributes_class="ConvFprop", node_class="")
        cfg = make_minimal_config(frontend=frontend)
        assert cfg._frontend_base_name == "ConvFprop"

    def test_frontend_base_name_from_node_class(self):
        frontend = make_frontend_config(attributes_class="", node_class="ConvFpropNode")
        cfg = make_minimal_config(frontend=frontend)
        assert cfg._frontend_base_name == "ConvFprop"

    def test_frontend_base_name_node_class_without_suffix(self):
        frontend = make_frontend_config(attributes_class="", node_class="ConvFprop")
        cfg = make_minimal_config(frontend=frontend)
        assert cfg._frontend_base_name == "ConvFprop"

    def test_frontend_base_name_from_name(self):
        frontend = make_frontend_config(attributes_class="", node_class="")
        cfg = make_minimal_config(name="ConvolutionFwd", frontend=frontend)
        assert cfg._frontend_base_name == "ConvolutionFwd"

    # --- Frontend filenames derived from _frontend_base_name ---

    def test_attributes_header_filename(self):
        frontend = make_frontend_config(attributes_class="ConvFpropAttributes")
        cfg = make_minimal_config(frontend=frontend)
        assert cfg.attributes_header_filename == "ConvFpropAttributes.hpp"

    def test_node_header_filename(self):
        frontend = make_frontend_config(attributes_class="ConvFpropAttributes")
        cfg = make_minimal_config(frontend=frontend)
        assert cfg.node_header_filename == "ConvFpropNode.hpp"

    def test_test_attributes_filename(self):
        frontend = make_frontend_config(attributes_class="ConvFpropAttributes")
        cfg = make_minimal_config(frontend=frontend)
        assert cfg.test_attributes_filename == "TestConvFpropAttributes.cpp"

    def test_test_node_filename(self):
        frontend = make_frontend_config(attributes_class="ConvFpropAttributes")
        cfg = make_minimal_config(frontend=frontend)
        assert cfg.test_node_filename == "TestConvFpropNode.cpp"

    def test_test_frontend_graph_filename(self):
        frontend = make_frontend_config(attributes_class="ConvFpropAttributes")
        cfg = make_minimal_config(frontend=frontend)
        assert cfg.test_frontend_graph_filename == "TestGraphConvFprop.cpp"

    # --- effective_graph_method_name ---

    def test_effective_graph_method_name_uses_override(self):
        frontend = make_frontend_config(graph_method_name="conv_fprop")
        cfg = make_minimal_config(frontend=frontend)
        assert cfg.effective_graph_method_name == "conv_fprop"

    def test_effective_graph_method_name_defaults_to_snake_case(self):
        frontend = make_frontend_config(graph_method_name="")
        cfg = make_minimal_config(name="ConvolutionFwd", frontend=frontend)
        assert cfg.effective_graph_method_name == "convolution_fwd"


# ---------------------------------------------------------------------------
# Task 2A.11: OperationConfig field filter properties
# ---------------------------------------------------------------------------


class TestOperationConfigFieldFilters:
    """Tests for OperationConfig field filter and boolean properties."""

    def _make_config_with_fields(
        self,
        tensor_fields=None,
        data_fields=None,
        tensor_array_fields=None,
    ):
        """Helper to create an OperationConfig with specific field lists."""
        return make_minimal_config(
            tensor_fields=tensor_fields or [],
            data_fields=data_fields or [],
            tensor_array_fields=tensor_array_fields or [],
        )

    # --- Tensor field filters ---

    def test_required_tensor_fields(self):
        fields = [
            make_tensor_field(name="x", required=True),
            make_tensor_field(name="b", required=False),
            make_tensor_field(name="w", required=True),
        ]
        cfg = self._make_config_with_fields(tensor_fields=fields)
        result = cfg.required_tensor_fields
        assert len(result) == 2
        assert [f.name for f in result] == ["x", "w"]

    def test_optional_tensor_fields(self):
        fields = [
            make_tensor_field(name="x", required=True),
            make_tensor_field(name="b", required=False),
        ]
        cfg = self._make_config_with_fields(tensor_fields=fields)
        result = cfg.optional_tensor_fields
        assert len(result) == 1
        assert result[0].name == "b"

    def test_required_tensor_fields_all_required(self):
        fields = [
            make_tensor_field(name="x", required=True),
            make_tensor_field(name="y", required=True),
        ]
        cfg = self._make_config_with_fields(tensor_fields=fields)
        assert len(cfg.required_tensor_fields) == 2
        assert len(cfg.optional_tensor_fields) == 0

    # --- Data field filters ---

    def test_required_data_fields(self):
        fields = [
            make_data_field(name="padding", type="vector_int64", required=True),
            make_data_field(name="alpha", type="scalar_float", required=False),
        ]
        cfg = self._make_config_with_fields(data_fields=fields)
        assert len(cfg.required_data_fields) == 1
        assert cfg.required_data_fields[0].name == "padding"

    def test_optional_data_fields(self):
        fields = [
            make_data_field(name="padding", type="vector_int64", required=True),
            make_data_field(name="alpha", type="scalar_float", required=False),
        ]
        cfg = self._make_config_with_fields(data_fields=fields)
        assert len(cfg.optional_data_fields) == 1
        assert cfg.optional_data_fields[0].name == "alpha"

    def test_enum_fields(self):
        fields = [
            make_data_field(name="data_type", type="enum"),
            make_data_field(name="padding", type="vector_int64"),
            make_data_field(name="mode", type="mode"),
        ]
        cfg = self._make_config_with_fields(data_fields=fields)
        result = cfg.enum_fields
        assert len(result) == 1
        assert result[0].name == "data_type"

    def test_mode_fields(self):
        fields = [
            make_data_field(name="conv_mode", type="mode"),
            make_data_field(name="padding", type="vector_int64"),
            make_data_field(name="pw_mode", type="mode"),
        ]
        cfg = self._make_config_with_fields(data_fields=fields)
        result = cfg.mode_fields
        assert len(result) == 2
        assert [f.name for f in result] == ["conv_mode", "pw_mode"]

    def test_has_mode_fields_true(self):
        fields = [make_data_field(name="conv_mode", type="mode")]
        cfg = self._make_config_with_fields(data_fields=fields)
        assert cfg.has_mode_fields is True

    def test_has_mode_fields_false(self):
        fields = [make_data_field(name="padding", type="vector_int64")]
        cfg = self._make_config_with_fields(data_fields=fields)
        assert cfg.has_mode_fields is False

    def test_vector_fields(self):
        fields = [
            make_data_field(name="padding", type="vector_int64"),
            make_data_field(name="stride", type="vector_int64"),
            make_data_field(name="alpha", type="scalar_float"),
        ]
        cfg = self._make_config_with_fields(data_fields=fields)
        result = cfg.vector_fields
        assert len(result) == 2
        assert [f.name for f in result] == ["padding", "stride"]

    def test_scalar_fields(self):
        fields = [
            make_data_field(name="alpha", type="scalar_float"),
            make_data_field(name="beta", type="scalar_int64"),
            make_data_field(name="flag", type="bool"),
            make_data_field(name="padding", type="vector_int64"),
        ]
        cfg = self._make_config_with_fields(data_fields=fields)
        result = cfg.scalar_fields
        assert len(result) == 3
        assert [f.name for f in result] == ["alpha", "beta", "flag"]

    def test_optional_scalar_fields(self):
        fields = [
            make_data_field(name="alpha", type="scalar_float", required=False),
            make_data_field(name="beta", type="scalar_float", required=True),
            make_data_field(name="padding", type="vector_int64", required=False),
        ]
        cfg = self._make_config_with_fields(data_fields=fields)
        result = cfg.optional_scalar_fields
        assert len(result) == 1
        assert result[0].name == "alpha"

    def test_non_shared_data_fields(self):
        fields = [
            make_data_field(name="mode", type="mode", shared=False),
            make_data_field(name="padding", type="vector_int64", shared=True),
            make_data_field(name="alpha", type="scalar_float", shared=False),
        ]
        cfg = self._make_config_with_fields(data_fields=fields)
        result = cfg.non_shared_data_fields
        assert len(result) == 2
        assert [f.name for f in result] == ["mode", "alpha"]

    def test_generatable_mode_fields(self):
        ed = EnumDef(values=[EnumValue(name="ADD", value=1)])
        fields = [
            make_data_field(name="pw_mode", type="mode", shared=False, enum_def=ed),
            make_data_field(name="conv_mode", type="mode", shared=True, enum_def=ed),
            make_data_field(name="no_def", type="mode", shared=False, enum_def=None),
            make_data_field(name="padding", type="vector_int64"),
        ]
        cfg = self._make_config_with_fields(data_fields=fields)
        result = cfg.generatable_mode_fields
        assert len(result) == 1
        assert result[0].name == "pw_mode"

    def test_graph_verifiable_data_fields(self):
        fields = [
            make_data_field(name="padding", type="vector_int64"),
            make_data_field(name="mode", type="mode"),
            make_data_field(name="data_type", type="enum"),
            make_data_field(name="alpha", type="scalar_float"),
            make_data_field(name="flag", type="bool"),
        ]
        cfg = self._make_config_with_fields(data_fields=fields)
        result = cfg.graph_verifiable_data_fields
        assert len(result) == 3
        assert [f.name for f in result] == ["padding", "mode", "data_type"]

    # --- Boolean properties ---

    def test_has_enum_fields_true(self):
        fields = [make_data_field(name="data_type", type="enum")]
        cfg = self._make_config_with_fields(data_fields=fields)
        assert cfg.has_enum_fields is True

    def test_has_enum_fields_false(self):
        fields = [make_data_field(name="padding", type="vector_int64")]
        cfg = self._make_config_with_fields(data_fields=fields)
        assert cfg.has_enum_fields is False

    def test_has_vector_fields_true(self):
        fields = [make_data_field(name="padding", type="vector_int64")]
        cfg = self._make_config_with_fields(data_fields=fields)
        assert cfg.has_vector_fields is True

    def test_has_vector_fields_false(self):
        fields = [make_data_field(name="mode", type="mode")]
        cfg = self._make_config_with_fields(data_fields=fields)
        assert cfg.has_vector_fields is False

    def test_has_tensor_array_fields_true(self):
        taf = [
            TensorArrayField(
                name="peer_stats", fbs_field="peer_stats_uid", attr_name="ATTR"
            )
        ]
        cfg = self._make_config_with_fields(tensor_array_fields=taf)
        assert cfg.has_tensor_array_fields is True

    def test_has_tensor_array_fields_false(self):
        cfg = self._make_config_with_fields(tensor_array_fields=[])
        assert cfg.has_tensor_array_fields is False

    # --- Derived list properties ---

    def test_all_tensor_names(self):
        fields = [
            make_tensor_field(name="x"),
            make_tensor_field(name="w"),
            make_tensor_field(name="y"),
        ]
        cfg = self._make_config_with_fields(tensor_fields=fields)
        assert cfg.all_tensor_names == ["x", "w", "y"]

    def test_all_tensor_names_empty(self):
        cfg = self._make_config_with_fields(tensor_fields=[])
        assert cfg.all_tensor_names == []

    def test_tensor_uid_list_from_test_data(self):
        from codegen.models import TestData

        fields = [
            make_tensor_field(name="x"),
            make_tensor_field(name="w"),
        ]
        td = TestData(tensor_uids={"x": 10, "w": 20})
        cfg = self._make_config_with_fields(tensor_fields=fields)
        cfg.test_data = td
        assert cfg.tensor_uid_list == [10, 20]

    def test_tensor_uid_list_fallback_to_index(self):
        from codegen.models import TestData

        fields = [
            make_tensor_field(name="x"),
            make_tensor_field(name="w"),
        ]
        td = TestData(tensor_uids={})
        cfg = self._make_config_with_fields(tensor_fields=fields)
        cfg.test_data = td
        # Fallback is i + 1
        assert cfg.tensor_uid_list == [1, 2]

    def test_tensor_attr_cases(self):
        fields = [
            make_tensor_field(name="x", attr_suffix="X"),
            make_tensor_field(name="w", attr_suffix="W"),
        ]
        cfg = make_minimal_config(
            operation_attr_prefix="HIPDNN_ATTR_OPERATION_CONV",
            tensor_fields=fields,
        )
        assert cfg.tensor_attr_cases == [
            "HIPDNN_ATTR_OPERATION_CONV_X",
            "HIPDNN_ATTR_OPERATION_CONV_W",
        ]

    def test_tensor_attr_cases_empty(self):
        cfg = self._make_config_with_fields(tensor_fields=[])
        assert cfg.tensor_attr_cases == []


# ---------------------------------------------------------------------------
# Task 2A.12: OperationConfig.tensor_field_frontend_map and reverse map
# ---------------------------------------------------------------------------


class TestTensorFieldFrontendMap:
    """Tests for tensor_field_frontend_map and frontend_to_tensor_field_map."""

    def test_exact_name_match(self):
        """Tensor field name matches frontend tensor name exactly."""
        tensor_fields = [
            make_tensor_field(name="x", frontend_getter="get_x()"),
        ]
        inputs = [make_frontend_tensor_config(name="x")]
        frontend = make_frontend_config(inputs=inputs)
        cfg = make_minimal_config(tensor_fields=tensor_fields, frontend=frontend)
        result = cfg.tensor_field_frontend_map
        assert "x" in result
        assert result["x"].name == "x"

    def test_getter_based_match(self):
        """Frontend getter 'get_x()' matches frontend tensor 'x' via getter_name."""
        tensor_fields = [
            make_tensor_field(name="tensor_x", frontend_getter="get_input()")
        ]
        inputs = [make_frontend_tensor_config(name="input", getter_name="get_input")]
        frontend = make_frontend_config(inputs=inputs)
        cfg = make_minimal_config(tensor_fields=tensor_fields, frontend=frontend)
        result = cfg.tensor_field_frontend_map
        assert "tensor_x" in result
        assert result["tensor_x"].name == "input"

    def test_abbreviated_in_to_input(self):
        """Backend 'in_0' maps to frontend 'input_0' via abbreviation matching."""
        tensor_fields = [make_tensor_field(name="in_0", frontend_getter="get_in_0()")]
        inputs = [
            make_frontend_tensor_config(name="input_0", getter_name="get_input_0")
        ]
        frontend = make_frontend_config(inputs=inputs)
        cfg = make_minimal_config(tensor_fields=tensor_fields, frontend=frontend)
        result = cfg.tensor_field_frontend_map
        assert "in_0" in result
        assert result["in_0"].name == "input_0"

    def test_abbreviated_out_to_output(self):
        """Backend 'out_0' maps to frontend 'output_0' via abbreviation matching."""
        tensor_fields = [make_tensor_field(name="out_0", frontend_getter="get_out_0()")]
        outputs = [
            make_frontend_tensor_config(name="output_0", getter_name="get_output_0")
        ]
        frontend = make_frontend_config(outputs=outputs)
        cfg = make_minimal_config(tensor_fields=tensor_fields, frontend=frontend)
        result = cfg.tensor_field_frontend_map
        assert "out_0" in result
        assert result["out_0"].name == "output_0"

    def test_no_match_excluded_from_map(self):
        """Tensor fields without a frontend match are absent from the map."""
        tensor_fields = [
            make_tensor_field(name="x", frontend_getter="get_x()"),
            make_tensor_field(name="unmatched", frontend_getter="get_unmatched()"),
        ]
        inputs = [make_frontend_tensor_config(name="x")]
        frontend = make_frontend_config(inputs=inputs)
        cfg = make_minimal_config(tensor_fields=tensor_fields, frontend=frontend)
        result = cfg.tensor_field_frontend_map
        assert "x" in result
        assert "unmatched" not in result

    def test_empty_frontend_returns_empty_map(self):
        """No frontend inputs/outputs returns empty map."""
        tensor_fields = [make_tensor_field(name="x")]
        frontend = FrontendConfig()
        cfg = make_minimal_config(tensor_fields=tensor_fields, frontend=frontend)
        assert cfg.tensor_field_frontend_map == {}

    def test_multiple_tensors_matched(self):
        """Multiple tensor fields matched across inputs and outputs."""
        tensor_fields = [
            make_tensor_field(name="x", frontend_getter="get_x()"),
            make_tensor_field(name="y", frontend_getter="get_y()"),
        ]
        inputs = [make_frontend_tensor_config(name="x")]
        outputs = [make_frontend_tensor_config(name="y")]
        frontend = make_frontend_config(inputs=inputs, outputs=outputs)
        cfg = make_minimal_config(tensor_fields=tensor_fields, frontend=frontend)
        result = cfg.tensor_field_frontend_map
        assert len(result) == 2
        assert result["x"].name == "x"
        assert result["y"].name == "y"

    # --- Reverse map ---

    def test_reverse_map_basic(self):
        """frontend_to_tensor_field_map is the reverse of tensor_field_frontend_map."""
        tensor_fields = [
            make_tensor_field(name="x", frontend_getter="get_x()"),
        ]
        inputs = [make_frontend_tensor_config(name="x")]
        frontend = make_frontend_config(inputs=inputs)
        cfg = make_minimal_config(tensor_fields=tensor_fields, frontend=frontend)
        result = cfg.frontend_to_tensor_field_map
        assert result == {"x": "x"}

    def test_reverse_map_abbreviated(self):
        """Reverse map handles abbreviated name mappings correctly."""
        tensor_fields = [
            make_tensor_field(name="in_0", frontend_getter="get_in_0()"),
            make_tensor_field(name="out_0", frontend_getter="get_out_0()"),
        ]
        inputs = [
            make_frontend_tensor_config(name="input_0", getter_name="get_input_0")
        ]
        outputs = [
            make_frontend_tensor_config(name="output_0", getter_name="get_output_0")
        ]
        frontend = make_frontend_config(inputs=inputs, outputs=outputs)
        cfg = make_minimal_config(tensor_fields=tensor_fields, frontend=frontend)
        result = cfg.frontend_to_tensor_field_map
        assert result == {"input_0": "in_0", "output_0": "out_0"}

    def test_reverse_map_empty_frontend(self):
        """Empty frontend yields empty reverse map."""
        cfg = make_minimal_config(frontend=FrontendConfig())
        assert cfg.frontend_to_tensor_field_map == {}

    def test_reverse_map_partial_match(self):
        """Unmatched tensors are absent from both maps."""
        tensor_fields = [
            make_tensor_field(name="x", frontend_getter="get_x()"),
            make_tensor_field(name="z", frontend_getter="get_z()"),
        ]
        inputs = [make_frontend_tensor_config(name="x")]
        frontend = make_frontend_config(inputs=inputs)
        cfg = make_minimal_config(tensor_fields=tensor_fields, frontend=frontend)
        forward = cfg.tensor_field_frontend_map
        reverse = cfg.frontend_to_tensor_field_map
        assert "x" in forward
        assert "z" not in forward
        assert reverse == {"x": "x"}


# ---------------------------------------------------------------------------
# Additional edge case tests
# ---------------------------------------------------------------------------


class TestOperationConfigEdgeCases:
    """Edge case tests across multiple OperationConfig properties."""

    def test_empty_tensor_fields_all_filters_empty(self):
        cfg = make_minimal_config(tensor_fields=[], data_fields=[])
        assert cfg.required_tensor_fields == []
        assert cfg.optional_tensor_fields == []
        assert cfg.all_tensor_names == []
        assert cfg.tensor_uid_list == []
        assert cfg.tensor_attr_cases == []

    def test_empty_data_fields_all_filters_empty(self):
        cfg = make_minimal_config(data_fields=[])
        assert cfg.required_data_fields == []
        assert cfg.optional_data_fields == []
        assert cfg.enum_fields == []
        assert cfg.mode_fields == []
        assert cfg.vector_fields == []
        assert cfg.scalar_fields == []
        assert cfg.optional_scalar_fields == []
        assert cfg.has_enum_fields is False
        assert cfg.has_vector_fields is False
        assert cfg.has_mode_fields is False

    def test_data_field_with_all_types(self):
        """Verify each type is categorized into exactly one filter."""
        types = ["vector_int64", "enum", "mode", "scalar_float", "scalar_int64", "bool"]
        for t in types:
            df = make_data_field(name=f"field_{t}", type=t)
            categories = sum([df.is_vector, df.is_enum, df.is_mode, df.is_scalar])
            # Each field type is classified into exactly one primary category
            assert categories == 1, f"Type {t} in {categories} categories (expected 1)"

    def test_packer_filename_empty_node_class_empty_packer_function(self):
        """When both packer_function and node_class are empty, fallback to name."""
        frontend = FrontendConfig()
        cfg = make_minimal_config(name="MyOp", frontend=frontend)
        assert cfg.packer_filename == "MyOpPacker.hpp"

    def test_integration_filename_node_class_without_node_suffix(self):
        frontend = make_frontend_config(node_class="BatchNormFwd")
        cfg = make_minimal_config(frontend=frontend)
        assert (
            cfg.test_integration_filename
            == "IntegrationBatchNormFwdDescriptorLowering.cpp"
        )
        assert (
            cfg.test_integration_lifting_filename
            == "IntegrationBatchNormFwdDescriptorLifting.cpp"
        )


class TestEffectiveConstantsInclude:
    """Tests for effective_constants_include and constants_namespace properties."""

    def test_returns_constants_include_when_set(self):
        td = make_test_data(constants_include="ConvFpropConstants")
        cfg = make_minimal_config(test_data=td)
        assert cfg.effective_constants_include == "ConvFpropConstants"

    def test_derives_from_name_when_not_set(self):
        td = make_test_data(constants_include="")
        cfg = make_minimal_config(name="BatchnormInference", test_data=td)
        assert cfg.effective_constants_include == "BatchnormInferenceConstants"

    def test_derives_from_name_when_default(self):
        cfg = make_minimal_config(name="ConvolutionBwd")
        assert cfg.effective_constants_include == "ConvolutionBwdConstants"


class TestTensorConstPrefix:
    """Tests for OperationConfig.tensor_const_prefix property."""

    def test_with_constants_include(self):
        """Pre-existing constants header uses short 'K_' prefix."""
        td = make_test_data(constants_include="ConvFpropConstants")
        cfg = make_minimal_config(test_data=td)
        assert cfg.tensor_const_prefix == "K_"

    def test_explicit_override_takes_precedence(self):
        """Explicit override beats the generic constants_include fallback."""
        td = make_test_data(
            constants_include="ConvFpropConstants", tensor_const_prefix="K_FPROP_"
        )
        cfg = make_minimal_config(test_data=td)
        assert cfg.tensor_const_prefix == "K_FPROP_"

    def test_without_constants_include(self):
        """No constants_include derives prefix from operation name."""
        cfg = make_minimal_config()
        assert cfg.tensor_const_prefix == "K_TESTOP_"
