# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for convert_miopen_shapes.py."""

import pytest

from dnn_benchmarking.tools.convert_miopen_shapes import (
    BNORM_FLAG_ALIASES,
    CONV_FLAG_ALIASES,
    ConvParams,
    build_bnorm_json,
    build_conv_json,
    conv_io_type,
    conv_out_dim,
    convert_line,
    is_flag,
    nchw_strides,
    ncdhw_strides,
    nhwc_strides,
    ndhwc_strides,
    normalize_args,
    parse_args,
    parse_line,
)


class TestStrideHelpers:
    """Tests for stride computation helpers."""

    def testnchw_strides_known_values(self) -> None:
        """NCHW strides: innermost dim is 1, outermost is product of inner dims."""
        # N=2, C=3, H=4, W=5 → [C*H*W, H*W, W, 1] = [60, 20, 5, 1]
        assert nchw_strides(2, 3, 4, 5) == [60, 20, 5, 1]

    def testnhwc_strides_known_values(self) -> None:
        """NHWC strides: C is innermost (stride 1), then W*C, H*W*C, etc."""
        # N=2, C=3, H=4, W=5 → [H*W*C, 1, W*C, C] = [60, 1, 15, 3]
        assert nhwc_strides(2, 3, 4, 5) == [60, 1, 15, 3]

    def testncdhw_strides_known_values(self) -> None:
        # N=1, C=2, D=3, H=4, W=5 → [C*D*H*W, D*H*W, H*W, W, 1] = [120, 60, 20, 5, 1]
        assert ncdhw_strides(1, 2, 3, 4, 5) == [120, 60, 20, 5, 1]

    def testndhwc_strides_known_values(self) -> None:
        # N=1, C=2, D=3, H=4, W=5 → [D*H*W*C, 1, H*W*C, W*C, C] = [120, 1, 40, 10, 2]
        assert ndhwc_strides(1, 2, 3, 4, 5) == [120, 1, 40, 10, 2]


class TestConvOutDim:
    """Tests for convolution output dimension formula."""

    def test_no_pad_stride1_dil1(self) -> None:
        # floor((16 + 0 - 1*(3-1) - 1) / 1 + 1) = floor(14/1 + 1) = 15
        assert conv_out_dim(16, 0, 1, 3, 1) == 14

    def test_with_pad(self) -> None:
        # floor((16 + 2 - 1*(3-1) - 1) / 1 + 1) = floor(15/1 + 1) = 16
        assert conv_out_dim(16, 1, 1, 3, 1) == 16

    def test_with_stride(self) -> None:
        # floor((16 + 0 - 1*(1-1) - 1) / 2 + 1) = floor(15/2 + 1) = 8
        assert conv_out_dim(16, 0, 1, 1, 2) == 8

    def test_with_dilation(self) -> None:
        # floor((16 + 0 - 2*(3-1) - 1) / 1 + 1) = floor((16 - 4 - 1)/1 + 1) = 12
        assert conv_out_dim(16, 0, 2, 3, 1) == 12

    def test_stride_and_pad_combined(self) -> None:
        # floor((16 + 2 - 1*(3-1) - 1) / 2 + 1) = floor(15/2 + 1) = 8
        assert conv_out_dim(16, 1, 1, 3, 2) == 8

    def test_kernel_larger_than_input_with_pad(self) -> None:
        # Input 3, kernel 5, pad 2, dil 1, stride 1:
        # floor((3 + 4 - 1*(5-1) - 1) / 1 + 1) = floor(2/1 + 1) = 3
        assert conv_out_dim(3, 2, 1, 5, 1) == 3

    def test_kernel_equal_input_no_pad(self) -> None:
        # floor((4 + 0 - 1*(4-1) - 1) / 1 + 1) = floor(0/1 + 1) = 1
        assert conv_out_dim(4, 0, 1, 4, 1) == 1


class TestIsFlag:
    """Tests for is_flag helper."""

    def test_normal_flag(self) -> None:
        assert is_flag("-n") is True

    def test_long_flag(self) -> None:
        assert is_flag("--layout") is True

    def test_negative_integer_is_not_flag(self) -> None:
        assert is_flag("-1") is False

    def test_negative_float_is_not_flag(self) -> None:
        assert is_flag("-0.5") is False

    def test_non_flag_token(self) -> None:
        assert is_flag("value") is False


class TestParseArgs:
    """Tests for parse_args."""

    def test_basic_flag_value_pairs(self) -> None:
        result = parse_args(["-n", "16", "-c", "64"])
        assert result == {"-n": "16", "-c": "64"}

    def test_boolean_flag(self) -> None:
        result = parse_args(["--verbose"])
        assert result == {"--verbose": "1"}

    def test_negative_value(self) -> None:
        # -p -1: -1 should be treated as a value, not a new flag
        result = parse_args(["-p", "-1"])
        assert result == {"-p": "-1"}

    def test_mixed(self) -> None:
        result = parse_args(["-n", "4", "--verbose", "-c", "32"])
        assert result["-n"] == "4"
        assert result["--verbose"] == "1"
        assert result["-c"] == "32"


class TestBuildConvJson:
    """Tests for build_conv_json via ConvParams."""

    def _make_params(self, **overrides) -> ConvParams:
        defaults = dict(
            N=2,
            C=32,
            H=8,
            W=8,
            K=64,
            R=3,
            S=3,
            pad_h=1,
            pad_w=1,
            stride_h=1,
            stride_w=1,
            dil_h=1,
            dil_w=1,
            groups=1,
            F=1,
            spatial_dim=2,
            in_layout="NCHW",
            fil_layout="NCHW",
            out_layout="NCHW",
        )
        defaults.update(overrides)
        return ConvParams(**defaults)

    def test_conv_fwd_node_type(self) -> None:
        p = self._make_params(F=1)
        graph = build_conv_json(p)
        assert graph["nodes"][0]["type"] == "ConvolutionFwdAttributes"

    def test_conv_fwd_tensor_wiring(self) -> None:
        p = self._make_params(F=1)
        graph = build_conv_json(p)
        node = graph["nodes"][0]
        assert node["inputs"] == {"x_tensor_uid": 1, "w_tensor_uid": 2}
        assert node["outputs"] == {"y_tensor_uid": 0}

    def test_conv_dgrad_node_type(self) -> None:
        p = self._make_params(F=2)
        graph = build_conv_json(p)
        assert graph["nodes"][0]["type"] == "ConvolutionBwdAttributes"

    def test_conv_dgrad_tensor_wiring(self) -> None:
        p = self._make_params(F=2)
        graph = build_conv_json(p)
        node = graph["nodes"][0]
        assert node["inputs"] == {"dy_tensor_uid": 1, "w_tensor_uid": 2}
        assert node["outputs"] == {"dx_tensor_uid": 0}

    def test_conv_wgrad_node_type(self) -> None:
        p = self._make_params(F=4)
        graph = build_conv_json(p)
        assert graph["nodes"][0]["type"] == "ConvolutionWrwAttributes"

    def test_conv_wgrad_tensor_wiring(self) -> None:
        p = self._make_params(F=4)
        graph = build_conv_json(p)
        node = graph["nodes"][0]
        assert node["inputs"] == {"dy_tensor_uid": 1, "x_tensor_uid": 2}
        assert node["outputs"] == {"dw_tensor_uid": 0}

    def test_nhwc_layout_produces_different_strides(self) -> None:
        p_nchw = self._make_params(in_layout="NCHW")
        p_nhwc = self._make_params(in_layout="NHWC")
        g_nchw = build_conv_json(p_nchw)
        g_nhwc = build_conv_json(p_nhwc)
        # Input tensor (uid=1) strides should differ
        strides_nchw = next(t["strides"] for t in g_nchw["tensors"] if t["uid"] == 1)
        strides_nhwc = next(t["strides"] for t in g_nhwc["tensors"] if t["uid"] == 1)
        assert strides_nchw != strides_nhwc

    def test_3d_conv_produces_5d_dims(self) -> None:
        p = ConvParams(
            N=1,
            C=4,
            H=4,
            W=4,
            K=8,
            R=3,
            S=3,
            pad_h=1,
            pad_w=1,
            stride_h=1,
            stride_w=1,
            dil_h=1,
            dil_w=1,
            groups=1,
            F=1,
            spatial_dim=3,
            in_layout="NCDHW",
            fil_layout="NCDHW",
            out_layout="NCDHW",
            D=4,
            D_f=3,
            pad_d=1,
            stride_d=1,
            dil_d=1,
        )
        graph = build_conv_json(p)
        x_tensor = next(t for t in graph["tensors"] if t["uid"] == 1)
        assert len(x_tensor["dims"]) == 5
        assert len(x_tensor["strides"]) == 5


class TestBuildBnormJson:
    """Tests for build_bnorm_json."""

    def _base_args(self, **overrides) -> dict:
        defaults = {"-n": "2", "-c": "64", "-H": "14", "-W": "14"}
        defaults.update(overrides)
        return defaults

    def test_inference_node_type(self) -> None:
        args = self._base_args(**{"--forw": "2"})
        graph = build_bnorm_json("bnorm", args)
        assert graph["nodes"][0]["type"] == "BatchnormInferenceAttributes"

    def test_inference_tensor_count(self) -> None:
        args = self._base_args(**{"--forw": "2"})
        graph = build_bnorm_json("bnorm", args)
        assert len(graph["tensors"]) == 6

    def test_fwd_training_node_type(self) -> None:
        args = self._base_args(**{"--forw": "1"})
        graph = build_bnorm_json("bnorm", args)
        assert graph["nodes"][0]["type"] == "BatchnormAttributes"

    def test_fwd_training_has_epsilon_tensor(self) -> None:
        args = self._base_args(**{"--forw": "1"})
        graph = build_bnorm_json("bnorm", args)
        node = graph["nodes"][0]
        assert "epsilon_tensor_uid" in node["inputs"]
        epsilon_uid = node["inputs"]["epsilon_tensor_uid"]
        epsilon_tensor = next(t for t in graph["tensors"] if t["uid"] == epsilon_uid)
        assert epsilon_tensor["dims"] == [1]

    def test_backward_node_type(self) -> None:
        args = self._base_args(**{"--back": "1"})
        graph = build_bnorm_json("bnorm", args)
        assert graph["nodes"][0]["type"] == "BatchnormBackwardAttributes"

    def test_backward_tensor_count(self) -> None:
        args = self._base_args(**{"--back": "1"})
        graph = build_bnorm_json("bnorm", args)
        assert len(graph["tensors"]) == 8

    def test_backward_has_mean_and_inv_variance(self) -> None:
        args = self._base_args(**{"--back": "1"})
        graph = build_bnorm_json("bnorm", args)
        node = graph["nodes"][0]
        assert "mean_tensor_uid" in node["inputs"]
        assert "inv_variance_tensor_uid" in node["inputs"]


class TestParseLine:
    """Tests for parse_line."""

    def test_blank_returns_none(self) -> None:
        assert parse_line("") is None
        assert parse_line("   ") is None

    def test_comment_returns_none(self) -> None:
        assert parse_line("# this is a comment") is None

    def test_valid_conv_line(self) -> None:
        line = "./bin/MIOpenDriver convbfp16 -n 2 -c 64 -H 8 -W 8 -k 128 -y 3 -x 3 -F 1"
        result = parse_line(line)
        assert result is not None
        operation, args = result
        assert operation == "convbfp16"
        assert args["-n"] == "2"
        assert args["-F"] == "1"

    def test_repeat_count_prefix_stripped(self) -> None:
        line = (
            "     5  ./bin/MIOpenDriver convbfp16 -n 1 -c 32 -H 4 -W 4 -k 32 -y 1 -x 1"
        )
        result = parse_line(line)
        assert result is not None
        operation, _ = result
        assert operation == "convbfp16"

    def test_too_short_returns_none(self) -> None:
        assert parse_line("./bin/MIOpenDriver") is None


class TestConvertLine:
    """Tests for convert_line dispatcher."""

    def test_unsupported_operation_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported operation"):
            convert_line("matmul", {}, "prefix")

    def test_conv_operation_succeeds(self) -> None:
        args = {
            "-n": "1",
            "-c": "8",
            "-H": "4",
            "-W": "4",
            "-k": "16",
            "-y": "1",
            "-x": "1",
            "-F": "1",
        }
        name_stem, graph = convert_line("convbfp16", args, "test")
        assert "conv" in name_stem
        assert graph["nodes"][0]["type"] == "ConvolutionFwdAttributes"

    def test_bnorm_operation_succeeds(self) -> None:
        args = {"-n": "1", "-c": "32", "-H": "8", "-W": "8", "--forw": "2"}
        name_stem, graph = convert_line("bnormbfp16", args, "test")
        assert "bnorm" in name_stem
        assert graph["nodes"][0]["type"] == "BatchnormInferenceAttributes"


class TestNormalizeArgs:
    """Tests for normalize_args helper."""

    def test_alternative_key_mapped_to_canonical(self) -> None:
        aliases = {"--long": "-s"}
        result = normalize_args({"--long": "42"}, aliases)
        assert result == {"-s": "42"}

    def test_canonical_key_preserved_when_no_conflict(self) -> None:
        aliases = {"--long": "-s"}
        result = normalize_args({"-s": "42"}, aliases)
        assert result == {"-s": "42"}

    def test_canonical_wins_when_both_present(self) -> None:
        aliases = {"--long": "-s"}
        result = normalize_args({"--long": "99", "-s": "42"}, aliases)
        assert result == {"-s": "42"}

    def test_unrelated_keys_untouched(self) -> None:
        aliases = {"--long": "-s"}
        result = normalize_args({"--other": "7"}, aliases)
        assert result == {"--other": "7"}


class TestConvFlagAliases:
    """Verify that long-form MIOpenDriver conv flags produce the same graph
    as their short-form equivalents."""

    _SHORT_ARGS = {
        "-n": "16",
        "-c": "96",
        "-H": "48",
        "-W": "32",
        "-k": "96",
        "-y": "3",
        "-x": "1",
        "-p": "1",
        "-q": "0",
        "-u": "1",
        "-v": "1",
        "-l": "1",
        "-j": "1",
        "-g": "1",
        "-F": "1",
    }

    _LONG_ARGS = {
        "--batchsize": "16",
        "--in_channels": "96",
        "--in_h": "48",
        "--in_w": "32",
        "--out_channels": "96",
        "--fil_h": "3",
        "--fil_w": "1",
        "--pad_h": "1",
        "--pad_w": "0",
        "--conv_stride_h": "1",
        "--conv_stride_w": "1",
        "--dilation_h": "1",
        "--dilation_w": "1",
        "--group_count": "1",
        "--forw": "1",
    }

    def test_conv_short_and_long_produce_same_params(self) -> None:
        short_p = ConvParams.from_args(self._SHORT_ARGS)
        long_p = ConvParams.from_args(self._LONG_ARGS)
        assert short_p == long_p

    def test_conv_short_and_long_produce_same_graph(self) -> None:
        short_p = ConvParams.from_args(self._SHORT_ARGS)
        long_p = ConvParams.from_args(self._LONG_ARGS)
        assert build_conv_json(short_p) == build_conv_json(long_p)

    def testconvert_line_with_long_flags(self) -> None:
        _, graph = convert_line("convbfp16", dict(self._LONG_ARGS), "test")
        assert graph["nodes"][0]["type"] == "ConvolutionFwdAttributes"

    def test_3d_short_flags(self) -> None:
        """Short forms -_, -!, -@, -$, -#, -^ for 3D conv params."""
        args = {
            "-n": "1",
            "-c": "4",
            "-H": "4",
            "-W": "4",
            "-k": "8",
            "-y": "3",
            "-x": "3",
            "-p": "1",
            "-q": "1",
            "-u": "1",
            "-v": "1",
            "-l": "1",
            "-j": "1",
            "-g": "1",
            "-F": "1",
            "-_": "3",
            "-!": "4",
            "-@": "3",
            "-$": "1",
            "-#": "1",
            "-^": "1",
        }
        p = ConvParams.from_args(args)
        assert p.spatial_dim == 3
        assert p.D == 4
        assert p.D_f == 3
        assert p.pad_d == 1
        assert p.stride_d == 1
        assert p.dil_d == 1

    def test_3d_short_and_long_match(self) -> None:
        short_3d = {
            "-n": "1",
            "-c": "4",
            "-H": "4",
            "-W": "4",
            "-k": "8",
            "-y": "3",
            "-x": "3",
            "-p": "1",
            "-q": "1",
            "-u": "1",
            "-v": "1",
            "-l": "1",
            "-j": "1",
            "-g": "1",
            "-F": "1",
            "-_": "3",
            "-!": "4",
            "-@": "3",
            "-$": "1",
            "-#": "1",
            "-^": "1",
        }
        long_3d = {
            "-n": "1",
            "-c": "4",
            "-H": "4",
            "-W": "4",
            "-k": "8",
            "-y": "3",
            "-x": "3",
            "-p": "1",
            "-q": "1",
            "-u": "1",
            "-v": "1",
            "-l": "1",
            "-j": "1",
            "-g": "1",
            "-F": "1",
            "--spatial_dim": "3",
            "--in_d": "4",
            "--fil_d": "3",
            "--pad_d": "1",
            "--conv_stride_d": "1",
            "--dilation_d": "1",
        }
        assert ConvParams.from_args(short_3d) == ConvParams.from_args(long_3d)

    def test_layout_short_flags(self) -> None:
        """Short forms -I, -f, -O for layout flags."""
        args = {
            "-n": "1",
            "-c": "4",
            "-H": "4",
            "-W": "4",
            "-k": "8",
            "-y": "1",
            "-x": "1",
            "-F": "1",
            "-I": "NHWC",
            "-f": "NHWC",
            "-O": "NHWC",
        }
        p = ConvParams.from_args(args)
        assert p.in_layout == "NHWC"
        assert p.fil_layout == "NHWC"
        assert p.out_layout == "NHWC"


class TestBnormFlagAliases:
    """Verify that long-form MIOpenDriver bnorm flags produce the same graph
    as their short-form equivalents."""

    def test_bnorm_long_form_dims(self) -> None:
        short_args = {"-n": "8", "-c": "64", "-H": "16", "-W": "16", "--forw": "2"}
        long_args = {
            "--batchsize": "8",
            "--in_channels": "64",
            "--in_h": "16",
            "--in_w": "16",
            "--forw": "2",
        }
        short_graph = build_bnorm_json("bnormbfp16", short_args)
        long_graph = build_bnorm_json("bnormbfp16", long_args)
        assert short_graph == long_graph

    def test_bnorm_short_forw_flag(self) -> None:
        """Using -F instead of --forw for bnorm."""
        args_long = {"-n": "4", "-c": "32", "-H": "8", "-W": "8", "--forw": "2"}
        args_short = {"-n": "4", "-c": "32", "-H": "8", "-W": "8", "-F": "2"}
        assert build_bnorm_json("bnorm", args_long) == build_bnorm_json(
            "bnorm", args_short
        )

    def test_bnorm_short_back_flag(self) -> None:
        """Using -b instead of --back for bnorm backward."""
        args_long = {"-n": "4", "-c": "32", "-H": "8", "-W": "8", "--back": "1"}
        args_short = {"-n": "4", "-c": "32", "-H": "8", "-W": "8", "-b": "1"}
        assert build_bnorm_json("bnorm", args_long) == build_bnorm_json(
            "bnorm", args_short
        )

    def test_bnorm_3d_long_depth(self) -> None:
        """Using --in_d instead of -D for 3D batchnorm."""
        args_short = {"-n": "2", "-c": "16", "-H": "8", "-W": "8", "-D": "4"}
        args_long = {"-n": "2", "-c": "16", "-H": "8", "-W": "8", "--in_d": "4"}
        assert build_bnorm_json("bnorm", args_short) == build_bnorm_json(
            "bnorm", args_long
        )


class TestConvDataTypes:
    """Verify conv operations map to correct hipDNN DataType enum strings."""

    _BASIC_ARGS = {
        "-n": "1",
        "-c": "8",
        "-H": "4",
        "-W": "4",
        "-k": "16",
        "-y": "1",
        "-x": "1",
        "-F": "1",
    }

    @pytest.mark.parametrize(
        "operation,expected_io_type",
        [
            ("conv", "float"),
            ("convfp16", "half"),
            ("convbfp16", "bfloat16"),
            ("convfp32", "float"),
        ],
    )
    def testconv_io_type_mapping(self, operation: str, expected_io_type: str) -> None:
        assert conv_io_type(operation) == expected_io_type

    @pytest.mark.parametrize(
        "operation,expected_io_type",
        [
            ("conv", "float"),
            ("convfp16", "half"),
            ("convbfp16", "bfloat16"),
        ],
    )
    def testconvert_line_sets_correct_io_type(
        self, operation: str, expected_io_type: str
    ) -> None:
        _, graph = convert_line(operation, dict(self._BASIC_ARGS), "test")
        assert graph["io_data_type"] == expected_io_type

    @pytest.mark.parametrize(
        "operation,expected_io_type",
        [
            ("conv", "float"),
            ("convfp16", "half"),
            ("convbfp16", "bfloat16"),
        ],
    )
    def test_conv_tensor_data_types(
        self, operation: str, expected_io_type: str
    ) -> None:
        _, graph = convert_line(operation, dict(self._BASIC_ARGS), "test")
        for tensor in graph["tensors"]:
            assert tensor["data_type"] == expected_io_type


class TestBnormDataTypes:
    """Verify bnorm operations map to correct hipDNN DataType enum strings."""

    _BASIC_ARGS = {"-n": "2", "-c": "32", "-H": "8", "-W": "8", "--forw": "2"}

    @pytest.mark.parametrize(
        "operation,expected_io_type,expected_stat_type",
        [
            ("bnorm", "float", "float"),
            ("bnormfp16", "half", "float"),
            ("bnormbfp16", "bfloat16", "float"),
        ],
    )
    def test_bnorm_inference_tensor_types(
        self, operation: str, expected_io_type: str, expected_stat_type: str
    ) -> None:
        graph = build_bnorm_json(operation, dict(self._BASIC_ARGS))
        assert graph["io_data_type"] == expected_io_type
        # x and y tensors should have io_type
        x_tensor = next(t for t in graph["tensors"] if t["name"] == "input_x")
        y_tensor = next(t for t in graph["tensors"] if t["name"] == "output_y")
        assert x_tensor["data_type"] == expected_io_type
        assert y_tensor["data_type"] == expected_io_type
        # stat tensors should be float
        mean_tensor = next(t for t in graph["tensors"] if t["name"] == "mean")
        assert mean_tensor["data_type"] == expected_stat_type

    @pytest.mark.parametrize(
        "operation,expected_scale_type",
        [
            ("bnorm", "float"),
            ("bnormfp16", "float"),
            ("bnormbfp16", "float"),
            ("bnormfp16fp32", "half"),
            ("bnormbfp16fp32", "bfloat16"),
        ],
    )
    def test_bnorm_scale_bias_types(
        self, operation: str, expected_scale_type: str
    ) -> None:
        graph = build_bnorm_json(operation, dict(self._BASIC_ARGS))
        scale_tensor = next(t for t in graph["tensors"] if t["name"] == "scale")
        assert scale_tensor["data_type"] == expected_scale_type
