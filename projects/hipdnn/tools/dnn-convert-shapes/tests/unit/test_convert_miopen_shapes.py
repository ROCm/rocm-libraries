# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for convert_miopen_shapes.py."""

import pytest

from dnn_convert_shapes import (
    BNORM_FLAG_ALIASES,
    CONV_FLAG_ALIASES,
    ConvMode,
    ConvParams,
    build_bnorm_json,
    build_conv_json,
    CONV_IO_TYPE,
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
from dnn_convert_shapes.strides import Layout


class TestStrideHelpers:
    """Tests for stride computation helpers."""

    def test_nchw_strides_known_values(self) -> None:
        """NCHW strides: innermost dim is 1, outermost is product of inner dims."""
        # N=2, C=3, H=4, W=5 → [C*H*W, H*W, W, 1] = [60, 20, 5, 1]
        assert nchw_strides(2, 3, 4, 5) == [60, 20, 5, 1]

    def test_nhwc_strides_known_values(self) -> None:
        """NHWC strides: C is innermost (stride 1), then W*C, H*W*C, etc."""
        # N=2, C=3, H=4, W=5 → [H*W*C, 1, W*C, C] = [60, 1, 15, 3]
        assert nhwc_strides(2, 3, 4, 5) == [60, 1, 15, 3]

    def test_ncdhw_strides_known_values(self) -> None:
        # N=1, C=2, D=3, H=4, W=5 → [C*D*H*W, D*H*W, H*W, W, 1] = [120, 60, 20, 5, 1]
        assert ncdhw_strides(1, 2, 3, 4, 5) == [120, 60, 20, 5, 1]

    def test_ndhwc_strides_known_values(self) -> None:
        # N=1, C=2, D=3, H=4, W=5 → [D*H*W*C, 1, H*W*C, W*C, C] = [120, 1, 40, 10, 2]
        assert ndhwc_strides(1, 2, 3, 4, 5) == [120, 1, 40, 10, 2]


class TestLayoutValidation:
    """Tests for layout validation in stride helpers."""

    def test_invalid_layout_raises(self) -> None:
        with pytest.raises(ValueError):
            Layout("HWCN")

    def test_valid_2d_layouts_accepted(self) -> None:
        from dnn_convert_shapes.strides import _input_strides

        _input_strides(Layout.NCHW, 2, 3, 4, 5)
        _input_strides(Layout.NHWC, 2, 3, 4, 5)

    def test_valid_3d_layouts_accepted(self) -> None:
        from dnn_convert_shapes.strides import _input_strides

        _input_strides(Layout.NCDHW, 2, 3, 4, 5, D=4)
        _input_strides(Layout.NDHWC, 2, 3, 4, 5, D=4)


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
            in_layout=Layout.NCHW,
            fil_layout=Layout.NCHW,
            out_layout=Layout.NCHW,
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
        p_nchw = self._make_params(in_layout=Layout.NCHW)
        p_nhwc = self._make_params(in_layout=Layout.NHWC)
        g_nchw = build_conv_json(p_nchw)
        g_nhwc = build_conv_json(p_nhwc)
        # Input tensor (uid=1) strides should differ
        strides_nchw = next(t["strides"] for t in g_nchw["tensors"] if t["uid"] == 1)
        strides_nhwc = next(t["strides"] for t in g_nhwc["tensors"] if t["uid"] == 1)
        assert strides_nchw != strides_nhwc

    def test_3d_conv_defaults_to_ncdhw_layout(self) -> None:
        args = {
            "-n": "1",
            "-c": "4",
            "-H": "4",
            "-W": "4",
            "-k": "8",
            "-y": "3",
            "-x": "3",
            "-F": "1",
            "--spatial_dim": "3",
            "--in_d": "4",
            "--fil_d": "3",
        }
        p = ConvParams.from_args(args)
        assert p.in_layout is Layout.NCDHW
        assert p.fil_layout is Layout.NCDHW
        assert p.out_layout is Layout.NCDHW

    def test_2d_conv_defaults_to_nchw_layout(self) -> None:
        args = {"-n": "1", "-c": "4", "-H": "4", "-W": "4", "-k": "8", "-F": "1"}
        p = ConvParams.from_args(args)
        assert p.in_layout is Layout.NCHW
        assert p.fil_layout is Layout.NCHW
        assert p.out_layout is Layout.NCHW

    def test_conv_mode_defaults_to_cross_correlation(self) -> None:
        p = self._make_params()
        graph = build_conv_json(p)
        assert graph["nodes"][0]["parameters"]["conv_mode"] == "CROSS_CORRELATION"

    def test_conv_mode_trans_produces_convolution(self) -> None:
        p = self._make_params(conv_mode=ConvMode.CONVOLUTION)
        graph = build_conv_json(p)
        assert graph["nodes"][0]["parameters"]["conv_mode"] == "CONVOLUTION"

    def test_conv_mode_parsed_from_args(self) -> None:
        args = {
            "-n": "1",
            "-c": "4",
            "-H": "4",
            "-W": "4",
            "-k": "8",
            "-F": "1",
            "-m": "trans",
        }
        p = ConvParams.from_args(args)
        assert p.conv_mode is ConvMode.CONVOLUTION

    def test_conv_mode_default_is_conv(self) -> None:
        args = {"-n": "1", "-c": "4", "-H": "4", "-W": "4", "-k": "8", "-F": "1"}
        p = ConvParams.from_args(args)
        assert p.conv_mode is ConvMode.CROSS_CORRELATION

    def test_conv_mode_long_flag(self) -> None:
        args = {
            "-n": "1",
            "-c": "4",
            "-H": "4",
            "-W": "4",
            "-k": "8",
            "-F": "1",
            "--mode": "trans",
        }
        p = ConvParams.from_args(args)
        assert p.conv_mode is ConvMode.CONVOLUTION

    def test_conv_mode_invalid_raises(self) -> None:
        args = {
            "-n": "1",
            "-c": "4",
            "-H": "4",
            "-W": "4",
            "-k": "8",
            "-F": "1",
            "-m": "invalid",
        }
        with pytest.raises(ValueError, match="Unknown conv mode"):
            ConvParams.from_args(args)

    def test_pad_mode_same_computes_padding(self) -> None:
        args = {
            "-n": "1",
            "-c": "3",
            "-H": "8",
            "-W": "8",
            "-k": "16",
            "-y": "3",
            "-x": "3",
            "-F": "1",
            "-z": "same",
        }
        p = ConvParams.from_args(args)
        assert p.pad_h == 1
        assert p.pad_w == 1

    def test_pad_mode_same_ignores_explicit_padding(self) -> None:
        args = {
            "-n": "1",
            "-c": "3",
            "-H": "8",
            "-W": "8",
            "-k": "16",
            "-y": "3",
            "-x": "3",
            "-F": "1",
            "-p": "99",
            "-q": "99",
            "-z": "same",
        }
        p = ConvParams.from_args(args)
        assert p.pad_h == 1
        assert p.pad_w == 1

    def test_pad_mode_valid_zeroes_padding(self) -> None:
        args = {
            "-n": "1",
            "-c": "3",
            "-H": "8",
            "-W": "8",
            "-k": "16",
            "-y": "3",
            "-x": "3",
            "-F": "1",
            "-p": "5",
            "-q": "5",
            "-z": "valid",
        }
        p = ConvParams.from_args(args)
        assert p.pad_h == 0
        assert p.pad_w == 0

    def test_pad_mode_default_uses_explicit_padding(self) -> None:
        args = {
            "-n": "1",
            "-c": "3",
            "-H": "8",
            "-W": "8",
            "-k": "16",
            "-y": "3",
            "-x": "3",
            "-F": "1",
            "-p": "2",
            "-q": "3",
        }
        p = ConvParams.from_args(args)
        assert p.pad_h == 2
        assert p.pad_w == 3

    def test_pad_mode_same_with_dilation(self) -> None:
        args = {
            "-n": "1",
            "-c": "3",
            "-H": "16",
            "-W": "16",
            "-k": "16",
            "-y": "3",
            "-x": "3",
            "-F": "1",
            "-l": "2",
            "-j": "2",
            "-z": "same",
        }
        p = ConvParams.from_args(args)
        assert p.pad_h == 2
        assert p.pad_w == 2

    def test_pad_mode_same_with_kernel_5(self) -> None:
        args = {
            "-n": "1",
            "-c": "3",
            "-H": "16",
            "-W": "16",
            "-k": "16",
            "-y": "5",
            "-x": "5",
            "-F": "1",
            "-z": "same",
        }
        p = ConvParams.from_args(args)
        assert p.pad_h == 2
        assert p.pad_w == 2

    def test_pad_mode_long_flag(self) -> None:
        args = {
            "-n": "1",
            "-c": "3",
            "-H": "8",
            "-W": "8",
            "-k": "16",
            "-y": "3",
            "-x": "3",
            "-F": "1",
            "--pad_mode": "valid",
        }
        p = ConvParams.from_args(args)
        assert p.pad_h == 0
        assert p.pad_w == 0

    def test_pad_mode_invalid_raises(self) -> None:
        args = {
            "-n": "1",
            "-c": "3",
            "-H": "8",
            "-W": "8",
            "-k": "16",
            "-y": "3",
            "-x": "3",
            "-F": "1",
            "-z": "bad",
        }
        with pytest.raises(ValueError, match="Unknown pad_mode"):
            ConvParams.from_args(args)

    def test_pad_mode_same_3d(self) -> None:
        args = {
            "-n": "1",
            "-c": "3",
            "-H": "8",
            "-W": "8",
            "-k": "16",
            "-y": "3",
            "-x": "3",
            "-F": "1",
            "--spatial_dim": "3",
            "--in_d": "8",
            "--fil_d": "3",
            "-z": "same",
        }
        p = ConvParams.from_args(args)
        assert p.pad_h == 1
        assert p.pad_w == 1
        assert p.pad_d == 1

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
            in_layout=Layout.NCDHW,
            fil_layout=Layout.NCDHW,
            out_layout=Layout.NCDHW,
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

    def test_per_activation_mode_scale_dims_match_input(self) -> None:
        args = self._base_args(**{"--forw": "2", "-m": "0"})
        graph = build_bnorm_json("bnorm", args)
        scale = next(t for t in graph["tensors"] if t["name"] == "scale")
        assert scale["dims"] == [1, 64, 14, 14]

    def test_spatial_mode_scale_dims_collapsed(self) -> None:
        args = self._base_args(**{"--forw": "2", "-m": "1"})
        graph = build_bnorm_json("bnorm", args)
        scale = next(t for t in graph["tensors"] if t["name"] == "scale")
        assert scale["dims"] == [1, 64, 1, 1]

    def test_per_activation_mode_is_default(self) -> None:
        args = self._base_args(**{"--forw": "2"})
        graph = build_bnorm_json("bnorm", args)
        scale = next(t for t in graph["tensors"] if t["name"] == "scale")
        assert scale["dims"] == [1, 64, 14, 14]

    def test_3d_default_layout_is_ncdhw(self) -> None:
        from dnn_convert_shapes.bnorm import BnormParams

        args = {"-n": "2", "-c": "16", "-H": "8", "-W": "8", "-D": "4"}
        p = BnormParams.from_args(args)
        assert p.layout is Layout.NCDHW

    def test_2d_default_layout_is_nchw(self) -> None:
        from dnn_convert_shapes.bnorm import BnormParams

        args = {"-n": "2", "-c": "16", "-H": "8", "-W": "8"}
        p = BnormParams.from_args(args)
        assert p.layout is Layout.NCHW


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
        results = convert_line("convbfp16", args, "test")
        assert len(results) == 1
        name_stem, graph = results[0]
        assert "conv" in name_stem
        assert graph["nodes"][0]["type"] == "ConvolutionFwdAttributes"

    def test_bnorm_operation_succeeds(self) -> None:
        args = {"-n": "1", "-c": "32", "-H": "8", "-W": "8", "--forw": "2"}
        results = convert_line("bnormbfp16", args, "test")
        assert len(results) == 1
        name_stem, graph = results[0]
        assert "bnorm" in name_stem
        assert graph["nodes"][0]["type"] == "BatchnormInferenceAttributes"

    _MULTI_DIR_ARGS = {
        "-n": "1",
        "-c": "8",
        "-H": "4",
        "-W": "4",
        "-k": "16",
        "-y": "1",
        "-x": "1",
    }

    def test_conv_f0_produces_three_graphs(self) -> None:
        args = {**self._MULTI_DIR_ARGS, "-F": "0"}
        results = convert_line("convbfp16", args, "test")
        assert len(results) == 3
        node_types = {graph["nodes"][0]["type"] for _, graph in results}
        assert node_types == {
            "ConvolutionFwdAttributes",
            "ConvolutionBwdAttributes",
            "ConvolutionWrwAttributes",
        }
        stems = [stem for stem, _ in results]
        assert len(set(stems)) == 3

    def test_conv_f3_produces_fwd_and_dgrad(self) -> None:
        args = {**self._MULTI_DIR_ARGS, "-F": "3"}
        results = convert_line("convbfp16", args, "test")
        assert len(results) == 2
        node_types = {graph["nodes"][0]["type"] for _, graph in results}
        assert node_types == {
            "ConvolutionFwdAttributes",
            "ConvolutionBwdAttributes",
        }

    def test_conv_f5_produces_fwd_and_wgrad(self) -> None:
        args = {**self._MULTI_DIR_ARGS, "-F": "5"}
        results = convert_line("convbfp16", args, "test")
        assert len(results) == 2
        node_types = {graph["nodes"][0]["type"] for _, graph in results}
        assert node_types == {
            "ConvolutionFwdAttributes",
            "ConvolutionWrwAttributes",
        }

    def test_conv_f6_produces_dgrad_and_wgrad(self) -> None:
        args = {**self._MULTI_DIR_ARGS, "-F": "6"}
        results = convert_line("convbfp16", args, "test")
        assert len(results) == 2
        node_types = {graph["nodes"][0]["type"] for _, graph in results}
        assert node_types == {
            "ConvolutionBwdAttributes",
            "ConvolutionWrwAttributes",
        }

    def test_conv_f7_produces_all_three(self) -> None:
        args = {**self._MULTI_DIR_ARGS, "-F": "7"}
        results = convert_line("convbfp16", args, "test")
        assert len(results) == 3
        node_types = {graph["nodes"][0]["type"] for _, graph in results}
        assert node_types == {
            "ConvolutionFwdAttributes",
            "ConvolutionBwdAttributes",
            "ConvolutionWrwAttributes",
        }


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

    def test_convert_line_with_long_flags(self) -> None:
        [(_, graph)] = convert_line("convbfp16", dict(self._LONG_ARGS), "test")
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
        assert p.in_layout is Layout.NHWC
        assert p.fil_layout is Layout.NHWC
        assert p.out_layout is Layout.NHWC


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
        ],
    )
    def test_conv_io_type_mapping(self, operation: str, expected_io_type: str) -> None:
        assert CONV_IO_TYPE[operation] == expected_io_type

    @pytest.mark.parametrize(
        "operation,expected_io_type",
        [
            ("conv", "float"),
            ("convfp16", "half"),
            ("convbfp16", "bfloat16"),
        ],
    )
    def test_convert_line_sets_correct_io_type(
        self, operation: str, expected_io_type: str
    ) -> None:
        [(_, graph)] = convert_line(operation, dict(self._BASIC_ARGS), "test")
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
        [(_, graph)] = convert_line(operation, dict(self._BASIC_ARGS), "test")
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


class TestConvUnsupportedWarnings:
    """Verify warnings for MIOpen features not representable in hipDNN graphs."""

    _BASE = {"-n": "1", "-c": "4", "-H": "4", "-W": "4", "-k": "8", "-F": "1"}

    def test_bias_flag_warns(self) -> None:
        args = {**self._BASE, "-b": "1"}
        with pytest.warns(UserWarning, match="Conv bias"):
            ConvParams.from_args(args)

    def test_tensor_vect_warns(self) -> None:
        args = {**self._BASE, "-Z": "1"}
        with pytest.warns(UserWarning, match="Tensor vectorization"):
            ConvParams.from_args(args)

    def test_trans_output_pad_warns(self) -> None:
        args = {**self._BASE, "-Y": "1"}
        with pytest.warns(UserWarning, match="Transpose output padding"):
            ConvParams.from_args(args)

    def test_vector_length_warns(self) -> None:
        args = {**self._BASE, "-L": "4"}
        with pytest.warns(UserWarning, match="vector_length"):
            ConvParams.from_args(args)

    def test_pad_val_warns(self) -> None:
        args = {**self._BASE, "-r": "1"}
        with pytest.warns(UserWarning, match="pad_val"):
            ConvParams.from_args(args)

    def test_cast_type_warns(self) -> None:
        args = {**self._BASE, "-R": "1"}
        with pytest.warns(UserWarning, match="Cast type.*wei_cast_type"):
            ConvParams.from_args(args)

    def test_cast_type_out_warns(self) -> None:
        args = {**self._BASE, "-T": "1"}
        with pytest.warns(UserWarning, match="Cast type.*out_cast_type"):
            ConvParams.from_args(args)

    def test_cast_type_in_warns(self) -> None:
        args = {**self._BASE, "-U": "1"}
        with pytest.warns(UserWarning, match="Cast type.*in_cast_type"):
            ConvParams.from_args(args)

    def test_vector_length_long_flag_warns(self) -> None:
        args = {**self._BASE, "--vector_length": "4"}
        with pytest.warns(UserWarning, match="vector_length"):
            ConvParams.from_args(args)

    def test_pad_val_long_flag_warns(self) -> None:
        args = {**self._BASE, "--pad_val": "1"}
        with pytest.warns(UserWarning, match="pad_val"):
            ConvParams.from_args(args)

    def test_no_warning_on_defaults(self) -> None:
        import warnings as w

        with w.catch_warnings():
            w.simplefilter("error")
            ConvParams.from_args(dict(self._BASE))


class TestBnormUnsupportedWarnings:
    """Verify warnings for MIOpen bnorm features not representable in hipDNN."""

    _BASE = {"-n": "2", "-c": "32", "-H": "8", "-W": "8"}

    def test_fused_activation_warns(self) -> None:
        args = {**self._BASE, "-f": "1"}
        with pytest.warns(UserWarning, match="Fused activation"):
            build_bnorm_json("bnorm", args)

    def test_non_default_alpha_warns(self) -> None:
        args = {**self._BASE, "-A": "0.5"}
        with pytest.warns(UserWarning, match="alpha.*beta"):
            build_bnorm_json("bnorm", args)

    def test_non_default_beta_warns(self) -> None:
        args = {**self._BASE, "-B": "1.0"}
        with pytest.warns(UserWarning, match="alpha.*beta"):
            build_bnorm_json("bnorm", args)

    def test_save_mode_warns(self) -> None:
        args = {**self._BASE, "-s": "1"}
        with pytest.warns(UserWarning, match="Save mode"):
            build_bnorm_json("bnorm", args)

    def test_run_mode_warns(self) -> None:
        args = {**self._BASE, "-r": "1"}
        with pytest.warns(UserWarning, match="Run mode"):
            build_bnorm_json("bnorm", args)

    def test_inverse_variance_flag_warns(self) -> None:
        args = {**self._BASE, "-I": "1"}
        with pytest.warns(UserWarning, match="Inverse variance"):
            build_bnorm_json("bnorm", args)

    def test_save_mode_long_flag_warns(self) -> None:
        args = {**self._BASE, "--save": "1"}
        with pytest.warns(UserWarning, match="Save mode"):
            build_bnorm_json("bnorm", args)

    def test_run_mode_long_flag_warns(self) -> None:
        args = {**self._BASE, "--run": "1"}
        with pytest.warns(UserWarning, match="Run mode"):
            build_bnorm_json("bnorm", args)

    def test_inverse_variance_long_flag_warns(self) -> None:
        args = {**self._BASE, "--inverse_variance": "1"}
        with pytest.warns(UserWarning, match="Inverse variance"):
            build_bnorm_json("bnorm", args)

    def test_no_warning_on_defaults(self) -> None:
        import warnings as w

        with w.catch_warnings():
            w.simplefilter("error")
            build_bnorm_json("bnorm", dict(self._BASE))
