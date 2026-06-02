# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for operation attribute accessors and aliases (no GPU required)."""

import hipdnn_frontend as hipdnn


class TestConvAttributeAliases:
    """The short Conv*Attributes names alias the long Convolution* classes."""

    def test_fprop_alias_identity(self):
        """ConvFpropAttributes is the same class as ConvolutionFpropAttributes."""
        assert hipdnn.ConvFpropAttributes is hipdnn.ConvolutionFpropAttributes

    def test_dgrad_alias_identity(self):
        """ConvDgradAttributes is the same class as ConvolutionDgradAttributes."""
        assert hipdnn.ConvDgradAttributes is hipdnn.ConvolutionDgradAttributes

    def test_wgrad_alias_identity(self):
        """ConvWgradAttributes is the same class as ConvolutionWgradAttributes."""
        assert hipdnn.ConvWgradAttributes is hipdnn.ConvolutionWgradAttributes


class TestConvAttributeChaining:
    """Convolution attribute setters chain and round-trip the name."""

    def test_fprop_setters_chain(self):
        """ConvFprop setters return self for chaining and store the name."""
        attrs = hipdnn.ConvFpropAttributes()
        result = (
            attrs.set_name("conv")
            .set_padding([1, 1])
            .set_stride([2, 2])
            .set_dilation([1, 1])
        )
        assert result is attrs
        assert attrs.get_name() == "conv"

    def test_dgrad_pre_post_padding_chain(self):
        """ConvDgrad pre/post padding setters chain and store the name."""
        attrs = hipdnn.ConvDgradAttributes()
        result = (
            attrs.set_name("dgrad")
            .set_pre_padding([1, 1])
            .set_post_padding([1, 1])
            .set_stride([1, 1])
            .set_dilation([1, 1])
        )
        assert result is attrs
        assert attrs.get_name() == "dgrad"

    def test_wgrad_pre_post_padding_chain(self):
        """ConvWgrad pre/post padding setters chain and store the name."""
        attrs = hipdnn.ConvWgradAttributes()
        result = (
            attrs.set_name("wgrad")
            .set_pre_padding([1, 1])
            .set_post_padding([1, 1])
            .set_stride([1, 1])
            .set_dilation([1, 1])
        )
        assert result is attrs
        assert attrs.get_name() == "wgrad"


class TestPointwiseAttributes:
    """Round-trip tests for PointwiseAttributes accessors."""

    def test_name_round_trip(self):
        """set_name()/get_name() round-trip."""
        attrs = hipdnn.PointwiseAttributes()
        attrs.set_name("relu")
        assert attrs.get_name() == "relu"

    def test_mode_round_trip(self):
        """set_mode()/get_mode() round-trip."""
        attrs = hipdnn.PointwiseAttributes()
        attrs.set_mode(hipdnn.PointwiseMode.RELU_FWD)
        assert attrs.get_mode() == hipdnn.PointwiseMode.RELU_FWD


class TestMatmulAttributes:
    """Round-trip tests for MatmulAttributes accessors."""

    def test_name_round_trip(self):
        """set_name()/get_name() round-trip."""
        attrs = hipdnn.MatmulAttributes()
        attrs.set_name("matmul")
        assert attrs.get_name() == "matmul"

    def test_compute_data_type_round_trip(self):
        """set_compute_data_type()/get_compute_data_type() round-trip."""
        attrs = hipdnn.MatmulAttributes()
        attrs.set_compute_data_type(hipdnn.DataType.FLOAT)
        assert attrs.get_compute_data_type() == hipdnn.DataType.FLOAT
