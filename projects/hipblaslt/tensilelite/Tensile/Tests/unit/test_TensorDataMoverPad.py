################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
import pytest

from Tensile.Components.TensorDataMover import TensorDataMoverLoad

pytestmark = pytest.mark.unit


class TestTensorDataMoverPadHelpers:
    """Unit tests for TDM pad / iterate-mode selection helpers."""

    def test_cal_pad_interval_1024(self) -> None:
        """LBSPP=1024 encodes as pad_interval=7 (the hardware maximum).

        Returns:
            None
        """
        assert TensorDataMoverLoad.calPadInterval(1024) == 7

    def test_needs_iterate_du128_vwb8(self) -> None:
        """DU=128 BF16 VWB=8 exceeds the non-iterate pad cap.

        Returns:
            None
        """
        assert TensorDataMoverLoad.needsIterateModeForPad(2048, 128, 2.0) is True

    def test_needs_iterate_du128_vwb4_halved(self) -> None:
        """Halved VWB=4 at DU=128 must use iterate when full VW=8 LBSPP exceeds cap.

        Returns:
            None
        """
        assert TensorDataMoverLoad.needsIterateModeForPad(1024, 128, 2.0) is True

    def test_no_iterate_du64_vwb8(self) -> None:
        """DU=64 BF16 VWB=8 stays on the non-iterate path.

        Returns:
            None
        """
        assert TensorDataMoverLoad.needsIterateModeForPad(1024, 64, 2.0) is False

    def test_no_iterate_du64_vwb4_when_full_vw_fits(self) -> None:
        """VWB=4 at DU=64 is safe when full VW=8 LBSPP still fits the 1024 B cap.

        Returns:
            None
        """
        assert TensorDataMoverLoad.needsIterateModeForPad(512, 64, 2.0) is False

    def test_no_iterate_lbspp_zero(self) -> None:
        """LBSPP=0 disables padding; iterate mode must not be forced.

        Returns:
            None
        """
        assert TensorDataMoverLoad.needsIterateModeForPad(0, 64, 2.0) is False

    def test_needs_iterate_fp4_non_integer_row_bytes(self) -> None:
        """fp4 (bpe=0.5) with non-integer DepthU*bpe must not under-estimate VW=8 LBSPP.

        Returns:
            None
        """
        assert TensorDataMoverLoad.needsIterateModeForPad(1024, 257, 0.5) is True

    def test_needs_iterate_bf6_halved_path(self) -> None:
        """bf6/fp6 (bpe=0.75) halved-VW path must detect full VW=8 LBSPP over cap.

        Returns:
            None
        """
        assert TensorDataMoverLoad.needsIterateModeForPad(768, 171, 0.75) is True

    def test_no_iterate_fp4_at_cap(self) -> None:
        """fp4 at exactly 1024 B VW=8 LBSPP stays non-iterate.

        Returns:
            None
        """
        assert TensorDataMoverLoad.needsIterateModeForPad(1024, 256, 0.5) is False
