// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "ActivationCommon.hpp"
#include "ConvolutionCommon.hpp"

namespace test_fusion_common
{

struct ConvBiasActivTestCase
{
    test_conv_common::ConvTestCase conv;
    bool doBias;
    test_activation_common::ActivTestCase activ;

    ConvBiasActivTestCase(test_conv_common::ConvTestCase&& convLocal,
                          bool doBiasLocal,
                          test_activation_common::ActivTestCase&& activLocal)
        : conv(std::move(convLocal))
        , doBias(doBiasLocal)
        , activ(activLocal)
    {
    }

    friend std::ostream& operator<<(std::ostream& ss, const ConvBiasActivTestCase& tc)
    {
        using namespace hipdnn_sdk::utilities;

        ss << "(conv:" << tc.conv;
        ss << " doBias:" << tc.doBias;
        ss << " activ:" << tc.activ;
        ss << ")";

        return ss;
    }
};

} // namespace test_fusion_common
