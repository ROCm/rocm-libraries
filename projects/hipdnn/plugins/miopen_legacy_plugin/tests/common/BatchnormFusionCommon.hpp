// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "ActivationCommon.hpp"
#include "BatchnormCommon.hpp"

#include <hipdnn_sdk/test_utilities/Seeds.hpp>

namespace test_bn_fusion_common
{

struct BnActivTestCase
{
    test_bn_common::BatchnormTestCase bn;
    test_activation_common::ActivTestCase activ;

    BnActivTestCase(test_bn_common::BatchnormTestCase bnLocal,
                    test_activation_common::ActivTestCase activLocal)
        : bn(std::move(bnLocal))
        , activ(activLocal)
    {
    }

    friend std::ostream& operator<<(std::ostream& ss, const BnActivTestCase& tc)
    {
        ss << "(bn:" << tc.bn;
        ss << " activ:" << tc.activ;
        ss << ")";
        return ss;
    }
};

inline std::vector<BnActivTestCase> generateBnActivTestCases(
    const std::vector<test_bn_common::BatchnormTestCase>& bnTestCases,
    const std::vector<test_activation_common::ActivTestCase>& activTestCases)
{
    std::vector<BnActivTestCase> result;
    result.reserve(bnTestCases.size() * activTestCases.size());

    for(const auto& bnCase : bnTestCases)
    {
        for(const auto& activCase : activTestCases)
        {
            result.emplace_back(bnCase, activCase);
        }
    }

    return result;
}

inline std::vector<BnActivTestCase>
    generateBnActivTestCases(const std::vector<test_bn_common::BatchnormTestCase>& bnTestCases,
                             const test_activation_common::ActivTestCase& activTestCase)
{
    std::vector<BnActivTestCase> result;
    result.reserve(bnTestCases.size());

    for(const auto& bnCase : bnTestCases)
    {
        result.emplace_back(bnCase, activTestCase);
    }

    return result;
}

inline std::vector<BnActivTestCase> getBnActivBwdTestCases()
{
    return generateBnActivTestCases(test_bn_common::getBnBwdTestCases(),
                                    test_activation_common::createBwdActivationTestCases());
}

inline std::vector<BnActivTestCase> getBnActiv3dBwdTestCases()
{
    return generateBnActivTestCases(test_bn_common::getBnBwd3dTestCases(),
                                    test_activation_common::createBwdActivationTestCases());
}

} // namespace test_bn_fusion_common
