// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// A deterministic problem has its own text perf-db record. A config tuned in deterministic mode
// (split-K = 1) is valid but slow in default mode, so a shared record lets the last mode tuned
// slow down the other.

#include <gtest/gtest.h>

#include "../tensor_holder.hpp"

#include <miopen/conv/problem_description.hpp>

#include <sstream>
#include <string>

namespace {

using miopen::conv::Direction;
using miopen::conv::ProblemDescription;

// The key the text perf-db stores the problem's record under (DbRecord::SerializeKey).
std::string PerfDbKey(const ProblemDescription& problem)
{
    std::ostringstream key;
    ProblemDescription::VisitAll(problem, [&](auto&& value, auto&&) {
        if(key.tellp() != 0)
            key << "x";
        key << value;
    });
    return key.str();
}

} // namespace

using CPU_ConvDeterministicPerfDbKey_NONE = ::testing::TestWithParam<int>;

TEST_P(CPU_ConvDeterministicPerfDbKey_NONE, DeterministicProblemHasItsOwnRecord)
{
    // The variable overrides the attribute of both descriptors. Then the test has one mode only.
    if(MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC)
        GTEST_SKIP() << "MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC is set";

    const auto conv_default = miopen::ConvolutionDescriptor{
        2, miopenConvolution, miopenPaddingDefault, {1, 1}, {1, 1}, {1, 1}};
    auto conv_det = conv_default;
    conv_det.attribute.Set(MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC, 1);

    const tensor<float> x{4, 8, 16, 16};
    const tensor<float> w{16, 8, 3, 3};
    const tensor<float> y{conv_default.GetForwardOutputTensor(x.desc, w.desc)};
    const auto problem_default =
        ProblemDescription{y.desc, w.desc, x.desc, conv_default, Direction::BackwardWeights};
    const auto problem_det =
        ProblemDescription{y.desc, w.desc, x.desc, conv_det, Direction::BackwardWeights};

    // Default-mode keys are unchanged, so existing user and system records stay valid.
    EXPECT_EQ(PerfDbKey(problem_default),
              "2x8x16x16x1x3x3x1x16x4x1x1x0x1x1x0x1x1x0x0x1xNCHWxFP32xW");
    EXPECT_EQ(PerfDbKey(problem_det), PerfDbKey(problem_default) + "xDET");
    // The sqlite system perf-db reads only Visit: its columns are the same in both modes.
    EXPECT_EQ(problem_det.FieldNames(), problem_default.FieldNames());
}

INSTANTIATE_TEST_SUITE_P(Quick, CPU_ConvDeterministicPerfDbKey_NONE, ::testing::Values(0));
