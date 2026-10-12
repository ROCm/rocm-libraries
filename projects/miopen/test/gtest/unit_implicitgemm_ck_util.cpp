/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <miopen/conv/problem_description.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/convolution.hpp>
#include <miopen/tensor.hpp>
#include "../../src/ck_impl/implicitgemm_ck_util.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace unit_implicitgemm_ck_util_test {
struct ParsingTestCase
{
    std::string instanceToCheck;
    std::string suffix;
    bool expectedSupported;
    bool checkSplitK;

    friend std::ostream& operator<<(std::ostream& os, const ParsingTestCase& tc)
    {
        return os << "(instanceToCheck: " << tc.instanceToCheck << " suffix: " << tc.suffix
                  << " expectedSupported: " << tc.expectedSupported
                  << " checkSplitK: " << tc.checkSplitK << ")";
    }
};

static std::vector<ParsingTestCase> GetTestCases()
{
    return {{"test0", "+5", true, true},
            {"test1", "+", false, true},
            {"test2", "", false, true},
            {"test3", "+2+3", false, true},
            {"test4", "+9999999999999999999999999", false, true},
            {"test5", "", true, false},
            {"test6", "+", false, false},
            {"test7", "+2", false, false},
            {"test8", "+2+3", false, false}};
}

using ProblemDescription = miopen::conv::ProblemDescription;

struct StubbedDeviceOp
{
    StubbedDeviceOp(const std::string& op) : typeString(op) {}

    std::string typeString;

    std::string GetTypeString() { return typeString; }
};

struct StubbedDeviceOps
{
    static std::vector<std::string> deviceOps;

    static std::vector<std::unique_ptr<StubbedDeviceOp>> GetInstances()
    {
        std::vector<std::unique_ptr<StubbedDeviceOp>> ops;
        ops.reserve(deviceOps.size());

        std::transform(deviceOps.begin(),
                       deviceOps.end(),
                       std::back_inserter(ops),
                       [&](auto& deviceOp) { return std::make_unique<StubbedDeviceOp>(deviceOp); });

        return ops;
    }
};

std::vector<std::string> StubbedDeviceOps::deviceOps = {};

struct StubbedCKArgs
{
    StubbedCKArgs(const ProblemDescription& /*problem*/) {}

    template <typename ConvPtr>
    bool IsSupportedBy(const ConvPtr&) const
    {
        return true;
    }

    template <typename ConvPtr>
    bool IsSupportedBySplitK(const ConvPtr&, int) const
    {
        return true;
    }
};

template <typename CKArgsType, typename DeviceOpType>
class CKArgParsingTest : public ::testing::TestWithParam<ParsingTestCase>
{
protected:
    void TestParsing()
    {
        auto testCase = GetParam();

        // Set up stubbed instances to match test
        DeviceOpType::deviceOps.clear();
        DeviceOpType::deviceOps.push_back(testCase.instanceToCheck);

        bool success;
        if(testCase.checkSplitK)
        {
            success = miopen::solver::
                IsCKArgsSupported<DeviceOpType, CKArgsType, ProblemDescription, true>(
                    ProblemDescription{}, testCase.instanceToCheck + testCase.suffix);
        }
        else
        {
            success = miopen::solver::IsCKArgsSupported<DeviceOpType, CKArgsType>(
                ProblemDescription{}, testCase.instanceToCheck + testCase.suffix);
        }
        EXPECT_EQ(success, testCase.expectedSupported);
    }
};
} // namespace unit_implicitgemm_ck_util_test

using namespace unit_implicitgemm_ck_util_test;

struct CPU_UnitTestImplicitGemmCKUtil_NONE : CKArgParsingTest<StubbedCKArgs, StubbedDeviceOps>
{
};

TEST_P(CPU_UnitTestImplicitGemmCKUtil_NONE, TestParsing)
{
#if MIOPEN_USE_COMPOSABLEKERNEL
    this->TestParsing();
#else
    GTEST_SKIP();
#endif
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestImplicitGemmCKUtil_NONE,
                         testing::ValuesIn(GetTestCases()));

// ---------------------------------------------------------------------------
// Regression: RequiresLargeTensorCKInstance must force a large-tensor (int64) CK
// instance for a grouped backward-weights problem whose flattened element count
// exceeds INT_MAX even though every individual length/stride still fits int32.
// Without the element-count gate the loader keeps such a problem eligible for a
// fast int32-indexing instance whose flat element offset overflows, silently
// returning wrong results. The gate is scoped to backward-weights: forward and
// backward-data verify correctly on the int32 instances. Shape is synthetic.
// ---------------------------------------------------------------------------
namespace {

miopen::conv::ProblemDescription MakeElemCountConvProblem(miopen::conv::Direction dir,
                                                          std::size_t n,
                                                          std::size_t c,
                                                          std::size_t h,
                                                          std::size_t w,
                                                          std::size_t k)
{
    const miopen::TensorDescriptor x{miopenHalf, miopenTensorNHWC, {n, c, h, w}};
    const miopen::TensorDescriptor weights{miopenHalf, miopenTensorNHWC, {k, c, 3, 3}};
    const miopen::ConvolutionDescriptor conv{2,
                                             miopenConvolution,
                                             miopenPaddingDefault,
                                             {1, 1}, // pads (SAME for 3x3)
                                             {1, 1}, // strides
                                             {1, 1}, // dilations
                                             {0, 0}, // trans output pads
                                             1,      // group count
                                             1.0f};  // lowp quant
    const miopen::TensorDescriptor y = conv.GetForwardOutputTensor(x, weights, miopenHalf);

    // ProblemDescription ctor: (in = x for fwd / y for backward*, weights,
    //                           out = y for fwd / x for backward*, conv, direction)
    if(dir == miopen::conv::Direction::Forward)
        return {x, weights, y, conv, dir};
    return {y, weights, x, conv, dir};
}

// 3D (NDHWC) sibling of MakeElemCountConvProblem. The element-count gate is
// rank-agnostic, so it must fire for 3D grouped wrw (ConvHipImplicitGemm3DGroupWrwXdlops)
// on the same >INT_MAX-element-extent condition as the 2D case.
miopen::conv::ProblemDescription MakeElemCount3DConvProblem(miopen::conv::Direction dir,
                                                            std::size_t n,
                                                            std::size_t c,
                                                            std::size_t d,
                                                            std::size_t h,
                                                            std::size_t w,
                                                            std::size_t k)
{
    const miopen::TensorDescriptor x{miopenHalf, miopenTensorNDHWC, {n, c, d, h, w}};
    const miopen::TensorDescriptor weights{miopenHalf, miopenTensorNDHWC, {k, c, 3, 3, 3}};
    const miopen::ConvolutionDescriptor conv{3,
                                             miopenConvolution,
                                             miopenPaddingDefault,
                                             {1, 1, 1}, // pads (SAME for 3x3x3)
                                             {1, 1, 1}, // strides
                                             {1, 1, 1}, // dilations
                                             {0, 0, 0}, // trans output pads
                                             1,         // group count
                                             1.0f};     // lowp quant
    const miopen::TensorDescriptor y = conv.GetForwardOutputTensor(x, weights, miopenHalf);

    if(dir == miopen::conv::Direction::Forward)
        return {x, weights, y, conv, dir};
    return {y, weights, x, conv, dir};
}

} // namespace

TEST(CPU_UnitTestImplicitGemmCKUtilLargeTensor_NONE, WrwElementCountOverflowGate)
{
    using miopen::conv::Direction;
    using miopen::solver::RequiresLargeTensorCKInstance;

    // N*C*H*W = 256*1024*128*128 = 4.29e9 > INT_MAX, yet every length and stride
    // (max stride = H*W*C = 16.78M) fits int32, so AllTensorsDimsFitIntoInt() is true
    // and only the element-count gate can flag it.
    constexpr std::size_t n = 256, c = 1024, hw = 128, k = 1024;

    EXPECT_TRUE(RequiresLargeTensorCKInstance(
        MakeElemCountConvProblem(Direction::BackwardWeights, n, c, hw, hw, k)))
        << "grouped wrw with >INT_MAX element count must require a large-tensor instance";

    EXPECT_FALSE(RequiresLargeTensorCKInstance(
        MakeElemCountConvProblem(Direction::Forward, n, c, hw, hw, k)))
        << "forward must stay on the (faster) int32 instances";

    EXPECT_FALSE(RequiresLargeTensorCKInstance(
        MakeElemCountConvProblem(Direction::BackwardData, n, c, hw, hw, k)))
        << "backward-data must stay on the (faster) int32 instances";

    // A small wrw problem (element count well within int32) stays on the int32 path.
    EXPECT_FALSE(RequiresLargeTensorCKInstance(
        MakeElemCountConvProblem(Direction::BackwardWeights, 1, 64, 32, 32, 64)));
}

TEST(CPU_UnitTestImplicitGemmCKUtilLargeTensor_NONE, Wrw3DElementCountOverflowGate)
{
    using miopen::conv::Direction;
    using miopen::solver::RequiresLargeTensorCKInstance;

    // N*C*D*H*W = 256*512*32*32*32 = 4.29e9 > INT_MAX, yet every length and stride
    // (max stride = C*D*H*W = 16.78M) fits int32, so AllTensorsDimsFitIntoInt() is true
    // and only the element-count gate can flag it. Mirrors WrwElementCountOverflowGate
    // for the 3D (NDHWC) grouped wrw path.
    constexpr std::size_t n = 256, c = 512, d = 32, hw = 32, k = 512;

    EXPECT_TRUE(RequiresLargeTensorCKInstance(
        MakeElemCount3DConvProblem(Direction::BackwardWeights, n, c, d, hw, hw, k)))
        << "3D grouped wrw with >INT_MAX element count must require a large-tensor instance";

    EXPECT_FALSE(RequiresLargeTensorCKInstance(
        MakeElemCount3DConvProblem(Direction::Forward, n, c, d, hw, hw, k)))
        << "3D forward must stay on the (faster) int32 instances";

    EXPECT_FALSE(RequiresLargeTensorCKInstance(
        MakeElemCount3DConvProblem(Direction::BackwardData, n, c, d, hw, hw, k)))
        << "3D backward-data must stay on the (faster) int32 instances";

    // A small 3D wrw problem (element count well within int32) stays on the int32 path.
    EXPECT_FALSE(RequiresLargeTensorCKInstance(
        MakeElemCount3DConvProblem(Direction::BackwardWeights, 1, 64, 8, 16, 16, 64)));
}
