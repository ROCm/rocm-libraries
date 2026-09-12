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

// =============================================================================
// Tests for IsCKSplitKSupportedGeneric
// =============================================================================
namespace unit_implicitgemm_ck_util_test {

/**
 * @brief Test case structure for IsCKSplitKSupportedGeneric
 */
struct SplitKGenericTestCase
{
    std::string kernelId;           // Kernel ID to check
    int splitK;                     // split_k value to validate
    bool expectedResult;            // Expected result from IsCKSplitKSupportedGeneric
    std::set<int> supportedSplitKs; // Set of split_k values this kernel supports

    friend std::ostream& operator<<(std::ostream& os, const SplitKGenericTestCase& tc)
    {
        os << "(kernelId: " << tc.kernelId << ", splitK: " << tc.splitK
           << ", expectedResult: " << std::boolalpha << tc.expectedResult << ")";
        return os;
    }
};

/**
 * @brief Stubbed CK Args that supports selective split_k values
 *
 * This allows us to test the IsCKSplitKSupportedGeneric validation logic
 * by controlling which (kernel, split_k) combinations are "supported".
 */
struct StubbedCKArgsWithSplitKValidation
{
    // Static storage for supported split_k values per kernel
    static std::map<std::string, std::set<int>> supportedSplitKByKernel;

    StubbedCKArgsWithSplitKValidation(const ProblemDescription& /*problem*/) {}

    template <typename ConvPtr>
    bool IsSupportedBy(const ConvPtr&) const
    {
        return true;
    }

    template <typename ConvPtr>
    bool IsSupportedBySplitK(const ConvPtr& conv_ptr, int split_k) const
    {
        const std::string kernel_id = conv_ptr->GetTypeString();
        auto it                     = supportedSplitKByKernel.find(kernel_id);
        if(it == supportedSplitKByKernel.end())
        {
            return false; // Kernel not found
        }
        return it->second.count(split_k) > 0; // Check if split_k is supported
    }
};

std::map<std::string, std::set<int>> StubbedCKArgsWithSplitKValidation::supportedSplitKByKernel =
    {};

/**
 * @brief Test cases for IsCKSplitKSupportedGeneric
 */
static std::vector<SplitKGenericTestCase> GetSplitKGenericTestCases()
{
    return {
        // Valid kernel with supported split_k values
        {"DeviceGroupedConvBwdDataMultipleD_Test", 1, true, {1, 2, 4, 8}},
        {"DeviceGroupedConvBwdDataMultipleD_Test", 2, true, {1, 2, 4, 8}},
        {"DeviceGroupedConvBwdDataMultipleD_Test", 4, true, {1, 2, 4, 8}},
        {"DeviceGroupedConvBwdDataMultipleD_Test", 8, true, {1, 2, 4, 8}},

        // Valid kernel with unsupported split_k values
        {"DeviceGroupedConvBwdDataMultipleD_Test", 16, false, {1, 2, 4, 8}},
        {"DeviceGroupedConvBwdDataMultipleD_Test", 32, false, {1, 2, 4, 8}},
        {"DeviceGroupedConvBwdDataMultipleD_Test", 128, false, {1, 2, 4, 8}},

        // Kernel that only supports split_k=1 (no split_k support)
        {"DeviceGroupedConvNoSplitK_Test", 1, true, {1}},
        {"DeviceGroupedConvNoSplitK_Test", 2, false, {1}},
        {"DeviceGroupedConvNoSplitK_Test", 4, false, {1}},

        // Non-existent kernel
        {"NonExistentKernel", 1, false, {}},
        {"NonExistentKernel", 4, false, {}},

        // Edge cases - boundary split_k values
        {"DeviceGroupedConvWrw_Test", 1, true, {1, 2, 4, 8, 16, 32, 64, 128}},
        {"DeviceGroupedConvWrw_Test", 128, true, {1, 2, 4, 8, 16, 32, 64, 128}},
        {"DeviceGroupedConvWrw_Test", 256, false, {1, 2, 4, 8, 16, 32, 64, 128}},
    };
}

/**
 * @brief Parameterized test fixture for IsCKSplitKSupportedGeneric
 */
class CPU_SplitKGenericTest_NONE : public ::testing::TestWithParam<SplitKGenericTestCase>
{
protected:
    void SetUp() override
    {
        // Clear state before each test
        StubbedDeviceOps::deviceOps.clear();
        StubbedCKArgsWithSplitKValidation::supportedSplitKByKernel.clear();
    }

    void TearDown() override
    {
        // Clean up after each test
        StubbedDeviceOps::deviceOps.clear();
        StubbedCKArgsWithSplitKValidation::supportedSplitKByKernel.clear();
    }
};

// cppcheck-suppress syntaxError
TEST_P(CPU_SplitKGenericTest_NONE, ValidatesSplitKSupport)
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    auto testCase = GetParam();

    // Set up the stubbed device ops to include the kernel we're testing
    if(!testCase.supportedSplitKs.empty())
    {
        StubbedDeviceOps::deviceOps.push_back(testCase.kernelId);
        StubbedCKArgsWithSplitKValidation::supportedSplitKByKernel[testCase.kernelId] =
            testCase.supportedSplitKs;
    }

    // Test IsCKSplitKSupported (single data type version)
    bool result =
        miopen::solver::IsCKSplitKSupported<StubbedDeviceOps, StubbedCKArgsWithSplitKValidation>(
            ProblemDescription{}, testCase.kernelId, testCase.splitK);

    EXPECT_EQ(result, testCase.expectedResult)
        << "Failed for kernel: " << testCase.kernelId << " split_k: " << testCase.splitK;
#else
    GTEST_SKIP();
#endif
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_SplitKGenericTest_NONE,
                         testing::ValuesIn(GetSplitKGenericTestCases()));

/**
 * @brief Direct unit tests for edge cases and specific scenarios
 */
class CPU_SplitKGenericDirectTest_NONE : public ::testing::Test
{
protected:
    void SetUp() override
    {
        StubbedDeviceOps::deviceOps.clear();
        StubbedCKArgsWithSplitKValidation::supportedSplitKByKernel.clear();
    }

    void TearDown() override
    {
        StubbedDeviceOps::deviceOps.clear();
        StubbedCKArgsWithSplitKValidation::supportedSplitKByKernel.clear();
    }
};

TEST_F(CPU_SplitKGenericDirectTest_NONE, EmptyKernelIdReturnsNotSupported)
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    StubbedDeviceOps::deviceOps.push_back("SomeKernel");
    StubbedCKArgsWithSplitKValidation::supportedSplitKByKernel["SomeKernel"] = {1, 2, 4};

    bool result =
        miopen::solver::IsCKSplitKSupported<StubbedDeviceOps, StubbedCKArgsWithSplitKValidation>(
            ProblemDescription{}, "", 1);

    EXPECT_FALSE(result) << "Empty kernel_id should return false";
#else
    GTEST_SKIP();
#endif
}

TEST_F(CPU_SplitKGenericDirectTest_NONE, ZeroSplitKValueHandledCorrectly)
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    const std::string kernelId = "TestKernel";
    StubbedDeviceOps::deviceOps.push_back(kernelId);
    // Only support positive split_k values
    StubbedCKArgsWithSplitKValidation::supportedSplitKByKernel[kernelId] = {1, 2, 4};

    bool result =
        miopen::solver::IsCKSplitKSupported<StubbedDeviceOps, StubbedCKArgsWithSplitKValidation>(
            ProblemDescription{}, kernelId, 0);

    // split_k=0 is not in the supported set, so should return false
    EXPECT_FALSE(result) << "split_k=0 should not be supported";
#else
    GTEST_SKIP();
#endif
}

TEST_F(CPU_SplitKGenericDirectTest_NONE, NegativeSplitKValueHandledCorrectly)
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    const std::string kernelId = "TestKernel";
    StubbedDeviceOps::deviceOps.push_back(kernelId);
    StubbedCKArgsWithSplitKValidation::supportedSplitKByKernel[kernelId] = {1, 2, 4};

    bool result =
        miopen::solver::IsCKSplitKSupported<StubbedDeviceOps, StubbedCKArgsWithSplitKValidation>(
            ProblemDescription{}, kernelId, -1);

    // split_k=-1 is not in the supported set, so should return false
    EXPECT_FALSE(result) << "Negative split_k should not be supported";
#else
    GTEST_SKIP();
#endif
}

TEST_F(CPU_SplitKGenericDirectTest_NONE, MultipleKernelsWithDifferentSplitKSupport)
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    const std::string kernel1 = "KernelA";
    const std::string kernel2 = "KernelB";

    StubbedDeviceOps::deviceOps.push_back(kernel1);
    StubbedDeviceOps::deviceOps.push_back(kernel2);

    // Kernel1 supports split_k = 1, 2, 4
    StubbedCKArgsWithSplitKValidation::supportedSplitKByKernel[kernel1] = {1, 2, 4};
    // Kernel2 supports split_k = 1, 8, 16
    StubbedCKArgsWithSplitKValidation::supportedSplitKByKernel[kernel2] = {1, 8, 16};

    // Test kernel1 with its supported values
    // Note: Use parentheses around function call to prevent macro from misinterpreting commas
    bool k1_s2 =
        miopen::solver::IsCKSplitKSupported<StubbedDeviceOps, StubbedCKArgsWithSplitKValidation>(
            ProblemDescription{}, kernel1, 2);
    EXPECT_TRUE(k1_s2);

    bool k1_s8 =
        miopen::solver::IsCKSplitKSupported<StubbedDeviceOps, StubbedCKArgsWithSplitKValidation>(
            ProblemDescription{}, kernel1, 8);
    EXPECT_FALSE(k1_s8);

    // Test kernel2 with its supported values
    bool k2_s8 =
        miopen::solver::IsCKSplitKSupported<StubbedDeviceOps, StubbedCKArgsWithSplitKValidation>(
            ProblemDescription{}, kernel2, 8);
    EXPECT_TRUE(k2_s8);

    bool k2_s4 =
        miopen::solver::IsCKSplitKSupported<StubbedDeviceOps, StubbedCKArgsWithSplitKValidation>(
            ProblemDescription{}, kernel2, 4);
    EXPECT_FALSE(k2_s4);
#else
    GTEST_SKIP();
#endif
}

} // namespace unit_implicitgemm_ck_util_test

// ---------------------------------------------------------------------------
// Regression: the gate keys on byte span, not element count, and applies to every
// direction (previously backward-weights only). Shapes below are chosen so every
// individual length/stride still fits int32 -- only the byte-span check can flag them.
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

// dtype-parameterized sibling of MakeElemCountConvProblem: fp32's larger byte-per-element
// ratio lets a shape cross the byte-span boundary while its element count stays well
// under INT_MAX, isolating the byte-span check from the element-count one.
miopen::conv::ProblemDescription MakeByteSpanConvProblem(miopenDataType_t dtype,
                                                         miopen::conv::Direction dir,
                                                         std::size_t n,
                                                         std::size_t c,
                                                         std::size_t h,
                                                         std::size_t w,
                                                         std::size_t k)
{
    const miopen::TensorDescriptor x{dtype, miopenTensorNHWC, {n, c, h, w}};
    const miopen::TensorDescriptor weights{dtype, miopenTensorNHWC, {k, c, 3, 3}};
    const miopen::ConvolutionDescriptor conv{2,
                                             miopenConvolution,
                                             miopenPaddingDefault,
                                             {1, 1}, // pads (SAME for 3x3)
                                             {1, 1}, // strides
                                             {1, 1}, // dilations
                                             {0, 0}, // trans output pads
                                             1,      // group count
                                             1.0f};  // lowp quant
    const miopen::TensorDescriptor y = conv.GetForwardOutputTensor(x, weights, dtype);

    if(dir == miopen::conv::Direction::Forward)
        return {x, weights, y, conv, dir};
    return {y, weights, x, conv, dir};
}

} // namespace

TEST(CPU_UnitTestImplicitGemmCKUtilLargeTensor_NONE, WrwElementCountOverflowGate)
{
    using miopen::conv::Direction;
    using miopen::solver::RequiresLargeTensorCKInstance;

    // N*C*H*W = 256*1024*128*128 = 4.29e9 elements; fp16 byte span ~8.58 GB, over INT_MAX
    // bytes despite every stride (max = H*W*C = 16.78M) fitting int32.
    constexpr std::size_t n = 256, c = 1024, hw = 128, k = 1024;

    EXPECT_TRUE(RequiresLargeTensorCKInstance(
        MakeElemCountConvProblem(Direction::BackwardWeights, n, c, hw, hw, k)))
        << "grouped wrw with >INT_MAX byte span must require a large-tensor instance";

    EXPECT_TRUE(RequiresLargeTensorCKInstance(
        MakeElemCountConvProblem(Direction::Forward, n, c, hw, hw, k)))
        << "forward with >INT_MAX byte span must require a large-tensor instance";

    EXPECT_TRUE(RequiresLargeTensorCKInstance(
        MakeElemCountConvProblem(Direction::BackwardData, n, c, hw, hw, k)))
        << "backward-data with >INT_MAX byte span must require a large-tensor instance";

    // A small wrw problem (element count well within int32) stays on the int32 path.
    EXPECT_FALSE(RequiresLargeTensorCKInstance(
        MakeElemCountConvProblem(Direction::BackwardWeights, 1, 64, 32, 32, 64)));
}

TEST(CPU_UnitTestImplicitGemmCKUtilLargeTensor_NONE, Wrw3DElementCountOverflowGate)
{
    using miopen::conv::Direction;
    using miopen::solver::RequiresLargeTensorCKInstance;

    // Mirrors WrwElementCountOverflowGate for the 3D (NDHWC) path: N*C*D*H*W =
    // 256*512*32*32*32 = 4.29e9 elements, fp16 byte span ~8.58 GB, over INT_MAX bytes
    // despite every stride (max = C*D*H*W = 16.78M) fitting int32.
    constexpr std::size_t n = 256, c = 512, d = 32, hw = 32, k = 512;

    EXPECT_TRUE(RequiresLargeTensorCKInstance(
        MakeElemCount3DConvProblem(Direction::BackwardWeights, n, c, d, hw, hw, k)))
        << "3D grouped wrw with >INT_MAX byte span must require a large-tensor instance";

    EXPECT_TRUE(RequiresLargeTensorCKInstance(
        MakeElemCount3DConvProblem(Direction::Forward, n, c, d, hw, hw, k)))
        << "3D forward with >INT_MAX byte span must require a large-tensor instance";

    EXPECT_TRUE(RequiresLargeTensorCKInstance(
        MakeElemCount3DConvProblem(Direction::BackwardData, n, c, d, hw, hw, k)))
        << "3D backward-data with >INT_MAX byte span must require a large-tensor instance";

    // A small 3D wrw problem (element count well within int32) stays on the int32 path.
    EXPECT_FALSE(RequiresLargeTensorCKInstance(
        MakeElemCount3DConvProblem(Direction::BackwardWeights, 1, 64, 8, 16, 16, 64)));
}

TEST(CPU_UnitTestImplicitGemmCKUtilLargeTensor_NONE, ByteSpanOverflowGate)
{
    using miopen::conv::Direction;
    using miopen::solver::RequiresLargeTensorCKInstance;

    // fp32, c=512, h=w=32, k=512, 3x3/pad1/stride1 (=> Ho=Wo=32, Y = {n,512,32,32}); element
    // count stays under INT_MAX on both sides of the boundary below, so only the byte span
    // (not GetElementSpace()) distinguishes them -- this is the case the original bug missed.
    // X and Y have equal span here (k=c, Ho=h, Wo=w):
    //   n=1024: elem = 536,870,912 (< INT_MAX); bytes = 2,147,483,648 (INT_MAX + 1) -> TRIGGER
    //   n=1023: elem = 536,346,624 (< INT_MAX); bytes = 2,145,386,496 (< INT_MAX)   -> no trigger
    constexpr std::size_t c = 512, hw = 32, k = 512;
    constexpr std::size_t n_over  = 1024; // byte span 2,147,483,648 > INT_MAX
    constexpr std::size_t n_under = 1023; // byte span 2,145,386,496 < INT_MAX

    // Over the 2 GiB byte boundary: every direction must require a large-tensor instance.
    EXPECT_TRUE(RequiresLargeTensorCKInstance(
        MakeByteSpanConvProblem(miopenFloat, Direction::Forward, n_over, c, hw, hw, k)))
        << "fwd: >INT_MAX byte span (element count still < INT_MAX) must require large-tensor";
    EXPECT_TRUE(RequiresLargeTensorCKInstance(
        MakeByteSpanConvProblem(miopenFloat, Direction::BackwardData, n_over, c, hw, hw, k)))
        << "bwd-data: >INT_MAX byte span must require large-tensor";
    EXPECT_TRUE(RequiresLargeTensorCKInstance(
        MakeByteSpanConvProblem(miopenFloat, Direction::BackwardWeights, n_over, c, hw, hw, k)))
        << "wrw: >INT_MAX byte span must require large-tensor";

    // One N below the boundary: byte span < INT_MAX, element count < INT_MAX, all strides
    // fit int32 -> every direction must stay on the int32 path.
    EXPECT_FALSE(RequiresLargeTensorCKInstance(
        MakeByteSpanConvProblem(miopenFloat, Direction::Forward, n_under, c, hw, hw, k)))
        << "fwd: byte span just under INT_MAX must stay on the int32 instances";
    EXPECT_FALSE(RequiresLargeTensorCKInstance(
        MakeByteSpanConvProblem(miopenFloat, Direction::BackwardData, n_under, c, hw, hw, k)))
        << "bwd-data: byte span just under INT_MAX must stay on the int32 instances";
    EXPECT_FALSE(RequiresLargeTensorCKInstance(
        MakeByteSpanConvProblem(miopenFloat, Direction::BackwardWeights, n_under, c, hw, hw, k)))
        << "wrw: byte span just under INT_MAX must stay on the int32 instances";
}
