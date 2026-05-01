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
