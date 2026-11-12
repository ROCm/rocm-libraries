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

static std::vector<std::unique_ptr<StubbedDeviceOp>>
MakeStubbedInstances(const std::vector<std::string>& names)
{
    std::vector<std::unique_ptr<StubbedDeviceOp>> ops;
    ops.reserve(names.size());

    std::transform(names.begin(), names.end(), std::back_inserter(ops), [&](auto& name) {
        return std::make_unique<StubbedDeviceOp>(name);
    });

    return ops;
}

// IsCKArgsSupported keeps the first GetInstances() result per DeviceOpType, so a
// stub's instance list must be fixed, as a CK instance factory's is.
struct StubbedDeviceOps
{
    static std::vector<std::unique_ptr<StubbedDeviceOp>> GetInstances()
    {
        std::vector<std::string> names;
        for(const auto& tc : GetTestCases())
            names.push_back(tc.instanceToCheck);
        return MakeStubbedInstances(names);
    }
};

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

// A DeviceOpType whose GetInstances() returns two instances that share one
// GetTypeString(). The cache (CachedCKInstanceIndex) must resolve the kernel_id
// to the first instance (lowest index), the same one FindConvPtrByID returns.
// ordinal records insertion order so a test can tell which instance was picked.
struct DuplicateTypeDeviceOp
{
    DuplicateTypeDeviceOp(const std::string& op, int order) : typeString(op), ordinal(order) {}

    std::string typeString;
    int ordinal;

    std::string GetTypeString() { return typeString; }
};

struct DuplicateTypeDeviceOps
{
    static std::vector<std::unique_ptr<DuplicateTypeDeviceOp>> GetInstances()
    {
        std::vector<std::unique_ptr<DuplicateTypeDeviceOp>> ops;
        ops.push_back(std::make_unique<DuplicateTypeDeviceOp>("duplicate_type", 0));
        ops.push_back(std::make_unique<DuplicateTypeDeviceOp>("duplicate_type", 1));
        return ops;
    }
};

// Only the first-enumerated "duplicate_type" instance (ordinal 0) is supported, so
// the validator returns true iff the cache resolved the id to the first instance.
struct DuplicateTypeCKArgs
{
    DuplicateTypeCKArgs(const ProblemDescription& /*problem*/) {}

    template <typename ConvPtr>
    bool IsSupportedBy(const ConvPtr& conv_ptr) const
    {
        return conv_ptr->ordinal == 0;
    }

    template <typename ConvPtr>
    bool IsSupportedBySplitK(const ConvPtr& conv_ptr, int) const
    {
        return conv_ptr->ordinal == 0;
    }
};

// Counts how many times GetInstances() runs. The cache must enumerate once per
// DeviceOpType for the whole process, so repeated validator calls add no calls.
struct CountingDeviceOp
{
    CountingDeviceOp(const std::string& op) : typeString(op) {}

    std::string typeString;

    std::string GetTypeString() { return typeString; }
};

struct CountingDeviceOps
{
    static int instanceCalls;

    static std::vector<std::unique_ptr<CountingDeviceOp>> GetInstances()
    {
        ++instanceCalls;
        std::vector<std::unique_ptr<CountingDeviceOp>> ops;
        ops.push_back(std::make_unique<CountingDeviceOp>("counting_kernel"));
        return ops;
    }
};
int CountingDeviceOps::instanceCalls = 0;

template <typename CKArgsType, typename DeviceOpType>
class CKArgParsingTest : public ::testing::TestWithParam<ParsingTestCase>
{
protected:
    void TestParsing()
    {
        auto testCase = GetParam();

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

// Regression guard for the instance cache's first-index invariant. When a
// DeviceOpType enumerates two instances with the same GetTypeString(), the cache
// must resolve the kernel_id to the first instance -- the one FindConvPtrByID
// returns -- because invoker construction and workspace sizing still call
// FindConvPtrByID and the two must agree. Only the first "duplicate_type" instance
// is supported here, so replacing the index's emplace with a last-index-wins
// insert would select the unsupported second instance and flip this to false.
TEST(CPU_UnitTestImplicitGemmCKUtilDuplicateType_NONE, FirstIndexWins)
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    const bool success =
        miopen::solver::IsCKArgsSupported<DuplicateTypeDeviceOps, DuplicateTypeCKArgs>(
            ProblemDescription{}, "duplicate_type");
    EXPECT_TRUE(success);
#else
    GTEST_SKIP();
#endif
}

// GetInstances() must be enumerated once per DeviceOpType and cached: validation
// of many (kernel, split_k) candidates, across the plain and split-k
// IsCKArgsSupported overloads and IsCKSplitKSupported, must not re-enumerate.
TEST(CPU_UnitTestImplicitGemmCKUtilInstanceCache_NONE, GetInstancesEnumeratedOnce)
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    CountingDeviceOps::instanceCalls = 0;
    const ProblemDescription problem{};

    miopen::solver::IsCKArgsSupported<CountingDeviceOps, StubbedCKArgs>(problem, "counting_kernel");
    miopen::solver::IsCKArgsSupported<CountingDeviceOps, StubbedCKArgs, ProblemDescription, true>(
        problem, "counting_kernel+1");
    miopen::solver::IsCKSplitKSupported<CountingDeviceOps, StubbedCKArgs>(
        problem, "counting_kernel", 1);
    for(int i = 0; i < 5; ++i)
        miopen::solver::IsCKArgsSupported<CountingDeviceOps, StubbedCKArgs>(problem,
                                                                            "counting_kernel");

    EXPECT_EQ(CountingDeviceOps::instanceCalls, 1);
#else
    GTEST_SKIP();
#endif
}

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
 * @brief Fixed instance list for the split_k tests
 *
 * Holds every kernel the parameterized cases register, plus the kernels the
 * direct tests use. "NonExistentKernel" has no supported split_k values and is
 * left out, so it is never found.
 */
struct StubbedSplitKDeviceOps
{
    static std::vector<std::unique_ptr<StubbedDeviceOp>> GetInstances()
    {
        std::vector<std::string> names = {"SomeKernel", "TestKernel", "KernelA", "KernelB"};
        for(const auto& tc : GetSplitKGenericTestCases())
        {
            if(!tc.supportedSplitKs.empty() &&
               std::find(names.begin(), names.end(), tc.kernelId) == names.end())
                names.push_back(tc.kernelId);
        }
        return MakeStubbedInstances(names);
    }
};

/**
 * @brief Parameterized test fixture for IsCKSplitKSupportedGeneric
 */
class CPU_SplitKGenericTest_NONE : public ::testing::TestWithParam<SplitKGenericTestCase>
{
protected:
    void SetUp() override
    {
        // Clear state before each test
        StubbedCKArgsWithSplitKValidation::supportedSplitKByKernel.clear();
    }

    void TearDown() override
    {
        // Clean up after each test
        StubbedCKArgsWithSplitKValidation::supportedSplitKByKernel.clear();
    }
};

// cppcheck-suppress syntaxError
TEST_P(CPU_SplitKGenericTest_NONE, ValidatesSplitKSupport)
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    auto testCase = GetParam();

    if(!testCase.supportedSplitKs.empty())
    {
        StubbedCKArgsWithSplitKValidation::supportedSplitKByKernel[testCase.kernelId] =
            testCase.supportedSplitKs;
    }

    // Test IsCKSplitKSupported (single data type version)
    bool result = miopen::solver::IsCKSplitKSupported<StubbedSplitKDeviceOps,
                                                      StubbedCKArgsWithSplitKValidation>(
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
    void SetUp() override { StubbedCKArgsWithSplitKValidation::supportedSplitKByKernel.clear(); }

    void TearDown() override { StubbedCKArgsWithSplitKValidation::supportedSplitKByKernel.clear(); }
};

TEST_F(CPU_SplitKGenericDirectTest_NONE, EmptyKernelIdReturnsNotSupported)
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    StubbedCKArgsWithSplitKValidation::supportedSplitKByKernel["SomeKernel"] = {1, 2, 4};

    bool result = miopen::solver::IsCKSplitKSupported<StubbedSplitKDeviceOps,
                                                      StubbedCKArgsWithSplitKValidation>(
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
    // Only support positive split_k values
    StubbedCKArgsWithSplitKValidation::supportedSplitKByKernel[kernelId] = {1, 2, 4};

    bool result = miopen::solver::IsCKSplitKSupported<StubbedSplitKDeviceOps,
                                                      StubbedCKArgsWithSplitKValidation>(
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
    const std::string kernelId                                           = "TestKernel";
    StubbedCKArgsWithSplitKValidation::supportedSplitKByKernel[kernelId] = {1, 2, 4};

    bool result = miopen::solver::IsCKSplitKSupported<StubbedSplitKDeviceOps,
                                                      StubbedCKArgsWithSplitKValidation>(
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

    // Kernel1 supports split_k = 1, 2, 4
    StubbedCKArgsWithSplitKValidation::supportedSplitKByKernel[kernel1] = {1, 2, 4};
    // Kernel2 supports split_k = 1, 8, 16
    StubbedCKArgsWithSplitKValidation::supportedSplitKByKernel[kernel2] = {1, 8, 16};

    // Test kernel1 with its supported values
    // Note: Use parentheses around function call to prevent macro from misinterpreting commas
    bool k1_s2 = miopen::solver::IsCKSplitKSupported<StubbedSplitKDeviceOps,
                                                     StubbedCKArgsWithSplitKValidation>(
        ProblemDescription{}, kernel1, 2);
    EXPECT_TRUE(k1_s2);

    bool k1_s8 = miopen::solver::IsCKSplitKSupported<StubbedSplitKDeviceOps,
                                                     StubbedCKArgsWithSplitKValidation>(
        ProblemDescription{}, kernel1, 8);
    EXPECT_FALSE(k1_s8);

    // Test kernel2 with its supported values
    bool k2_s8 = miopen::solver::IsCKSplitKSupported<StubbedSplitKDeviceOps,
                                                     StubbedCKArgsWithSplitKValidation>(
        ProblemDescription{}, kernel2, 8);
    EXPECT_TRUE(k2_s8);

    bool k2_s4 = miopen::solver::IsCKSplitKSupported<StubbedSplitKDeviceOps,
                                                     StubbedCKArgsWithSplitKValidation>(
        ProblemDescription{}, kernel2, 4);
    EXPECT_FALSE(k2_s4);
#else
    GTEST_SKIP();
#endif
}

} // namespace unit_implicitgemm_ck_util_test

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
