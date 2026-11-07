// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "engines/plans/HipblasltMxScaleLayout.hpp"

using namespace hipblaslt_plugin;

namespace
{

size_t roundUp(size_t value, size_t multiple)
{
    return ((value + multiple - 1) / multiple) * multiple;
}

// Host references, written from the index formulas of mxDataGenerator's
// PreSwizzle.hpp. Input is the natural K-contiguous scale [mn][kb].

std::vector<uint8_t>
    transposeHost(const std::vector<uint8_t>& in, size_t rows, size_t cols) // [rows][cols]
{
    std::vector<uint8_t> out(in.size());
    for(size_t r = 0; r < rows; ++r)
    {
        for(size_t c = 0; c < cols; ++c)
        {
            out[c * rows + r] = in[r * cols + c];
        }
    }
    return out;
}

// preSwizzleScalesGFX950: view(MNp/32, 2, 16, Kbp/8, 2, 4).permute(0, 3, 5, 2, 4, 1).
std::vector<uint8_t> swizzleGfx950Host(const std::vector<uint8_t>& in, size_t mn, size_t kb)
{
    const size_t mnPadded = roundUp(mn, 32);
    const size_t kbPadded = roundUp(kb, 8);
    std::vector<uint8_t> out(mnPadded * kbPadded, 0);
    for(size_t r = 0; r < mn; ++r)
    {
        for(size_t c = 0; c < kb; ++c)
        {
            const size_t d0 = r / 32;
            const size_t d1 = (r % 32) / 16;
            const size_t d2 = r % 16;
            const size_t d3 = c / 8;
            const size_t d4 = (c % 8) / 4;
            const size_t d5 = c % 4;
            const size_t idx = d1 + 2 * d4 + 4 * d2 + 64 * d5 + 256 * d3 + 32 * kbPadded * d0;
            out[idx] = in[r * kb + c];
        }
    }
    return out;
}

// preSwizzleScalesGFX1250 (mxBlock 32): [mn][Kbp/4][4] -> [Kbp/4][mn][4].
std::vector<uint8_t> swizzleGfx1250Host(const std::vector<uint8_t>& in, size_t mn, size_t kb)
{
    const size_t kbPadded = roundUp(kb, 4);
    std::vector<uint8_t> out(mn * kbPadded, 0);
    for(size_t m = 0; m < mn; ++m)
    {
        for(size_t k = 0; k < kb; ++k)
        {
            out[(k / 4) * mn * 4 + m * 4 + k % 4] = in[m * kb + k];
        }
    }
    return out;
}

struct DeviceFree
{
    void operator()(void* ptr) const
    {
        static_cast<void>(hipFree(ptr));
    }
};
using DevicePtr = std::unique_ptr<void, DeviceFree>;

DevicePtr deviceAlloc(size_t bytes)
{
    void* ptr = nullptr;
    EXPECT_EQ(hipMalloc(&ptr, bytes == 0 ? 1 : bytes), hipSuccess);
    return DevicePtr(ptr);
}

} // namespace

// ===========================================================================
// Planning (CPU — only host-side hipBLASLt layout/descriptor objects)
// ===========================================================================

TEST(TestMxScaleLayout, ScaleModeForFormat)
{
    EXPECT_EQ(scaleModeFor(MxScaleFormat::NATURAL), HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
    EXPECT_EQ(scaleModeFor(MxScaleFormat::SWIZZLED_GFX1250),
              HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
    EXPECT_EQ(scaleModeFor(MxScaleFormat::SWIZZLED_GFX950),
              HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT);
}

TEST(TestMxScaleLayout, ScaleBytesIncludePadding)
{
    EXPECT_EQ(mxScaleBytes(MxScaleFormat::NATURAL, 48, 5), 240u);
    EXPECT_EQ(mxScaleBytes(MxScaleFormat::SWIZZLED_GFX950, 48, 5), 64u * 8u);
    EXPECT_EQ(mxScaleBytes(MxScaleFormat::SWIZZLED_GFX950, 64, 16), 64u * 16u);
    EXPECT_EQ(mxScaleBytes(MxScaleFormat::SWIZZLED_GFX1250, 48, 5), 48u * 8u);
    EXPECT_EQ(mxScaleBytes(MxScaleFormat::SWIZZLED_GFX1250, 48, 8), 48u * 8u);
}

// rocRoller reads natural scales in the operand's order, so a scale that already
// has it is handed over untouched, with no workspace.
TEST(TestMxScaleLayout, NaturalMatchingOrderIsPassThrough)
{
    for(const auto order : {MxOrder::K_CONTIGUOUS, MxOrder::MN_CONTIGUOUS})
    {
        MxScalePrep const prep(order, order, MxScaleFormat::NATURAL, 64, 8);
        EXPECT_TRUE(prep.isPassThrough());
        EXPECT_EQ(prep.workspaceSize(), 0u);
        EXPECT_EQ(prep.describe(), "pass-through");
    }
}

// rocRoller reads a scale stored in the other order once hipBLASLt has transposed it.
TEST(TestMxScaleLayout, NaturalMismatchedOrderTransposes)
{
    for(const auto order : {MxOrder::K_CONTIGUOUS, MxOrder::MN_CONTIGUOUS})
    {
        const auto operandOrder
            = order == MxOrder::K_CONTIGUOUS ? MxOrder::MN_CONTIGUOUS : MxOrder::K_CONTIGUOUS;
        MxScalePrep const prep(order, operandOrder, MxScaleFormat::NATURAL, 48, 4);
        EXPECT_EQ(prep.method(), MxScalePrep::Method::TRANSPOSE) << toString(order);
        EXPECT_EQ(prep.describe(), "transpose");
        EXPECT_GE(prep.workspaceSize(), 48u * 4u);
    }
}

// A vector scale (one row or one column) reads the same in either order.
TEST(TestMxScaleLayout, NaturalVectorScaleIsPassThrough)
{
    MxScalePrep const prep(
        MxOrder::K_CONTIGUOUS, MxOrder::MN_CONTIGUOUS, MxScaleFormat::NATURAL, 64, 1);
    EXPECT_TRUE(prep.isPassThrough());
}

// The swizzled formats are Tensile's, and Tensile only has K-contiguous (TN) MX
// kernels.
TEST(TestMxScaleLayout, SwizzledFormatsRequireKContiguousOperand)
{
    for(const auto format : {MxScaleFormat::SWIZZLED_GFX950, MxScaleFormat::SWIZZLED_GFX1250})
    {
        EXPECT_THROW(MxScalePrep(MxOrder::K_CONTIGUOUS, MxOrder::MN_CONTIGUOUS, format, 64, 8),
                     hipdnn_plugin_sdk::HipdnnPluginException)
            << toString(format);
    }
}

TEST(TestMxScaleLayout, RejectsEmptyScale)
{
    EXPECT_THROW(
        MxScalePrep(MxOrder::K_CONTIGUOUS, MxOrder::K_CONTIGUOUS, MxScaleFormat::NATURAL, 0, 4),
        hipdnn_plugin_sdk::HipdnnPluginException);
}

// ===========================================================================
// Execution (GPU) — byte-exact against the host references above
// ===========================================================================

// A device and a hipBLASLt handle. The relayout kernel compiles for whatever device
// this is, so every case runs on every device.
template <typename Base>
class MxScaleLayoutGpuFixture : public Base
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        ASSERT_EQ(hipblasLtCreate(&_handle), HIPBLAS_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            EXPECT_EQ(hipblasLtDestroy(_handle), HIPBLAS_STATUS_SUCCESS);
        }
    }

    hipblasLtHandle_t _handle = nullptr;
};

class TestGpuMxScaleKernel : public MxScaleLayoutGpuFixture<::testing::Test>
{
};

// hipBLASLt has no function for the swizzled layouts, so the provider's relayout
// kernel writes them, one launch per scale, padding included.
TEST_F(TestGpuMxScaleKernel, SwizzledFormatsUseTheRelayoutKernel)
{
    for(const auto format : {MxScaleFormat::SWIZZLED_GFX950, MxScaleFormat::SWIZZLED_GFX1250})
    {
        for(const auto order : {MxOrder::K_CONTIGUOUS, MxOrder::MN_CONTIGUOUS})
        {
            MxScalePrep const prep(order, MxOrder::K_CONTIGUOUS, format, 48, 5);
            EXPECT_EQ(prep.method(), MxScalePrep::Method::RELAYOUT_KERNEL) << toString(format);
            EXPECT_EQ(prep.describe(), "relayout kernel");
            EXPECT_GE(prep.workspaceSize(), mxScaleBytes(format, 48, 5)) << toString(format);
        }
    }
}

// A workspace that is not 4-byte aligned is written byte by byte.
TEST_F(TestGpuMxScaleKernel, SwizzlesIntoAnUnalignedWorkspace)
{
    constexpr size_t MN = 48;
    constexpr size_t KB = 5;
    std::vector<uint8_t> natural(MN * KB);
    for(size_t i = 0; i < natural.size(); ++i)
    {
        natural[i] = static_cast<uint8_t>(i * 7 + 1);
    }
    auto source = deviceAlloc(natural.size());
    ASSERT_EQ(hipMemcpy(source.get(), natural.data(), natural.size(), hipMemcpyHostToDevice),
              hipSuccess);

    for(const auto format : {MxScaleFormat::SWIZZLED_GFX950, MxScaleFormat::SWIZZLED_GFX1250})
    {
        const auto expected = format == MxScaleFormat::SWIZZLED_GFX950
                                  ? swizzleGfx950Host(natural, MN, KB)
                                  : swizzleGfx1250Host(natural, MN, KB);
        MxScalePrep const prep(MxOrder::K_CONTIGUOUS, MxOrder::K_CONTIGUOUS, format, MN, KB);
        auto workspace = deviceAlloc(prep.workspaceSize() + 4);
        ASSERT_EQ(hipMemset(workspace.get(), 0xA5, prep.workspaceSize() + 4), hipSuccess);
        void* unaligned = static_cast<char*>(workspace.get()) + 1;

        const void* result = prep.run(_handle, nullptr, source.get(), unaligned);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        std::vector<uint8_t> actual(expected.size());
        ASSERT_EQ(hipMemcpy(actual.data(), result, actual.size(), hipMemcpyDeviceToHost),
                  hipSuccess);
        EXPECT_EQ(actual, expected) << toString(format);
    }
}

// (format, source order, operand order, mn, kBlocks)
using ScalePrepCase = std::tuple<MxScaleFormat, MxOrder, MxOrder, int64_t, int64_t>;

class TestGpuMxScaleLayout : public MxScaleLayoutGpuFixture<::testing::TestWithParam<ScalePrepCase>>
{
};

TEST_P(TestGpuMxScaleLayout, MatchesHostReference)
{
    const auto [format, sourceOrder, operandOrder, mnSigned, kbSigned] = GetParam();
    const auto mn = static_cast<size_t>(mnSigned);
    const auto kb = static_cast<size_t>(kbSigned);

    // Random natural scale bytes, plus the byte patterns a float round-trip would
    // corrupt (0x80 is -0 in the top byte of a word, 0xFF/0x7F build NaNs).
    std::mt19937 rng(static_cast<uint32_t>(mn * 1000 + kb));
    std::uniform_int_distribution<int> byteDist(0, 255);
    std::vector<uint8_t> natural(mn * kb);
    for(auto& value : natural)
    {
        value = static_cast<uint8_t>(byteDist(rng));
    }
    const std::vector<uint8_t> specials = {0x00, 0x80, 0x7F, 0xFF, 0xFE, 0x01};
    for(size_t i = 0; i < specials.size() && i < natural.size(); ++i)
    {
        natural[i] = specials[i];
    }

    // What the user hands in (natural data in the requested order) and what
    // hipBLASLt must receive: natural scales in the operand's order, or a swizzle
    // (only for K-contiguous operands).
    const auto source
        = sourceOrder == MxOrder::K_CONTIGUOUS ? natural : transposeHost(natural, mn, kb);
    std::vector<uint8_t> expected;
    switch(format)
    {
    case MxScaleFormat::SWIZZLED_GFX950:
        expected = swizzleGfx950Host(natural, mn, kb);
        break;
    case MxScaleFormat::SWIZZLED_GFX1250:
        expected = swizzleGfx1250Host(natural, mn, kb);
        break;
    case MxScaleFormat::NATURAL:
    default:
        expected = operandOrder == MxOrder::K_CONTIGUOUS ? natural : transposeHost(natural, mn, kb);
        break;
    }

    MxScalePrep const prep(sourceOrder, operandOrder, format, mnSigned, kbSigned);

    auto deviceSource = deviceAlloc(source.size());
    ASSERT_EQ(hipMemcpy(deviceSource.get(), source.data(), source.size(), hipMemcpyHostToDevice),
              hipSuccess);
    // Fill the workspace with garbage so missing padding writes are caught.
    auto workspace = deviceAlloc(prep.workspaceSize());
    ASSERT_EQ(
        hipMemset(workspace.get(), 0xA5, prep.workspaceSize() == 0 ? 1 : prep.workspaceSize()),
        hipSuccess);

    const void* result = prep.run(_handle, nullptr, deviceSource.get(), workspace.get());
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    std::vector<uint8_t> actual(expected.size());
    ASSERT_EQ(hipMemcpy(actual.data(), result, actual.size(), hipMemcpyDeviceToHost), hipSuccess);

    size_t mismatches = 0;
    size_t firstMismatch = actual.size();
    for(size_t i = 0; i < actual.size(); ++i)
    {
        if(actual[i] != expected[i])
        {
            firstMismatch = mismatches == 0 ? i : firstMismatch;
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0u) << toString(format) << " from " << toString(sourceOrder) << " for a "
                              << toString(operandOrder) << " operand, mn=" << mn << " kb=" << kb
                              << " (" << prep.describe() << ") first mismatch at byte "
                              << firstMismatch;
}

// The A and B scales of one GEMM share a single relayout launch when both need
// the same target; each must still come out exactly as it would alone, even with
// different shapes and source orders.
class TestGpuMxScaleLayoutPair : public MxScaleLayoutGpuFixture<::testing::Test>
{
};

TEST_F(TestGpuMxScaleLayoutPair, FusedRunMatchesHostReference)
{
    struct Scale
    {
        MxOrder order;
        size_t mn;
        size_t kb;
    };
    // Two and three 64-wide tiles along K, so the second scale's tiles start mid-grid.
    const std::array<Scale, 2> scales
        = {Scale{MxOrder::K_CONTIGUOUS, 48, 65}, Scale{MxOrder::MN_CONTIGUOUS, 96, 130}};

    for(const auto format : {MxScaleFormat::SWIZZLED_GFX950, MxScaleFormat::SWIZZLED_GFX1250})
    {
        std::vector<std::vector<uint8_t>> expected;
        std::vector<DevicePtr> sources;
        std::vector<DevicePtr> workspaces;
        std::vector<MxScalePrep> preps;
        std::mt19937 rng(7);
        for(const auto& scale : scales)
        {
            std::vector<uint8_t> natural(scale.mn * scale.kb);
            for(auto& value : natural)
            {
                value = static_cast<uint8_t>(rng());
            }
            expected.push_back(format == MxScaleFormat::SWIZZLED_GFX950
                                   ? swizzleGfx950Host(natural, scale.mn, scale.kb)
                                   : swizzleGfx1250Host(natural, scale.mn, scale.kb));
            const auto source = scale.order == MxOrder::K_CONTIGUOUS
                                    ? natural
                                    : transposeHost(natural, scale.mn, scale.kb);
            sources.push_back(deviceAlloc(source.size()));
            ASSERT_EQ(
                hipMemcpy(
                    sources.back().get(), source.data(), source.size(), hipMemcpyHostToDevice),
                hipSuccess);
            preps.emplace_back(scale.order,
                               MxOrder::K_CONTIGUOUS,
                               format,
                               static_cast<int64_t>(scale.mn),
                               static_cast<int64_t>(scale.kb));
            workspaces.push_back(deviceAlloc(preps.back().workspaceSize()));
            ASSERT_EQ(hipMemset(workspaces.back().get(), 0xA5, preps.back().workspaceSize()),
                      hipSuccess);
        }

        const auto [outA, outB] = MxScalePrep::runPair(preps[0],
                                                       sources[0].get(),
                                                       workspaces[0].get(),
                                                       preps[1],
                                                       sources[1].get(),
                                                       workspaces[1].get(),
                                                       _handle,
                                                       nullptr);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const std::array<const void*, 2> outputs = {outA, outB};
        for(size_t i = 0; i < outputs.size(); ++i)
        {
            std::vector<uint8_t> actual(expected[i].size());
            ASSERT_EQ(hipMemcpy(actual.data(), outputs[i], actual.size(), hipMemcpyDeviceToHost),
                      hipSuccess);
            EXPECT_EQ(actual, expected[i]) << toString(format) << " scale " << i;
        }
    }
}

// Names like "SWIZZLED_GFX950_K_CONTIGUOUS_48x5", so one format can be selected with
// --gtest_filter (e.g. "*NATURAL*"); the instantiation prefix names the operand order.
namespace
{
std::string scalePrepCaseName(const ::testing::TestParamInfo<ScalePrepCase>& info)
{
    const auto [format, order, operandOrder, mn, kBlocks] = info.param;
    static_cast<void>(operandOrder);
    return std::string(toString(format)) + "_" + toString(order) + "_" + std::to_string(mn) + "x"
           + std::to_string(kBlocks);
}
} // namespace

// K-contiguous operands: every format.
INSTANTIATE_TEST_SUITE_P(
    ,
    TestGpuMxScaleLayout,
    ::testing::Combine(::testing::Values(MxScaleFormat::NATURAL,
                                         MxScaleFormat::SWIZZLED_GFX950,
                                         MxScaleFormat::SWIZZLED_GFX1250),
                       ::testing::Values(MxOrder::K_CONTIGUOUS, MxOrder::MN_CONTIGUOUS),
                       ::testing::Values(MxOrder::K_CONTIGUOUS),
                       ::testing::Values<int64_t>(1, 16, 17, 32, 48, 64, 96, 256, 1000),
                       ::testing::Values<int64_t>(1, 3, 4, 5, 8, 12, 16, 33, 65, 128)),
    scalePrepCaseName);

// M/N-contiguous operands (the default NT layout): rocRoller reads the scale in the
// operand's order, so a K-contiguous scale is transposed to [kBlocks][mn].
INSTANTIATE_TEST_SUITE_P(
    MnContiguousOperand,
    TestGpuMxScaleLayout,
    ::testing::Combine(::testing::Values(MxScaleFormat::NATURAL),
                       ::testing::Values(MxOrder::K_CONTIGUOUS, MxOrder::MN_CONTIGUOUS),
                       ::testing::Values(MxOrder::MN_CONTIGUOUS),
                       ::testing::Values<int64_t>(1, 16, 17, 32, 48, 64, 96, 256, 1000),
                       ::testing::Values<int64_t>(1, 3, 4, 5, 8, 12, 16, 33, 65, 128)),
    scalePrepCaseName);
