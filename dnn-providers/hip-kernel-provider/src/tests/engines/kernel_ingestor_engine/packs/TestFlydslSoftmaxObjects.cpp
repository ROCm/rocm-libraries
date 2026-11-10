// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>

#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/types/Half.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSoftmax.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslNormDeviceTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSoftmaxTestGraphs.hpp"

/**
 * @file TestFlydslSoftmaxObjects.cpp
 * @brief Every FlyDSL softmax object this device's shard carries, launched through the
 *        pack's own matchers and dispatch handler against hipDNN's CPU reference
 *        (CpuFpReferenceSoftmax, computing in f64) at the widths it serves: one vector
 *        (or element), odd and mid widths, its `n_max`, a baked object's own width, and
 *        Qwen3.8's 248320-wide vocabulary for the unbounded ones. The rows hold fully
 *        masked rows (every element -inf: s 0, stats -inf), partly masked ones, a NaN row
 *        (s and stats NaN), wide-range ones and a row count that leaves the last
 *        multi-row block partly empty; stats are off, natural-log and log2 by turns, and
 *        padded row pitches and an in-place s are covered. Outputs start as NaN, so an
 *        element nothing writes fails.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using data_objects::DataType;
using hipdnn_data_sdk::utilities::Tensor;
using Bf16 = hipdnn_data_sdk::types::bfloat16;
using F16 = hipdnn_data_sdk::types::half;

constexpr int64_t VOCAB = 248320;

std::string stringField(const KernelDefinition& kernel, const char* field)
{
    const auto* value = tryGetMetadataField<std::string>(kernel.metadata, field);
    return value == nullptr ? std::string() : *value;
}

int64_t intField(const KernelDefinition& kernel, const char* field)
{
    const auto* value = tryGetMetadataField<int64_t>(kernel.metadata, field);
    return value == nullptr ? -1 : *value;
}

/// One launch of one object.
struct SoftmaxCase
{
    int64_t rows = 13;
    int64_t columns = 0;
    /// Elements between rows of p and of s; 0 means the width (packed).
    int64_t pRowStride = 0;
    int64_t sRowStride = 0;
    /// 0: no stats; 1: natural log; 2: log2.
    int statsMode = 0;
    /// s is p's own buffer.
    bool inPlace = false;
};

std::string describe(const SoftmaxCase& c)
{
    return std::to_string(c.rows) + "x" + std::to_string(c.columns) + " pitch "
           + std::to_string(c.pRowStride) + "/" + std::to_string(c.sRowStride) + " stats "
           + std::to_string(c.statsMode) + (c.inPlace ? " in place" : "");
}

/// The launches for @p kernel: the widths it serves, each with stats in turn.
std::vector<SoftmaxCase> casesFor(const KernelDefinition& kernel, bool sameDtype)
{
    const auto bakedN = intField(kernel, "N");
    const auto nMax = intField(kernel, "n_max");
    const auto nAlign = intField(kernel, "n_align");
    std::vector<int64_t> widths;
    if(bakedN > 0)
    {
        widths = {bakedN};
    }
    else
    {
        const std::vector<int64_t> candidates
            = nAlign > 1 ? std::vector<int64_t>{nAlign, 17 * nAlign, 256, 1000, 2048, 6000, VOCAB}
                         : std::vector<int64_t>{1, 7, 1001, 2047, VOCAB + 1};
        for(const auto width : candidates)
        {
            if(width % nAlign == 0 && (nMax == 0 || width <= nMax))
            {
                widths.push_back(width);
            }
        }
        if(nMax > 0 && widths.back() != nMax)
        {
            widths.push_back(nMax);
        }
    }

    std::vector<SoftmaxCase> cases;
    int statsMode = 0;
    for(const auto width : widths)
    {
        SoftmaxCase c;
        c.columns = width;
        c.rows = width >= VOCAB ? 5 : 13;
        c.statsMode = statsMode;
        statsMode = (statsMode + 1) % 3;
        cases.push_back(c);
    }
    // Padded pitches, p's and s's different, at the narrowest width.
    SoftmaxCase padded = cases.front();
    padded.pRowStride = padded.columns + 2 * nAlign;
    padded.sRowStride = padded.columns + 5 * nAlign;
    padded.statsMode = 1;
    cases.push_back(padded);
    if(sameDtype)
    {
        SoftmaxCase inPlace = cases.front();
        inPlace.inPlace = true;
        inPlace.statsMode = 2;
        cases.push_back(inPlace);
    }
    return cases;
}

/// Row r's logits: every 5th row (from 3) fully masked, the next partly masked, row 1
/// holding a NaN, row 2 spread wide; the rest normal(0, 3).
float logit(int64_t row, int64_t column, int64_t columns, std::mt19937& generator)
{
    std::normal_distribution<float> normal(0.0F, 3.0F);
    const auto value = normal(generator);
    if(row % 5 == 3)
    {
        return -std::numeric_limits<float>::infinity();
    }
    if(row % 5 == 4 && column % 3 != 0)
    {
        return -std::numeric_limits<float>::infinity();
    }
    if(row == 1 && column == columns / 2)
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return row == 2 ? 30.0F * value : value;
}

double toleranceFor(DataType type)
{
    if(type == DataType::BFLOAT16)
    {
        return 1e-2;
    }
    return type == DataType::HALF ? 2e-3 : 1e-4;
}

/// The smallest magnitude below which an output in @p type is compared absolutely: its
/// subnormals, which the kernel may flush.
double absoluteFloorFor(DataType type)
{
    return type == DataType::HALF ? 1e-7 : 1e-30;
}

template <typename P, typename S>
void expectCaseMatches(const KernelDefinition& kernel,
                       DataType pType,
                       DataType sType,
                       const SoftmaxCase& softmaxCase)
{
    SCOPED_TRACE(kernel.name + " on " + describe(softmaxCase));
    const auto rows = softmaxCase.rows;
    const auto columns = softmaxCase.columns;
    const auto pPitch = softmaxCase.pRowStride == 0 ? columns : softmaxCase.pRowStride;
    const auto unpackedPitch = softmaxCase.sRowStride == 0 ? columns : softmaxCase.sRowStride;
    const auto sPitch = softmaxCase.inPlace ? pPitch : unpackedPitch;

    SoftmaxGraphSpec spec;
    spec.pDataType = pType;
    spec.sDataType = sType;
    spec.pDims = {rows, columns};
    spec.pStrides = std::vector<int64_t>{pPitch, 1};
    spec.sStrides = std::vector<int64_t>{sPitch, 1};
    spec.stats = softmaxCase.statsMode != 0;
    spec.statsLog2 = softmaxCase.statsMode == 2;

    const GraphFixture fixture(buildFlydslSoftmaxGraph(spec), packedArchDeviceProperties());
    const auto bound = matchesGraph(FLYDSL_SOFTMAX, fixture.context());
    ASSERT_TRUE(bound.has_value()) << "the graph this object is meant for is declined";
    ASSERT_TRUE(matchesKernel(FLYDSL_SOFTMAX, fixture.context(), kernel, *bound))
        << "the object declines the graph it is meant for";

    const auto& handler = dispatchHandler(FLYDSL_SOFTMAX);
    const auto prepared = handler.prepare(fixture.context(), *bound, kernel);
    ASSERT_NE(prepared, nullptr);

    const std::vector<int64_t> dims{rows, columns};
    Tensor<P> p(dims, {pPitch, 1});
    Tensor<S> s(dims, {sPitch, 1});
    Tensor<float> stats({rows, 1});
    Tensor<double> expectedStats({rows, 1});
    std::mt19937 generator(static_cast<unsigned>(columns * 131 + rows));
    for(int64_t r = 0; r < rows; ++r)
    {
        for(int64_t j = 0; j < columns; ++j)
        {
            p.setHostValue(static_cast<P>(logit(r, j, columns, generator)), r, j);
        }
    }
    p.memory().markHostModified();
    s.fillWithSentinelValue();
    stats.fillWithSentinelValue();

    // In f64, so the comparison measures the kernel rather than an f32 reference's own sum.
    Tensor<double> pWide(dims, {pPitch, 1});
    for(int64_t r = 0; r < rows; ++r)
    {
        for(int64_t j = 0; j < columns; ++j)
        {
            pWide.setHostValue(static_cast<double>(static_cast<float>(p.getHostValue(r, j))), r, j);
        }
    }
    Tensor<double> expectedWide(dims, {sPitch, 1});
    hipdnn_test_sdk::utilities::CpuFpReferenceSoftmax::forward<double, double, double, double>(
        pWide, expectedWide, spec.stats ? &expectedStats : nullptr, spec.statsLog2);

    std::vector<hipdnnPluginDeviceBuffer_t> arguments{
        {FLYDSL_SOFTMAX_P_UID, p.memory().deviceData()},
        {FLYDSL_SOFTMAX_S_UID,
         softmaxCase.inPlace ? p.memory().deviceData() : s.memory().deviceData()}};
    if(spec.stats)
    {
        arguments.push_back({FLYDSL_SOFTMAX_STATS_UID, stats.memory().deviceData()});
    }
    const Handle handle;
    handler.launch(
        handle, *prepared, arguments.data(), static_cast<uint32_t>(arguments.size()), nullptr);
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());

    const auto read = [&](int64_t r, int64_t j) -> double {
        if(softmaxCase.inPlace)
        {
            return static_cast<double>(static_cast<float>(p.getHostValue(r, j)));
        }
        return static_cast<double>(static_cast<float>(s.getHostValue(r, j)));
    };
    // Non-const host access copies the device's results back; the reads below are const.
    if(softmaxCase.inPlace)
    {
        p.memory().markDeviceModified();
        static_cast<void>(p.memory().hostData());
    }
    else
    {
        s.memory().markDeviceModified();
        static_cast<void>(s.memory().hostData());
    }
    if(spec.stats)
    {
        stats.memory().markDeviceModified();
        static_cast<void>(stats.memory().hostData());
    }

    const double tolerance = toleranceFor(sType);
    const double floor = absoluteFloorFor(sType);
    size_t mismatches = 0;
    for(int64_t r = 0; r < rows; ++r)
    {
        for(int64_t j = 0; j < columns; ++j)
        {
            const auto want = expectedWide.getHostValue(r, j);
            const auto got = read(r, j);
            const bool ok = std::isnan(want)
                                ? std::isnan(got)
                                : std::isfinite(got)
                                      && std::abs(got - want) <= tolerance * std::abs(want) + floor;
            if(!ok && ++mismatches <= 5)
            {
                ADD_FAILURE() << "s[" << r << ", " << j << "] = " << got << ", expected " << want;
            }
        }
        if(spec.stats)
        {
            const auto want = expectedStats.getHostValue(r, 0);
            const auto got = static_cast<double>(stats.getHostValue(r, 0));
            const bool ok = [&] {
                if(std::isnan(want))
                {
                    return std::isnan(got);
                }
                if(std::isinf(want))
                {
                    return got == want;
                }
                return std::abs(got - want) <= 2e-5 * std::max(1.0, std::abs(want));
            }();
            if(!ok && ++mismatches <= 5)
            {
                ADD_FAILURE() << "stats[" << r << "] = " << got << ", expected " << want;
            }
        }
    }
    EXPECT_EQ(mismatches, 0U);
}

template <typename P>
void expectObjectMatches(const KernelDefinition& kernel, DataType pType)
{
    const auto outDtype = stringField(kernel, "out_dtype");
    const bool sameDtype = outDtype == stringField(kernel, "dtype");
    for(const auto& softmaxCase : casesFor(kernel, sameDtype))
    {
        if(outDtype == "bf16")
        {
            expectCaseMatches<P, Bf16>(kernel, pType, DataType::BFLOAT16, softmaxCase);
        }
        else if(outDtype == "f16")
        {
            expectCaseMatches<P, F16>(kernel, pType, DataType::HALF, softmaxCase);
        }
        else if(outDtype == "f32")
        {
            expectCaseMatches<P, float>(kernel, pType, DataType::FLOAT, softmaxCase);
        }
        else
        {
            ADD_FAILURE() << kernel.name << " declares out_dtype '" << outDtype << "'";
            return;
        }
    }
}

TEST(TestGpuFlydslSoftmaxObjects, EveryObjectMatchesTheCpuReference)
{
    SKIP_IF_NO_DEVICES();
    const auto kernels = stagedKernels(FLYDSL_SOFTMAX.engineName);
    if(kernels.empty())
    {
        reportUnstaged("softmax", "this build staged no softmax objects for this device");
        return;
    }

    for(const auto& kernel : kernels)
    {
        const auto dtype = stringField(kernel, "dtype");
        if(dtype == "bf16")
        {
            expectObjectMatches<Bf16>(kernel, DataType::BFLOAT16);
        }
        else if(dtype == "f16")
        {
            expectObjectMatches<F16>(kernel, DataType::HALF);
        }
        else if(dtype == "f32")
        {
            expectObjectMatches<float>(kernel, DataType::FLOAT);
        }
        else
        {
            ADD_FAILURE() << kernel.name << " declares dtype '" << dtype << "'";
        }
    }
    RecordProperty("objects", static_cast<int>(kernels.size()));
}

/// More bytes of p than one launch addresses: the dispatch runs several, one per slice of
/// whole rows. 2200 f32 rows of the vocabulary, 2.2 GB, on the unbounded vector object.
TEST(TestGpuFlydslSoftmaxObjects, SplitsAGraphPastOneLaunchIntoRowSlices)
{
    SKIP_IF_NO_DEVICES();
    const auto kernels = stagedKernels(FLYDSL_SOFTMAX.engineName);
    const KernelDefinition* chosen = nullptr;
    for(const auto& kernel : kernels)
    {
        if(stringField(kernel, "dtype") == "f32" && stringField(kernel, "out_dtype") == "bf16"
           && intField(kernel, "N") == 0 && intField(kernel, "n_max") == 0
           && intField(kernel, "n_align") > 1)
        {
            chosen = &kernel;
        }
    }
    if(chosen == nullptr)
    {
        reportUnstaged("softmax", "this build staged no unbounded f32 softmax object");
        return;
    }
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    ASSERT_EQ(hipSuccess, hipMemGetInfo(&freeBytes, &totalBytes));
    constexpr int64_t ROWS = 2200;
    const auto needed = static_cast<size_t>(ROWS * VOCAB * (4 + 2) + ROWS * 4);
    if(freeBytes < needed + (size_t{1} << 30))
    {
        GTEST_SKIP() << "needs " << needed << " bytes of device memory";
    }

    SoftmaxGraphSpec spec;
    spec.pDataType = DataType::FLOAT;
    spec.sDataType = DataType::BFLOAT16;
    spec.pDims = {ROWS, VOCAB};
    spec.stats = true;
    const GraphFixture fixture(buildFlydslSoftmaxGraph(spec), packedArchDeviceProperties());
    const auto bound = matchesGraph(FLYDSL_SOFTMAX, fixture.context());
    ASSERT_TRUE(bound.has_value());
    ASSERT_TRUE(matchesKernel(FLYDSL_SOFTMAX, fixture.context(), *chosen, *bound));
    const auto& handler = dispatchHandler(FLYDSL_SOFTMAX);
    const auto prepared = handler.prepare(fixture.context(), *bound, *chosen);

    // Each row r holds one hot logit at column r, the rest 0, so every row's answer is
    // known in closed form and a row the launches miss or misplace shows. p and s live on
    // the device only, written and read where the check looks, so the test moves bytes
    // rather than gigabytes across the bus.
    using DeviceBuffer = std::unique_ptr<void, hipError_t (*)(void*)>;
    const auto allocate = [](size_t bytes) {
        void* pointer = nullptr;
        return hipMalloc(&pointer, bytes) == hipSuccess ? DeviceBuffer(pointer, &hipFree)
                                                        : DeviceBuffer(nullptr, &hipFree);
    };
    const auto p = allocate(static_cast<size_t>(ROWS * VOCAB) * sizeof(float));
    const auto s = allocate(static_cast<size_t>(ROWS * VOCAB) * sizeof(Bf16));
    ASSERT_NE(p, nullptr);
    ASSERT_NE(s, nullptr);
    auto* pBytes = static_cast<char*>(p.get());
    auto* sBytes = static_cast<char*>(s.get());
    ASSERT_EQ(hipSuccess, hipMemset(pBytes, 0, static_cast<size_t>(ROWS * VOCAB) * sizeof(float)));
    // 0xFFFF is a bf16 NaN, so an element nothing writes fails.
    ASSERT_EQ(hipSuccess,
              hipMemset(sBytes, 0xFF, static_cast<size_t>(ROWS * VOCAB) * sizeof(Bf16)));
    const float hotLogit = 10.0F;
    for(int64_t r = 0; r < ROWS; ++r)
    {
        ASSERT_EQ(hipSuccess,
                  hipMemcpy(pBytes + static_cast<size_t>(r * VOCAB + r) * sizeof(float),
                            &hotLogit,
                            sizeof(float),
                            hipMemcpyHostToDevice));
    }
    Tensor<float> stats({ROWS, 1});
    stats.fillWithSentinelValue();

    const std::vector<hipdnnPluginDeviceBuffer_t> arguments{
        {FLYDSL_SOFTMAX_P_UID, pBytes},
        {FLYDSL_SOFTMAX_S_UID, sBytes},
        {FLYDSL_SOFTMAX_STATS_UID, stats.memory().deviceData()}};
    const Handle handle;
    handler.launch(
        handle, *prepared, arguments.data(), static_cast<uint32_t>(arguments.size()), nullptr);
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
    stats.memory().markDeviceModified();
    static_cast<void>(stats.memory().hostData());
    const auto sAt = [sBytes](int64_t r, int64_t j) {
        Bf16 value(0.0F);
        const auto copied = hipMemcpy(&value,
                                      sBytes + static_cast<size_t>(r * VOCAB + j) * sizeof(Bf16),
                                      sizeof(Bf16),
                                      hipMemcpyDeviceToHost);
        return copied == hipSuccess ? static_cast<double>(static_cast<float>(value))
                                    : std::numeric_limits<double>::quiet_NaN();
    };

    const double sum = std::exp(10.0) + static_cast<double>(VOCAB - 1);
    const double hot = std::exp(10.0) / sum;
    const double cold = 1.0 / sum;
    const double lse = std::log(sum);
    size_t mismatches = 0;
    for(int64_t r = 0; r < ROWS; ++r)
    {
        for(const auto j : {int64_t{0}, r, VOCAB - 1, (r + 1) % VOCAB})
        {
            const auto want = j == r ? hot : cold;
            const auto got = sAt(r, j);
            if(!(std::abs(got - want) <= 1e-2 * want) && ++mismatches <= 5)
            {
                ADD_FAILURE() << "s[" << r << ", " << j << "] = " << got << ", expected " << want;
            }
        }
        const auto got = static_cast<double>(stats.getHostValue(r, 0));
        if(!(std::abs(got - lse) <= 2e-5 * lse) && ++mismatches <= 5)
        {
            ADD_FAILURE() << "stats[" << r << "] = " << got << ", expected " << lse;
        }
    }
    EXPECT_EQ(mismatches, 0U);
}

/// s overlapping p anywhere but exactly, or stats overlapping either, is refused at launch:
/// one row's writes would clobber another's reads.
TEST(TestGpuFlydslSoftmaxObjects, RefusesAnOverlapItCannotComputeInPlace)
{
    SKIP_IF_NO_DEVICES();
    const auto kernels = stagedKernels(FLYDSL_SOFTMAX.engineName);
    const KernelDefinition* chosen = nullptr;
    for(const auto& kernel : kernels)
    {
        if(stringField(kernel, "dtype") == "bf16" && stringField(kernel, "out_dtype") == "f32"
           && intField(kernel, "N") == 0 && intField(kernel, "n_max") == 0)
        {
            chosen = &kernel;
        }
    }
    if(chosen == nullptr)
    {
        reportUnstaged("softmax", "this build staged no unbounded bf16-to-f32 softmax object");
        return;
    }
    SoftmaxGraphSpec spec;
    spec.sDataType = DataType::FLOAT;
    spec.pDims = {4, 512};
    const GraphFixture fixture(buildFlydslSoftmaxGraph(spec), packedArchDeviceProperties());
    const auto bound = matchesGraph(FLYDSL_SOFTMAX, fixture.context());
    ASSERT_TRUE(bound.has_value());
    const auto& handler = dispatchHandler(FLYDSL_SOFTMAX);
    const auto prepared = handler.prepare(fixture.context(), *bound, *chosen);

    Tensor<float> buffer({4, 512});
    const std::vector<hipdnnPluginDeviceBuffer_t> arguments{
        {FLYDSL_SOFTMAX_P_UID, buffer.memory().deviceData()},
        {FLYDSL_SOFTMAX_S_UID, buffer.memory().deviceData()}};
    const Handle handle;
    EXPECT_THROW(
        handler.launch(
            handle, *prepared, arguments.data(), static_cast<uint32_t>(arguments.size()), nullptr),
        hipdnn_plugin_sdk::HipdnnPluginException);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
