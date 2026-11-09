// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <nlohmann/json.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvIgemmNchwProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestGpuConvIgemmNchw.cpp
 * @brief The channels-first backward-data implicit-GEMM kernel (CONV_IGEMM_NCHW_BWD) on the
 *        device, through the engine's own graph match, kernel match, prepare and launch, against
 *        hipDNN's CPU reference convolution; and host checks of the shipped catalog.
 *
 * Every config of the installed catalog that the kernel match admits on this device runs on
 * every case, in fp32, fp16 and bf16. The cases cover every shipped stride (2, 3, 4, 5, 6, 8),
 * filters that fill their last tap row and filters that do not, padding below, at and past the
 * stride (every pad mod stride class the tiles distinguish), 1-D, 2-D and 3-D graphs with
 * height and depth rows, channel counts no tile divides, and widths spanning several tiles.
 */
namespace
{

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_naive_test;
using conv_naive::ConvDirection;
using hipdnn_data_sdk::utilities::Tensor;
using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_test_sdk::utilities::CpuFpReferenceConvolution;

using Half = hipdnn_data_sdk::types::half;
using BFloat16 = hipdnn_data_sdk::types::bfloat16;

constexpr PackSymbols PACK{"",
                           "hipkernel.conv_igemm_nchw_bwd.graph_match",
                           "",
                           "hipkernel.conv_igemm_nchw.kernel_match",
                           "",
                           "hipkernel.conv_igemm_nchw.dispatch",
                           "",
                           "",
                           ""};

/// The shipped catalog, as kernel definitions: read from the production descriptor tree beside
/// the provider (the unit tests' own descriptor root holds only test sets), so these cases run
/// exactly the configs that ship.
const std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition>& catalog()
{
    static const auto s_catalog = [] {
        std::vector<std::filesystem::path> roots;
        {
            const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter unset(
                "HIPDNN_DESCRIPTOR_DIR", "");
            roots.push_back(descriptorSearchDirectory());
        }
        std::error_code exeError;
        for(auto dir = std::filesystem::read_symlink("/proc/self/exe", exeError).parent_path();
            !exeError && !dir.empty() && dir != dir.root_path();
            dir = dir.parent_path())
        {
            roots.push_back(dir / "lib" / "hipdnn_plugins" / "engines"
                            / HIPKERNELPROVIDER_DESCRIPTOR_SUBDIR);
        }
        const std::string file = "conv_igemm_nchw_bwd.kdp.json";
        std::filesystem::path found;
        for(const auto& root : roots)
        {
            std::error_code error;
            for(auto it = std::filesystem::recursive_directory_iterator(root, error);
                !error && found.empty() && it != std::filesystem::recursive_directory_iterator();
                it.increment(error))
            {
                if(it->path().filename() == file)
                {
                    found = it->path();
                }
            }
            if(!found.empty())
            {
                break;
            }
        }
        if(found.empty())
        {
            throw std::runtime_error("no " + file + " beside the provider or this test binary");
        }
        std::ifstream stream(found);
        const auto pack = nlohmann::json::parse(stream);
        std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition> kernels;
        for(const auto& descriptor : pack.at("kernelDescriptors"))
        {
            hipdnn_plugin_sdk::ingestor::KernelDefinition kernel;
            kernel.name = descriptor.at("name").get<std::string>();
            kernel.source.kind = hipdnn_plugin_sdk::ingestor::KernelSourceKind::EMBEDDED_SOURCE;
            kernel.source.sourceFile
                = descriptor.at("kernel_source").at("source_file").get<std::string>();
            kernel.source.entryPoint
                = descriptor.at("kernel_source").at("entry_point").get<std::string>();
            for(const auto& [key, value] : descriptor.at("metadata").items())
            {
                if(value.is_string())
                {
                    kernel.metadata[key] = value.get<std::string>();
                }
                else
                {
                    kernel.metadata[key] = value.get<int64_t>();
                }
            }
            kernel.priority = descriptor.at("priority").get<int64_t>();
            kernels.push_back(std::move(kernel));
        }
        return kernels;
    }();
    return s_catalog;
}

void forEachIndex(const std::vector<int64_t>& dims,
                  const std::function<void(const std::vector<int64_t>&)>& visit)
{
    std::vector<int64_t> index(dims.size(), 0);
    while(true)
    {
        visit(index);
        size_t axis = dims.size();
        while(axis > 0)
        {
            --axis;
            if(++index[axis] < dims[axis])
            {
                break;
            }
            index[axis] = 0;
            if(axis == 0)
            {
                return;
            }
        }
    }
}

int64_t offsetOf(const std::vector<int64_t>& index, const std::vector<int64_t>& strides)
{
    int64_t offset = 0;
    for(size_t axis = 0; axis < index.size(); ++axis)
    {
        offset += index[axis] * strides[axis];
    }
    return offset;
}

template <typename T>
float storageEpsilon()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return 1.2e-7f;
    }
    else if constexpr(std::is_same_v<T, Half>)
    {
        return 9.8e-4f;
    }
    else
    {
        return 7.9e-3f;
    }
}

template <typename T>
DataType dataTypeOf()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return DataType::FLOAT;
    }
    else if constexpr(std::is_same_v<T, Half>)
    {
        return DataType::HALF;
    }
    else
    {
        return DataType::BFLOAT16;
    }
}

struct DeviceCase
{
    std::string name;
    ConvShape shape;
};

/// A ConvTranspose1d-shaped backward data along the width: @p rows the identity axes' extents
/// (empty for a 1-D graph), filter S, stride, padding (pre, post).
ConvShape upsample(int64_t c,
                   int64_t k,
                   std::vector<int64_t> rows,
                   int64_t width,
                   int64_t filter,
                   int64_t stride,
                   int64_t pre,
                   int64_t post)
{
    ConvShape shape;
    shape.n = 2;
    shape.c = c;
    shape.k = k;
    shape.groups = 1;
    shape.input = rows;
    shape.input.push_back(width);
    shape.filter.assign(rows.size(), 1);
    shape.filter.push_back(filter);
    shape.prePadding.assign(rows.size(), 0);
    shape.prePadding.push_back(pre);
    shape.postPadding.assign(rows.size(), 0);
    shape.postPadding.push_back(post);
    shape.stride.assign(rows.size(), 1);
    shape.stride.push_back(stride);
    shape.dilation.assign(rows.size() + 1, 1);
    return shape;
}

// Channel counts are not multiples of the channel tiles; widths are not multiples of the dx
// tiles, and the 600-wide case spans several of them.
std::vector<DeviceCase> deviceCases()
{
    return {
        {"OneDK8S4", upsample(40, 72, {}, 37, 8, 4, 2, 2)},
        {"K4S2", upsample(24, 40, {1}, 61, 4, 2, 1, 1)},
        {"K3S2Partial", upsample(16, 32, {1}, 40, 3, 2, 1, 1)},
        {"K7S3", upsample(21, 19, {1}, 50, 7, 3, 2, 2)},
        {"K8S4RowsPadPastStride", upsample(24, 16, {3}, 50, 8, 4, 6, 1)},
        {"K8S4Wide", upsample(36, 48, {1}, 600, 8, 4, 2, 2)},
        {"K9S5", upsample(9, 11, {1}, 33, 9, 5, 2, 2)},
        {"K10S5Pad0", upsample(16, 24, {1}, 64, 10, 5, 0, 0)},
        {"K12S6", upsample(20, 24, {1}, 90, 12, 6, 3, 3)},
        {"K16S8", upsample(24, 20, {1}, 70, 16, 8, 4, 4)},
        {"K13S8PadAtStride", upsample(18, 9, {1}, 77, 13, 8, 8, 0)},
        {"ThreeDRowsK4S2", upsample(16, 24, {2, 2}, 30, 4, 2, 1, 1)},
        {"Channels70K7S3", upsample(70, 100, {1}, 90, 7, 3, 2, 2)},
        // dy one wide (hipconv #15149 review: a stride normalized for an output extent of 1):
        // dx 9 from a K8 S4 filter, its last row reached by no tap.
        {"K8S4OutputOne", upsample(24, 16, {1}, 9, 8, 4, 0, 0)},
        {"RowsK4S2OutputOne", upsample(16, 24, {3}, 5, 4, 2, 0, 0)},
    };
}

template <typename T>
void runCase(ConvShape shape)
{
    shape.dataType = dataTypeOf<T>();
    const auto spec = specFor(shape, ConvDirection::BACKWARD_DATA);
    const GraphFixture fixture(buildGraph(spec), currentDeviceProperties());
    const auto bound = matchesGraph(PACK, fixture.context());
    ASSERT_TRUE(bound.has_value()) << "the engine refused a case it should serve";

    Tensor<T> x(spec.x.dims, spec.x.strides);
    Tensor<T> w(spec.w.dims, spec.w.strides);
    Tensor<T> y(spec.y.dims, spec.y.strides);
    constexpr unsigned int SEED = 4321;
    y.fillTensorWithRandomValues(-1.0f, 1.0f, SEED);
    w.fillTensorWithRandomValues(-1.0f, 1.0f, SEED + 1);
    Tensor<T> reference(x.dims(), x.strides());
    CpuFpReferenceConvolution::dgrad(
        reference, w, y, shape.stride, shape.dilation, shape.prePadding, shape.postPadding);
    // Each dx element sums (taps of its phase x output channels) products of values in [-1, 1].
    int64_t reductionLength = shape.k;
    for(const auto extent : shape.filter)
    {
        reductionLength *= extent;
    }
    const float accumulationBound = 2e-6f * static_cast<float>(reductionLength);

    const auto& handler = dispatchHandler(PACK);
    int ran = 0;
    for(const auto& kernel : catalog())
    {
        if(!matchesKernel(PACK, fixture.context(), kernel, *bound))
        {
            continue;
        }
        SCOPED_TRACE(kernel.name);
        ++ran;
        const auto prepared = handler.prepare(fixture.context(), *bound, kernel);
        ASSERT_NE(prepared, nullptr);
        EXPECT_EQ(handler.workspaceBytes(fixture.context(), *bound, kernel), size_t{0});

        x.fillWithSentinelValue();
        const std::array<hipdnnPluginDeviceBuffer_t, 3> buffers{
            hipdnnPluginDeviceBuffer_t{X_UID, x.rawDeviceData()},
            hipdnnPluginDeviceBuffer_t{W_UID, w.rawDeviceData()},
            hipdnnPluginDeviceBuffer_t{Y_UID, y.rawDeviceData()}};
        const Handle handle;
        handler.launch(
            handle, *prepared, buffers.data(), static_cast<uint32_t>(buffers.size()), nullptr);
        ASSERT_EQ(hipStreamSynchronize(handle.getStream()), hipSuccess);
        x.markDeviceModified();

        const auto* actual = static_cast<const T*>(x.rawHostData());
        const auto* expected = static_cast<const T*>(reference.rawHostData());
        int64_t mismatches = 0;
        forEachIndex(x.dims(), [&](const std::vector<int64_t>& index) {
            const float got = static_cast<float>(actual[offsetOf(index, x.strides())]);
            const float want = static_cast<float>(expected[offsetOf(index, reference.strides())]);
            const float tolerance
                = accumulationBound + 2.0f * storageEpsilon<T>() * (std::fabs(want) + 1e-3f);
            if(!(std::fabs(got - want) <= tolerance) && ++mismatches <= 5)
            {
                ADD_FAILURE() << "at linear index " << offsetOf(index, x.strides()) << ": got "
                              << got << ", want " << want << ", tolerance " << tolerance;
            }
        });
        EXPECT_EQ(mismatches, 0);
    }
    EXPECT_GE(ran, 1) << "no catalog config serves this case";
}

class TestGpuConvIgemmNchw : public ::testing::TestWithParam<DeviceCase>
{
};

TEST_P(TestGpuConvIgemmNchw, Float)
{
    SKIP_IF_NO_DEVICES();
    runCase<float>(GetParam().shape);
}

TEST_P(TestGpuConvIgemmNchw, Half)
{
    SKIP_IF_NO_DEVICES();
    runCase<Half>(GetParam().shape);
}

TEST_P(TestGpuConvIgemmNchw, BFloat16)
{
    SKIP_IF_NO_DEVICES();
    runCase<BFloat16>(GetParam().shape);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvIgemmNchw,
                         ::testing::ValuesIn(deviceCases()),
                         [](const auto& info) { return info.param.name; });

// ---- Host-only: the shipped catalog ----

// The launch is sized from the problem alone, so a device that reports no units (hipconv #15149
// review: zero CUs threw) is served exactly as a counted one, without throwing.
TEST(TestConvIgemmNchwCatalog, AnUncountedDeviceIsServedAsACountedOne)
{
    auto shape = upsample(24, 40, {1}, 61, 4, 2, 1, 1);
    shape.dataType = DataType::HALF;
    for(const char* arch : {"gfx942:sramecc+:xnack-", "gfx1100", "gfx1201"})
    {
        auto counted = testDeviceProperties();
        counted.gcnArchName = arch;
        counted.multiProcessorCount = 64;
        auto uncounted = counted;
        uncounted.multiProcessorCount = 0;
        const auto spec = specFor(shape, ConvDirection::BACKWARD_DATA);
        const GraphFixture withUnits(buildGraph(spec), counted);
        const GraphFixture withoutUnits(buildGraph(spec), uncounted);
        const auto bound = matchesGraph(PACK, withUnits.context());
        ASSERT_TRUE(bound.has_value());
        ASSERT_TRUE(matchesGraph(PACK, withoutUnits.context()).has_value());
        int served = 0;
        for(const auto& kernel : catalog())
        {
            const bool admitted = matchesKernel(PACK, withUnits.context(), kernel, *bound);
            served += admitted ? 1 : 0;
            EXPECT_EQ(matchesKernel(PACK, withoutUnits.context(), kernel, *bound), admitted)
                << arch << " " << kernel.name;
        }
        EXPECT_GT(served, 0) << arch;
    }
}

std::string_view stringOf(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                          const char* field)
{
    return kernel.getStringMetadata(field);
}

TEST(TestConvIgemmNchwCatalog, KnobsTellApartEveryConfigOfAGraph)
{
    // Candidate enumeration identifies a config by its integer knobs: within one dtype (what
    // one graph can admit) no two configs may share every knob.
    static constexpr std::array<const char*, 13> KNOBS{"atom_id",
                                                       "bmq",
                                                       "bn",
                                                       "bk",
                                                       "wm",
                                                       "wn",
                                                       "wp",
                                                       "tq",
                                                       "tc",
                                                       "vec",
                                                       "stride",
                                                       "taps",
                                                       "stages"};
    std::set<std::vector<int64_t>> seen[3];
    for(const auto& kernel : catalog())
    {
        SCOPED_TRACE(kernel.name);
        const auto dtype = stringOf(kernel, "dtype");
        const int slot = dtype == "FLOAT" ? 0 : dtype == "HALF" ? 1 : 2;
        std::vector<int64_t> tuple;
        for(const char* knob : KNOBS)
        {
            tuple.push_back(kernel.getIntMetadata(knob));
        }
        EXPECT_TRUE(seen[slot].insert(tuple).second);
    }
}

TEST(TestConvIgemmNchwCatalog, AtomIdsNameTheirAtomsAndEveryConfigFitsLds)
{
    std::set<int64_t> strides;
    for(const auto& kernel : catalog())
    {
        SCOPED_TRACE(kernel.name);
        conv_igemm_nchw::NchwConfig config;
        config.atom = std::string(stringOf(kernel, "atom"));
        config.atomId = kernel.getIntMetadata("atom_id");
        config.bmq = kernel.getIntMetadata("bmq");
        config.bn = kernel.getIntMetadata("bn");
        config.bk = kernel.getIntMetadata("bk");
        config.vec = kernel.getIntMetadata("vec");
        config.stride = kernel.getIntMetadata("stride");
        config.taps = kernel.getIntMetadata("taps");
        config.stages = kernel.getIntMetadata("stages");
        EXPECT_EQ(conv_igemm_nchw::atomIdOf(config.atom), config.atomId);
        EXPECT_EQ(stringOf(kernel, "direction"), "BWD");
        const int elementBytes = stringOf(kernel, "dtype") == "FLOAT" ? 4 : 2;
        EXPECT_LE(conv_igemm_nchw::ldsBytesOf(config, elementBytes), 65536);
        strides.insert(config.stride);
    }
    // Every stride of the vocoder ConvTranspose1d family ships.
    EXPECT_EQ(strides, (std::set<int64_t>{2, 3, 4, 5, 6, 8}));
}

} // namespace

#endif
