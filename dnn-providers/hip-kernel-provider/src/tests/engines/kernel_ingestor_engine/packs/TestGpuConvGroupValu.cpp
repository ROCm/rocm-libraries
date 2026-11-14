// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <nlohmann/json.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvGroupValuProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestGpuConvGroupValu.cpp
 * @brief CONV_GROUP_VALU_FWD / CONV_GROUP_VALU_BWD on the device, through the engines' own graph
 *        match, kernel match, prepare and launch, against hipDNN's CPU reference (fprop / dgrad
 *        accumulated in double).
 *
 * Every config of the catalog shipped for this device's architecture that serves a case runs on
 * it, in fp32, fp16 and bf16 (a layout no config ships for on the architecture is skipped):
 * groups of 4 and 8 channels (odd batch), both layouts, ragged widths, width stride 2,
 * 3-D 3x3x3 at stride 1 and 2, true convolution, height stride and dilation, an inf in the input
 * at a row's last column (only the outputs the reference reaches it with may be non-finite), and
 * a forward over an input past 2 GiB of small images (sampled).
 *
 * Tolerance: fp32 sums over the reduction, so each output is off by at most a few units of fp32
 * rounding of the sum of its terms' magnitudes, plus two units of the element type's rounding.
 */
namespace
{

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_naive_test;
using conv_naive::ConvDirection;
using hipdnn_data_sdk::utilities::Tensor;
using hipdnn_flatbuffers_sdk::data_objects::ConvMode;
using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_test_sdk::utilities::CpuFpReferenceConvolution;

using Half = hipdnn_data_sdk::types::half;
using BFloat16 = hipdnn_data_sdk::types::bfloat16;

PackSymbols packFor(ConvDirection direction)
{
    return {"",
            direction == ConvDirection::FORWARD ? "hipkernel.conv_group_valu_fwd.graph_match"
                                                : "hipkernel.conv_group_valu_bwd.graph_match",
            "",
            "hipkernel.conv_group_valu.kernel_match",
            "",
            "hipkernel.conv_group_valu.dispatch",
            "",
            "",
            ""};
}

/// The shipped catalogs' configs whose pack lists @p arch (or every arch), read from the
/// production descriptor tree beside the provider (or beside this binary).
std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition> catalog(const std::string& arch,
                                                                   const std::string& prefix)
{
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
    std::vector<std::filesystem::path> files;
    for(const auto& root : roots)
    {
        std::error_code error;
        for(auto it = std::filesystem::recursive_directory_iterator(root, error);
            !error && it != std::filesystem::recursive_directory_iterator();
            it.increment(error))
        {
            const auto name = it->path().filename().string();
            if(name.rfind(prefix, 0) == 0 && name.size() > 9
               && name.substr(name.size() - 9) == ".kdp.json")
            {
                files.push_back(it->path());
            }
        }
        if(!files.empty())
        {
            break;
        }
    }
    if(files.empty())
    {
        throw std::runtime_error("no " + prefix
                                 + " catalog beside the provider or this test binary");
    }
    std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition> kernels;
    for(const auto& file : files)
    {
        std::ifstream stream(file);
        const auto json = nlohmann::json::parse(stream);
        std::vector<std::string> archs;
        for(const auto& entry : json.at("arch"))
        {
            archs.push_back(entry.get<std::string>());
        }
        if(!hipdnn_plugin_sdk::ingestor::archSupports(archs, arch))
        {
            continue;
        }
        for(const auto& descriptor : json.at("kernelDescriptors"))
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
    }
    return kernels;
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
void reflectFilter(Tensor<T>& filter, Tensor<T>& reflected)
{
    const auto& dims = filter.dims();
    const auto* source = static_cast<const T*>(filter.rawHostData());
    auto* target = static_cast<T*>(reflected.rawHostData());
    forEachIndex(dims, [&](const std::vector<int64_t>& index) {
        auto mirror = index;
        for(size_t axis = 2; axis < dims.size(); ++axis)
        {
            mirror[axis] = dims[axis] - 1 - index[axis];
        }
        target[offsetOf(mirror, reflected.strides())] = source[offsetOf(index, filter.strides())];
    });
    reflected.markHostModified();
}

template <typename T>
void absolute(Tensor<T>& from, Tensor<T>& to)
{
    const auto* source = static_cast<const T*>(from.rawHostData());
    auto* target = static_cast<T*>(to.rawHostData());
    forEachIndex(from.dims(), [&](const std::vector<int64_t>& index) {
        const auto v = static_cast<float>(source[offsetOf(index, from.strides())]);
        target[offsetOf(index, to.strides())] = static_cast<T>(std::fabs(v));
    });
    to.markHostModified();
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

ConvShape conv(int64_t n,
               int64_t c,
               int64_t k,
               int64_t groups,
               std::vector<int64_t> input,
               std::vector<int64_t> filter,
               const std::vector<int64_t>& pad,
               std::vector<int64_t> stride,
               bool channelsLast)
{
    ConvShape shape;
    shape.n = n;
    shape.c = c;
    shape.k = k;
    shape.groups = groups;
    shape.input = std::move(input);
    shape.filter = std::move(filter);
    shape.prePadding = pad;
    shape.postPadding = pad;
    shape.stride = std::move(stride);
    shape.dilation.assign(shape.input.size(), 1);
    shape.channelsLast = channelsLast;
    return shape;
}

struct DeviceCase
{
    std::string name;
    ConvShape shape;
    ConvDirection direction = ConvDirection::FORWARD;
    bool infAtRowEnd = false;
};

std::vector<DeviceCase> deviceCases()
{
    std::vector<DeviceCase> cases;
    const auto both = [&cases](const std::string& name, const ConvShape& shape, bool inf = false) {
        cases.push_back({"Fwd" + name, shape, ConvDirection::FORWARD, inf});
        cases.push_back({"Bwd" + name, shape, ConvDirection::BACKWARD_DATA, inf});
    };
    both("Grouped4Nchw", conv(3, 16, 16, 4, {14, 14}, {3, 3}, {1, 1}, {1, 1}, false));
    both("Grouped8Nhwc", conv(3, 32, 16, 4, {9, 13}, {3, 3}, {1, 1}, {1, 1}, true));
    both("RaggedNchw", conv(2, 8, 8, 2, {13, 11}, {3, 3}, {1, 1}, {1, 1}, false));
    both("Stride2Nchw", conv(2, 16, 16, 4, {15, 17}, {3, 3}, {1, 1}, {2, 2}, false));
    both("Stride2Nhwc", conv(2, 16, 16, 4, {15, 16}, {3, 3}, {1, 1}, {2, 2}, true));
    both("ThreeD", conv(2, 8, 8, 2, {5, 6, 7}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, false));
    both("ThreeDStride2", conv(2, 8, 16, 2, {7, 8, 9}, {3, 3, 3}, {1, 1, 1}, {2, 2, 2}, false));
    auto convolution = conv(2, 16, 16, 4, {10, 12}, {3, 3}, {1, 1}, {1, 1}, false);
    convolution.mode = ConvMode::CONVOLUTION;
    both("ConvolutionMode", convolution);
    auto height = conv(2, 8, 8, 2, {17, 16}, {3, 3}, {2, 1}, {2, 1}, false);
    height.dilation = {2, 1};
    both("HeightStrideDilation", height);
    // Backward data at unit stride with a height dilation (the padding span - 1 - pad).
    auto bwdDil = conv(2, 8, 8, 2, {11, 12}, {3, 3}, {2, 1}, {1, 1}, false);
    bwdDil.dilation = {2, 1};
    cases.push_back({"BwdHeightDilation", bwdDil, ConvDirection::BACKWARD_DATA, false});
    both("InfAtRowEndNchw", conv(2, 8, 8, 2, {7, 13}, {3, 3}, {1, 1}, {1, 1}, false), true);
    both("InfAtRowEndNhwc", conv(2, 8, 8, 2, {7, 13}, {3, 3}, {1, 1}, {1, 1}, true), true);
    return cases;
}

template <typename T>
void runCase(ConvShape shape, ConvDirection direction, bool infAtRowEnd)
{
    shape.dataType = dataTypeOf<T>();
    const bool bwd = direction == ConvDirection::BACKWARD_DATA;
    const auto spec = specFor(shape, direction);
    const auto device = currentDeviceProperties();
    const GraphFixture fixture(buildGraph(spec), device);
    const auto pack = packFor(direction);
    const auto bound = matchesGraph(pack, fixture.context());
    ASSERT_TRUE(bound.has_value()) << "the engine refused a case it should serve";

    Tensor<T> x(spec.x.dims, spec.x.strides);
    Tensor<T> w(spec.w.dims, spec.w.strides);
    Tensor<T> y(spec.y.dims, spec.y.strides);
    constexpr unsigned int SEED = 4242;
    Tensor<T>& in = bwd ? y : x;
    Tensor<T>& out = bwd ? x : y;
    in.fillTensorWithRandomValues(-1.0f, 1.0f, SEED);
    w.fillTensorWithRandomValues(-1.0f, 1.0f, SEED + 1);
    if(infAtRowEnd)
    {
        const auto& d = in.dims();
        std::vector<int64_t> at(d.size(), 0);
        at[0] = d[0] - 1;
        at[1] = d[1] - 1;
        at[d.size() - 2] = d[d.size() - 2] / 2;
        at[d.size() - 1] = d[d.size() - 1] - 1;
        static_cast<T*>(in.rawHostData())[offsetOf(at, in.strides())]
            = static_cast<T>(std::numeric_limits<float>::infinity());
        in.markHostModified();
    }
    const bool reflect = shape.mode == ConvMode::CONVOLUTION;
    Tensor<T> reflectedW(w.dims(), w.strides());
    if(reflect)
    {
        reflectFilter(w, reflectedW);
    }
    Tensor<T> aIn(in.dims(), in.strides());
    Tensor<T> aW(w.dims(), w.strides());
    absolute(in, aIn);
    absolute(reflect ? reflectedW : w, aW);
    Tensor<T> reference(out.dims(), out.strides());
    Tensor<T> magnitude(out.dims(), out.strides());
    if(bwd)
    {
        CpuFpReferenceConvolution::dgrad<T, T, T, double>(reference,
                                                          reflect ? reflectedW : w,
                                                          y,
                                                          shape.stride,
                                                          shape.dilation,
                                                          shape.prePadding,
                                                          shape.postPadding);
        CpuFpReferenceConvolution::dgrad<T, T, T, double>(
            magnitude, aW, aIn, shape.stride, shape.dilation, shape.prePadding, shape.postPadding);
    }
    else
    {
        CpuFpReferenceConvolution::fprop<T, T, T, double>(x,
                                                          reflect ? reflectedW : w,
                                                          reference,
                                                          shape.stride,
                                                          shape.dilation,
                                                          shape.prePadding,
                                                          shape.postPadding);
        CpuFpReferenceConvolution::fprop<T, T, T, double>(
            aIn, aW, magnitude, shape.stride, shape.dilation, shape.prePadding, shape.postPadding);
    }
    const auto* expected = static_cast<const T*>(reference.rawHostData());
    const auto* bounds = static_cast<const T*>(magnitude.rawHostData());

    const auto kernels
        = catalog(device.gcnArchName, bwd ? "conv_group_valu_bwd_" : "conv_group_valu_fwd_");
    const std::string layout = shape.channelsLast ? "nhwc" : "nchw";
    if(std::none_of(kernels.begin(), kernels.end(), [&](const auto& kernel) {
           return kernel.getStringMetadata("layout") == layout;
       }))
    {
        GTEST_SKIP() << "no " << layout << " config ships for " << device.gcnArchName;
    }
    const auto& handler = dispatchHandler(pack);
    int ran = 0;
    for(const auto& kernel : kernels)
    {
        if(!matchesKernel(pack, fixture.context(), kernel, *bound))
        {
            continue;
        }
        SCOPED_TRACE(kernel.name);
        ++ran;
        const auto prepared = handler.prepare(fixture.context(), *bound, kernel);
        ASSERT_NE(prepared, nullptr);
        const size_t workspaceBytes = handler.workspaceBytes(fixture.context(), *bound, kernel);
        ASSERT_GT(workspaceBytes, 0u);
        void* workspace = nullptr;
        ASSERT_EQ(hipMalloc(&workspace, workspaceBytes), hipSuccess);
        out.fillWithSentinelValue();
        const std::array<hipdnnPluginDeviceBuffer_t, 3> buffers{
            hipdnnPluginDeviceBuffer_t{X_UID, x.rawDeviceData()},
            hipdnnPluginDeviceBuffer_t{W_UID, w.rawDeviceData()},
            hipdnnPluginDeviceBuffer_t{Y_UID, y.rawDeviceData()}};
        const Handle handle;
        handler.launch(
            handle, *prepared, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace);
        ASSERT_EQ(hipStreamSynchronize(handle.getStream()), hipSuccess);
        ASSERT_EQ(hipFree(workspace), hipSuccess);
        out.markDeviceModified();
        const auto* actual = static_cast<const T*>(out.rawHostData());
        int64_t mismatches = 0;
        int64_t nonFinite = 0;
        forEachIndex(out.dims(), [&](const std::vector<int64_t>& index) {
            const auto got = static_cast<float>(actual[offsetOf(index, out.strides())]);
            const auto want = static_cast<float>(expected[offsetOf(index, reference.strides())]);
            const auto mag = static_cast<float>(bounds[offsetOf(index, magnitude.strides())]);
            if(!std::isfinite(want))
            {
                // An output the inf reaches: it must not come back finite.
                ++nonFinite;
                if(std::isfinite(got) && ++mismatches <= 5)
                {
                    ADD_FAILURE() << "at linear index " << offsetOf(index, out.strides())
                                  << ": got " << got << ", want " << want;
                }
                return;
            }
            const float tolerance
                = 8.0f * 1.2e-7f * mag + 2.0f * storageEpsilon<T>() * std::fabs(want) + 1e-6f;
            if(!(std::fabs(got - want) <= tolerance) && ++mismatches <= 5)
            {
                ADD_FAILURE() << "at linear index " << offsetOf(index, out.strides()) << ": got "
                              << got << ", want " << want << ", tolerance " << tolerance;
            }
        });
        EXPECT_EQ(mismatches, 0);
        if(infAtRowEnd)
        {
            EXPECT_GT(nonFinite, 0) << "the inf reached no output: the case does not test it";
        }
    }
    EXPECT_GE(ran, 1) << "no catalog config serves this case";
}

class TestGpuConvGroupValu : public ::testing::TestWithParam<DeviceCase>
{
};

TEST_P(TestGpuConvGroupValu, Float)
{
    SKIP_IF_NO_DEVICES();
    runCase<float>(GetParam().shape, GetParam().direction, GetParam().infAtRowEnd);
}

TEST_P(TestGpuConvGroupValu, Half)
{
    SKIP_IF_NO_DEVICES();
    runCase<Half>(GetParam().shape, GetParam().direction, GetParam().infAtRowEnd);
}

TEST_P(TestGpuConvGroupValu, BFloat16)
{
    SKIP_IF_NO_DEVICES();
    runCase<BFloat16>(GetParam().shape, GetParam().direction, GetParam().infAtRowEnd);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvGroupValu,
                         ::testing::ValuesIn(deviceCases()),
                         [](const auto& info) { return info.param.name; });

/// Forward over x past 2 GiB of small images (4100 of 16 x 128 x 128 halves, 3x3 stride 2 into 4
/// channels): every pass's 64-bit image base, 32-bit offsets within it. Sampled outputs checked
/// on the host.
TEST(TestGpuConvGroupValuLarge, HalfForwardPast2GiBSampled)
{
    SKIP_IF_NO_DEVICES();
    const auto device = currentDeviceProperties();
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    ASSERT_EQ(hipMemGetInfo(&freeBytes, &totalBytes), hipSuccess);
    if(freeBytes < (size_t{3} << 30))
    {
        GTEST_SKIP() << "needs 3 GiB of device memory";
    }
    auto shape = conv(4100, 16, 4, 1, {128, 128}, {3, 3}, {1, 1}, {2, 2}, false);
    shape.dataType = DataType::HALF;
    const auto spec = specFor(shape, ConvDirection::FORWARD);
    const GraphFixture fixture(buildGraph(spec), device);
    const auto pack = packFor(ConvDirection::FORWARD);
    const auto bound = matchesGraph(pack, fixture.context());
    ASSERT_TRUE(bound.has_value());

    const int64_t n = 4100;
    const int64_t c = 16;
    const int64_t k = 4;
    const int64_t h = 128;
    const int64_t wd = 128;
    const int64_t ho = 64;
    const int64_t wo = 64;
    const int64_t xCount = n * c * h * wd;
    const int64_t yCount = n * k * ho * wo;
    // Exact in fp16 and in every fp32 partial sum: x integers in [-3, 3], w in [-2, 1] / 8.
    const auto xValue = [](int64_t i) {
        return static_cast<float>(static_cast<int>(((i * 2654435761LL) >> 7) & 7) - 3);
    };
    const auto wValue = [](int64_t i) {
        return static_cast<float>(static_cast<int>((i * 7 + 3) & 3) - 2) / 8.0f;
    };
    std::vector<Half> hx(static_cast<size_t>(xCount));
    for(int64_t i = 0; i < xCount; ++i)
    {
        hx[static_cast<size_t>(i)] = static_cast<Half>(xValue(i));
    }
    std::vector<Half> hw(static_cast<size_t>(k * c * 9));
    for(size_t i = 0; i < hw.size(); ++i)
    {
        hw[i] = static_cast<Half>(wValue(static_cast<int64_t>(i)));
    }
    void* dx = nullptr;
    void* dy = nullptr;
    void* dw = nullptr;
    ASSERT_EQ(hipMalloc(&dx, static_cast<size_t>(xCount) * 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&dy, static_cast<size_t>(yCount) * 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&dw, hw.size() * 2), hipSuccess);
    ASSERT_EQ(hipMemcpy(dx, hx.data(), static_cast<size_t>(xCount) * 2, hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dw, hw.data(), hw.size() * 2, hipMemcpyHostToDevice), hipSuccess);
    hx.clear();
    hx.shrink_to_fit();

    const auto& handler = dispatchHandler(pack);
    int ran = 0;
    for(const auto& kernel : catalog(device.gcnArchName, "conv_group_valu_fwd_"))
    {
        if(!matchesKernel(pack, fixture.context(), kernel, *bound))
        {
            continue;
        }
        SCOPED_TRACE(kernel.name);
        ++ran;
        const auto prepared = handler.prepare(fixture.context(), *bound, kernel);
        const size_t workspaceBytes = handler.workspaceBytes(fixture.context(), *bound, kernel);
        void* workspace = nullptr;
        ASSERT_EQ(hipMalloc(&workspace, workspaceBytes), hipSuccess);
        ASSERT_EQ(hipMemset(dy, 0xff, static_cast<size_t>(yCount) * 2), hipSuccess);
        const std::array<hipdnnPluginDeviceBuffer_t, 3> buffers{
            hipdnnPluginDeviceBuffer_t{X_UID, dx},
            hipdnnPluginDeviceBuffer_t{W_UID, dw},
            hipdnnPluginDeviceBuffer_t{Y_UID, dy}};
        const Handle handle;
        handler.launch(handle, *prepared, buffers.data(), 3, workspace);
        ASSERT_EQ(hipStreamSynchronize(handle.getStream()), hipSuccess);
        ASSERT_EQ(hipFree(workspace), hipSuccess);
        for(const auto& [in, kk, oy, ox] : std::vector<std::array<int64_t, 4>>{
                {0, 0, 0, 0}, {4099, 3, 63, 63}, {2050, 1, 17, 40}, {3333, 2, 63, 0}})
        {
            double want = 0.0;
            for(int64_t cc = 0; cc < c; ++cc)
            {
                for(int64_t r = 0; r < 3; ++r)
                {
                    const int64_t iy = oy * 2 - 1 + r;
                    if(iy < 0 || iy >= h)
                    {
                        continue;
                    }
                    for(int64_t s = 0; s < 3; ++s)
                    {
                        const int64_t ix = ox * 2 - 1 + s;
                        if(ix < 0 || ix >= wd)
                        {
                            continue;
                        }
                        want += static_cast<double>(xValue(((in * c + cc) * h + iy) * wd + ix))
                                * static_cast<double>(static_cast<float>(
                                    hw[static_cast<size_t>(((kk * c + cc) * 3 + r) * 3 + s)]));
                    }
                }
            }
            Half got;
            ASSERT_EQ(hipMemcpy(&got,
                                static_cast<const Half*>(dy) + ((in * k + kk) * ho + oy) * wo + ox,
                                2,
                                hipMemcpyDeviceToHost),
                      hipSuccess);
            EXPECT_NEAR(static_cast<float>(got), want, 2.0 * 9.8e-4 * std::fabs(want) + 1e-3)
                << "y[" << in << "][" << kk << "][" << oy << "][" << ox << "]";
        }
    }
    EXPECT_GE(ran, 1);
    ASSERT_EQ(hipFree(dx), hipSuccess);
    ASSERT_EQ(hipFree(dy), hipSuccess);
    ASSERT_EQ(hipFree(dw), hipSuccess);
}

} // namespace

#endif
