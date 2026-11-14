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
#include "engines/kernel_ingestor_engine/packs/ConvGroupValuWrwProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestGpuConvGroupValuWrw.cpp
 * @brief CONV_GROUP_VALU_WRW on the device, through the engine's own graph match, kernel match,
 *        prepare and launch, against hipDNN's CPU reference wgrad accumulated in double.
 *
 * Every config of the catalog shipped for this device's architecture that serves a case runs on
 * it, in fp32, fp16 and bf16 (a layout no config ships for on the architecture is skipped): grouped layers of 4 channels a group (odd batch), both layouts,
 * dense layers with ragged rows (widths no multiple of a lane's columns), width stride 2, 1x1
 * (2-D and 1-D), 5x5, true convolution, height stride / dilation / padding, several splits folded in order, and a
 * tensor past 2 GiB of small images (sampled).
 *
 * Tolerance: the sum is fp32 over the images and positions, so each element is off by at most a
 * few units of fp32 rounding of the sum of its terms' magnitudes, plus two units of the element
 * type's rounding of the output.
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

PackSymbols pack()
{
    return {"",
            "hipkernel.conv_group_valu_wrw.graph_match",
            "",
            "hipkernel.conv_group_valu_wrw.kernel_match",
            "",
            "hipkernel.conv_group_valu_wrw.dispatch",
            "",
            "",
            ""};
}

/// The shipped catalogs' configs whose pack lists @p arch (or every arch), read from the
/// production descriptor tree beside the provider (or beside this binary).
std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition> catalog(const std::string& arch)
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
            if(name.rfind("conv_group_valu_wrw_", 0) == 0 && name.size() > 9
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
        throw std::runtime_error("no conv_group_valu_wrw catalog beside the provider or this "
                                 "test binary");
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

struct DeviceCase
{
    std::string name;
    ConvShape shape;
};

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

std::vector<DeviceCase> deviceCases()
{
    std::vector<DeviceCase> cases;
    // ResNeXt-style groups of 4 (odd batch), both layouts.
    cases.push_back({"Grouped4Nchw", conv(3, 16, 16, 4, {14, 14}, {3, 3}, {1, 1}, {1, 1}, false)});
    cases.push_back({"Grouped4Nhwc", conv(3, 16, 16, 4, {14, 14}, {3, 3}, {1, 1}, {1, 1}, true)});
    // Dense, widths no multiple of a lane's columns (ragged rows), rows of 13 and 11.
    cases.push_back({"DenseRaggedNchw", conv(2, 8, 8, 1, {13, 11}, {3, 3}, {1, 1}, {1, 1}, false)});
    cases.push_back({"DenseRaggedNhwc", conv(2, 8, 8, 1, {13, 11}, {3, 3}, {1, 1}, {1, 1}, true)});
    // Width stride 2 (odd input width), groups of 4.
    cases.push_back({"Stride2Nchw", conv(2, 8, 8, 2, {15, 17}, {3, 3}, {1, 1}, {2, 2}, false)});
    cases.push_back({"Stride2Nhwc", conv(2, 8, 8, 2, {15, 16}, {3, 3}, {1, 1}, {2, 2}, true)});
    // 1x1 (2-D and 1-D) and 5x5.
    cases.push_back({"PointwiseNchw", conv(3, 16, 16, 1, {9, 10}, {1, 1}, {0, 0}, {1, 1}, false)});
    cases.push_back({"PointwiseNhwc", conv(3, 16, 16, 1, {9, 10}, {1, 1}, {0, 0}, {1, 1}, true)});
    cases.push_back({"OneDPointwise", conv(5, 16, 16, 1, {37}, {1}, {0}, {1}, false)});
    cases.push_back({"FiveByFiveNchw", conv(2, 8, 8, 1, {12, 16}, {5, 5}, {2, 2}, {1, 1}, false)});
    // True convolution, grouped.
    auto convolution = conv(2, 16, 16, 4, {10, 12}, {3, 3}, {1, 1}, {1, 1}, false);
    convolution.mode = ConvMode::CONVOLUTION;
    cases.push_back({"ConvolutionMode", convolution});
    // Height stride, dilation and padding are runtime.
    auto height = conv(2, 8, 8, 2, {17, 16}, {3, 3}, {2, 1}, {2, 1}, false);
    height.dilation = {2, 1};
    cases.push_back({"HeightStrideDilation", height});
    // Enough rows for several splits, folded in order.
    cases.push_back({"Splits", conv(33, 8, 8, 2, {20, 16}, {3, 3}, {1, 1}, {1, 1}, false)});
    // 3-D: a tile per depth tap, rows over the output depth; depth padding, stride, dilation.
    cases.push_back(
        {"ThreeDNchw", conv(2, 8, 8, 2, {5, 6, 7}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, false)});
    cases.push_back(
        {"ThreeDNhwc", conv(2, 8, 8, 2, {5, 6, 8}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, true)});
    cases.push_back(
        {"ThreeDStride2", conv(3, 8, 16, 2, {7, 8, 9}, {3, 3, 3}, {1, 1, 1}, {2, 2, 2}, false)});
    auto depthDilation = conv(2, 8, 8, 2, {9, 6, 8}, {3, 3, 3}, {2, 1, 1}, {1, 1, 1}, false);
    depthDilation.dilation = {2, 1, 1};
    cases.push_back({"ThreeDDepthDilation", depthDilation});
    auto convolution3d = conv(2, 8, 8, 2, {4, 6, 8}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, false);
    convolution3d.mode = ConvMode::CONVOLUTION;
    cases.push_back({"ThreeDConvolutionMode", convolution3d});
    return cases;
}

template <typename T>
/// @p infAtRowEnd: x holds +inf at the last column of one row of one image and channel. With a
/// width that is no multiple of a lane's columns, that column is also under output columns past
/// the row; the reference never multiplies it by them, so the taps it does not reach stay finite.
void runCase(ConvShape shape, bool infAtRowEnd = false)
{
    shape.dataType = dataTypeOf<T>();
    const auto spec = specFor(shape, ConvDirection::BACKWARD_WEIGHTS);
    const auto device = currentDeviceProperties();
    const GraphFixture fixture(buildGraph(spec), device);
    const auto bound = matchesGraph(pack(), fixture.context());
    ASSERT_TRUE(bound.has_value()) << "the engine refused a case it should serve";

    Tensor<T> x(spec.x.dims, spec.x.strides);
    Tensor<T> w(spec.w.dims, spec.w.strides);
    Tensor<T> y(spec.y.dims, spec.y.strides);
    constexpr unsigned int SEED = 9876;
    x.fillTensorWithRandomValues(-1.0f, 1.0f, SEED);
    y.fillTensorWithRandomValues(-1.0f, 1.0f, SEED + 1);
    if(infAtRowEnd)
    {
        const auto& d = spec.x.dims;
        const std::vector<int64_t> at{d[0] - 1, d[1] - 1, d[2] / 2, d[3] - 1};
        static_cast<T*>(x.rawHostData())[offsetOf(at, x.strides())]
            = static_cast<T>(std::numeric_limits<float>::infinity());
        x.markHostModified();
    }
    Tensor<T> reference(w.dims(), w.strides());
    Tensor<T> magnitude(w.dims(), w.strides());
    Tensor<T> ax(x.dims(), x.strides());
    Tensor<T> ay(y.dims(), y.strides());
    absolute(x, ax);
    absolute(y, ay);
    const bool reflect = shape.mode == ConvMode::CONVOLUTION;
    const auto wgrad = [&](Tensor<T>& xs, Tensor<T>& ys, Tensor<T>& out) {
        if(reflect)
        {
            // The graph's filter is applied reflected, so its gradient is the reflection.
            Tensor<T> dw(w.dims(), w.strides());
            CpuFpReferenceConvolution::wgrad<T, T, T, double>(
                xs, dw, ys, shape.stride, shape.dilation, shape.prePadding, shape.postPadding);
            reflectFilter(dw, out);
        }
        else
        {
            CpuFpReferenceConvolution::wgrad<T, T, T, double>(
                xs, out, ys, shape.stride, shape.dilation, shape.prePadding, shape.postPadding);
        }
    };
    wgrad(x, y, reference);
    wgrad(ax, ay, magnitude);
    const auto* expected = static_cast<const T*>(reference.rawHostData());
    const auto* bounds = static_cast<const T*>(magnitude.rawHostData());

    const auto kernels = catalog(device.gcnArchName);
    const std::string layout = shape.channelsLast ? "nhwc" : "nchw";
    if(std::none_of(kernels.begin(), kernels.end(), [&](const auto& kernel) {
           return kernel.getStringMetadata("layout") == layout;
       }))
    {
        GTEST_SKIP() << "no " << layout << " config ships for " << device.gcnArchName;
    }
    const auto& handler = dispatchHandler(pack());
    int ran = 0;
    for(const auto& kernel : kernels)
    {
        if(!matchesKernel(pack(), fixture.context(), kernel, *bound))
        {
            continue;
        }
        SCOPED_TRACE(kernel.name);
        ++ran;
        const auto prepared = handler.prepare(fixture.context(), *bound, kernel);
        ASSERT_NE(prepared, nullptr);
        const size_t workspaceBytes = handler.workspaceBytes(fixture.context(), *bound, kernel);
        void* workspace = nullptr;
        if(workspaceBytes > 0)
        {
            ASSERT_EQ(hipMalloc(&workspace, workspaceBytes), hipSuccess);
        }
        w.fillWithSentinelValue();
        const std::array<hipdnnPluginDeviceBuffer_t, 3> buffers{
            hipdnnPluginDeviceBuffer_t{X_UID, x.rawDeviceData()},
            hipdnnPluginDeviceBuffer_t{W_UID, w.rawDeviceData()},
            hipdnnPluginDeviceBuffer_t{Y_UID, y.rawDeviceData()}};
        const Handle handle;
        handler.launch(
            handle, *prepared, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace);
        ASSERT_EQ(hipStreamSynchronize(handle.getStream()), hipSuccess);
        if(workspace != nullptr)
        {
            ASSERT_EQ(hipFree(workspace), hipSuccess);
        }
        w.markDeviceModified();
        const auto* actual = static_cast<const T*>(w.rawHostData());
        int64_t mismatches = 0;
        int64_t nonFinite = 0;
        forEachIndex(w.dims(), [&](const std::vector<int64_t>& index) {
            const auto got = static_cast<float>(actual[offsetOf(index, w.strides())]);
            const auto want = static_cast<float>(expected[offsetOf(index, reference.strides())]);
            const auto mag = static_cast<float>(bounds[offsetOf(index, magnitude.strides())]);
            if(!std::isfinite(want))
            {
                // A tap the inf reaches: it must not come back finite.
                ++nonFinite;
                if(std::isfinite(got) && ++mismatches <= 5)
                {
                    ADD_FAILURE() << "at linear index " << offsetOf(index, w.strides()) << ": got "
                                  << got << ", want " << want;
                }
                return;
            }
            const float tolerance
                = 8.0f * 1.2e-7f * mag + 2.0f * storageEpsilon<T>() * std::fabs(want) + 1e-6f;
            if(!(std::fabs(got - want) <= tolerance) && ++mismatches <= 5)
            {
                ADD_FAILURE() << "at linear index " << offsetOf(index, w.strides()) << ": got "
                              << got << ", want " << want << ", tolerance " << tolerance;
            }
        });
        EXPECT_EQ(mismatches, 0);
        if(infAtRowEnd)
        {
            EXPECT_GT(nonFinite, 0) << "the inf reached no tap: the case does not test it";
        }
    }
    EXPECT_GE(ran, 1) << "no catalog config serves this case";
}

class TestGpuConvGroupValuWrw : public ::testing::TestWithParam<DeviceCase>
{
};

TEST_P(TestGpuConvGroupValuWrw, Float)
{
    SKIP_IF_NO_DEVICES();
    runCase<float>(GetParam().shape);
}

TEST_P(TestGpuConvGroupValuWrw, Half)
{
    SKIP_IF_NO_DEVICES();
    runCase<Half>(GetParam().shape);
}

TEST_P(TestGpuConvGroupValuWrw, BFloat16)
{
    SKIP_IF_NO_DEVICES();
    runCase<BFloat16>(GetParam().shape);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvGroupValuWrw,
                         ::testing::ValuesIn(deviceCases()),
                         [](const auto& info) { return info.param.name; });

/// An inf in x at the last column of a row 13 wide: every tile's lanes own 2, 4 or 8 columns, so
/// the last lane of a row also holds output columns past the row, whose window reaches that
/// column. Only the taps the reference reaches it with may be non-finite.
class TestGpuConvGroupValuWrwInf : public ::testing::TestWithParam<bool>
{
};

TEST_P(TestGpuConvGroupValuWrwInf, Float)
{
    SKIP_IF_NO_DEVICES();
    runCase<float>(conv(2, 8, 8, 2, {7, 13}, {3, 3}, {1, 1}, {1, 1}, GetParam()), true);
}

TEST_P(TestGpuConvGroupValuWrwInf, Half)
{
    SKIP_IF_NO_DEVICES();
    runCase<Half>(conv(2, 8, 8, 2, {7, 13}, {3, 3}, {1, 1}, {1, 1}, GetParam()), true);
}

TEST_P(TestGpuConvGroupValuWrwInf, BFloat16)
{
    SKIP_IF_NO_DEVICES();
    runCase<BFloat16>(conv(2, 8, 8, 2, {7, 13}, {3, 3}, {1, 1}, {1, 1}, GetParam()), true);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvGroupValuWrwInf,
                         ::testing::Values(false, true),
                         [](const auto& info) { return info.param ? "Nhwc" : "Nchw"; });

/// x past 2 GiB of small images (4100 of 16 x 128 x 128 halves, 3x3 stride 2): every pass's
/// 64-bit image base, 32-bit offsets within it. The reference is sampled: a few dw elements
/// summed on the host.
TEST(TestGpuConvGroupValuWrwLarge, HalfPast2GiBSampled)
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
    const auto spec = specFor(shape, ConvDirection::BACKWARD_WEIGHTS);
    const GraphFixture fixture(buildGraph(spec), device);
    const auto bound = matchesGraph(pack(), fixture.context());
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
    // Exact in fp16 and in every fp32 partial sum: x integers in [-3, 3], dy in [-2, 1] / 128.
    const auto xValue = [](int64_t i) {
        return static_cast<float>(static_cast<int>(((i * 2654435761LL) >> 7) & 7) - 3);
    };
    const auto yValue = [](int64_t i) {
        // Multiples of 2^-7: every sum stays exact in fp32 and within fp16's range.
        return static_cast<float>(static_cast<int>(((i * 40503LL + 17) >> 5) & 3) - 2) / 128.0f;
    };
    std::vector<Half> hx(static_cast<size_t>(xCount));
    for(int64_t i = 0; i < xCount; ++i)
    {
        hx[static_cast<size_t>(i)] = static_cast<Half>(xValue(i));
    }
    std::vector<Half> hy(static_cast<size_t>(yCount));
    for(int64_t i = 0; i < yCount; ++i)
    {
        hy[static_cast<size_t>(i)] = static_cast<Half>(yValue(i));
    }
    void* dx = nullptr;
    void* dy = nullptr;
    void* dw = nullptr;
    ASSERT_EQ(hipMalloc(&dx, static_cast<size_t>(xCount) * 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&dy, static_cast<size_t>(yCount) * 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&dw, static_cast<size_t>(k * c * 9) * 2), hipSuccess);
    ASSERT_EQ(hipMemcpy(dx, hx.data(), static_cast<size_t>(xCount) * 2, hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dy, hy.data(), static_cast<size_t>(yCount) * 2, hipMemcpyHostToDevice),
              hipSuccess);
    hx.clear();
    hx.shrink_to_fit();

    const auto& handler = dispatchHandler(pack());
    int ran = 0;
    for(const auto& kernel : catalog(device.gcnArchName))
    {
        if(!matchesKernel(pack(), fixture.context(), kernel, *bound))
        {
            continue;
        }
        SCOPED_TRACE(kernel.name);
        ++ran;
        const auto prepared = handler.prepare(fixture.context(), *bound, kernel);
        const size_t workspaceBytes = handler.workspaceBytes(fixture.context(), *bound, kernel);
        void* workspace = nullptr;
        if(workspaceBytes > 0)
        {
            ASSERT_EQ(hipMalloc(&workspace, workspaceBytes), hipSuccess);
        }
        ASSERT_EQ(hipMemset(dw, 0xff, static_cast<size_t>(k * c * 9) * 2), hipSuccess);
        const std::array<hipdnnPluginDeviceBuffer_t, 3> buffers{
            hipdnnPluginDeviceBuffer_t{X_UID, dx},
            hipdnnPluginDeviceBuffer_t{W_UID, dw},
            hipdnnPluginDeviceBuffer_t{Y_UID, dy}};
        const Handle handle;
        handler.launch(handle, *prepared, buffers.data(), 3, workspace);
        ASSERT_EQ(hipStreamSynchronize(handle.getStream()), hipSuccess);
        if(workspace != nullptr)
        {
            ASSERT_EQ(hipFree(workspace), hipSuccess);
        }
        std::vector<Half> got(static_cast<size_t>(k * c * 9));
        ASSERT_EQ(hipMemcpy(got.data(), dw, got.size() * 2, hipMemcpyDeviceToHost), hipSuccess);
        // Sampled elements: the corners of the filter for a few (k, c).
        for(const auto& [kk, cc, r, s] : std::vector<std::array<int64_t, 4>>{
                {0, 0, 0, 0}, {3, 15, 2, 2}, {1, 7, 1, 0}, {2, 9, 0, 2}})
        {
            double want = 0.0;
            for(int64_t in = 0; in < n; ++in)
            {
                for(int64_t oy = 0; oy < ho; ++oy)
                {
                    const int64_t iy = oy * 2 - 1 + r;
                    if(iy < 0 || iy >= h)
                    {
                        continue;
                    }
                    for(int64_t ox = 0; ox < wo; ++ox)
                    {
                        const int64_t ix = ox * 2 - 1 + s;
                        if(ix < 0 || ix >= wd)
                        {
                            continue;
                        }
                        want += static_cast<double>(xValue(((in * c + cc) * h + iy) * wd + ix))
                                * static_cast<double>(static_cast<float>(
                                    hy[static_cast<size_t>(((in * k + kk) * ho + oy) * wo + ox)]));
                    }
                }
            }
            const auto value
                = static_cast<float>(got[static_cast<size_t>(((kk * c + cc) * 3 + r) * 3 + s)]);
            EXPECT_NEAR(value, want, 2.0 * 9.8e-4 * std::fabs(want) + 1e-3)
                << "dw[" << kk << "][" << cc << "][" << r << "][" << s << "]";
        }
    }
    EXPECT_GE(ran, 1);
    ASSERT_EQ(hipFree(dx), hipSuccess);
    ASSERT_EQ(hipFree(dy), hipSuccess);
    ASSERT_EQ(hipFree(dw), hipSuccess);
}

} // namespace

#endif
