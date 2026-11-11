// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <algorithm>
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

#include "compilation/KernelCompileOptions.hpp"
#include "compilation/Program.hpp"
#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvIgemmProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvStemBwdProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestGpuConvStemBwd.cpp
 * @brief The stem backward-data kernels (CONV_STEM_BWD) on the device, through the engine's own
 *        graph match, kernel match, workspace sizing, prepare and launch (ConvStemBwdTransform,
 *        then ConvStemBwd), against hipDNN's CPU reference in fp64; and host checks of the
 *        shipped catalog.
 *
 * Every config of the installed catalog that the kernel match admits on this device runs on
 * every case, in fp32, fp16 and bf16: every window the catalog covers (7x7, 6x6, 5x5, 3x3 and
 * 2x2 at stride 2, 3x3 / 5x5 / 7x7 / 1x1 at stride 1, 4x4 and 11x11 at stride 4), 1, 3 and 4
 * input channels, channels-first and channels-last dy (dx and the filter in either layout),
 * rows of dy that are and are not whole vectors, output channel counts no chunk divides,
 * partial tiles, asymmetric padding, true convolution and a 3-D graph of depth 1. Each compiled
 * config's static LDS is checked against the host's arithmetic (ldsBytesOf) once.
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

constexpr PackSymbols PACK{"",
                           "hipkernel.conv_stem_bwd.graph_match",
                           "",
                           "hipkernel.conv_stem_bwd.kernel_match",
                           "",
                           "hipkernel.conv_stem_bwd.dispatch",
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
        // Packs per dtype and atom family (conv_stem_bwd_<dtype>_<family>.kdp.json), all in the
        // first root that has any.
        const auto isPack = [](const std::string& name) {
            const std::string prefix = "conv_stem_bwd_", suffix = ".kdp.json";
            return name.size() > prefix.size() + suffix.size()
                   && name.compare(0, prefix.size(), prefix) == 0
                   && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
        };
        std::vector<std::filesystem::path> found;
        for(const auto& root : roots)
        {
            std::error_code error;
            for(auto it = std::filesystem::recursive_directory_iterator(root, error);
                !error && it != std::filesystem::recursive_directory_iterator();
                it.increment(error))
            {
                if(isPack(it->path().filename().string()))
                {
                    found.push_back(it->path());
                }
            }
            if(!found.empty())
            {
                break;
            }
        }
        std::sort(found.begin(), found.end());
        for(const char* dtype : {"conv_stem_bwd_f32_", "conv_stem_bwd_f16_", "conv_stem_bwd_bf16_"})
        {
            if(std::none_of(found.begin(), found.end(), [&](const std::filesystem::path& path) {
                   return path.filename().string().rfind(dtype, 0) == 0;
               }))
            {
                throw std::runtime_error(std::string("no ") + dtype
                                         + "*.kdp.json beside the provider or this test binary");
            }
        }
        std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition> kernels;
        for(const auto& path : found)
        {
            std::ifstream stream(path);
            const auto pack = nlohmann::json::parse(stream);
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
        }
        return kernels;
    }();
    return s_catalog;
}

conv_stem_bwd::StemBwdConfig configOf(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel)
{
    conv_stem_bwd::StemBwdConfig c;
    c.atom = std::string(kernel.getStringMetadata("atom"));
    c.atomId = kernel.getIntMetadata("atom_id");
    c.layout = kernel.getIntMetadata("layout");
    c.vec = kernel.getIntMetadata("vec");
    c.u = kernel.getIntMetadata("u");
    c.v = kernel.getIntMetadata("v");
    c.nb = kernel.getIntMetadata("nb");
    c.kc = kernel.getIntMetadata("kc");
    c.th = kernel.getIntMetadata("th");
    c.fm = kernel.getIntMetadata("fm");
    c.sh = kernel.getIntMetadata("sh");
    c.sw = kernel.getIntMetadata("sw");
    c.ft = kernel.getIntMetadata("ft");
    c.sd = kernel.getIntMetadata("sd");
    return c;
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

/// @p from with every spatial axis reflected (true convolution's filter as cross-correlation's).
template <typename T>
void reflectFilter(Tensor<T>& from, Tensor<T>& to)
{
    const auto& dims = from.dims();
    const auto* src = static_cast<const T*>(from.rawHostData());
    auto* dst = static_cast<T*>(to.rawHostData());
    forEachIndex(dims, [&](const std::vector<int64_t>& index) {
        auto reflected = index;
        for(size_t axis = 2; axis < dims.size(); ++axis)
        {
            reflected[axis] = dims[axis] - 1 - index[axis];
        }
        dst[offsetOf(reflected, to.strides())] = src[offsetOf(index, from.strides())];
    });
}

struct DeviceCase
{
    std::string name;
    ConvShape shape;
    /// dx and the filter stored in the other layout than dy.
    bool otherLayouts = false;
};

ConvShape stem(int64_t c,
               int64_t k,
               std::vector<int64_t> input,
               int64_t filter,
               int64_t stride,
               int64_t padding,
               bool channelsLast = false,
               int64_t n = 2)
{
    ConvShape shape;
    shape.n = n;
    shape.c = c;
    shape.k = k;
    shape.input = std::move(input);
    shape.filter = {filter, filter};
    shape.prePadding = shape.postPadding = {padding, padding};
    shape.stride = {stride, stride};
    shape.dilation = {1, 1};
    shape.channelsLast = channelsLast;
    return shape;
}

/// The width whose dy rows are 32 long (whole vectors in every element type).
int64_t wholeRows(int64_t filter, int64_t stride, int64_t padding)
{
    return 31 * stride + filter - 2 * padding;
}

std::vector<DeviceCase> deviceCases()
{
    std::vector<DeviceCase> cases{
        {"Stem7x7S2Nchw", stem(3, 24, {37, 53}, 7, 2, 3)},
        {"Stem7x7S2NchwWholeRows", stem(3, 40, {29, wholeRows(7, 2, 3)}, 7, 2, 3)},
        {"Stem7x7S2Nhwc", stem(3, 24, {37, 53}, 7, 2, 3, true)},
        {"Stem7x7S2OneChannel", stem(1, 16, {21, wholeRows(7, 2, 3)}, 7, 2, 3)},
        {"Stem7x7S2FourChannelsNhwc", stem(4, 72, {19, 41}, 7, 2, 3, true, 1)},
        {"Stem7x7S2OddOutputChannels", stem(3, 7, {23, 30}, 7, 2, 3)},
        {"Stem3x3S2Pad1Nchw", stem(3, 32, {29, 41}, 3, 2, 1)},
        {"Stem3x3S2Pad1WholeRows", stem(3, 32, {17, wholeRows(3, 2, 1)}, 3, 2, 1)},
        {"Stem3x3S2Pad1Nhwc", stem(3, 16, {29, 41}, 3, 2, 1, true)},
        {"Stem3x3S2Pad0", stem(3, 16, {20, 33}, 3, 2, 0)},
        {"Stem6x6S2Pad2", stem(3, 32, {30, wholeRows(6, 2, 2)}, 6, 2, 2)},
        {"Stem5x5S2Pad2Nhwc", stem(4, 16, {21, 27}, 5, 2, 2, true)},
        {"Stem2x2S2", stem(3, 16, {20, 30}, 2, 2, 0)},
        {"Stem3x3S1Pad1", stem(3, 16, {12, 30}, 3, 1, 1)},
        {"Stem3x3S1Pad1Nhwc", stem(1, 24, {12, 32}, 3, 1, 1, true)},
        {"Stem5x5S1Pad2", stem(3, 16, {11, 19}, 5, 1, 2)},
        {"Stem7x7S1Pad3", stem(4, 8, {9, 21}, 7, 1, 3)},
        {"Pointwise1x1", stem(4, 40, {7, 32}, 1, 1, 0, true)},
        {"Patchify4x4S4", stem(3, 32, {32, 48}, 4, 4, 0)},
        {"Patchify4x4S4Nhwc", stem(3, 16, {28, 36}, 4, 4, 0, true)},
        {"Stem11x11S4Pad2", stem(3, 16, {35, 51}, 11, 4, 2)},
    };
    auto asymmetric = stem(3, 16, {26, 37}, 7, 2, 3);
    asymmetric.prePadding = {3, 3};
    asymmetric.postPadding = {2, 4};
    cases.push_back({"AsymmetricPadding", asymmetric});
    auto convolution = stem(3, 24, {21, 34}, 7, 2, 3);
    convolution.mode = ConvMode::CONVOLUTION;
    cases.push_back({"TrueConvolution", convolution});
    auto convolutionNhwc = stem(3, 16, {21, 34}, 3, 2, 1, true);
    convolutionNhwc.mode = ConvMode::CONVOLUTION;
    cases.push_back({"TrueConvolutionNhwc", convolutionNhwc});
    cases.push_back({"DxAndFilterChannelsLastUnderNchwDy", stem(3, 24, {23, 37}, 7, 2, 3), true});
    cases.push_back(
        {"DxAndFilterChannelsFirstUnderNhwcDy", stem(3, 24, {23, 37}, 3, 2, 1, true), true});
    auto threeD = stem(3, 16, {1, 21, 30}, 7, 2, 3);
    threeD.filter = {1, 7, 7};
    threeD.prePadding = threeD.postPadding = {0, 3, 3};
    threeD.stride = {1, 2, 2};
    threeD.dilation = {1, 1, 1};
    cases.push_back({"ThreeDOfDepthOne", threeD});
    // Stride 1 (space-to-depth configs too): odd extents (a half super-pixel at each far
    // edge), padding on one side per axis, true convolution.
    cases.push_back({"Stem3x3S1OddExtents", stem(3, 16, {13, 31}, 3, 1, 1)});
    // Stride 1, dy rows whole vectors: space-to-depth from column vectors too.
    cases.push_back({"Stem3x3S1WholeRows", stem(3, 24, {13, 32}, 3, 1, 1)});
    cases.push_back({"Stem5x5S1WholeRows", stem(3, 16, {11, 40}, 5, 1, 2)});
    cases.push_back({"Stem7x7S1WholeRows", stem(4, 8, {9, 32}, 7, 1, 3)});
    auto oneSided = stem(3, 16, {12, 22}, 3, 1, 0);
    oneSided.prePadding = {0, 2};
    oneSided.postPadding = {2, 0};
    cases.push_back({"Stem3x3S1OneSidedPadding", oneSided});
    auto convolutionS1 = stem(3, 16, {11, 19}, 5, 1, 2, true);
    convolutionS1.mode = ConvMode::CONVOLUTION;
    cases.push_back({"Stem5x5S1TrueConvolutionNhwc", convolutionS1});
    // 3-D stems: the dx planes in the grid, the depth taps in the reduction (unit depth stride).
    const auto stem3d = [](int64_t c,
                           int64_t k,
                           std::vector<int64_t> input,
                           int64_t taps,
                           int64_t filter,
                           int64_t stride,
                           int64_t padding,
                           bool channelsLast = false) {
        auto shape = stem(c, k, std::move(input), filter, stride, padding, channelsLast);
        shape.filter = {taps, filter, filter};
        shape.prePadding = shape.postPadding = {taps / 2, padding, padding};
        shape.stride = {1, stride, stride};
        shape.dilation = {1, 1, 1};
        return shape;
    };
    cases.push_back({"Stem1x7x7S2Planes", stem3d(3, 16, {3, 21, 30}, 1, 7, 2, 3)});
    cases.push_back({"Stem3x7x7S2Taps", stem3d(3, 24, {4, 21, 30}, 3, 7, 2, 3)});
    cases.push_back({"Stem3x7x7S2TapsNhwc", stem3d(3, 16, {3, 17, 26}, 3, 7, 2, 3, true)});
    cases.push_back({"Stem3x3x3S1OneChannel", stem3d(1, 16, {4, 10, 20}, 3, 3, 1, 1)});
    // Depth padding on one side only: the planes past it read no dy plane.
    auto leading = stem3d(3, 16, {3, 21, 30}, 3, 7, 2, 3);
    leading.prePadding = {2, 3, 3};
    leading.postPadding = {0, 3, 3};
    cases.push_back({"Stem3x7x7S2LeadingDepthPadding", leading});
    auto convolution3d = stem3d(3, 16, {3, 21, 30}, 3, 7, 2, 3);
    convolution3d.mode = ConvMode::CONVOLUTION;
    cases.push_back({"Stem3x7x7S2TrueConvolution", convolution3d});
    return cases;
}

/// The static LDS of @p kernel's compiled ConvStemBwd entry, from the program the pack compiled
/// (the module cache's entry for the same source and definitions).
int ldsOfCompiled(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                  const conv_stem_bwd::StemBwdConfig& config,
                  DataType dataType,
                  const std::string& arch)
{
    const auto info = conv_igemm::atomNamed(config.atom); // "fma": no atom template
    if(!info.has_value())
    {
        throw std::runtime_error("unknown atom " + config.atom);
    }
    compilation::KernelCompileOptions options(arch);
    for(const auto& [name, value] :
        conv_stem_bwd::compileDefinesFor(dataType, config, info->kernelType))
    {
        options.add(name, value);
    }
    const auto program = HipMlopsKernelCompiler::moduleCache().getOrLoad(
        kernel.source.sourceFile, std::vector<std::string>(options));
    if(!program)
    {
        throw std::runtime_error("the program did not compile");
    }
    int bytes = -1;
    const auto function = program->getKernel("ConvStemBwd");
    if(hipFuncGetAttribute(&bytes, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function) != hipSuccess)
    {
        throw std::runtime_error("hipFuncGetAttribute failed");
    }
    return bytes;
}

template <typename T>
void runCase(const DeviceCase& deviceCase)
{
    auto shape = deviceCase.shape;
    shape.dataType = dataTypeOf<T>();
    auto spec = specFor(shape, ConvDirection::BACKWARD_DATA);
    if(deviceCase.otherLayouts)
    {
        spec.x.strides = packedStrides(spec.x.dims, !shape.channelsLast);
        spec.w.strides = packedStrides(spec.w.dims, !shape.channelsLast);
    }
    const auto device = currentDeviceProperties();
    const GraphFixture fixture(buildGraph(spec), device);
    const auto bound = matchesGraph(PACK, fixture.context());
    ASSERT_TRUE(bound.has_value()) << "the engine refused a case it should serve";

    Tensor<T> x(spec.x.dims, spec.x.strides);
    Tensor<T> w(spec.w.dims, spec.w.strides);
    Tensor<T> y(spec.y.dims, spec.y.strides);
    constexpr unsigned int SEED = 4321;
    y.fillTensorWithRandomValues(-1.0f, 1.0f, SEED);
    w.fillTensorWithRandomValues(-1.0f, 1.0f, SEED + 1);
    Tensor<T> reference(x.dims(), x.strides());
    if(shape.mode == ConvMode::CONVOLUTION)
    {
        Tensor<T> reflected(w.dims(), w.strides());
        reflectFilter(w, reflected);
        CpuFpReferenceConvolution::dgrad<T, T, T, double>(reference,
                                                          reflected,
                                                          y,
                                                          shape.stride,
                                                          shape.dilation,
                                                          shape.prePadding,
                                                          shape.postPadding);
    }
    else
    {
        CpuFpReferenceConvolution::dgrad<T, T, T, double>(
            reference, w, y, shape.stride, shape.dilation, shape.prePadding, shape.postPadding);
    }
    // Each dx element sums (taps of its phase x output channels) products of values in [-1, 1],
    // in fp32.
    int64_t reductionLength = shape.k;
    for(const auto extent : shape.filter)
    {
        reductionLength *= extent;
    }
    const float accumulationBound = 2e-6f * static_cast<float>(reductionLength);

    static std::set<std::string> s_ldsChecked;
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
        const size_t workspaceBytes = handler.workspaceBytes(fixture.context(), *bound, kernel);
        ASSERT_GT(workspaceBytes, 0u);

        const auto config = configOf(kernel);
        if(s_ldsChecked.insert(kernel.name).second)
        {
            const int compiled = ldsOfCompiled(kernel, config, shape.dataType, device.gcnArchName);
            // Nothing but the two staged tiles takes LDS.
            EXPECT_EQ(compiled, conv_stem_bwd::ldsBytesOf(config, static_cast<int>(sizeof(T))));
            EXPECT_LE(compiled, conv_stem_bwd::ldsLimitOf(device.gcnArchName));
        }

        void* workspace = nullptr;
        ASSERT_EQ(hipMalloc(&workspace, workspaceBytes), hipSuccess);
        // A stale workspace must not leak into the result: the transform writes all of it.
        ASSERT_EQ(hipMemset(workspace, 0x7f, workspaceBytes), hipSuccess);
        x.fillWithSentinelValue();
        const std::array<hipdnnPluginDeviceBuffer_t, 3> buffers{
            hipdnnPluginDeviceBuffer_t{X_UID, x.rawDeviceData()},
            hipdnnPluginDeviceBuffer_t{W_UID, w.rawDeviceData()},
            hipdnnPluginDeviceBuffer_t{Y_UID, y.rawDeviceData()}};
        const Handle handle;
        handler.launch(
            handle, *prepared, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace);
        ASSERT_EQ(hipStreamSynchronize(handle.getStream()), hipSuccess);
        ASSERT_EQ(hipFree(workspace), hipSuccess);
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
    // Every device with a matrix core runs 16-bit; every device runs fp32 (FMA).
    EXPECT_GE(ran, 1) << "no catalog config serves this case";
}

class TestGpuConvStemBwd : public ::testing::TestWithParam<DeviceCase>
{
};

TEST_P(TestGpuConvStemBwd, Float)
{
    SKIP_IF_NO_DEVICES();
    runCase<float>(GetParam());
}

TEST_P(TestGpuConvStemBwd, Half)
{
    SKIP_IF_NO_DEVICES();
    runCase<Half>(GetParam());
}

TEST_P(TestGpuConvStemBwd, BFloat16)
{
    SKIP_IF_NO_DEVICES();
    runCase<BFloat16>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvStemBwd,
                         ::testing::ValuesIn(deviceCases()),
                         [](const auto& info) { return info.param.name; });

// ---- Host-only: the shipped catalog ----

TEST(TestConvStemBwdCatalog, KnobsTellApartEveryConfigOfAGraph)
{
    // Candidate enumeration identifies a config by its integer knobs: within one dtype (what
    // one graph can admit) no two configs may share every knob.
    static constexpr std::array<const char*, 13> KNOBS{
        "atom_id", "layout", "vec", "u", "v", "nb", "kc", "th", "fm", "sh", "sw", "ft", "sd"};
    std::set<std::vector<int64_t>> seen[3];
    for(const auto& kernel : catalog())
    {
        SCOPED_TRACE(kernel.name);
        const auto dtype = kernel.getStringMetadata("dtype");
        const int slot = dtype == "FLOAT" ? 0 : dtype == "HALF" ? 1 : 2;
        std::vector<int64_t> tuple;
        for(const char* knob : KNOBS)
        {
            tuple.push_back(kernel.getIntMetadata(knob));
        }
        EXPECT_TRUE(seen[slot].insert(tuple).second);
    }
}

TEST(TestConvStemBwdCatalog, AtomIdsNameTheirAtomsAndEveryConfigFitsItsArchitectures)
{
    static constexpr std::array<const char*, 7> ARCHS{
        "gfx908", "gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1151", "gfx1201"};
    std::set<std::string> dtypes;
    for(const auto& kernel : catalog())
    {
        SCOPED_TRACE(kernel.name);
        const auto config = configOf(kernel);
        EXPECT_EQ(conv_stem_bwd::atomIdOf(config.atom), config.atomId);
        EXPECT_EQ(kernel.getStringMetadata("direction"), "BWD");
        EXPECT_EQ(kernel.source.entryPoint, "ConvStemBwd");
        const auto dtype = std::string(kernel.getStringMetadata("dtype"));
        dtypes.insert(dtype);
        const int elementBytes = dtype == "FLOAT" ? 4 : 2;
        EXPECT_TRUE(config.vec == 1 || config.vec * elementBytes == 16);
        EXPECT_LE(config.nb, conv_stem_bwd::MAX_COLUMNS);
        int archs = 0;
        for(const char* arch : ARCHS)
        {
            if(!conv_igemm::atomRunsOn(config.atom, arch))
            {
                continue;
            }
            ++archs;
            EXPECT_LE(conv_stem_bwd::ldsBytesOf(config, elementBytes),
                      conv_stem_bwd::ldsLimitOf(arch))
                << arch;
        }
        EXPECT_GT(archs, 0) << "a config no architecture runs";
    }
    EXPECT_EQ(dtypes, (std::set<std::string>{"FLOAT", "HALF", "BFLOAT16"}));
}

} // namespace

#endif
