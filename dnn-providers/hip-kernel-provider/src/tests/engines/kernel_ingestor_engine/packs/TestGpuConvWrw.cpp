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
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
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
#include "engines/kernel_ingestor_engine/packs/ConvWrwProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestGpuConvWrw.cpp
 * @brief The matrix-core backward-weights kernels (CONV_WRW) on the device, through the engine's
 *        own graph match, kernel match, workspace sizing, prepare and launch (ConvWrw, then
 *        ConvWrwReduce; ConvWrw alone where it writes dw at one split), against hipDNN's CPU reference in fp64; and host checks of the shipped
 *        catalog.
 *
 * Every config of the installed catalog that the kernel match admits on this device runs on
 * every case, in fp32, fp16 and bf16 (matrix-core loops, gfx950's DMA ring and transpose reads,
 * the fp32 FMA ring wherever widths are whole vectors, and the fp32 FMA output rows for any
 * stride and any width, and gfx950's direct loop on 16-bit channels-last unit-stride cases):
 * channels-first and channels-last, 1-D, 2-D and 3-D,
 * groups, stride 2, dilation (2, 3 along the width, 12 for the deeplab rows), padding below, at
 * and past the filter, odd channel counts and partial tiles, both dw layouts under either
 * activation layout, and true convolution. Each compiled config's static LDS is checked against
 * the host's arithmetic (ldsBytesOf) once.
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
                           "hipkernel.conv_wrw.graph_match",
                           "",
                           "hipkernel.conv_wrw.kernel_match",
                           "",
                           "hipkernel.conv_wrw.dispatch",
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
        // Packs per dtype and kernel family (conv_wrw_<dtype>_<family>.kdp.json), all in the
        // first root that has any.
        const auto isPack = [](const std::string& name) {
            const std::string prefix = "conv_wrw_", suffix = ".kdp.json";
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
        // Every dtype ships packs; each dtype's prefix must be present.
        for(const char* dtype : {"conv_wrw_f32_", "conv_wrw_f16_", "conv_wrw_bf16_"})
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

conv_wrw::WrwConfig configOf(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel)
{
    conv_wrw::WrwConfig c;
    c.atom = std::string(kernel.getStringMetadata("atom"));
    c.atomId = kernel.getIntMetadata("atom_id");
    c.loop = kernel.getIntMetadata("loop");
    c.layout = kernel.getIntMetadata("layout");
    c.bk = kernel.getIntMetadata("bk");
    c.bc = kernel.getIntMetadata("bc");
    c.bq = kernel.getIntMetadata("bq");
    c.fr = kernel.getIntMetadata("fr");
    c.s = kernel.getIntMetadata("s");
    c.sw = kernel.getIntMetadata("sw");
    c.dw = kernel.getIntMetadata("dw");
    c.wk = kernel.getIntMetadata("wk");
    c.wc = kernel.getIntMetadata("wc");
    c.tk = kernel.getIntMetadata("tk");
    c.tc = kernel.getIntMetadata("tc");
    c.bpu = kernel.getIntMetadata("bpu");
    c.vec = kernel.getIntMetadata("vec");
    c.xcd = kernel.getIntMetadata("xcd");
    c.align = kernel.getIntMetadata("align");
    c.ns = kernel.getIntMetadata("ns");
    c.wq = kernel.getIntMetadata("wq");
    c.dws = kernel.getIntMetadata("dws");
    c.pb = kernel.getIntMetadata("pb");
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

/// dw of true convolution: cross-correlation's with every spatial axis reflected.
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
    /// dw stored in the other layout than x and dy.
    bool otherFilterLayout = false;
    /// Fewer than 8 reduction steps (one split) into dw of contiguous channels: every config
    /// but loop 5 writes dw itself, without workspace or reduce.
    bool writesDw = false;
    /// Served by the 16-bit matrix atoms only: no fp32 or gfx906 (FMA) configs.
    bool matrix16Only = false;
    /// No gfx906 (FMA) configs.
    bool noGfx906 = false;
    /// Served only where a few configs ship (block-based buffers): skipped where none does.
    bool someArchs = false;
    /// Served by the generic loop alone, which ships on the matrix atoms of gfx908 to gfx950
    /// and gfx12: skipped on gfx906 and gfx11.
    bool genericOnly = false;
    /// x's element strides (the generic loop's strided operands); empty: packed.
    std::vector<int64_t> xStrides{};
};

ConvShape conv(int64_t c,
               int64_t k,
               std::vector<int64_t> input,
               std::vector<int64_t> filter,
               std::vector<int64_t> pre,
               std::vector<int64_t> post,
               std::vector<int64_t> stride,
               std::vector<int64_t> dilation,
               bool channelsLast = false,
               int64_t groups = 1,
               int64_t n = 2)
{
    ConvShape shape;
    shape.n = n;
    shape.c = c;
    shape.k = k;
    shape.groups = groups;
    shape.input = std::move(input);
    shape.filter = std::move(filter);
    shape.prePadding = std::move(pre);
    shape.postPadding = std::move(post);
    shape.stride = std::move(stride);
    shape.dilation = std::move(dilation);
    shape.channelsLast = channelsLast;
    return shape;
}

// Widths are not multiples of the position tiles and channel counts not of the channel tiles
// (partial tiles), unless a case targets the aligned or channels-last vector configs.
std::vector<DeviceCase> deviceCases()
{
    std::vector<DeviceCase> cases{
        {"Nchw3x3", conv(24, 40, {10, 12}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1})},
        {"Nchw3x3Aligned", conv(16, 32, {9, 16}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1})},
        {"Nhwc3x3", conv(24, 40, {10, 12}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, true)},
        {"Nhwc3x3Wide", conv(64, 96, {21, 70}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, true)},
        // Output channels that are not whole vectors (dy units at any element), images whose
        // last dy element ends mid-dword.
        {"Nhwc3x3ThreeOutputChannels",
         conv(32, 3, {21, 45}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, true)},
        {"Nhwc1x1FiveOutputChannels",
         conv(16, 5, {13, 37}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, true, 1, 3)},
        {"ThreeDNhwc3x3x3",
         conv(16, 32, {4, 6, 40}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, true)},
        {"GroupedNchw3x3", conv(48, 32, {7, 9}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, false, 2)},
        {"GroupedNhwc3x3", conv(24, 48, {7, 9}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, true, 3)},
        {"ThreeD3x3x3",
         conv(8, 16, {4, 6, 7}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1})},
        {"OneD1x7", conv(16, 24, {50}, {7}, {3}, {3}, {1}, {1})},
        {"OneD1x7Nhwc", conv(16, 24, {50}, {7}, {3}, {3}, {1}, {1}, true)},
        {"OneD1x3Dilation5", conv(32, 64, {150}, {3}, {5}, {5}, {1}, {5})},
        {"OneD1x3Dilation3", conv(32, 64, {150}, {3}, {3}, {3}, {1}, {3})},
        {"Dilation6", conv(16, 24, {20, 26}, {3, 3}, {6, 6}, {6, 6}, {1, 1}, {6, 6})},
        {"OneD1x4", conv(32, 48, {90}, {4}, {0}, {0}, {1}, {1})},
        {"OneD1x4Stride2", conv(16, 32, {90}, {4}, {1}, {1}, {2}, {1})},
        {"OneD1x16Stride8", conv(32, 64, {203}, {16}, {4}, {4}, {8}, {1})},
        {"ThreeD2x2x2Stride2",
         conv(8, 16, {6, 8, 10}, {2, 2, 2}, {0, 0, 0}, {0, 0, 0}, {2, 2, 2}, {1, 1, 1})},
        {"Deeplab3x3Dilation36",
         conv(16, 16, {76, 76}, {3, 3}, {36, 36}, {36, 36}, {1, 1}, {36, 36}, false, 1, 1)},
        {"Wide1x11Dilation3", conv(16, 16, {3, 40}, {1, 11}, {0, 15}, {0, 15}, {1, 1}, {1, 3})},
        {"Dilation2", conv(16, 24, {11, 13}, {3, 3}, {2, 2}, {2, 2}, {1, 1}, {2, 2})},
        {"Deeplab3x3Dilation12",
         conv(16, 16, {30, 30}, {3, 3}, {12, 12}, {12, 12}, {1, 1}, {12, 12}, false, 1, 1)},
        {"Deeplab3x3Dilation12Nhwc",
         conv(16, 16, {30, 30}, {3, 3}, {12, 12}, {12, 12}, {1, 1}, {12, 12}, true, 1, 1)},
        {"Stride2", conv(24, 40, {13, 15}, {3, 3}, {1, 1}, {1, 1}, {2, 2}, {1, 1})},
        {"Stride2Nhwc", conv(24, 40, {13, 15}, {3, 3}, {1, 1}, {1, 1}, {2, 2}, {1, 1}, true)},
        {"Stride2Pad0OddChannels", conv(5, 7, {9, 10}, {3, 3}, {0, 0}, {0, 0}, {2, 2}, {1, 1})},
        {"Stem7x7Stride2", conv(3, 16, {23, 30}, {7, 7}, {3, 3}, {3, 3}, {2, 2}, {1, 1})},
        {"Stem7x7Stride2K64", conv(3, 72, {37, 70}, {7, 7}, {3, 3}, {3, 3}, {2, 2}, {1, 1})},
        {"Stem3x3Stride1C5", conv(5, 40, {21, 77}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1})},
        {"Stem3x3Stride2ThreeD",
         conv(3, 24, {5, 13, 21}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {2, 2, 2}, {1, 1, 1})},
        // Channels-last stems: input channels not whole vectors (small-C rows, layout 1).
        {"Stem7x7Stride2Nhwc", conv(3, 72, {37, 70}, {7, 7}, {3, 3}, {3, 3}, {2, 2}, {1, 1}, true)},
        {"Stem3x3Stride2NhwcOneChannel",
         conv(1, 24, {21, 33}, {3, 3}, {1, 1}, {1, 1}, {2, 2}, {1, 1}, true)},
        {"Stem3x3Stride1C5Nhwc",
         conv(5, 40, {21, 77}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, true)},
        {"Stem3x3Stride2ThreeDNhwc",
         conv(3, 24, {5, 13, 21}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {2, 2, 2}, {1, 1, 1}, true)},
        {"Stem7x7Stride2NhwcGrouped",
         conv(6, 16, {17, 29}, {7, 7}, {3, 3}, {3, 3}, {2, 2}, {1, 1}, true, 2)},
        {"Stride2Pointwise1x1Nhwc",
         conv(16, 32, {9, 11}, {1, 1}, {0, 0}, {0, 0}, {2, 2}, {1, 1}, true)},
        {"Pointwise1x1Nhwc", conv(64, 136, {6, 9}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, true)},
        {"Pointwise1x1OddChannels", conv(70, 33, {6, 7}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, {1, 1})},
        {"Pad0OddChannels", conv(5, 7, {7, 9}, {3, 3}, {0, 0}, {0, 0}, {1, 1}, {1, 1})},
        // The row reduce: channel runs of 64, 64 and 8 (K C R S).
        {"Nchw3x3ThreeChannelRuns", conv(136, 24, {6, 9}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1})},
        // A filter as wide as its map (FCN's point_output, as 1-D): one output position per
        // image.
        {"PointOutput1x7", conv(24, 40, {7}, {7}, {0}, {0}, {1}, {1}, false, 1, 8)},
        {"PaddingPastTheFilter", conv(8, 16, {6, 9}, {3, 3}, {3, 4}, {3, 4}, {1, 1}, {1, 1})},
        {"ManySteps", conv(64, 96, {20, 70}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, false, 1, 4)},
    };
    auto convolution = conv(16, 16, {7, 9}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, false, 2);
    convolution.mode = ConvMode::CONVOLUTION;
    cases.push_back({"TrueConvolutionGrouped", convolution});
    auto convolutionNhwc = conv(16, 32, {7, 9}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, true);
    convolutionNhwc.mode = ConvMode::CONVOLUTION;
    cases.push_back({"TrueConvolutionNhwc", convolutionNhwc});
    cases.push_back({"FilterKrscUnderNchw",
                     conv(24, 40, {10, 12}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}),
                     true});
    cases.push_back({"FilterKcrsUnderNhwc",
                     conv(24, 40, {10, 12}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, true),
                     true});
    // One image of 3 rows of 9: at most 5 steps (3 rows, or 3 + 2 on a 3-row ring, one width
    // chunk), one split.
    cases.push_back({"OneSplitNhwc3x3",
                     conv(24, 40, {3, 9}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, true, 1, 1),
                     false,
                     true});
    cases.push_back({"OneSplitGroupedNhwc3x3",
                     conv(48, 64, {3, 9}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, true, 2, 1),
                     false,
                     true});
    cases.push_back({"OneSplitFilterKrscUnderNchw",
                     conv(24, 40, {3, 9}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, false, 1, 1),
                     true,
                     true});
    cases.push_back({"OneSplitPointwise1x1OddChannels",
                     conv(70, 33, {3, 7}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, false, 1, 1),
                     false,
                     true});
    cases.push_back({"OneSplitPointwise1x1Nhwc",
                     conv(64, 136, {3, 9}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, true, 1, 1),
                     false,
                     true});
    auto oneSplitConvolution
        = conv(16, 32, {3, 9}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, true, 1, 1);
    oneSplitConvolution.mode = ConvMode::CONVOLUTION;
    cases.push_back({"OneSplitTrueConvolutionNhwc", oneSplitConvolution, false, true});
    // Inception's asymmetric and 5x5 rows (channels-first, 16-bit matrix atoms): 7x1 and 3x1 on
    // the ring (3x1 also at a whole-vector width for the aligned configs), 5x5 on the rows and
    // the ring, and 5x5 over a 5x5 map (point_output: one output position per image).
    const std::vector<std::pair<std::string, ConvShape>> matrix16{
        {"Inception7x1", conv(24, 40, {13, 9}, {7, 1}, {3, 0}, {3, 0}, {1, 1}, {1, 1})},
        {"Inception3x1", conv(24, 40, {8, 9}, {3, 1}, {1, 0}, {1, 0}, {1, 1}, {1, 1})},
        {"Inception3x1Aligned", conv(24, 40, {8, 16}, {3, 1}, {1, 0}, {1, 0}, {1, 1}, {1, 1})},
        {"Inception5x5", conv(16, 24, {11, 13}, {5, 5}, {2, 2}, {2, 2}, {1, 1}, {1, 1})},
        {"PointOutput5x5", conv(16, 24, {5, 5}, {5, 5}, {0, 0}, {0, 0}, {1, 1}, {1, 1})},
    };
    for(const auto& [name, shape] : matrix16)
    {
        cases.push_back({name, shape, false, false, true});
    }
    // Narrow outputs (packed rows where they ship, gfx950 / gfx12): several images' 7^2 rows,
    // stride 2 with and without padding (17-column rows: 3 a 64-position step, a partial last
    // group), 3-D, and a grouped true convolution.
    cases.push_back({"PackedRows7x7",
                     conv(24, 40, {7, 7}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, false, 1, 3)});
    cases.push_back(
        {"PackedRowsStride2", conv(24, 40, {14, 14}, {3, 3}, {1, 1}, {1, 1}, {2, 2}, {1, 1})});
    cases.push_back({"PackedRowsStride2Pad0",
                     conv(24, 40, {35, 35}, {3, 3}, {0, 0}, {0, 0}, {2, 2}, {1, 1}, false, 1, 1)});
    cases.push_back(
        {"PackedRowsThreeD",
         conv(16, 32, {3, 5, 6}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1})});
    auto packedConvolution = conv(32, 48, {6, 7}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, false, 2);
    packedConvolution.mode = ConvMode::CONVOLUTION;
    cases.push_back({"PackedRowsGroupedConvolution", packedConvolution});
    // 5x5x5 volumes (AIPYTORCH-1356): unit stride on the ring / fp32 rows, stride 2 on the
    // output rows, and few input channels on the small-C rows (16-bit). No gfx906 configs.
    cases.push_back(
        {"ThreeD5x5x5",
         conv(16, 24, {7, 9, 10}, {5, 5, 5}, {2, 2, 2}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}),
         false,
         false,
         false,
         true});
    cases.push_back(
        {"ThreeD5x5x5Stride2",
         conv(16, 24, {9, 11, 13}, {5, 5, 5}, {1, 1, 1}, {1, 1, 1}, {2, 2, 2}, {1, 1, 1}),
         false,
         false,
         false,
         true});
    // One x image past 2 GiB in 16-bit (64 channels of 4100 x 4200; past 4 GiB in fp32): only
    // the block-based buffer configs (pb 1) serve it, each block's 32 (16-bit) or 16 (fp32)
    // channel planes within the 32-bit range. One output channel keeps the CPU reference short.
    cases.push_back({"LargeImageBlockBuffers",
                     conv(64, 1, {4100, 4200}, {3, 3}, {0, 0}, {0, 0}, {2, 2}, {1, 1}, false, 1, 1),
                     false,
                     false,
                     false,
                     true,
                     true});
    cases.push_back({"ThreeD5x5x5FewChannels",
                     conv(8, 24, {6, 7, 9}, {5, 5, 5}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}),
                     false,
                     false,
                     false,
                     true});
    // The generic loop (runtime filter, strides, dilation): channels-last 16-bit widths no
    // specialized config has (5x5, 1x9, 1x5x5, dilation 4), and strided x (padded rows: no
    // packed layout) in either form.
    const auto generic
        = [&](const std::string& name, const ConvShape& shape, std::vector<int64_t> xStrides = {}) {
              DeviceCase deviceCase{name, shape};
              deviceCase.genericOnly = true;
              deviceCase.xStrides = std::move(xStrides);
              cases.push_back(deviceCase);
          };
    generic("Generic5x5Nhwc", conv(24, 40, {13, 17}, {5, 5}, {2, 2}, {2, 2}, {1, 1}, {1, 1}, true));
    generic("Generic1x9Nhwc", conv(32, 64, {150}, {9}, {4}, {4}, {1}, {1}, true));
    // Dilated width taps (the shared window reads them at offsets of 2).
    generic("Generic1x5Dilation2Nhwc", conv(32, 32, {90}, {5}, {4}, {4}, {1}, {2}, true));
    generic(
        "Generic1x5x5ThreeDNhwcGrouped",
        conv(32, 32, {3, 9, 11}, {1, 5, 5}, {0, 2, 2}, {0, 2, 2}, {1, 1, 1}, {1, 1, 1}, true, 2));
    generic("GenericDilation4Nhwc",
            conv(16, 16, {40, 40}, {3, 3}, {4, 4}, {4, 4}, {1, 1}, {4, 4}, true, 1, 1));
    generic("Generic7x7Stride2Nhwc",
            conv(24, 24, {19, 23}, {7, 7}, {3, 3}, {3, 3}, {2, 2}, {1, 1}, true));
    // 24 channels of 10 x 12, rows 16 apart (channels-first elements).
    generic("GenericPaddedRowsNchw",
            conv(24, 40, {10, 12}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}),
            {24 * 10 * 16, 10 * 16, 16, 1});
    // Channels-last, positions 32 channels apart (24 used): channel vectors.
    generic("GenericPaddedChannelsNhwc",
            conv(24, 40, {10, 12}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, true),
            {10 * 12 * 32, 1, 12 * 32, 32});
    // One channels-last x image of 2^30 elements (2 GiB in 16-bit, 4 GiB in fp32): the generic
    // loop's block-based buffers (pb 1), rebased each step. 1x1 stride 4 keeps the CPU
    // reference short.
    {
        DeviceCase largeCl{
            "LargeImageGenericBlockBuffersNhwc",
            conv(16384, 8, {256, 256}, {1, 1}, {0, 0}, {0, 0}, {4, 4}, {1, 1}, true, 1, 1)};
        largeCl.someArchs = true;
        cases.push_back(largeCl);
    }
    // Positions 26 apart: not whole vectors, channels-last elements.
    generic("GenericPaddedChannelsNhwcElements",
            conv(24, 40, {10, 12}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, true),
            {10 * 12 * 26, 1, 12 * 26, 26});
    return cases;
}

/// The static LDS of @p kernel's compiled ConvWrw entry, from the program the pack compiled (the
/// module cache's entry for the same source and definitions).
int ldsOfCompiled(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                  const conv_wrw::WrwConfig& config,
                  DataType dataType,
                  const std::string& arch)
{
    const auto info = conv_igemm::atomNamed(config.atom); // "fma": no atom template
    if(!info.has_value())
    {
        throw std::runtime_error("unknown atom " + config.atom);
    }
    compilation::KernelCompileOptions options(arch);
    for(const auto& [name, value] : conv_wrw::compileDefinesFor(dataType, config, info->kernelType))
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
    const auto function = program->getKernel("ConvWrw");
    if(hipFuncGetAttribute(&bytes, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function) != hipSuccess)
    {
        throw std::runtime_error("hipFuncGetAttribute failed");
    }
    return bytes;
}

/// Whether this device is expected to serve every case in @p dataType: the MFMA families from
/// gfx90a (gfx908's bf16 atom ships for the common filters only), gfx906 on the FMA loops (every
/// element type), RDNA in 16-bit only.
bool expectServed(const std::string& arch, DataType dataType)
{
    const auto base = std::string(hipdnn_plugin_sdk::stripArchFeatures(arch));
    // gfx906: the FMA loops, every element type.
    if(base == "gfx90a" || base == "gfx942" || base == "gfx950" || base == "gfx906")
    {
        return true;
    }
    if(base.rfind("gfx11", 0) == 0 || base.rfind("gfx12", 0) == 0)
    {
        return dataType != DataType::FLOAT;
    }
    return false;
}

template <typename T>
void runCase(const DeviceCase& deviceCase)
{
    auto shape = deviceCase.shape;
    shape.dataType = dataTypeOf<T>();
    auto spec = specFor(shape, ConvDirection::BACKWARD_WEIGHTS);
    if(deviceCase.otherFilterLayout)
    {
        spec.w.strides = packedStrides(spec.w.dims, !shape.channelsLast);
    }
    if(!deviceCase.xStrides.empty())
    {
        spec.x.strides = deviceCase.xStrides;
    }
    const auto device = currentDeviceProperties();
    const GraphFixture fixture(buildGraph(spec), device);
    const auto bound = matchesGraph(PACK, fixture.context());
    ASSERT_TRUE(bound.has_value()) << "the engine refused a case it should serve";
    // Cases only a few configs serve skip before building their (large) tensors.
    if(deviceCase.someArchs
       && std::none_of(catalog().begin(), catalog().end(), [&](const auto& kernel) {
              return matchesKernel(PACK, fixture.context(), kernel, *bound);
          }))
    {
        GTEST_SKIP() << "no catalog config for this case on " << device.gcnArchName;
    }

    Tensor<T> x(spec.x.dims, spec.x.strides);
    Tensor<T> w(spec.w.dims, spec.w.strides);
    Tensor<T> y(spec.y.dims, spec.y.strides);
    constexpr unsigned int SEED = 4321;
    x.fillTensorWithRandomValues(-1.0f, 1.0f, SEED);
    y.fillTensorWithRandomValues(-1.0f, 1.0f, SEED + 1);

    Tensor<T> reference(spec.w.dims, spec.w.strides);
    const bool reflect = shape.mode == ConvMode::CONVOLUTION;
    Tensor<T> crossCorrelation(spec.w.dims, spec.w.strides);
    CpuFpReferenceConvolution::wgrad<T, T, T, double>(x,
                                                      reflect ? crossCorrelation : reference,
                                                      y,
                                                      shape.stride,
                                                      shape.dilation,
                                                      shape.prePadding,
                                                      shape.postPadding);
    if(reflect)
    {
        reflectFilter(crossCorrelation, reference);
    }
    // Each dw element sums (images x output positions) products of values in [-1, 1], in fp32.
    int64_t reduction = spec.y.dims[0];
    for(size_t axis = 2; axis < spec.y.dims.size(); ++axis)
    {
        reduction *= spec.y.dims[axis];
    }
    const float accumulationBound = 2e-6f * static_cast<float>(reduction);

    static std::set<std::string> s_ldsChecked;
    const auto& handler = dispatchHandler(PACK);
    int ran = 0;
    int wroteDw = 0;
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
        const auto config = configOf(kernel);
        // No workspace: the kernel writes dw itself (never loop 5).
        if(workspaceBytes == 0)
        {
            EXPECT_NE(config.loop, conv_wrw::LOOP_DIRECT);
            ++wroteDw;
        }
        if(s_ldsChecked.insert(kernel.name).second)
        {
            const int compiled = ldsOfCompiled(kernel, config, shape.dataType, device.gcnArchName);
            const auto host = conv_wrw::ldsBytesOf(config, static_cast<int>(sizeof(T)));
            // The declared LDS is the host's; the kernel never takes more than the device has.
            EXPECT_GE(compiled, host);
            EXPECT_LE(compiled, conv_wrw::ldsLimitOf(device.gcnArchName));
            if(config.loop == conv_wrw::LOOP_RING_DMA || config.loop == conv_wrw::LOOP_DIRECT)
            {
                // The unused stages / ring slots are one-element arrays, which may or may not be
                // allocated.
                EXPECT_LE(compiled, host + 32);
            }
            else
            {
                // Nothing else takes LDS (a private array indexed through a pointer would be
                // promoted into the LDS left over).
                EXPECT_EQ(compiled, host);
            }
        }

        void* workspace = nullptr;
        if(workspaceBytes > 0)
        {
            ASSERT_EQ(hipMalloc(&workspace, workspaceBytes), hipSuccess);
            // Stale partials must not leak into the result.
            ASSERT_EQ(hipMemset(workspace, 0x7f, workspaceBytes), hipSuccess);
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
        const auto* expected = static_cast<const T*>(reference.rawHostData());
        int64_t mismatches = 0;
        forEachIndex(w.dims(), [&](const std::vector<int64_t>& index) {
            const float got = static_cast<float>(actual[offsetOf(index, w.strides())]);
            const float want = static_cast<float>(expected[offsetOf(index, reference.strides())]);
            const float tolerance
                = accumulationBound + 2.0f * storageEpsilon<T>() * (std::fabs(want) + 1e-3f);
            if(!(std::fabs(got - want) <= tolerance) && ++mismatches <= 5)
            {
                ADD_FAILURE() << "at linear index " << offsetOf(index, w.strides()) << ": got "
                              << got << ", want " << want << ", tolerance " << tolerance;
            }
        });
        EXPECT_EQ(mismatches, 0);
    }
    const auto base = std::string(hipdnn_plugin_sdk::stripArchFeatures(device.gcnArchName));
    const bool gfx906 = base == "gfx906";
    const bool fmaOrFp32 = shape.dataType == DataType::FLOAT || gfx906;
    if(ran == 0
       && (!expectServed(device.gcnArchName, shape.dataType)
           || (deviceCase.matrix16Only && fmaOrFp32) || (deviceCase.noGfx906 && gfx906)
           || deviceCase.someArchs
           || (deviceCase.genericOnly && (gfx906 || base.rfind("gfx11", 0) == 0))))
    {
        GTEST_SKIP() << "no catalog config for this case on " << device.gcnArchName;
    }
    EXPECT_GE(ran, 1) << "no catalog config serves this case";
    if(deviceCase.writesDw)
    {
        EXPECT_GE(wroteDw, 1) << "no config wrote dw itself";
    }
}

class TestGpuConvWrw : public ::testing::TestWithParam<DeviceCase>
{
};

TEST_P(TestGpuConvWrw, Float)
{
    SKIP_IF_NO_DEVICES();
    runCase<float>(GetParam());
}

TEST_P(TestGpuConvWrw, Half)
{
    SKIP_IF_NO_DEVICES();
    runCase<Half>(GetParam());
}

TEST_P(TestGpuConvWrw, BFloat16)
{
    SKIP_IF_NO_DEVICES();
    runCase<BFloat16>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvWrw,
                         ::testing::ValuesIn(deviceCases()),
                         [](const auto& info) { return info.param.name; });

// ---- Host-only: the shipped catalog ----

std::string_view stringOf(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                          const char* field)
{
    return kernel.getStringMetadata(field);
}

TEST(TestConvWrwCatalog, KnobsTellApartEveryConfigOfAGraph)
{
    // Candidate enumeration identifies a config by its integer knobs: within one dtype (what
    // one graph can admit) no two configs may share every knob.
    static constexpr std::array<const char*, 22> KNOBS{
        "atom_id", "loop", "layout", "bk",  "bc",  "bq",    "fr", "s",   "sw", "dw",  "wk",
        "wc",      "tk",   "tc",     "vec", "xcd", "align", "ns", "bpu", "wq", "dws", "pb"};
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

TEST(TestConvWrwCatalog, EveryGenericSpecKeyParses)
{
    // A malformed key declines its generic config everywhere: the generator's keys must all be
    // the host's grammar.
    auto shape = conv(24, 40, {10, 12}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1});
    shape.dataType = DataType::FLOAT;
    const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::BACKWARD_WEIGHTS)));
    const auto problem = conv_wrw::parseWrwProblem(fixture.context());
    ASSERT_TRUE(problem.has_value());
    int keys = 0;
    for(const auto& kernel : catalog())
    {
        if(kernel.getIntMetadata("loop") != conv_wrw::LOOP_GENERIC)
        {
            continue;
        }
        SCOPED_TRACE(kernel.name);
        const std::string spec(kernel.getStringMetadata("spec"));
        size_t from = 0;
        while(from < spec.size())
        {
            const size_t to = std::min(spec.find(';', from), spec.size());
            const auto key = spec.substr(from, to - from);
            EXPECT_TRUE(conv_wrw::specServes(*problem, key, "gfx1201").has_value()) << key;
            ++keys;
            from = to + 1;
        }
    }
    EXPECT_GT(keys, 0);
}

TEST(TestConvWrwCatalog, AtomIdsNameTheirAtomsAndEveryConfigFitsItsArchitectures)
{
    static constexpr std::array<const char*, 7> ARCHS{
        "gfx908", "gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1151", "gfx1201"};
    std::set<int64_t> loops;
    for(const auto& kernel : catalog())
    {
        SCOPED_TRACE(kernel.name);
        const auto config = configOf(kernel);
        EXPECT_EQ(conv_wrw::atomIdOf(config.atom), config.atomId);
        EXPECT_EQ(stringOf(kernel, "direction"), "WRW");
        EXPECT_EQ(kernel.source.entryPoint, "ConvWrw");
        const auto dtype = stringOf(kernel, "dtype");
        const int elementBytes = dtype == "FLOAT" ? 4 : 2;
        // 16-byte vectors; the FMA output rows also take scalar units.
        if(config.loop != conv_wrw::LOOP_ROWS_FMA || config.vec != 1)
        {
            EXPECT_EQ(config.vec * elementBytes, 16);
        }
        int archs = 0;
        for(const char* arch : ARCHS)
        {
            const bool gfx950Only = config.loop == conv_wrw::LOOP_RING_DMA
                                    || config.loop == conv_wrw::LOOP_DIRECT
                                    || config.layout == conv_wrw::LAYOUT_CHANNELS_LAST_TR;
            if(!conv_igemm::atomRunsOn(config.atom, arch)
               || (gfx950Only && std::string(arch) != "gfx950"))
            {
                continue;
            }
            ++archs;
            EXPECT_LE(conv_wrw::ldsBytesOf(config, elementBytes), conv_wrw::ldsLimitOf(arch))
                << arch;
        }
        EXPECT_GT(archs, 0) << "a config no architecture runs";
        loops.insert(config.loop);
    }
    EXPECT_EQ(loops, (std::set<int64_t>{0, 1, 2, 3, 4, 5, 6, 7, 8}));
}

} // namespace

#endif
