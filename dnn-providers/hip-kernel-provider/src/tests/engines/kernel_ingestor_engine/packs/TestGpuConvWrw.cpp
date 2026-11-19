// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
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
#include "engines/kernel_ingestor_engine/packs/ConvDeterminismTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvIgemmProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvTf32Timing.hpp"
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
 * groups, stride 2, dilation (2, 3 along the width, 4 to 36 for the deeplab rows in both layouts),
 * padding below, at
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
            const std::string prefix = "conv_wrw_";
            const std::string suffix = ".kdp.json";
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
    c.math = kernel.getIntMetadata("math");
    c.flat = kernel.metadata.count("flat") != 0 ? kernel.getIntMetadata("flat") : 0;
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
    /// Served by the generic loop alone, which ships on every matrix atom (gfx908 to gfx950,
    /// gfx11 and gfx12): skipped on gfx906.
    bool genericOnly = false;
    /// x's element strides (the generic loop's strided operands); empty: packed. The default
    /// initializer lets the cases' aggregate initializations omit it without
    /// -Wmissing-field-initializers.
    std::vector<int64_t> xStrides{}; // NOLINT(readability-redundant-member-init)
    /// Served on gfx908 in 16-bit too (bf16 on its half-rate atoms): output channels that are
    /// not whole vectors, the small-C rows and the generic loop.
    bool gfx908 = false;
    /// Nonzero: exercise this native gfx1250 loop alone against the independent reference.
    int64_t onlyLoop = 0;
    bool dilationRows = false;
    /// Targeted matrix family whose native catalog exists only for 16-bit operands.
    bool sixteenBitOnly = false;
    bool directChannelsFirstStore = false;
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
        // HiFi-GAN's output layer: one output channel, 1x7, in both layouts.
        {"OneD1x7OneOutputChannel", conv(32, 1, {300}, {7}, {3}, {3}, {1}, {1})},
        {"OneD1x7OneOutputChannelNhwc", conv(32, 1, {300}, {7}, {3}, {3}, {1}, {1}, true)},
        // 32-channel groups (ResNeXt) channels-last.
        {"Grouped32Nhwc3x3", conv(64, 64, {7, 9}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, true, 2)},
        // A 3-D VAE output layer (3 channels from many), channels-last.
        {"ThreeDNhwc3x3x3ThreeOutputChannels",
         conv(40, 3, {4, 7, 37}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, true)},
        {"ThreeDNhwc3x3x3",
         conv(16, 32, {4, 6, 40}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, true)},
        {"GroupedNchw3x3", conv(48, 32, {7, 9}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, false, 2)},
        {"GroupedNhwc3x3", conv(24, 48, {7, 9}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, true, 3)},
        {"ThreeD3x3x3",
         conv(8, 16, {4, 6, 7}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1})},
        // Small 3-D maps whose rows fit a ring stage's narrow steps (16 and 8 positions), with
        // stages that cross tracks and images, in both layouts.
        {"ThreeDNhwc3x3x3Rows16",
         conv(64, 64, {3, 7, 16}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, true)},
        {"ThreeDNhwc3x3x3Rows8",
         conv(32, 64, {4, 5, 8}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, true)},
        {"ThreeD3x3x3Rows8",
         conv(64, 32, {4, 5, 8}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1})},
        {"OneD1x7", conv(16, 24, {50}, {7}, {3}, {3}, {1}, {1})},
        {"OneD1x7Nhwc", conv(16, 24, {50}, {7}, {3}, {3}, {1}, {1}, true)},
        {"OneD1x3Dilation5", conv(32, 64, {150}, {3}, {5}, {5}, {1}, {5})},
        {"OneD1x3Dilation3", conv(32, 64, {150}, {3}, {3}, {3}, {1}, {3})},
        // HiFi-GAN's dilated 1-D rows channels-last (their own specialized configs).
        {"OneD1x3Dilation5Nhwc", conv(32, 64, {150}, {3}, {5}, {5}, {1}, {5}, true)},
        {"OneD1x7Dilation3Nhwc", conv(32, 48, {130}, {7}, {9}, {9}, {1}, {3}, true)},
        {"OneD1x11Dilation5Nhwc", conv(16, 32, {170}, {11}, {25}, {25}, {1}, {5}, true, 1, 3)},
        {"Dilation6", conv(16, 24, {20, 26}, {3, 3}, {6, 6}, {6, 6}, {1, 1}, {6, 6})},
        {"OneD1x4", conv(32, 48, {90}, {4}, {0}, {0}, {1}, {1})},
        {"OneD1x4Stride2", conv(16, 32, {90}, {4}, {1}, {1}, {2}, {1})},
        {"OneD1x16Stride8", conv(32, 64, {203}, {16}, {4}, {4}, {8}, {1})},
        {"ThreeD2x2x2Stride2",
         conv(8, 16, {6, 8, 10}, {2, 2, 2}, {0, 0, 0}, {0, 0, 0}, {2, 2, 2}, {1, 1, 1})},
        // Width strides dividing the filter width over widths they divide: also served as the
        // width fold's stride-1 problem (HiFi-GAN's 1x16 stride 8, padding 4; patch-like 2x2
        // stride 2 channels-last).
        {"OneD1x16Stride8Fold", conv(32, 64, {256}, {16}, {4}, {4}, {8}, {1})},
        {"OneD1x16Stride8FoldNhwc", conv(32, 64, {256}, {16}, {4}, {4}, {8}, {1}, true)},
        {"Fold2x2Stride2Nhwc",
         conv(16, 32, {12, 16}, {2, 2}, {0, 0}, {0, 0}, {2, 2}, {1, 1}, true)},
        {"Deeplab3x3Dilation36",
         conv(16, 16, {76, 76}, {3, 3}, {36, 36}, {36, 36}, {1, 1}, {36, 36}, false, 1, 1)},
        {"Wide1x11Dilation3", conv(16, 16, {3, 40}, {1, 11}, {0, 15}, {0, 15}, {1, 1}, {1, 3})},
        {"Dilation2", conv(16, 24, {11, 13}, {3, 3}, {2, 2}, {2, 2}, {1, 1}, {2, 2})},
        {"Deeplab3x3Dilation12",
         conv(16, 16, {30, 30}, {3, 3}, {12, 12}, {12, 12}, {1, 1}, {12, 12}, false, 1, 1)},
        {"Deeplab3x3Dilation12Nhwc",
         conv(16, 16, {30, 30}, {3, 3}, {12, 12}, {12, 12}, {1, 1}, {12, 12}, true, 1, 1)},
        // Channels-last deeplab and SSD dilations on their own specialized configs (none
        // falls to the BLAS path): d4 with partial channel tiles, d6, d24 over three images
        // (ring tracks of two lengths) and d36 on a 65 x 65 map.
        {"Deeplab3x3Dilation4Nhwc",
         conv(24, 40, {21, 26}, {3, 3}, {4, 4}, {4, 4}, {1, 1}, {4, 4}, true, 1, 2)},
        {"Dilation6Nhwc", conv(16, 24, {20, 26}, {3, 3}, {6, 6}, {6, 6}, {1, 1}, {6, 6}, true)},
        {"Deeplab3x3Dilation24ThreeImagesNhwc",
         conv(24, 40, {41, 45}, {3, 3}, {24, 24}, {24, 24}, {1, 1}, {24, 24}, true, 1, 3)},
        {"Deeplab65Dilation36Nhwc",
         conv(16, 24, {65, 65}, {3, 3}, {36, 36}, {36, 36}, {1, 1}, {36, 36}, true, 1, 1)},
        // Ring tracks of two lengths (ringTrackOf): dilation 24 on 41 rows (classes below 17
        // one step longer) over three images, so splits start in long and short tracks and in
        // a later image; and dilation 36 on a 65 x 65 map (2 dy rows below class 29, else 1)
        // with partial channel tiles.
        {"Deeplab3x3Dilation24ThreeImages",
         conv(24, 40, {41, 45}, {3, 3}, {24, 24}, {24, 24}, {1, 1}, {24, 24}, false, 1, 3)},
        {"Deeplab65Dilation36PartialTiles",
         conv(24, 40, {65, 65}, {3, 3}, {36, 36}, {36, 36}, {1, 1}, {36, 36}, false, 1, 2)},
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
        // Channels-first planes read as one row (9 x 13 = 117 positions: chunks run across row
        // ends, the last partial), three images, partial channel tiles; and a 3-D plane.
        {"Pointwise1x1PlaneAsOneRow",
         conv(40, 72, {9, 13}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, false, 1, 3)},
        {"Pointwise1x1x1PlaneAsOneRow",
         conv(24, 40, {3, 5, 7}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1})},
        {"Pad0OddChannels", conv(5, 7, {7, 9}, {3, 3}, {0, 0}, {0, 0}, {1, 1}, {1, 1})},
        // The row reduce: channel runs of 64, 64 and 8 (K C R S).
        {"Nchw3x3ThreeChannelRuns", conv(136, 24, {6, 9}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1})},
        // A filter as wide as its map (FCN's point_output, as 1-D): one output position per
        // image.
        {"PointOutput1x7", conv(24, 40, {7}, {7}, {0}, {0}, {1}, {1}, false, 1, 8)},
        {"PaddingPastTheFilter", conv(8, 16, {6, 9}, {3, 3}, {3, 4}, {3, 4}, {1, 1}, {1, 1})},
        {"ManySteps", conv(64, 96, {20, 70}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, false, 1, 4)},
    };
    // Small-C window tiles: a partial BC8 tile, two tiles with a channel tail, and the
    // Cg16 admission boundary, independently of batch and group count. Restrict the loop
    // so a generic config cannot stand in for the native small-C numerical path.
    for(const int64_t stride : {1, 2})
    {
        for(size_t variant = 0; variant < 3; ++variant)
        {
            constexpr std::array<int64_t, 3> CHANNELS{5, 9, 16};
            constexpr std::array<int64_t, 3> OUTPUT_CHANNELS{17, 23, 31};
            constexpr std::array<int64_t, 3> HEIGHTS{3, 11, 9};
            constexpr std::array<int64_t, 3> WIDTHS{31, 65, 129};
            const int64_t groups = static_cast<int64_t>(variant) + 1;
            auto shape = conv(CHANNELS[variant] * groups,
                              OUTPUT_CHANNELS[variant] * groups,
                              {HEIGHTS[variant], WIDTHS[variant]},
                              {3, 3},
                              {1, 1},
                              {1, 1},
                              {stride, stride},
                              {1, 1},
                              false,
                              groups,
                              groups);
            if(variant == 2)
            {
                shape.mode = ConvMode::CONVOLUTION;
            }
            // The short mixed-filter case has fewer than eight row-chunk steps:
            // loop6 must write dw without partials, including at stride two.
            DeviceCase deviceCase{"NativeSmallCStride" + std::to_string(stride) + "Cg"
                                      + std::to_string(CHANNELS[variant]),
                                  shape};
            deviceCase.otherFilterLayout = variant == 0;
            deviceCase.onlyLoop = 6;
            deviceCase.sixteenBitOnly = true;
            cases.push_back(std::move(deviceCase));
        }
    }

    // Whole-vector and clamped vector loads, just below/at/above both new position tiles.
    // Both layouts see each width, while per-group channels cross BK64/BC64 and retain tails.
    for(const bool channelsLast : {false, true})
    {
        constexpr std::array<int64_t, 6> WIDTHS{127, 128, 129, 255, 256, 257};
        constexpr std::array<int64_t, 3> CHANNELS{24, 72, 80};
        constexpr std::array<int64_t, 3> OUTPUT_CHANNELS{40, 80, 72};
        for(size_t variant = 0; variant < WIDTHS.size(); ++variant)
        {
            const size_t channelVariant = variant % CHANNELS.size();
            const int64_t groups = static_cast<int64_t>(channelVariant) + 1;
            const bool oneSplit = variant < 2;
            DeviceCase deviceCase{"NativePointwiseRing"
                                      + std::string(channelsLast ? "NhwcW" : "NchwW")
                                      + std::to_string(WIDTHS[variant]),
                                  conv(CHANNELS[channelVariant] * groups,
                                       OUTPUT_CHANNELS[channelVariant] * groups,
                                       {oneSplit ? 3 : 9, WIDTHS[variant]},
                                       {1, 1},
                                       {0, 0},
                                       {0, 0},
                                       {1, 1},
                                       {1, 1},
                                       channelsLast,
                                       groups,
                                       oneSplit ? 1 : 3)};
            deviceCase.onlyLoop = 1;
            cases.push_back(std::move(deviceCase));
        }
    }
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
    // r(2+1)d's temporal 3x1x1, channels-first, input channels that are not whole vectors.
    cases.push_back(
        {"ThreeD3x1x1UnalignedChannels",
         conv(46, 16, {4, 6, 7}, {3, 1, 1}, {1, 0, 0}, {1, 0, 0}, {1, 1, 1}, {1, 1, 1})});
    // Inception's 5x5 channels-last (the gfx11 output rows; the generic loop elsewhere).
    cases.push_back({"Inception5x5Nhwc",
                     conv(16, 24, {11, 13}, {5, 5}, {2, 2}, {2, 2}, {1, 1}, {1, 1}, true),
                     false,
                     false,
                     true});
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
    // 5x5x5 volumes: unit stride on the ring / fp32 rows, stride 2 on the
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
    // specialized config has (5x5, 1x9, 1x5x5, 3x3 dilation 3), and strided x (padded rows: no
    // packed layout) in either form.
    const auto generic
        = [&](const std::string& name, const ConvShape& shape, std::vector<int64_t> xStrides = {}) {
              DeviceCase deviceCase{name, shape};
              deviceCase.genericOnly = true;
              deviceCase.xStrides = std::move(xStrides);
              cases.push_back(deviceCase);
          };
    generic("Generic5x5Nhwc", conv(24, 40, {13, 17}, {5, 5}, {2, 2}, {2, 2}, {1, 1}, {1, 1}, true));
    // r(2+1)d's temporal 3x1x1 with input channels that are not whole vectors (230, 45): the
    // generic loop's channels-last elements. (Channels-first: the specialized tier.)
    generic("Generic3x1x1UnalignedChannelsNdhwc",
            conv(46, 16, {4, 6, 7}, {3, 1, 1}, {1, 0, 0}, {1, 0, 0}, {1, 1, 1}, {1, 1, 1}, true));
    generic("Generic1x9Nhwc", conv(32, 64, {150}, {9}, {4}, {4}, {1}, {1}, true));
    // Dilated width taps (the shared window reads them at offsets of 2).
    generic("Generic1x5Dilation2Nhwc", conv(32, 32, {90}, {5}, {4}, {4}, {1}, {2}, true));
    generic(
        "Generic1x5x5ThreeDNhwcGrouped",
        conv(32, 32, {3, 9, 11}, {1, 5, 5}, {0, 2, 2}, {0, 2, 2}, {1, 1, 1}, {1, 1, 1}, true, 2));
    generic("GenericDilation3Nhwc",
            conv(16, 16, {40, 40}, {3, 3}, {3, 3}, {3, 3}, {1, 1}, {3, 3}, true, 1, 1));
    generic("Generic7x7Stride2Nhwc",
            conv(24, 24, {19, 23}, {7, 7}, {3, 3}, {3, 3}, {2, 2}, {1, 1}, true));
    // Channels-first and channels-last geometries no specialized config has: 1-D 9 taps at
    // stride 3 and dilation 2, 3-D 3x5x3 with mixed padding and dilation, a 4x4 stride-4
    // patchify, and 3-D stride 2 dilation 2 channels-last.
    generic("Generic1x9Stride3Dilation2Nchw", conv(24, 40, {100}, {9}, {4}, {4}, {3}, {2}));
    generic("Generic3x5x3ThreeDNchw",
            conv(16, 24, {5, 7, 9}, {3, 5, 3}, {1, 2, 1}, {1, 2, 1}, {1, 2, 1}, {1, 1, 2}));
    generic("GenericPatchify4x4Stride4Nchw",
            conv(16, 32, {16, 20}, {4, 4}, {0, 0}, {0, 0}, {4, 4}, {1, 1}));
    generic("Generic3x3x3Stride2Dilation2Ndhwc",
            conv(16, 32, {7, 9, 11}, {3, 3, 3}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}, true));
    // ASPP-class dilations no specialized config has (3x3 d18 and d30, 1x3 d12), in both layouts:
    // the generic loop serves them on gfx11 and on gfx12's matrix atoms (elsewhere it declines
    // them for BLAS).
    const auto aspp = [&](const std::string& name, const ConvShape& shape) {
        DeviceCase deviceCase{name, shape};
        deviceCase.genericOnly = true;
        deviceCase.someArchs = true;
        cases.push_back(deviceCase);
    };
    aspp("GenericDilation18Nhwc",
         conv(16, 24, {40, 44}, {3, 3}, {18, 18}, {18, 18}, {1, 1}, {18, 18}, true));
    aspp("GenericDilation18Nchw",
         conv(16, 24, {40, 44}, {3, 3}, {18, 18}, {18, 18}, {1, 1}, {18, 18}));
    aspp("GenericDilation30PartialTilesNchw",
         conv(24, 40, {65, 65}, {3, 3}, {30, 30}, {30, 30}, {1, 1}, {30, 30}, false, 1, 1));
    aspp("GenericOneD1x3Dilation12Nhwc", conv(32, 48, {130}, {3}, {12}, {12}, {1}, {12}, true));
    // 24 channels of 10 x 12, rows 16 apart (channels-first elements).
    generic("GenericPaddedRowsNchw",
            conv(24, 40, {10, 12}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}),
            {int64_t{24} * 10 * 16, int64_t{10} * 16, 16, 1});
    // Channels-last, positions 32 channels apart (24 used): channel vectors.
    generic("GenericPaddedChannelsNhwc",
            conv(24, 40, {10, 12}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, true),
            {int64_t{10} * 12 * 32, 1, int64_t{12} * 32, 32});
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
    // Three channels-last images of 4096 x 320^2 (2.5 GiB in 16-bit, 5 GiB in fp32): the
    // tensor is past 2 GiB, each image within it, so every config reads its image through a
    // 64-bit base. 1x1 stride 4 keeps the CPU reference short.
    {
        DeviceCase largeBatch{
            "LargeBatchGenericNhwc",
            conv(4096, 8, {320, 320}, {1, 1}, {0, 0}, {0, 0}, {4, 4}, {1, 1}, true, 1, 3)};
        largeBatch.someArchs = true;
        cases.push_back(largeBatch);
    }
    // Positions 26 apart: not whole vectors, channels-last elements.
    generic("GenericPaddedChannelsNhwcElements",
            conv(24, 40, {10, 12}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, true),
            {int64_t{10} * 12 * 26, 1, int64_t{12} * 26, 26});
    for(auto& deviceCase : cases)
    {
        static const std::set<std::string> s_onGfx908{"Nhwc3x3ThreeOutputChannels",
                                                      "Nhwc1x1FiveOutputChannels",
                                                      "ThreeDNhwc3x3x3ThreeOutputChannels",
                                                      "Stem7x7Stride2",
                                                      "Stem7x7Stride2K64",
                                                      "Stem3x3Stride1C5",
                                                      "Stem3x3Stride2ThreeD",
                                                      "Stem7x7Stride2Nhwc",
                                                      "Stem3x3Stride2NhwcOneChannel",
                                                      "Stem3x3Stride1C5Nhwc",
                                                      "Stem3x3Stride2ThreeDNhwc"};
        deviceCase.gfx908 = deviceCase.genericOnly || s_onGfx908.count(deviceCase.name) != 0;
    }
    const auto nativeCase = [&](std::string name, ConvShape shape, bool filterOther, int64_t loop) {
        DeviceCase test{std::move(name), std::move(shape), filterOther};
        test.onlyLoop = loop;
        cases.push_back(std::move(test));
    };
    nativeCase(
        "GenericThreeDGroupedTails",
        conv(10, 14, {5, 9, 17}, {2, 4, 5}, {1, 4, 6}, {0, 1, 2}, {2, 3, 2}, {2, 2, 3}, false, 2),
        true,
        conv_wrw::LOOP_NATIVE_GENERIC);
    nativeCase(
        "GenericThreeDGroupedTailsNdhwc",
        conv(10, 14, {5, 9, 17}, {2, 4, 5}, {1, 4, 6}, {0, 1, 2}, {2, 3, 2}, {2, 2, 3}, true, 2),
        true,
        conv_wrw::LOOP_NATIVE_GENERIC);
    nativeCase("GenericWideStridedFilter",
               conv(3, 5, {57}, {17}, {15}, {4}, {3}, {2}, false, 1, 3),
               false,
               conv_wrw::LOOP_NATIVE_GENERIC);
    nativeCase("GenericMostlyPadding",
               conv(1, 1, {1, 1}, {3, 3}, {9, 7}, {0, 0}, {1, 1}, {1, 1}),
               false,
               conv_wrw::LOOP_NATIVE_GENERIC);
    auto genericConvolution = conv(5, 7, {17, 23}, {3, 3}, {6, 6}, {6, 6}, {1, 1}, {6, 6}, true);
    genericConvolution.mode = ConvMode::CONVOLUTION;
    nativeCase("GenericDilation6MixedLayoutConvolution",
               genericConvolution,
               true,
               conv_wrw::LOOP_NATIVE_GENERIC);
    auto genericThreeDConvolution
        = conv(10, 14, {5, 9, 17}, {2, 4, 5}, {1, 4, 6}, {0, 1, 2}, {2, 3, 2}, {2, 2, 3}, true, 2);
    genericThreeDConvolution.mode = ConvMode::CONVOLUTION;
    nativeCase("GenericThreeDGroupedConvolution",
               genericThreeDConvolution,
               true,
               conv_wrw::LOOP_NATIVE_GENERIC);
    nativeCase("TapTiledPatchNhwc",
               conv(3, 19, {35, 49}, {16, 16}, {2, 1}, {1, 3}, {16, 16}, {1, 1}, true, 1, 3),
               false,
               conv_wrw::LOOP_TAP_TILED);
    auto tapTiledGrouped
        = conv(10, 14, {5, 9, 17}, {2, 4, 5}, {1, 4, 6}, {0, 1, 2}, {2, 3, 2}, {2, 2, 3}, true, 2);
    tapTiledGrouped.mode = ConvMode::CONVOLUTION;
    nativeCase("TapTiledGroupedThreeDMixedFilter", tapTiledGrouped, true, conv_wrw::LOOP_TAP_TILED);
    auto smallKGrouped = conv(
        10, 6, {4, 7, 11}, {2, 3, 3}, {1, 2, 2}, {0, 1, 2}, {1, 2, 1}, {1, 1, 2}, true, 2, 3);
    smallKGrouped.mode = ConvMode::CONVOLUTION;
    nativeCase("SmallKGroupedThreeDMixedFilter", smallKGrouped, true, conv_wrw::LOOP_SMALL_K);
    nativeCase("SmallKMostlyPaddingDirect",
               conv(1, 1, {1}, {3}, {5}, {4}, {1}, {2}, true, 1, 1),
               false,
               conv_wrw::LOOP_SMALL_K);
    nativeCase("SmallKPointwiseChannelTail",
               conv(129, 4, {3, 19}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, true, 1, 3),
               false,
               conv_wrw::LOOP_SMALL_K);
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
    const auto function = program->getKernel(kernel.source.entryPoint);
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
    if(base == "gfx1250")
    {
        return true;
    }
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

/// The configs a case runs: the graph's own element type, or three-bf16 (math 1: fp32 graphs,
/// CONV_WRW_TF32's configs, the bf16 kernel on split operands).
enum class Math
{
    NATIVE,
    THREE_BF16,
};

/// +inf and -inf planted into @p input (the tensor the reduction reads, besides the filter).
template <typename T>
void plantInfinities(Tensor<T>& input)
{
    int64_t count = 1;
    for(const auto extent : input.dims())
    {
        count *= extent;
    }
    auto* data = static_cast<T*>(input.rawHostData());
    data[0] = static_cast<T>(INFINITY);
    data[count / 2] = static_cast<T>(-INFINITY);
    input.markHostModified();
}

/// A result matches its reference: within @p tolerance where the reference is finite, the same
/// infinity, or NaN where the reference is NaN.
bool matches(float got, float want, float tolerance)
{
    if(std::isnan(want))
    {
        return std::isnan(got);
    }
    if(std::isinf(want))
    {
        return got == want;
    }
    return std::fabs(got - want) <= tolerance;
}

/// The reduce's dw is the fixed-order sum of the partials @p workspace holds after a launch
/// of @p launch, bit for bit: per element, split group j (0..3) adds splits j, j + 4, ... in
/// order from +0, then ((g0 + g1) + g2) + g3, rounded once to dw's type (both reduces, quad and
/// tile, sum in this order; a NaN matches any NaN). @p dw is the device dw in its layout, taps
/// reflected under flip.
template <typename TO>
void expectReduceIsFixedOrderSum(const conv_wrw::WrwLaunch& launch,
                                 const void* workspace,
                                 const void* dw,
                                 const std::vector<int64_t>& dwDims,
                                 const std::vector<int64_t>& dwStrides)
{
    const auto& r = launch.reduce;
    const size_t partialFloats
        = static_cast<size_t>(r.splits) * static_cast<size_t>(r.partialElems);
    std::vector<float> partials(partialFloats);
    ASSERT_EQ(
        hipMemcpy(partials.data(),
                  static_cast<const char*>(workspace) + (launch.split ? launch.splitBytes : 0),
                  partialFloats * sizeof(float),
                  hipMemcpyDeviceToHost),
        hipSuccess);
    int64_t count = 1;
    for(const auto extent : dwDims)
    {
        count *= extent;
    }
    // The last element's offset: dw may be packed in either filter layout.
    int64_t span = 0;
    for(size_t axis = 0; axis < dwDims.size(); ++axis)
    {
        span += (dwDims[axis] - 1) * dwStrides[axis];
    }
    std::vector<TO> got(static_cast<size_t>(span + 1));
    ASSERT_EQ(hipMemcpy(got.data(), dw, got.size() * sizeof(TO), hipMemcpyDeviceToHost),
              hipSuccess);
    int64_t mismatches = 0;
    const int64_t groups = r.groups;
    const int64_t taps = r.taps;
    const int64_t kg = r.kg;
    const int64_t cg = r.cg;
    const int64_t splits = r.splits;
    // A value's bits (a NaN or a signed zero compares by its encoding).
    const auto bitsOf = [](const TO& v) {
        std::conditional_t<sizeof(TO) == 4, uint32_t, uint16_t> u = 0;
        std::memcpy(&u, &v, sizeof(u));
        return u;
    };
    for(int64_t g = 0; g < groups; ++g)
    {
        for(int64_t tap = 0; tap < taps; ++tap)
        {
            for(int64_t k = 0; k < kg; ++k)
            {
                for(int64_t c = 0; c < cg; ++c)
                {
                    const int64_t at = ((g * taps + tap) * kg + k) * cg + c;
                    std::array<float, 4> group{0.0f, 0.0f, 0.0f, 0.0f};
                    for(int64_t sp = 0; sp < splits; ++sp)
                    {
                        group[static_cast<size_t>(sp % 4)]
                            += partials[static_cast<size_t>(sp * r.partialElems + at)];
                    }
                    float sum = group[0];
                    sum += group[1];
                    sum += group[2];
                    sum += group[3];
                    const TO want = static_cast<TO>(sum);
                    // dw index: (g kg + k, c, taps in row-major order over dims 2..). The width
                    // fold's folded (channel j C + c, tap (outer, tau)) is the graph's channel c
                    // and width tap (tau - pre) s + j + pad, absent outside the filter.
                    int64_t dwc = c;
                    int64_t dwtapIndex = tap;
                    bool flip = r.flip != 0;
                    int64_t dwTaps = taps;
                    if(launch.fold)
                    {
                        const auto& f = launch.reduceFold;
                        const int64_t j = c / f.c;
                        const int64_t t = (tap % f.fw - f.pre) * f.s + j + f.pad;
                        if(t < 0 || t >= f.filterW)
                        {
                            continue;
                        }
                        dwc = c - j * f.c;
                        dwtapIndex = tap / f.fw * f.filterW + t;
                        flip = f.flip != 0;
                        dwTaps = static_cast<int64_t>(f.outer) * f.filterW;
                    }
                    const int64_t wtap = flip ? dwTaps - 1 - dwtapIndex : dwtapIndex;
                    int64_t offset = (g * kg + k) * dwStrides[0] + dwc * dwStrides[1];
                    int64_t rest = wtap;
                    for(size_t axis = dwDims.size() - 1; axis >= 2; --axis)
                    {
                        offset += (rest % dwDims[axis]) * dwStrides[axis];
                        rest /= dwDims[axis];
                    }
                    const TO have = got[static_cast<size_t>(offset)];
                    // NaN (+inf meeting -inf) carries no defined payload: any NaN matches.
                    const bool bothNan = std::isnan(static_cast<float>(want))
                                         && std::isnan(static_cast<float>(have));
                    if(!bothNan && bitsOf(want) != bitsOf(have) && ++mismatches <= 5)
                    {
                        ADD_FAILURE() << "reduce at dw offset " << offset << " (g " << g << ", k "
                                      << k << ", c " << c << ", tap " << tap
                                      << "): not the fixed-order sum of " << splits << " splits";
                    }
                }
            }
        }
    }
    const char* reduce = "quad";
    if(r.tile == 1)
    {
        reduce = "tile";
    }
    else if(r.tile == 2)
    {
        reduce = "rows";
    }
    else if(r.tile == 3)
    {
        reduce = "deep";
    }
    else if(r.tile == 4)
    {
        reduce = "fold";
    }
    EXPECT_EQ(mismatches, 0) << "of " << count << " dw elements (reduce " << reduce << ")";
}

/// Runs every config of @p kernels (default: the shipped catalog) the kernel match admits on
/// the case, against the CPU reference.
template <typename T>
void runCase(const DeviceCase& deviceCase,
             Math math = Math::NATIVE,
             bool infinities = false,
             const std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition>* kernels = nullptr)
{
    const auto& configs = kernels != nullptr ? *kernels : catalog();
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
    if((deviceCase.onlyLoop != 0
        && hipdnn_plugin_sdk::stripArchFeatures(device.gcnArchName) != "gfx1250")
       || (deviceCase.sixteenBitOnly && shape.dataType == DataType::FLOAT))
    {
        GTEST_SKIP() << "targeted native WRW family requires gfx1250 and its supported dtype";
    }
    if(deviceCase.dilationRows
       && (dataTypeOf<T>() != DataType::HALF
           || hipdnn_plugin_sdk::stripArchFeatures(device.gcnArchName) != "gfx1250"))
    {
        GTEST_SKIP() << "measured dilation rows require native gfx1250 HALF";
    }
    const GraphFixture fixture(buildGraph(spec), device);
    const auto bound = matchesGraph(PACK, fixture.context());
    ASSERT_TRUE(bound.has_value()) << "the engine refused a case it should serve";
    // Cases only a few configs serve skip before building their (large) tensors.
    if(deviceCase.someArchs
       && std::none_of(configs.begin(), configs.end(), [&](const auto& kernel) {
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
    if(infinities)
    {
        plantInfinities(x);
    }
    std::optional<Tensor<T>> alternateX;
    std::optional<Tensor<T>> alternateY;
    std::optional<Tensor<double>> unroundedReference;
    const bool independentRevisions
        = deviceCase.dilationRows || deviceCase.directChannelsFirstStore;
    if(independentRevisions)
    {
        alternateX.emplace(spec.x.dims, spec.x.strides);
        alternateY.emplace(spec.y.dims, spec.y.strides);
        unroundedReference.emplace(spec.w.dims, spec.w.strides);
    }

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
    // Three-bf16 drops lo x lo and rounds each lo to bf16: about 2^-16 of each product, K2's and
    // K1's three-bf16 bound.
    const float accumulationBound
        = (math == Math::THREE_BF16 ? 3e-5f : 2e-6f) * static_cast<float>(reduction);

    static std::set<std::string> s_ldsChecked;
    const auto& handler = dispatchHandler(PACK);
    int ran = 0;
    int wroteDw = 0;
    for(const auto& kernel : configs)
    {
        if(!matchesKernel(PACK, fixture.context(), kernel, *bound)
           || (kernel.getIntMetadata("math") == 1) != (math == Math::THREE_BF16))
        {
            continue;
        }
        // Loop 10 has its own strict numerical suite below, never the legacy scaled bound.
        if(configOf(kernel).loop == conv_wrw::LOOP_ACCURATE)
        {
            continue;
        }
        if(deviceCase.onlyLoop != 0 && configOf(kernel).loop != deviceCase.onlyLoop)
        {
            continue;
        }
        if(deviceCase.dilationRows)
        {
            const auto candidate = configOf(kernel);
            if(candidate.loop != conv_wrw::LOOP_ROWS || candidate.atom != "wmma125_f16"
               || candidate.layout != 0 || candidate.bk != 64 || candidate.bc != 64
               || (candidate.bq != 32 && candidate.bq != 128) || candidate.wk != 2
               || candidate.wc != 2 || candidate.bpu != 2 || candidate.fr != 1 || candidate.s != 3
               || candidate.sw != 1 || candidate.dw != 24 || candidate.vec != 8
               || candidate.dws != 1)
            {
                continue;
            }
        }
        if(deviceCase.directChannelsFirstStore && configOf(kernel).dws != 2)
        {
            continue;
        }
        // Known open parity item: gfx950's direct loop (5) turns an infinity in x into
        // NaN in other dw elements (0 x inf from positions outside the window), on 16-bit graphs
        // too; left out of the infinity runs until that is fixed.
        if(infinities && kernel.getIntMetadata("loop") == conv_wrw::LOOP_DIRECT)
        {
            continue;
        }
        SCOPED_TRACE(kernel.name);
        ++ran;
        const auto prepared = handler.prepare(fixture.context(), *bound, kernel);
        ASSERT_NE(prepared, nullptr);
        const size_t workspaceBytes = handler.workspaceBytes(fixture.context(), *bound, kernel);
        const auto config = configOf(kernel);
        if(deviceCase.directChannelsFirstStore)
        {
            const auto problem = conv_wrw::parseWrwProblem(fixture.context());
            ASSERT_TRUE(problem.has_value());
            const auto launch = conv_wrw::wrwLaunchFor(*problem, config, device);
            ASSERT_TRUE(launch.has_value());
            EXPECT_EQ(workspaceBytes == 0, launch->args.splits == 1);
        }
        // No workspace: the kernel writes dw itself (never loop 5).
        if(workspaceBytes == 0)
        {
            EXPECT_NE(config.loop, conv_wrw::LOOP_DIRECT);
            ++wroteDw;
        }
        if(s_ldsChecked.insert(kernel.name).second)
        {
            const int compiled = ldsOfCompiled(kernel, config, shape.dataType, device.gcnArchName);
            // Three-bf16 compiles the bf16 kernel.
            const auto host
                = conv_wrw::ldsBytesOf(config, config.math == 1 ? 2 : static_cast<int>(sizeof(T)));
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
        const Handle handle;
        const int revisions = independentRevisions ? 3 : 1;
        for(int revision = 0; revision < revisions; ++revision)
        {
            SCOPED_TRACE(revision);
            // Revision one rebinds fresh allocations; revision two mutates the original pair.
            auto& currentX = revision == 1 ? *alternateX : x;
            auto& currentY = revision == 1 ? *alternateY : y;
            if(revision != 0)
            {
                currentX.fillTensorWithRandomValues(
                    -1.0f, 1.0f, SEED + 2U * static_cast<unsigned>(revision));
                currentY.fillTensorWithRandomValues(
                    -1.0f, 1.0f, SEED + 2U * static_cast<unsigned>(revision) + 1U);
                CpuFpReferenceConvolution::wgrad<T, T, T, double>(currentX,
                                                                  reflect ? crossCorrelation
                                                                          : reference,
                                                                  currentY,
                                                                  shape.stride,
                                                                  shape.dilation,
                                                                  shape.prePadding,
                                                                  shape.postPadding);
                if(reflect)
                {
                    reflectFilter(crossCorrelation, reference);
                }
            }
            if(independentRevisions)
            {
                CpuFpReferenceConvolution::wgrad<T, double, T, double>(currentX,
                                                                       *unroundedReference,
                                                                       currentY,
                                                                       shape.stride,
                                                                       shape.dilation,
                                                                       shape.prePadding,
                                                                       shape.postPadding);
            }
            if(workspace != nullptr)
            {
                ASSERT_EQ(hipMemset(workspace, 0x7f, workspaceBytes), hipSuccess);
            }
            w.fillWithSentinelValue();
            const std::array<hipdnnPluginDeviceBuffer_t, 3> buffers{
                hipdnnPluginDeviceBuffer_t{X_UID, currentX.rawDeviceData()},
                hipdnnPluginDeviceBuffer_t{W_UID, w.rawDeviceData()},
                hipdnnPluginDeviceBuffer_t{Y_UID, currentY.rawDeviceData()}};
            handler.launch(handle,
                           *prepared,
                           buffers.data(),
                           static_cast<uint32_t>(buffers.size()),
                           workspace);
            ASSERT_EQ(hipStreamSynchronize(handle.getStream()), hipSuccess);
            // Deterministic: a second launch gives the same bits (the UED's RFC 0025 claim).
            EXPECT_TRUE(conv_determinism_test::relaunchReproducesOutput(
                [&] {
                    handler.launch(handle,
                                   *prepared,
                                   buffers.data(),
                                   static_cast<uint32_t>(buffers.size()),
                                   workspace);
                },
                handle.getStream(),
                w,
                workspace,
                workspaceBytes));
            if(workspace != nullptr)
            {
                // The partials of the last launch and its dw: the reduce's sum, bit for bit.
                const auto problem = conv_wrw::parseWrwProblem(fixture.context());
                ASSERT_TRUE(problem.has_value());
                const auto launch = conv_wrw::wrwLaunchFor(*problem, config, device);
                ASSERT_TRUE(launch.has_value());
                if(!launch->dwStore)
                {
                    if(config.math == 1)
                    {
                        expectReduceIsFixedOrderSum<float>(
                            *launch, workspace, w.rawDeviceData(), w.dims(), w.strides());
                    }
                    else
                    {
                        expectReduceIsFixedOrderSum<T>(
                            *launch, workspace, w.rawDeviceData(), w.dims(), w.strides());
                    }
                }
            }
            w.markDeviceModified();

            const auto* actual = static_cast<const T*>(w.rawHostData());
            const auto* expected = static_cast<const T*>(reference.rawHostData());
            const double* independent
                = independentRevisions
                      ? static_cast<const double*>(unroundedReference->rawHostData())
                      : nullptr;
            int64_t mismatches = 0;
            forEachIndex(w.dims(), [&](const std::vector<int64_t>& index) {
                const auto got = static_cast<float>(actual[offsetOf(index, w.strides())]);
                const auto legacyWant
                    = static_cast<float>(expected[offsetOf(index, reference.strides())]);
                const float legacyTolerance
                    = accumulationBound
                      + 2.0f * storageEpsilon<T>() * (std::fabs(legacyWant) + 1e-3f);
                int64_t independentOffset = 0;
                if(independentRevisions)
                {
                    for(size_t axis = 0; axis < index.size(); ++axis)
                    {
                        independentOffset
                            += (reflect && axis >= 2 ? w.dims()[axis] - 1 - index[axis]
                                                     : index[axis])
                               * unroundedReference->strides()[axis];
                    }
                }
                const double want = independentRevisions ? independent[independentOffset]
                                                         : static_cast<double>(legacyWant);
                const double tolerance = independentRevisions
                                             ? 0.002 + 0.002 * std::fabs(want)
                                             : static_cast<double>(legacyTolerance);
                const bool passes = independentRevisions
                                        ? std::fabs(static_cast<double>(got) - want) <= tolerance
                                        : matches(got, legacyWant, legacyTolerance);
                if(!passes && ++mismatches <= 5)
                {
                    ADD_FAILURE() << "at linear index " << offsetOf(index, w.strides()) << ": got "
                                  << got << ", want " << want << ", tolerance " << tolerance;
                }
            });
            EXPECT_EQ(mismatches, 0);
        }
        if(workspace != nullptr)
        {
            ASSERT_EQ(hipFree(workspace), hipSuccess);
        }
    }
    // Three-bf16 is admitted only where it measured faster (threeBf16Pays).
    if(ran == 0 && math == Math::THREE_BF16)
    {
        const auto problem = conv_wrw::parseWrwProblem(fixture.context());
        if(problem.has_value() && !conv_wrw::threeBf16Pays(*problem, device.gcnArchName))
        {
            GTEST_SKIP() << "three-bf16 is not admitted for this class on " << device.gcnArchName;
        }
    }
    const auto base = std::string(hipdnn_plugin_sdk::stripArchFeatures(device.gcnArchName));
    const bool gfx906 = base == "gfx906";
    const bool fmaOrFp32 = shape.dataType == DataType::FLOAT || gfx906;
    const bool on908 = base == "gfx908" && shape.dataType != DataType::FLOAT && deviceCase.gfx908;
    if(ran == 0 && !on908
       && (!expectServed(device.gcnArchName, shape.dataType)
           || (deviceCase.matrix16Only && fmaOrFp32) || (deviceCase.noGfx906 && gfx906)
           || deviceCase.someArchs || (deviceCase.genericOnly && gfx906)))
    {
        GTEST_SKIP() << "no catalog config for this case on " << device.gcnArchName;
    }
    EXPECT_GE(ran, deviceCase.dilationRows ? 2 : 1) << "no catalog config serves this case";
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

/// +inf and -inf in x on the generic loop's cases, every element type: each dw element an
/// infinity reaches is that infinity (NaN where both meet), as the reference gives, whichever
/// configs serve the case.
class TestGpuConvWrwGenericInfinities : public ::testing::TestWithParam<DeviceCase>
{
};

TEST_P(TestGpuConvWrwGenericInfinities, Float)
{
    SKIP_IF_NO_DEVICES();
    runCase<float>(GetParam(), Math::NATIVE, true);
}

TEST_P(TestGpuConvWrwGenericInfinities, Half)
{
    SKIP_IF_NO_DEVICES();
    runCase<Half>(GetParam(), Math::NATIVE, true);
}

TEST_P(TestGpuConvWrwGenericInfinities, BFloat16)
{
    SKIP_IF_NO_DEVICES();
    runCase<BFloat16>(GetParam(), Math::NATIVE, true);
}

std::vector<DeviceCase> genericInfinityCases()
{
    std::vector<DeviceCase> cases;
    for(const auto& deviceCase : deviceCases())
    {
        if(deviceCase.genericOnly)
        {
            cases.push_back(deviceCase);
        }
    }
    return cases;
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvWrwGenericInfinities,
                         ::testing::ValuesIn(genericInfinityCases()),
                         [](const auto& info) { return info.param.name; });

/// +inf and -inf in x on the row ring's cases, every element type, on the matrix-core ring's
/// configs (loop 1): the one-step ring and its stages of narrow steps (both layouts) carry an
/// infinity to every dw element it reaches, as the reference gives.
class TestGpuConvWrwRingInfinities : public ::testing::TestWithParam<DeviceCase>
{
};

/// The catalog's matrix-core ring configs.
const std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition>& ringCatalog()
{
    static const auto s_ring = [] {
        std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition> out;
        for(const auto& kernel : catalog())
        {
            if(configOf(kernel).loop == conv_wrw::LOOP_RING)
            {
                out.push_back(kernel);
            }
        }
        return out;
    }();
    return s_ring;
}

TEST_P(TestGpuConvWrwRingInfinities, Float)
{
    SKIP_IF_NO_DEVICES();
    runCase<float>(GetParam(), Math::NATIVE, true, &ringCatalog());
}

TEST_P(TestGpuConvWrwRingInfinities, Half)
{
    SKIP_IF_NO_DEVICES();
    runCase<Half>(GetParam(), Math::NATIVE, true, &ringCatalog());
}

TEST_P(TestGpuConvWrwRingInfinities, BFloat16)
{
    SKIP_IF_NO_DEVICES();
    runCase<BFloat16>(GetParam(), Math::NATIVE, true, &ringCatalog());
}

std::vector<DeviceCase> ringInfinityCases()
{
    std::vector<DeviceCase> cases;
    for(const auto& deviceCase : deviceCases())
    {
        if(deviceCase.name == "Nhwc3x3" || deviceCase.name == "Nchw3x3"
           || deviceCase.name.rfind("ThreeDNhwc3x3x3Rows", 0) == 0
           || deviceCase.name == "ThreeD3x3x3Rows8")
        {
            // Skipped where no ring config serves the type (fp32 on RDNA and gfx906: FMA loops).
            auto ringCase = deviceCase;
            ringCase.someArchs = true;
            cases.push_back(ringCase);
        }
    }
    return cases;
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvWrwRingInfinities,
                         ::testing::ValuesIn(ringInfinityCases()),
                         [](const auto& info) { return info.param.name; });

/// gfx950's channels-last ring kept channels-last (loop 1, layout 2) at 128 x 128 tiles, a
/// geometry the kernel match admits but the catalog does not ship: two k-fragments a wave (w2x4,
/// 288 accumulators a lane, so it spills) and one (w4x4), 64 positions a step, on 320 channels
/// (partial k and c tiles: the last tile's second k-wave and last two c-waves hold no channel).
/// A FlyDSL port of this loop computed wrong dw here; the HIP kernel must not.
template <typename T>
void runWideTilesPartialChannels()
{
    const std::string atom = std::is_same_v<T, Half> ? "mfma32x16_f16" : "mfma32x16_bf16";
    const auto base = std::find_if(catalog().begin(), catalog().end(), [&](const auto& kernel) {
        return kernel.getStringMetadata("atom") == atom
               && kernel.getIntMetadata("loop") == conv_wrw::LOOP_RING
               && kernel.getIntMetadata("layout") == conv_wrw::LAYOUT_CHANNELS_LAST_TR
               && kernel.getIntMetadata("fr") == 3 && kernel.getIntMetadata("s") == 3
               && kernel.getIntMetadata("dw") == 1 && kernel.getIntMetadata("pb") == 0
               && kernel.getIntMetadata("math") == 0;
    });
    ASSERT_NE(base, catalog().end()) << "no channels-last ring config of " << atom;
    std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition> kernels;
    // (bk, bc, bq, wk, wc)
    for(const auto& [bk, bc, bq, wk, wc] :
        std::vector<std::array<int64_t, 5>>{{128, 128, 64, 2, 4}, {128, 128, 64, 4, 4}})
    {
        auto kernel = *base;
        kernel.metadata["bk"] = bk;
        kernel.metadata["bc"] = bc;
        kernel.metadata["bq"] = bq;
        kernel.metadata["wk"] = wk;
        kernel.metadata["wc"] = wc;
        kernel.metadata["bpu"] = int64_t{1};
        kernel.name = "wide." + std::to_string(bk) + "x" + std::to_string(bc) + "x"
                      + std::to_string(bq) + ".w" + std::to_string(wk) + "x" + std::to_string(wc);
        kernels.push_back(std::move(kernel));
    }
    DeviceCase deviceCase{"WideTiles320",
                          conv(320, 320, {16, 18}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, true)};
    // Admitted on gfx950 only (transpose reads, LDS past 64 KiB): skipped elsewhere.
    deviceCase.someArchs = true;
    runCase<T>(deviceCase, Math::NATIVE, false, &kernels);
}

/// The deep quad reduce (ConvWrwReduceDeep): many splits into few quads (c8 k8 3x3: 144 quads,
/// 16 a block), against the CPU reference and, bit for bit, the quad reduce's fixed-order sum.
/// Up to four admitted configs whose launch takes it.
template <typename T>
void runDeepReduce()
{
    const DeviceCase deviceCase{
        "DeepReduce", conv(8, 8, {16, 16}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, false, 1, 64)};
    auto shape = deviceCase.shape;
    shape.dataType = dataTypeOf<T>();
    const auto spec = specFor(shape, ConvDirection::BACKWARD_WEIGHTS);
    const auto device = currentDeviceProperties();
    const GraphFixture fixture(buildGraph(spec), device);
    const auto bound = matchesGraph(PACK, fixture.context());
    ASSERT_TRUE(bound.has_value());
    const auto problem = conv_wrw::parseWrwProblem(fixture.context());
    ASSERT_TRUE(problem.has_value());
    std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition> kernels;
    for(const auto& kernel : catalog())
    {
        if(kernels.size() >= 4 || !matchesKernel(PACK, fixture.context(), kernel, *bound)
           || kernel.getIntMetadata("math") != 0)
        {
            continue;
        }
        const auto launch = conv_wrw::wrwLaunchFor(*problem, configOf(kernel), device);
        if(launch.has_value() && !launch->dwStore && launch->reduce.tile == 3)
        {
            EXPECT_GE(launch->reduce.splits, K3_REDUCE_DEEP_MIN_SPLITS);
            EXPECT_EQ(launch->reduceBlock, 4u * static_cast<unsigned int>(launch->reduce.quads));
            kernels.push_back(kernel);
        }
    }
    if(kernels.empty())
    {
        GTEST_SKIP() << "no config takes the deep reduce here on " << device.gcnArchName;
    }
    runCase<T>(deviceCase, Math::NATIVE, false, &kernels);
}

TEST(TestGpuConvWrwDeepReduce, Float)
{
    SKIP_IF_NO_DEVICES();
    runDeepReduce<float>();
}

TEST(TestGpuConvWrwDeepReduce, Half)
{
    SKIP_IF_NO_DEVICES();
    runDeepReduce<Half>();
}

TEST(TestGpuConvWrwDeepReduce, BFloat16)
{
    SKIP_IF_NO_DEVICES();
    runDeepReduce<BFloat16>();
}

/// The width fold (widthFoldOf): up to four admitted configs per case whose launch runs the
/// folded stride-1 problem (staging channels-first x, the fold reduce writing dw), against the
/// CPU reference and, bit for bit, the fixed-order sum of the folded partials. True convolution
/// reflects the graph's taps.
template <typename T>
void runFold(const DeviceCase& deviceCase)
{
    auto shape = deviceCase.shape;
    shape.dataType = dataTypeOf<T>();
    const auto spec = specFor(shape, ConvDirection::BACKWARD_WEIGHTS);
    const auto device = currentDeviceProperties();
    const GraphFixture fixture(buildGraph(spec), device);
    const auto bound = matchesGraph(PACK, fixture.context());
    ASSERT_TRUE(bound.has_value());
    const auto problem = conv_wrw::parseWrwProblem(fixture.context());
    ASSERT_TRUE(problem.has_value());
    ASSERT_TRUE(conv_wrw::widthFoldOf(*problem).has_value());
    std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition> kernels;
    for(const auto& kernel : catalog())
    {
        if(kernels.size() >= 4 || !matchesKernel(PACK, fixture.context(), kernel, *bound)
           || kernel.getIntMetadata("math") != 0)
        {
            continue;
        }
        const auto launch = conv_wrw::wrwLaunchFor(*problem, configOf(kernel), device);
        if(launch.has_value() && launch->fold)
        {
            EXPECT_EQ(launch->reduce.tile, 4);
            EXPECT_EQ(launch->foldStage, problem->channelsFirst);
            kernels.push_back(kernel);
        }
    }
    if(kernels.empty())
    {
        GTEST_SKIP() << "no config serves the fold here on " << device.gcnArchName;
    }
    runCase<T>(deviceCase, Math::NATIVE, false, &kernels);
}

std::vector<DeviceCase> foldCases()
{
    auto convolution = conv(16, 24, {64}, {4}, {1}, {1}, {2}, {1});
    convolution.mode = ConvMode::CONVOLUTION;
    return {{"OneD1x16Stride8", conv(32, 64, {256}, {16}, {4}, {4}, {8}, {1})},
            {"OneD1x16Stride8Nhwc", conv(32, 64, {256}, {16}, {4}, {4}, {8}, {1}, true)},
            {"TwoD2x2Stride2", conv(16, 32, {12, 16}, {2, 2}, {0, 0}, {0, 0}, {2, 2}, {1, 1})},
            {"TwoD2x2Stride2Nhwc",
             conv(16, 32, {12, 16}, {2, 2}, {0, 0}, {0, 0}, {2, 2}, {1, 1}, true)},
            {"OneD1x4Stride2Convolution", convolution},
            // A width the stride does not divide: the staged tail is zero.
            {"OneD1x16Stride8RaggedWidth", conv(32, 64, {250}, {16}, {4}, {4}, {8}, {1})}};
}

class TestGpuConvWrwFold : public ::testing::TestWithParam<DeviceCase>
{
};

TEST_P(TestGpuConvWrwFold, Float)
{
    SKIP_IF_NO_DEVICES();
    runFold<float>(GetParam());
}

TEST_P(TestGpuConvWrwFold, Half)
{
    SKIP_IF_NO_DEVICES();
    runFold<Half>(GetParam());
}

TEST_P(TestGpuConvWrwFold, BFloat16)
{
    SKIP_IF_NO_DEVICES();
    runFold<BFloat16>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvWrwFold,
                         ::testing::ValuesIn(foldCases()),
                         [](const auto& info) { return info.param.name; });

TEST(TestGpuConvWrwWideTiles, HalfPartialChannelTiles)
{
    SKIP_IF_NO_DEVICES();
    runWideTilesPartialChannels<Half>();
}

TEST(TestGpuConvWrwWideTiles, BFloat16PartialChannelTiles)
{
    SKIP_IF_NO_DEVICES();
    runWideTilesPartialChannels<BFloat16>();
}

/// The three-bf16 fp32 configs (CONV_WRW_TF32's): devices with bf16 matrix cores where three
/// bf16 products beat exact fp32 (gfx90a, gfx942, gfx950, gfx11, gfx12; not gfx908).
class TestGpuConvWrwThreeBf16 : public ::testing::TestWithParam<DeviceCase>
{
};

TEST_P(TestGpuConvWrwThreeBf16, Float)
{
    SKIP_IF_NO_DEVICES();
    // Held: stripArchFeatures returns a view into the name.
    const auto properties = currentDeviceProperties();
    const std::string arch(hipdnn_plugin_sdk::stripArchFeatures(properties.gcnArchName));
    if(arch != "gfx90a" && arch != "gfx942" && arch != "gfx950" && arch.rfind("gfx11", 0) != 0
       && arch.rfind("gfx12", 0) != 0)
    {
        GTEST_SKIP() << "no three-bf16 configs on " << arch;
    }
    runCase<float>(GetParam(), Math::THREE_BF16);
}

/// +inf and -inf in x: every dw element an infinity reaches is that infinity (NaN where both
/// meet), as exact fp32 gives; the split must not turn them into NaN.
TEST_P(TestGpuConvWrwThreeBf16, FloatInfinities)
{
    SKIP_IF_NO_DEVICES();
    const auto properties = currentDeviceProperties();
    const std::string arch(hipdnn_plugin_sdk::stripArchFeatures(properties.gcnArchName));
    if(arch != "gfx90a" && arch != "gfx942" && arch != "gfx950" && arch.rfind("gfx11", 0) != 0
       && arch.rfind("gfx12", 0) != 0)
    {
        GTEST_SKIP() << "no three-bf16 configs on " << arch;
    }
    runCase<float>(GetParam(), Math::THREE_BF16, true);
}

/// Layouts, 1-D to 3-D, stride, dilation, groups, stems (small-C rows), pointwise, true
/// convolution, padding past the filter, and many reduction steps (several splits).
std::vector<DeviceCase> threeBf16Cases()
{
    static const std::set<std::string> s_names{"Nchw3x3",
                                               "Nhwc3x3",
                                               "Nhwc3x3Wide",
                                               "ThreeD3x3x3",
                                               "ThreeDNhwc3x3x3",
                                               "OneD1x7",
                                               "Stride2",
                                               "Stride2Nhwc",
                                               "Dilation2",
                                               "GroupedNhwc3x3",
                                               "Stem7x7Stride2",
                                               "Stem7x7Stride2Nhwc",
                                               "Pointwise1x1Nhwc",
                                               "Pointwise1x1OddChannels",
                                               "PaddingPastTheFilter",
                                               "ManySteps",
                                               "TrueConvolutionNhwc",
                                               "FilterKrscUnderNchw"};
    std::vector<DeviceCase> cases;
    for(const auto& deviceCase : deviceCases())
    {
        if(s_names.count(deviceCase.name) != 0)
        {
            cases.push_back(deviceCase);
        }
    }
    if(cases.size() != s_names.size())
    {
        throw std::logic_error("a three-bf16 case names no device case");
    }
    // Shapes three-bf16 is admitted for (threeBf16Pays): CDNA's 128+ input and 64+ output
    // channels at 3x3 unit stride, and RDNA's 256+ channels (the stems above cover RDNA's few).
    cases.push_back(
        {"Wide128Nchw3x3", conv(128, 64, {10, 12}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1})});
    cases.push_back(
        {"Wide128Nhwc3x3", conv(128, 64, {10, 12}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, true)});
    cases.push_back(
        {"Wide128Ncdhw3x3x3",
         conv(128, 64, {4, 6, 7}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1})});
    cases.push_back(
        {"Wide256Nchw3x3", conv(256, 72, {6, 9}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1})});
    cases.push_back({"Wide256Nhwc3x3Grouped",
                     conv(512, 128, {6, 9}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, true, 2)});
    return cases;
}

/// Opt-in (K3_TF32_TIMING=<shape CSV>): the best exact fp32 and the best three-bf16 config per
/// shape. See ConvTf32Timing.hpp.
TEST(TestGpuConvWrwTiming, ExactAgainstThreeBf16)
{
    SKIP_IF_NO_DEVICES();
    conv_tf32_timing::timeByForm(
        PACK, ConvDirection::BACKWARD_WEIGHTS, catalog(), "K3_TF32_TIMING", [](const auto& kernel) {
            return std::string(kernel.getIntMetadata("math") == 1 ? "bf16x3" : "exact");
        });
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvWrwThreeBf16,
                         ::testing::ValuesIn(threeBf16Cases()),
                         [](const auto& info) { return info.param.name; });

class TestGpuConvWrwDilationRows : public ::testing::TestWithParam<int64_t>
{
};

TEST_P(TestGpuConvWrwDilationRows, HalfAgainstUnroundedDouble)
{
    SKIP_IF_NO_DEVICES();
    DeviceCase test{
        "DilationRows",
        conv(16, 24, {5, GetParam()}, {3, 3}, {24, 24}, {24, 24}, {1, 1}, {24, 24}, false, 1, 1)};
    test.dilationRows = true;
    runCase<Half>(test);
}

INSTANTIATE_TEST_SUITE_P(, TestGpuConvWrwDilationRows, ::testing::Values(32, 65, 128));

struct AccurateCase
{
    std::string name;
    ConvShape shape;
    bool filterLast = false;
    int partitions = 0; ///< 1: one partition, 2: multiple, 0: either
    bool productCancellation = false;
    bool splitCancellation = false;
};

std::vector<AccurateCase> accurateCases()
{
    std::vector<AccurateCase> cases;
    for(const bool channelsLast : {false, true})
    {
        for(const bool filterLast : {false, true})
        {
            for(const auto mode : {ConvMode::CROSS_CORRELATION, ConvMode::CONVOLUTION})
            {
                auto shape = conv(10,
                                  14,
                                  {5, 9, 17},
                                  {2, 4, 5},
                                  {1, 4, 6},
                                  {0, 1, 2},
                                  {2, 3, 2},
                                  {2, 2, 3},
                                  channelsLast,
                                  2);
                shape.mode = mode;
                cases.push_back(
                    {std::string(channelsLast ? "Ndhwc" : "Ncdhw")
                         + (filterLast ? "FilterLast" : "FilterFirst")
                         + (mode == ConvMode::CONVOLUTION ? "Convolution" : "Correlation"),
                     shape,
                     filterLast});
            }
        }
    }
    for(const bool channelsLast : {false, true})
    {
        cases.push_back({channelsLast ? "LongCancellationNwc" : "LongCancellationNcw",
                         conv(3, 5, {65539}, {1}, {0}, {0}, {1}, {1}, channelsLast, 1, 1),
                         !channelsLast,
                         2});
    }
    cases.push_back({"LongProductCancellation",
                     conv(3, 5, {65538}, {1}, {0}, {0}, {1}, {1}, false, 1, 1),
                     true,
                     2,
                     true});
    cases.push_back({"LongSplitCancellation",
                     conv(3, 5, {65536}, {1}, {0}, {0}, {1}, {1}, false, 1, 1),
                     true,
                     2,
                     false,
                     true});
    cases.push_back({"SinglePartitionTwoD",
                     conv(3, 5, {2, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, true, 1, 1),
                     false,
                     1});
    cases.push_back(
        {"MostlyPaddingThreeD",
         conv(2, 3, {1, 1, 1}, {2, 3, 2}, {3, 4, 3}, {0, 0, 0}, {2, 1, 2}, {2, 2, 1}, false, 1, 1),
         true,
         1});
    return cases;
}

void fillAccurateOperands(Tensor<float>& x,
                          Tensor<float>& dy,
                          int revision,
                          const AccurateCase& test)
{
    auto* xp = static_cast<float*>(x.rawHostData());
    forEachIndex(x.dims(), [&](const auto& index) {
        int64_t position = index[0];
        for(size_t axis = 2; axis < index.size(); ++axis)
        {
            position = position * x.dims()[axis] + index[axis];
        }
        float value = 0.0f;
        if(test.splitCancellation)
        {
            value = 1.0f + float(index[1]) / 16.0f;
        }
        else if(test.productCancellation)
        {
            value = 4096.0f + float(index[1] + revision) + (position % 2 == 0 ? 1.0f : 0.0f);
        }
        else
        {
            value = 1.0f + float((position / 4) % 7) / 16.0f + float(index[1]) / 32.0f
                    + float(revision) / 128.0f;
        }
        xp[offsetOf(index, x.strides())] = value;
    });
    auto* yp = static_cast<float*>(dy.rawHostData());
    constexpr std::array<float, 4> CANCELLATION{16777216.0f, 1.0f, -16777216.0f, 1.0f};
    forEachIndex(dy.dims(), [&](const auto& index) {
        int64_t position = index[0];
        for(size_t axis = 2; axis < index.size(); ++axis)
        {
            position = position * dy.dims()[axis] + index[axis];
        }
        float value = 0.0f;
        if(test.splitCancellation)
        {
            if(position < 32768)
            {
                value = 16777216.0f;
            }
            else if(position == 65535)
            {
                value = -16777216.0f + float(revision + 1);
            }
            else
            {
                value = -16777216.0f;
            }
        }
        else if(test.productCancellation)
        {
            value = position % 2 == 0 ? 4097.0f + float(2 * revision)
                                      : -4098.0f - float(2 * revision);
        }
        else
        {
            value = CANCELLATION[size_t((position + revision) % 4)]
                    * (1.0f + float(index[1] % 3) / 8.0f);
        }
        yp[offsetOf(index, dy.strides())] = value;
    });
    x.markHostModified();
    dy.markHostModified();
}

// Independent direct convolution from graph dimensions/strides, not the K3 launch geometry
// or partial layout. Keep the authority in double, including the comparison (no FP32 oracle).
std::vector<double> accurateCpuWeights(const ConvGraphSpec& spec,
                                       const Tensor<float>& x,
                                       const Tensor<float>& dy,
                                       int64_t groups)
{
    const auto* xp = x.memory().hostData();
    const auto* yp = dy.memory().hostData();
    size_t elements = 1;
    for(const auto extent : spec.w.dims)
    {
        elements *= size_t(extent);
    }
    std::vector<double> reference(elements, 0.0);
    auto outputPositions = spec.y.dims;
    outputPositions[1] = 1;
    const int64_t kg = spec.y.dims[1] / groups;
    forEachIndex(spec.w.dims, [&](const auto& weight) {
        std::vector<int64_t> xi(spec.x.dims.size(), 0);
        std::vector<int64_t> yi(spec.y.dims.size(), 0);
        xi[1] = (weight[0] / kg) * spec.w.dims[1] + weight[1];
        yi[1] = weight[0];
        double sum = 0.0;
        forEachIndex(outputPositions, [&](const auto& position) {
            xi[0] = yi[0] = position[0];
            bool inside = true;
            for(size_t axis = 2; axis < xi.size(); ++axis)
            {
                const auto tap = spec.mode == ConvMode::CONVOLUTION
                                     ? spec.w.dims[axis] - 1 - weight[axis]
                                     : weight[axis];
                xi[axis] = position[axis] * spec.stride[axis - 2] + tap * spec.dilation[axis - 2]
                           - spec.prePadding[axis - 2];
                yi[axis] = position[axis];
                inside = inside && xi[axis] >= 0 && xi[axis] < spec.x.dims[axis];
            }
            if(inside)
            {
                sum += double(xp[offsetOf(xi, spec.x.strides)])
                       * double(yp[offsetOf(yi, spec.y.strides)]);
            }
        });
        reference[size_t(offsetOf(weight, spec.w.strides))] = sum;
    });
    return reference;
}

class TestGpuConvWrwAccurate : public ::testing::TestWithParam<AccurateCase>
{
};

TEST_P(TestGpuConvWrwAccurate, StrictDoubleTruthAndPreparedRebinding)
{
    SKIP_IF_NO_DEVICES();
    const auto& test = GetParam();
    auto shape = test.shape;
    shape.dataType = DataType::FLOAT;
    auto spec = specFor(shape, ConvDirection::BACKWARD_WEIGHTS);
    spec.w.strides = packedStrides(spec.w.dims, test.filterLast);
    const auto device = currentDeviceProperties();
    if(hipdnn_plugin_sdk::stripArchFeatures(device.gcnArchName) != "gfx1250")
    {
        GTEST_SKIP() << "FP64 WRW is qualified only on gfx1250";
    }
    const GraphFixture fixture(buildGraph(spec), device);
    const auto bound = matchesGraph(PACK, fixture.context());
    ASSERT_TRUE(bound.has_value());
    const auto problem = conv_wrw::parseWrwProblem(fixture.context());
    ASSERT_TRUE(problem.has_value());
    Tensor<float> x0(spec.x.dims, spec.x.strides);
    Tensor<float> x1(spec.x.dims, spec.x.strides);
    Tensor<float> y0(spec.y.dims, spec.y.strides);
    Tensor<float> y1(spec.y.dims, spec.y.strides);
    Tensor<float> w0(spec.w.dims, spec.w.strides);
    Tensor<float> w1(spec.w.dims, spec.w.strides);
    const auto& handler = dispatchHandler(PACK);
    const Handle handle;
    int ran = 0;
    for(const auto& kernel : catalog())
    {
        const auto config = configOf(kernel);
        if(config.loop != conv_wrw::LOOP_ACCURATE
           || !matchesKernel(PACK, fixture.context(), kernel, *bound))
        {
            continue;
        }
        SCOPED_TRACE(kernel.name);
        ++ran;
        const auto launch = conv_wrw::wrwLaunchFor(*problem, config, device);
        ASSERT_TRUE(launch.has_value());
        if(test.partitions == 1)
        {
            ASSERT_EQ(launch->args.splits, 1);
        }
        if(test.partitions == 2)
        {
            ASSERT_GT(launch->args.splits, 1);
        }
        const auto prepared = handler.prepare(fixture.context(), *bound, kernel);
        ASSERT_NE(prepared, nullptr);
        const size_t bytes = handler.workspaceBytes(fixture.context(), *bound, kernel);
        ASSERT_LE(bytes, size_t(conv_wrw::MAX_WORKSPACE_BYTES));
        void* workspace = nullptr;
        ASSERT_EQ(hipMalloc(&workspace, bytes), hipSuccess);
        std::array<hipdnnPluginDeviceBuffer_t, 3> previous{};
        for(int revision = 0; revision < 3; ++revision)
        {
            SCOPED_TRACE(revision);
            auto& x = revision == 0 ? x0 : x1;
            auto& y = revision == 0 ? y0 : y1;
            auto& w = revision == 0 ? w0 : w1;
            fillAccurateOperands(x, y, revision, test);
            const auto reference = accurateCpuWeights(spec, x, y, shape.groups);
            w.fillWithSentinelValue();
            const std::array<hipdnnPluginDeviceBuffer_t, 3> buffers{
                hipdnnPluginDeviceBuffer_t{X_UID, x.rawDeviceData()},
                hipdnnPluginDeviceBuffer_t{W_UID, w.rawDeviceData()},
                hipdnnPluginDeviceBuffer_t{Y_UID, y.rawDeviceData()}};
            if(revision == 2)
            {
                for(size_t i = 0; i < buffers.size(); ++i)
                {
                    EXPECT_EQ(buffers[i].ptr, previous[i].ptr);
                }
            }
            previous = buffers;
            ASSERT_EQ(hipMemset(workspace, 0x7f, bytes), hipSuccess);
            handler.launch(handle, *prepared, buffers.data(), uint32_t(buffers.size()), workspace);
            ASSERT_EQ(hipStreamSynchronize(handle.getStream()), hipSuccess);
            EXPECT_TRUE(conv_determinism_test::relaunchReproducesOutput(
                [&] {
                    handler.launch(
                        handle, *prepared, buffers.data(), uint32_t(buffers.size()), workspace);
                },
                handle.getStream(),
                w,
                workspace,
                bytes));
            w.markDeviceModified();
            const auto* actual = static_cast<const float*>(w.rawHostData());
            int64_t mismatches = 0;
            for(size_t i = 0; i < reference.size(); ++i)
            {
                const double tolerance = 1e-4 + 1e-4 * std::fabs(reference[i]);
                if(!(std::fabs(double(actual[i]) - reference[i]) <= tolerance) && ++mismatches <= 5)
                {
                    ADD_FAILURE() << "dw[" << i << "] got " << actual[i] << ", reference "
                                  << reference[i] << ", tolerance " << tolerance;
                }
            }
            EXPECT_EQ(mismatches, 0);
        }
        ASSERT_EQ(hipFree(workspace), hipSuccess);
    }
    ASSERT_GT(ran, 0) << "the accuracy candidate must be reachable through real prepare/launch";
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvWrwAccurate,
                         ::testing::ValuesIn(accurateCases()),
                         [](const auto& info) { return info.param.name; });

TEST(TestGpuConvWrwDirectChannelsFirstStore, GroupedReflectedAndSplitTransitions)
{
    SKIP_IF_NO_DEVICES();
    if(hipdnn_plugin_sdk::stripArchFeatures(currentDeviceProperties().gcnArchName) != "gfx1250")
    {
        GTEST_SKIP() << "native direct channels-first store requires gfx1250";
    }
    for(const bool manySplits : {false, true})
    {
        auto shape = conv(15,
                          21,
                          manySplits ? std::vector<int64_t>{8, 16} : std::vector<int64_t>{4, 4},
                          {3, 3},
                          {1, 1},
                          {1, 1},
                          {1, 1},
                          {1, 1},
                          false,
                          3,
                          manySplits ? 16 : 1);
        shape.mode = ConvMode::CONVOLUTION;
        DeviceCase test{"DirectChannelsFirstStore", shape};
        test.onlyLoop = conv_wrw::LOOP_TAP_TILED;
        test.directChannelsFirstStore = true;
        runCase<Half>(test);
    }
    auto shape = conv(
        19, 21, {2, 2, 4}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, false, 1, 1);
    DeviceCase test{"DirectChannelsFirstStoreThreeD", shape};
    test.onlyLoop = conv_wrw::LOOP_TAP_TILED;
    test.directChannelsFirstStore = true;
    runCase<Half>(test);
}

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
    static constexpr std::array<const char*, 24> KNOBS{
        "atom_id", "loop",  "layout", "bk", "bc",  "bq", "fr",   "s",
        "sw",      "dw",    "wk",     "wc", "tk",  "tc", "vec",  "xcd",
        "ns",      "align", "bpu",    "wq", "dws", "pb", "math", "flat"};
    std::array<std::set<std::vector<int64_t>>, 3> seen;
    for(const auto& kernel : catalog())
    {
        SCOPED_TRACE(kernel.name);
        const auto dtype = stringOf(kernel, "dtype");
        size_t slot = 2;
        if(dtype == "FLOAT")
        {
            slot = 0;
        }
        else if(dtype == "HALF")
        {
            slot = 1;
        }
        std::vector<int64_t> tuple;
        tuple.reserve(KNOBS.size());
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

TEST(TestConvWrwCatalog, EveryMatrixArchHasAGenericTierPerDtypeAndLayout)
{
    // The generic loop is the coverage floor: on every architecture with matrix cores, each
    // element type has generic configs reading elements (layout 0) and channel vectors
    // (layout 1), and their block-based buffer forms for images past 2 GiB. fp32 on RDNA runs
    // the FMA form.
    static constexpr std::array<const char*, 6> ARCHS{
        "gfx908", "gfx90a", "gfx942", "gfx950", "gfx1101", "gfx1201"};
    for(const char* arch : ARCHS)
    {
        for(const char* dtype : {"FLOAT", "HALF", "BFLOAT16"})
        {
            // gfx908 has no bf16 generic configs (its bf16 atom ships for common filters).
            if(std::string(arch) == "gfx908" && std::string(dtype) == "BFLOAT16")
            {
                continue;
            }
            // RDNA fp32 runs the FMA form, whose pack lists the RDNA architectures: a build for
            // other targets installs no copy of it.
            const bool rdna = std::string(arch).rfind("gfx11", 0) == 0
                              || std::string(arch).rfind("gfx12", 0) == 0;
            if(rdna && std::string(dtype) == "FLOAT"
               && std::none_of(catalog().begin(), catalog().end(), [](const auto& kernel) {
                      return kernel.getStringMetadata("atom") == "fma"
                             && kernel.getIntMetadata("loop") == conv_wrw::LOOP_GENERIC;
                  }))
            {
                continue;
            }
            for(const int64_t layout :
                {conv_wrw::LAYOUT_CHANNELS_FIRST, conv_wrw::LAYOUT_CHANNELS_LAST})
            {
                for(const int64_t pb : {int64_t{0}, int64_t{1}})
                {
                    SCOPED_TRACE(std::string(arch) + " " + dtype + " layout "
                                 + std::to_string(layout) + " pb " + std::to_string(pb));
                    const bool found
                        = std::any_of(catalog().begin(), catalog().end(), [&](const auto& kernel) {
                              const auto config = configOf(kernel);
                              return config.loop == conv_wrw::LOOP_GENERIC && config.math == 0
                                     && config.layout == layout && config.pb == pb
                                     && stringOf(kernel, "dtype") == dtype
                                     && conv_igemm::atomRunsOn(config.atom, arch);
                          });
                    EXPECT_TRUE(found);
                }
            }
        }
    }
}

TEST(TestConvWrwCatalog, DilatedRowsHaveSpecializedConfigsInBothLayouts)
{
    // A dilated filter row the specialized tier misses in one layout falls to the generic tier,
    // which declines ASPP-class dilations except on gfx11 and gfx12, and then to the BLAS path
    // (30x slower on gfx1100 for channels-last fp16 deeplab). Every matrix-core arch serves every
    // deeplab / SSD 3x3 dilation and HiFi-GAN's dilated 1-D rows in both layouts with a
    // specialized config.
    static constexpr std::array<const char*, 6> ARCHS{
        "gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1151", "gfx1201"};
    // (filter width, dilation): 3x3 deeplab / SSD rows (positive width), 1 x S rows (negative).
    static constexpr std::array<std::pair<int64_t, int64_t>, 11> ROWS{{{3, 4},
                                                                       {3, 6},
                                                                       {3, 12},
                                                                       {3, 24},
                                                                       {3, 36},
                                                                       {-3, 3},
                                                                       {-3, 5},
                                                                       {-7, 3},
                                                                       {-7, 5},
                                                                       {-11, 3},
                                                                       {-11, 5}}};
    const auto& kernels = catalog();
    for(const auto dataType : {DataType::HALF, DataType::BFLOAT16})
    {
        for(const bool channelsLast : {false, true})
        {
            for(const auto& [width, d] : ROWS)
            {
                const int64_t taps = width < 0 ? -width : width;
                const int64_t pad = (taps - 1) * d / 2;
                auto shape
                    = width < 0
                          ? conv(64, 64, {200}, {taps}, {pad}, {pad}, {1}, {d}, channelsLast, 1, 2)
                          : conv(64,
                                 64,
                                 {40, 40},
                                 {3, 3},
                                 {d, d},
                                 {d, d},
                                 {1, 1},
                                 {d, d},
                                 channelsLast,
                                 1,
                                 2);
                shape.dataType = dataType;
                const GraphFixture fixture(
                    buildGraph(specFor(shape, ConvDirection::BACKWARD_WEIGHTS)));
                const auto problem = conv_wrw::parseWrwProblem(fixture.context());
                ASSERT_TRUE(problem.has_value());
                for(const char* arch : ARCHS)
                {
                    SCOPED_TRACE(std::string(arch) + (channelsLast ? " nhwc" : " nchw") + " width "
                                 + std::to_string(width) + " d" + std::to_string(d) + " "
                                 + (dataType == DataType::HALF ? "fp16" : "bf16"));
                    auto device = testDeviceProperties();
                    device.gcnArchName = arch;
                    device.warpSize = std::string(arch).rfind("gfx1", 0) == 0 ? 32 : 64;
                    device.multiProcessorCount = 64;
                    const bool served = std::any_of(
                        kernels.begin(), kernels.end(), [&](const auto& kernel) {
                            const auto config = configOf(kernel);
                            return config.loop != conv_wrw::LOOP_GENERIC && config.math == 0
                                   && config.atom != "fma"
                                   && conv_wrw::wrwLaunchFor(*problem, config, device).has_value();
                        });
                    EXPECT_TRUE(served);
                }
            }
        }
    }
}

TEST(TestConvWrwCatalog, AtomIdsNameTheirAtomsAndEveryConfigFitsItsArchitectures)
{
    static constexpr std::array<const char*, 8> ARCHS{
        "gfx908", "gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1151", "gfx1201", "gfx1250"};
    for(const auto& kernel : catalog())
    {
        SCOPED_TRACE(kernel.name);
        const auto config = configOf(kernel);
        EXPECT_EQ(conv_wrw::atomIdOf(config.atom), config.atomId);
        EXPECT_EQ(stringOf(kernel, "direction"), "WRW");
        const auto dtype = stringOf(kernel, "dtype");
        // Three-bf16 configs (math 1) are bf16 kernels on fp32 graphs.
        const int elementBytes = dtype == "FLOAT" && config.math == 0 ? 4 : 2;
        EXPECT_TRUE(config.math == 0 || (config.math == 1 && dtype == "FLOAT"));
        // 16-byte vectors; the FMA output rows also take scalar units.
        if((config.loop != conv_wrw::LOOP_ROWS_FMA && config.loop != conv_wrw::LOOP_NATIVE_GENERIC
            && config.loop != conv_wrw::LOOP_TAP_TILED && config.loop != conv_wrw::LOOP_SMALL_K
            && config.loop != conv_wrw::LOOP_ACCURATE)
           || config.vec != 1)
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
    }
}

} // namespace

#endif
