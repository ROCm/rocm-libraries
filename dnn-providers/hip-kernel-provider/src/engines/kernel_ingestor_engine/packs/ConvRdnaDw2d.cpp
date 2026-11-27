// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "engines/kernel_ingestor_engine/packs/ConvRdnaDw2d.hpp"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <string>

#include "engines/kernel_ingestor_engine/IngestorLaunchRestore.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::conv_rdna::dw2d
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::KernelArgument;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::MetadataValues;
using hipdnn_plugin_sdk::ingestor::SavedLaunchInputs;

namespace
{

constexpr std::string_view FAMILY_FIELD = "kernel_family";
constexpr std::string_view VARIANT_FIELD = "variant";
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view LAYOUT_FIELD = "layout";
constexpr std::string_view FILTER_H_FIELD = "filter_h";
constexpr std::string_view FILTER_W_FIELD = "filter_w";
constexpr std::string_view STRIDE_FIELD = "stride";
constexpr std::string_view DILATION_FIELD = "dilation";
constexpr std::string_view DW2D_FAMILY = "dw2d_direct";
constexpr std::string_view SPECIALIZED_VARIANT = "specialized";
constexpr std::string_view GENERIC_VARIANT = "generic";

constexpr std::string_view N_VALUE = "n";
constexpr std::string_view C_VALUE = "c";
constexpr std::string_view K_VALUE = "k";
// Per spatial axis (height, width).
constexpr std::array<std::string_view, 2> IN_VALUES{"in_h", "in_w"};
constexpr std::array<std::string_view, 2> FILTER_VALUES{"filter_h", "filter_w"};
constexpr std::array<std::string_view, 2> OUT_VALUES{"out_h", "out_w"};
constexpr std::array<std::string_view, 2> PAD_VALUES{"pad_h", "pad_w"};
constexpr std::array<std::string_view, 2> STRIDE_VALUES{"stride_h", "stride_w"};
constexpr std::array<std::string_view, 2> DILATION_VALUES{"dilation_h", "dilation_w"};
constexpr std::string_view GENERIC_VALUE = "generic";
constexpr std::string_view LANE_ON_DILATION_GRID_VALUE = "lane_on_dilation_grid";
constexpr std::string_view OUT_PER_LANE_VALUE = "out_per_lane";

/// A specialized shape: filter (height, width), stride and dilation (equal along H and W), with
/// pre-padding dilation x (filter - 1) / 2 along each axis.
struct Shape
{
    int filterH;
    int filterW;
    int stride;
    int dilation;
};

// Must match configs/hip_conv_fwd_dw2d_rdna.yaml.
constexpr std::array<Shape, 7> UNDILATED_SHAPES{{{3, 3, 1, 1},
                                                 {3, 3, 2, 1},
                                                 {5, 5, 1, 1},
                                                 {5, 5, 2, 1},
                                                 {7, 7, 1, 1},
                                                 {7, 7, 2, 1},
                                                 {1, 31, 1, 1}}};
constexpr std::array<Shape, 6> DILATED_SHAPES{
    {{3, 3, 1, 2}, {3, 3, 1, 4}, {3, 3, 1, 6}, {3, 3, 1, 12}, {3, 3, 1, 18}, {5, 5, 1, 2}}};

// Must match kernels/ConvFwdDepthwise2d.cpp and the generic kernel of
// ../conv_dw3d_rdna/kernels/ConvFwdDepthwise3d.cpp.
constexpr int64_t ROW_CHUNK = 8;
constexpr int UNDILATED_OUT_PER_LANE = 2;
constexpr int64_t GENERIC_TILE_W = 4;
// The specialized kernel walks its tiles in a grid-stride loop; more workgroups than this did
// not help in MIOpen's measurements.
constexpr int64_t MAX_SPECIALIZED_BLOCKS = 2048;
constexpr int64_t MAX_GRID_X_ITEMS = 0x7fffffff;

bool isShape(const ConvProblem& problem, const Shape& shape)
{
    return problem.filter[1] == shape.filterH && problem.filter[2] == shape.filterW
           && problem.stride[1] == shape.stride && problem.stride[2] == shape.stride
           && problem.dilation[1] == shape.dilation && problem.dilation[2] == shape.dilation
           && problem.pad[1] == shape.dilation * (shape.filterH - 1) / 2
           && problem.pad[2] == shape.dilation * (shape.filterW - 1) / 2;
}

int64_t ceilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

/// The product of @p factors (each at least 1), or @p cap + 1 once it would pass @p cap, so
/// values near INT_MAX (reachable through a saved plan) cannot wrap it.
int64_t cappedProduct(std::initializer_list<int64_t> factors, int64_t cap)
{
    int64_t product = 1;
    for(const int64_t factor : factors)
    {
        if(factor > cap / product)
        {
            return cap + 1;
        }
        product *= factor;
    }
    return product;
}

/// The tiles of one output image in the specialized kernel's walk: lanes over row chunks and
/// column groups. On the dilation grid a lane's outputs are a lane step apart, so the tiles come
/// in step-sized groups. Saturates past INT_MAX, the kernel's counter width.
int64_t tilesPerImage(const ConvProblem& problem, const Geometry& geometry)
{
    int64_t rowStep = 1;
    int64_t colStep = 1;
    if(geometry.laneOnDilationGrid)
    {
        rowStep = problem.dilation[1] / std::gcd(problem.stride[1], problem.dilation[1]);
        colStep = problem.dilation[2] / std::gcd(problem.stride[2], problem.dilation[2]);
    }
    const int64_t tilesW = ceilDiv(problem.out[2], geometry.outPerLane * colStep) * colStep;
    const int64_t rowTiles = ceilDiv(problem.out[1], ROW_CHUNK * rowStep) * rowStep;
    return cappedProduct({tilesW, rowTiles}, std::numeric_limits<int>::max());
}

bool launchFits(const ConvProblem& problem, const Geometry& geometry)
{
    if(!geometry.generic)
    {
        // The grid is capped; the kernel counts the tiles of one image and the (n, c) pairs in
        // int.
        constexpr int64_t INT_LIMIT = std::numeric_limits<int>::max();
        return tilesPerImage(problem, geometry) <= INT_LIMIT
               && cappedProduct({problem.n, problem.k}, INT_LIMIT) <= INT_LIMIT;
    }
    return gridXBlocks(problem, geometry) * BLOCK_SIZE <= MAX_GRID_X_ITEMS;
}

/// Whether @p geometry is one the bundles build.
bool isBuiltGeometry(const Geometry& geometry)
{
    if(geometry.generic)
    {
        return !geometry.laneOnDilationGrid && geometry.outPerLane == GENERIC_TILE_W;
    }
    return geometry.laneOnDilationGrid ? (geometry.outPerLane == 2 || geometry.outPerLane == 4)
                                       : geometry.outPerLane == UNDILATED_OUT_PER_LANE;
}

/// The IO_DTYPE of the specialized kernel for a dtype.
std::string ioDtype(data_objects::DataType dtype)
{
    if(dtype == data_objects::DataType::FLOAT)
    {
        return "float";
    }
    return dtype == data_objects::DataType::HALF ? "__half" : "__hip_bfloat16";
}

/// The DW_DTYPE of the generic kernel for a dtype.
int genericDtypeCode(data_objects::DataType dtype)
{
    if(dtype == data_objects::DataType::FLOAT)
    {
        return 0;
    }
    return dtype == data_objects::DataType::HALF ? 1 : 2;
}

} // namespace

Geometry geometryOf(const ConvProblem& problem)
{
    if(problem.layout == Layout::CHANNELS_FIRST && problem.k == problem.c)
    {
        for(const auto& shape : UNDILATED_SHAPES)
        {
            if(isShape(problem, shape))
            {
                return Geometry{false, false, UNDILATED_OUT_PER_LANE};
            }
        }
        for(const auto& shape : DILATED_SHAPES)
        {
            if(isShape(problem, shape))
            {
                // Four outputs amortize the 16-bit conversion of the window; in fp32 two were
                // faster on gfx1151 (MIOpen's ConvDirectDepthwiseFwd2DDilated).
                return Geometry{
                    false, true, problem.dtype == data_objects::DataType::FLOAT ? 2 : 4};
            }
        }
    }
    return Geometry{true, false, static_cast<int>(GENERIC_TILE_W)};
}

int64_t gridXBlocks(const ConvProblem& problem, const Geometry& geometry)
{
    // The products saturate, so launchFits refuses rather than wraps.
    if(geometry.generic)
    {
        return ceilDiv(
            cappedProduct(
                {problem.n, problem.k, problem.out[1], ceilDiv(problem.out[2], GENERIC_TILE_W)},
                MAX_GRID_X_ITEMS),
            BLOCK_SIZE);
    }
    const int64_t tiles = cappedProduct({tilesPerImage(problem, geometry), problem.n, problem.k},
                                        MAX_SPECIALIZED_BLOCKS * BLOCK_SIZE);
    return std::min(ceilDiv(tiles, BLOCK_SIZE), MAX_SPECIALIZED_BLOCKS);
}

bool serves(const ConvProblem& problem)
{
    return problem.spatialRank <= 2 && problem.groups >= 2 && problem.groups == problem.c
           && problem.k % problem.c == 0 && launchFits(problem, geometryOf(problem));
}

bool isVariantOf(const KernelDefinition& kernel, const ConvProblem& problem)
{
    if(kernel.getStringMetadata(std::string(FAMILY_FIELD)) != DW2D_FAMILY
       || kernel.getStringMetadata(std::string(DTYPE_FIELD))
              != data_objects::EnumNameDataType(problem.dtype)
       || kernel.getStringMetadata(std::string(LAYOUT_FIELD)) != layoutName(problem.layout))
    {
        return false;
    }
    if(geometryOf(problem).generic)
    {
        return kernel.getStringMetadata(std::string(VARIANT_FIELD)) == GENERIC_VARIANT;
    }
    const auto intField
        = [&kernel](std::string_view field) { return kernel.getIntMetadata(std::string(field)); };
    return kernel.getStringMetadata(std::string(VARIANT_FIELD)) == SPECIALIZED_VARIANT
           && intField(FILTER_H_FIELD) == problem.filter[1]
           && intField(FILTER_W_FIELD) == problem.filter[2]
           && intField(STRIDE_FIELD) == problem.stride[1]
           && intField(DILATION_FIELD) == problem.dilation[1];
}

void addCompileOptions(compilation::KernelCompileOptions& options,
                       const ConvProblem& problem,
                       const Geometry& geometry)
{
    if(geometry.generic)
    {
        options.add("DW_GENERIC", 1);
        options.add("DW_DTYPE", genericDtypeCode(problem.dtype));
        options.add("DW_NDHWC", problem.layout == Layout::CHANNELS_LAST ? 1 : 0);
        return;
    }
    options.add("IO_DTYPE", ioDtype(problem.dtype));
    options.add("FILTER_H", problem.filter[1]);
    options.add("FILTER_W", problem.filter[2]);
    options.add("PAD_H", problem.pad[1]);
    options.add("PAD_W", problem.pad[2]);
    options.add("STRIDE_H", problem.stride[1]);
    options.add("STRIDE_W", problem.stride[2]);
    options.add("DILATION_H", problem.dilation[1]);
    options.add("DILATION_W", problem.dilation[2]);
    options.add("OUT_PER_LANE", geometry.outPerLane);
    options.add("LANE_ON_DILATION_GRID", geometry.laneOnDilationGrid ? 1 : 0);
}

/// Mirrors the argument lists of the two kernels and launch() below one for one.
const std::vector<KernelArgument>& kernelSignature(bool generic)
{
    static const KernelArgument s_buffer{
        "global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
    static const KernelArgument s_scalar{"by_value", static_cast<uint32_t>(sizeof(int)), 0, ""};
    const auto make = [](size_t scalars) {
        std::vector<KernelArgument> signature{s_buffer, s_buffer, s_buffer};
        signature.insert(signature.end(), scalars, s_scalar);
        return signature;
    };
    // Specialized: N, C, iH, iW, oH, oW. Generic (the dw3d kernel): N, C, Di, Hi, Wi, K, Do, Ho,
    // Wo, Z, Y, X, sd, sh, sw, pd, ph, pw, dd, dh, dw.
    static const std::vector<KernelArgument> s_specialized = make(6);
    static const std::vector<KernelArgument> s_generic = make(21);
    return generic ? s_generic : s_specialized;
}

void launch(const compilation::IRunnableKernel& kernel,
            hipStream_t stream,
            const void* x,
            const void* w,
            void* y,
            const ConvProblem& p,
            const Geometry& geometry)
{
    if(geometry.generic)
    {
        // The dw3d generic kernel with depth 1: unit depth extent, filter, stride and
        // dilation, no depth padding.
        kernel.launch(stream,
                      x,
                      w,
                      y,
                      p.n,
                      p.c,
                      1,
                      p.in[1],
                      p.in[2],
                      p.k,
                      1,
                      p.out[1],
                      p.out[2],
                      1,
                      p.filter[1],
                      p.filter[2],
                      1,
                      p.stride[1],
                      p.stride[2],
                      0,
                      p.pad[1],
                      p.pad[2],
                      1,
                      p.dilation[1],
                      p.dilation[2]);
        return;
    }
    // Filter, padding, stride and dilation are compiled into the specialized variant.
    kernel.launch(stream, x, w, y, p.n, p.c, p.in[1], p.in[2], p.out[1], p.out[2]);
}

void addProblemLaunchValues(MetadataValues& values,
                            const ConvProblem& problem,
                            const Geometry& geometry)
{
    values[std::string(N_VALUE)] = int64_t{problem.n};
    values[std::string(C_VALUE)] = int64_t{problem.c};
    values[std::string(K_VALUE)] = int64_t{problem.k};
    values[std::string(GENERIC_VALUE)] = int64_t{geometry.generic ? 1 : 0};
    values[std::string(LANE_ON_DILATION_GRID_VALUE)] = int64_t{geometry.laneOnDilationGrid ? 1 : 0};
    values[std::string(OUT_PER_LANE_VALUE)] = int64_t{geometry.outPerLane};
    for(size_t axis = 0; axis < 2; ++axis)
    {
        values[std::string(IN_VALUES[axis])] = int64_t{problem.in[axis + 1]};
        values[std::string(FILTER_VALUES[axis])] = int64_t{problem.filter[axis + 1]};
        values[std::string(OUT_VALUES[axis])] = int64_t{problem.out[axis + 1]};
        values[std::string(PAD_VALUES[axis])] = int64_t{problem.pad[axis + 1]};
        values[std::string(STRIDE_VALUES[axis])] = int64_t{problem.stride[axis + 1]};
        values[std::string(DILATION_VALUES[axis])] = int64_t{problem.dilation[axis + 1]};
    }
}

std::vector<std::string_view> problemLaunchValueNames()
{
    std::vector<std::string_view> names{
        N_VALUE, C_VALUE, K_VALUE, GENERIC_VALUE, LANE_ON_DILATION_GRID_VALUE, OUT_PER_LANE_VALUE};
    for(const auto* axisNames :
        {&IN_VALUES, &FILTER_VALUES, &OUT_VALUES, &PAD_VALUES, &STRIDE_VALUES, &DILATION_VALUES})
    {
        names.insert(names.end(), axisNames->begin(), axisNames->end());
    }
    return names;
}

void readProblemLaunchValues(const SavedLaunchInputs& inputs,
                             ConvProblem& problem,
                             Geometry& geometry)
{
    constexpr int64_t INT_MAX_VALUE = std::numeric_limits<int>::max();
    const auto positive = [&inputs](std::string_view name) {
        return static_cast<int>(requireLaunchInt(inputs, name, 1, INT_MAX_VALUE));
    };

    problem.spatialRank = 2;
    problem.n = positive(N_VALUE);
    problem.c = positive(C_VALUE);
    problem.k = positive(K_VALUE);
    problem.groups = problem.c;
    for(size_t axis = 0; axis < 2; ++axis)
    {
        problem.in[axis + 1] = positive(IN_VALUES[axis]);
        problem.filter[axis + 1] = positive(FILTER_VALUES[axis]);
        problem.out[axis + 1] = positive(OUT_VALUES[axis]);
        problem.pad[axis + 1]
            = static_cast<int>(requireLaunchInt(inputs, PAD_VALUES[axis], 0, INT_MAX_VALUE));
        problem.stride[axis + 1] = positive(STRIDE_VALUES[axis]);
        problem.dilation[axis + 1] = positive(DILATION_VALUES[axis]);
    }
    geometry.generic = requireLaunchInt(inputs, GENERIC_VALUE, 0, 1) == 1;
    geometry.laneOnDilationGrid = requireLaunchInt(inputs, LANE_ON_DILATION_GRID_VALUE, 0, 1) == 1;
    geometry.outPerLane = static_cast<int>(requireLaunchInt(inputs, OUT_PER_LANE_VALUE, 1, 4));

    if(problem.k % problem.c != 0 || !isBuiltGeometry(geometry) || !launchFits(problem, geometry))
    {
        serialization::refuseIngestorPlan(
            serialization::IngestorPlanRefusal::INCOMPATIBLE,
            "launch values of dispatch '" + inputs.dispatchSymbol
                + "' give a K that is not a multiple of C, a geometry the engine does not "
                  "build, or a grid beyond the launch limits");
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_rdna::dw2d

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
