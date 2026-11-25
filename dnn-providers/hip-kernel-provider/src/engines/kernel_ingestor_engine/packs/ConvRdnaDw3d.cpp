// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "engines/kernel_ingestor_engine/packs/ConvRdnaDw3d.hpp"

#include <array>
#include <initializer_list>
#include <limits>
#include <string>

#include "engines/kernel_ingestor_engine/IngestorLaunchRestore.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::conv_rdna::dw3d
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
constexpr std::string_view FILTER_D_FIELD = "filter_d";
constexpr std::string_view FILTER_H_FIELD = "filter_h";
constexpr std::string_view FILTER_W_FIELD = "filter_w";
constexpr std::string_view STRIDE_H_FIELD = "stride_h";
constexpr std::string_view STRIDE_W_FIELD = "stride_w";
constexpr std::string_view TILE_D_FIELD = "tile_d";
constexpr std::string_view TILE_W_FIELD = "tile_w";
constexpr std::string_view UNIFORM_K_FIELD = "uniform_k";
constexpr std::string_view DW3D_FAMILY = "dw3d_direct";
constexpr std::string_view SPECIALIZED_VARIANT = "specialized";
constexpr std::string_view GENERIC_VARIANT = "generic";

constexpr std::string_view N_VALUE = "n";
constexpr std::string_view C_VALUE = "c";
constexpr std::string_view K_VALUE = "k";
// Per spatial axis (depth, height, width).
constexpr std::array<std::string_view, 3> IN_VALUES{"in_d", "in_h", "in_w"};
constexpr std::array<std::string_view, 3> FILTER_VALUES{"filter_d", "filter_h", "filter_w"};
constexpr std::array<std::string_view, 3> OUT_VALUES{"out_d", "out_h", "out_w"};
constexpr std::array<std::string_view, 3> PAD_VALUES{"pad_d", "pad_h", "pad_w"};
constexpr std::array<std::string_view, 3> STRIDE_VALUES{"stride_d", "stride_h", "stride_w"};
constexpr std::array<std::string_view, 3> DILATION_VALUES{"dilation_d", "dilation_h", "dilation_w"};
constexpr std::string_view GENERIC_VALUE = "generic";
constexpr std::string_view TILE_D_VALUE = "tile_d";
constexpr std::string_view TILE_W_VALUE = "tile_w";
constexpr std::string_view UNIFORM_K_VALUE = "uniform_k";

// The filters (depth, height, width) the bundles have specialized variants for, each with H / W
// stride 1 and 2 and H / W dilation 1. Must match the variants of
// configs/hip_conv_fwd_dw3d_rdna.yaml.
constexpr std::array<std::array<int, 3>, 4> SPECIALIZED_FILTERS{
    {{3, 3, 3}, {5, 5, 5}, {3, 1, 1}, {1, 3, 3}}};

// Must match kernels/ConvFwdDepthwise3d.cpp.
constexpr int64_t TILE_H = 4;
constexpr int64_t GENERIC_TILE_W = 4;
constexpr unsigned int BLOCK_SIZE = 256;
constexpr unsigned int UNIFORM_K_BLOCK_SIZE = 64;
// HSA grid limits: work-items along x, workgroups along y.
constexpr int64_t MAX_GRID_X_ITEMS = 0x7fffffff;
constexpr int64_t MAX_GRID_Y = 0xffff;

bool isSpecializedShape(const ConvProblem& problem)
{
    bool filterBuilt = false;
    for(const auto& filter : SPECIALIZED_FILTERS)
    {
        filterBuilt = filterBuilt
                      || (problem.filter[0] == filter[0] && problem.filter[1] == filter[1]
                          && problem.filter[2] == filter[2]);
    }
    const bool strideBuilt = problem.stride[1] == problem.stride[2]
                             && (problem.stride[1] == 1 || problem.stride[1] == 2);
    return filterBuilt && strideBuilt && problem.dilation[1] == 1 && problem.dilation[2] == 1;
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

/// Threads the kernel indexes along x (all of them unless uniform-k, one volume otherwise).
int64_t gridXItems(const ConvProblem& problem, const Geometry& geometry)
{
    // Saturates at MAX_GRID_X_ITEMS + 1, so launchFits refuses rather than wraps.
    if(geometry.generic)
    {
        return cappedProduct({problem.n,
                              problem.k,
                              problem.out[0],
                              problem.out[1],
                              ceilDiv(problem.out[2], GENERIC_TILE_W)},
                             MAX_GRID_X_ITEMS);
    }
    const int64_t tiles = cappedProduct({ceilDiv(problem.out[0], geometry.tileD),
                                         ceilDiv(problem.out[1], TILE_H),
                                         ceilDiv(problem.out[2], geometry.tileW)},
                                        MAX_GRID_X_ITEMS);
    return geometry.uniformK ? tiles
                             : cappedProduct({problem.n, problem.k, tiles}, MAX_GRID_X_ITEMS);
}

bool launchFits(const ConvProblem& problem, const Geometry& geometry)
{
    return gridXBlocks(problem, geometry) * blockSize(geometry) <= MAX_GRID_X_ITEMS
           && gridY(problem, geometry) <= MAX_GRID_Y;
}

/// Whether @p geometry is one the bundles build: the generic tile, or a specialized tile.
bool isBuiltGeometry(const Geometry& geometry)
{
    if(geometry.generic)
    {
        return geometry.tileD == 1 && geometry.tileW == GENERIC_TILE_W && !geometry.uniformK;
    }
    return (geometry.tileD == 1 || (geometry.tileD == 2 && geometry.uniformK))
           && (geometry.tileW == 4 || geometry.tileW == 8);
}

/// The kernel's DW_DTYPE for a dtype.
int dtypeCode(data_objects::DataType dtype)
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
    const bool channelsLast = problem.layout == Layout::CHANNELS_LAST;
    // The NCDHW specializations use the uniform-k grid only, whose y extent is N x K.
    if(!isSpecializedShape(problem)
       || (!channelsLast && int64_t{problem.n} * problem.k > MAX_GRID_Y))
    {
        return Geometry{true, 1, static_cast<int>(GENERIC_TILE_W), false};
    }
    Geometry geometry;
    geometry.uniformK = !channelsLast;
    // Two output depths share their input planes; it needs unit depth stride and dilation.
    geometry.tileD = !channelsLast && problem.stride[0] == 1 && problem.dilation[0] == 1 ? 2 : 1;
    // A wider tile reuses more of each input row for 5-wide filters, except with two fp32
    // output depths (#15100's measurements on gfx1151).
    if(geometry.tileD == 2 && problem.dtype == data_objects::DataType::FLOAT)
    {
        geometry.tileW = 4;
    }
    else
    {
        geometry.tileW = problem.filter[2] >= 5 ? 8 : 4;
    }
    return geometry;
}

unsigned int blockSize(const Geometry& geometry)
{
    return geometry.uniformK ? UNIFORM_K_BLOCK_SIZE : BLOCK_SIZE;
}

int64_t gridXBlocks(const ConvProblem& problem, const Geometry& geometry)
{
    return ceilDiv(gridXItems(problem, geometry), blockSize(geometry));
}

int64_t gridY(const ConvProblem& problem, const Geometry& geometry)
{
    return geometry.uniformK ? int64_t{problem.n} * problem.k : 1;
}

bool serves(const ConvProblem& problem)
{
    return problem.spatialRank == 3 && problem.groups >= 2 && problem.groups == problem.c
           && problem.k % problem.c == 0 && launchFits(problem, geometryOf(problem));
}

bool isVariantOf(const KernelDefinition& kernel, const ConvProblem& problem)
{
    if(kernel.getStringMetadata(std::string(FAMILY_FIELD)) != DW3D_FAMILY
       || kernel.getStringMetadata(std::string(DTYPE_FIELD))
              != data_objects::EnumNameDataType(problem.dtype)
       || kernel.getStringMetadata(std::string(LAYOUT_FIELD)) != layoutName(problem.layout))
    {
        return false;
    }
    const auto geometry = geometryOf(problem);
    if(geometry.generic)
    {
        return kernel.getStringMetadata(std::string(VARIANT_FIELD)) == GENERIC_VARIANT;
    }
    const auto intField
        = [&kernel](std::string_view field) { return kernel.getIntMetadata(std::string(field)); };
    return kernel.getStringMetadata(std::string(VARIANT_FIELD)) == SPECIALIZED_VARIANT
           && intField(FILTER_D_FIELD) == problem.filter[0]
           && intField(FILTER_H_FIELD) == problem.filter[1]
           && intField(FILTER_W_FIELD) == problem.filter[2]
           && intField(STRIDE_H_FIELD) == problem.stride[1]
           && intField(STRIDE_W_FIELD) == problem.stride[2]
           && intField(TILE_D_FIELD) == geometry.tileD && intField(TILE_W_FIELD) == geometry.tileW
           && intField(UNIFORM_K_FIELD) == (geometry.uniformK ? 1 : 0);
}

void addCompileOptions(compilation::KernelCompileOptions& options,
                       const ConvProblem& problem,
                       const Geometry& geometry,
                       bool hasBias)
{
    options.add("DW_DTYPE", dtypeCode(problem.dtype));
    options.add("DW_NDHWC", problem.layout == Layout::CHANNELS_LAST ? 1 : 0);
    if(hasBias)
    {
        options.add("DW_HAS_BIAS", 1);
    }
    if(geometry.generic)
    {
        options.add("DW_GENERIC", 1);
        return;
    }
    options.add("DW_KZ", problem.filter[0]);
    options.add("DW_KY", problem.filter[1]);
    options.add("DW_KX", problem.filter[2]);
    options.add("DW_SH", problem.stride[1]);
    options.add("DW_SW", problem.stride[2]);
    options.add("DW_TD", geometry.tileD);
    options.add("DW_TW", geometry.tileW);
    options.add("DW_UK", geometry.uniformK ? 1 : 0);
}

/// Mirrors the argument lists of the two kernels and launch() below one for one.
const std::vector<KernelArgument>& kernelSignature(bool generic, bool hasBias)
{
    static const KernelArgument s_buffer{
        "global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
    static const KernelArgument s_scalar{"by_value", static_cast<uint32_t>(sizeof(int)), 0, ""};
    const auto make = [](size_t buffers, size_t scalars) {
        std::vector<KernelArgument> signature(buffers, s_buffer);
        signature.insert(signature.end(), scalars, s_scalar);
        return signature;
    };
    // Specialized: N, C, Di, Hi, Wi, K, Do, Ho, Wo, sd, pd, ph, pw, dd. Generic: N, C, Di, Hi,
    // Wi, K, Do, Ho, Wo, Z, Y, X, sd, sh, sw, pd, ph, pw, dd, dh, dw. The bias pointer follows
    // `out`.
    static const std::vector<KernelArgument> s_specialized = make(3, 14);
    static const std::vector<KernelArgument> s_generic = make(3, 21);
    static const std::vector<KernelArgument> s_specializedBias = make(4, 14);
    static const std::vector<KernelArgument> s_genericBias = make(4, 21);
    if(hasBias)
    {
        return generic ? s_genericBias : s_specializedBias;
    }
    return generic ? s_generic : s_specialized;
}

void launch(const compilation::IRunnableKernel& kernel,
            hipStream_t stream,
            const void* x,
            const void* w,
            void* y,
            const ConvProblem& p,
            const Geometry& geometry,
            const void* bias)
{
    const auto run = [&](auto&&... buffers) {
        if(geometry.generic)
        {
            kernel.launch(stream,
                          buffers...,
                          p.n,
                          p.c,
                          p.in[0],
                          p.in[1],
                          p.in[2],
                          p.k,
                          p.out[0],
                          p.out[1],
                          p.out[2],
                          p.filter[0],
                          p.filter[1],
                          p.filter[2],
                          p.stride[0],
                          p.stride[1],
                          p.stride[2],
                          p.pad[0],
                          p.pad[1],
                          p.pad[2],
                          p.dilation[0],
                          p.dilation[1],
                          p.dilation[2]);
            return;
        }
        // The filter, H / W stride and dilation are compiled into the specialized variant.
        kernel.launch(stream,
                      buffers...,
                      p.n,
                      p.c,
                      p.in[0],
                      p.in[1],
                      p.in[2],
                      p.k,
                      p.out[0],
                      p.out[1],
                      p.out[2],
                      p.stride[0],
                      p.pad[0],
                      p.pad[1],
                      p.pad[2],
                      p.dilation[0]);
    };
    if(bias != nullptr)
    {
        run(x, w, y, bias);
    }
    else
    {
        run(x, w, y);
    }
}

void addProblemLaunchValues(MetadataValues& values,
                            const ConvProblem& problem,
                            const Geometry& geometry)
{
    values[std::string(N_VALUE)] = int64_t{problem.n};
    values[std::string(C_VALUE)] = int64_t{problem.c};
    values[std::string(K_VALUE)] = int64_t{problem.k};
    values[std::string(GENERIC_VALUE)] = int64_t{geometry.generic ? 1 : 0};
    values[std::string(TILE_D_VALUE)] = int64_t{geometry.tileD};
    values[std::string(TILE_W_VALUE)] = int64_t{geometry.tileW};
    values[std::string(UNIFORM_K_VALUE)] = int64_t{geometry.uniformK ? 1 : 0};
    for(size_t axis = 0; axis < 3; ++axis)
    {
        values[std::string(IN_VALUES[axis])] = int64_t{problem.in[axis]};
        values[std::string(FILTER_VALUES[axis])] = int64_t{problem.filter[axis]};
        values[std::string(OUT_VALUES[axis])] = int64_t{problem.out[axis]};
        values[std::string(PAD_VALUES[axis])] = int64_t{problem.pad[axis]};
        values[std::string(STRIDE_VALUES[axis])] = int64_t{problem.stride[axis]};
        values[std::string(DILATION_VALUES[axis])] = int64_t{problem.dilation[axis]};
    }
}

std::vector<std::string_view> problemLaunchValueNames()
{
    std::vector<std::string_view> names{
        N_VALUE, C_VALUE, K_VALUE, GENERIC_VALUE, TILE_D_VALUE, TILE_W_VALUE, UNIFORM_K_VALUE};
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

    problem.spatialRank = 3;
    problem.n = positive(N_VALUE);
    problem.c = positive(C_VALUE);
    problem.k = positive(K_VALUE);
    problem.groups = problem.c;
    for(size_t axis = 0; axis < 3; ++axis)
    {
        problem.in[axis] = positive(IN_VALUES[axis]);
        problem.filter[axis] = positive(FILTER_VALUES[axis]);
        problem.out[axis] = positive(OUT_VALUES[axis]);
        problem.pad[axis]
            = static_cast<int>(requireLaunchInt(inputs, PAD_VALUES[axis], 0, INT_MAX_VALUE));
        problem.stride[axis] = positive(STRIDE_VALUES[axis]);
        problem.dilation[axis] = positive(DILATION_VALUES[axis]);
    }
    geometry.generic = requireLaunchInt(inputs, GENERIC_VALUE, 0, 1) == 1;
    geometry.tileD = static_cast<int>(requireLaunchInt(inputs, TILE_D_VALUE, 1, 2));
    geometry.tileW = static_cast<int>(requireLaunchInt(inputs, TILE_W_VALUE, 1, 8));
    geometry.uniformK = requireLaunchInt(inputs, UNIFORM_K_VALUE, 0, 1) == 1;

    if(problem.k % problem.c != 0 || !isBuiltGeometry(geometry) || !launchFits(problem, geometry))
    {
        serialization::refuseIngestorPlan(
            serialization::IngestorPlanRefusal::INCOMPATIBLE,
            "launch values of dispatch '" + inputs.dispatchSymbol
                + "' give a K that is not a multiple of C, a tile the engine does not build, or "
                  "a grid beyond the launch limits");
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_rdna::dw3d

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
