// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Tile ranking for the rocRoller-off path. This is chooseSolutionIndexParameters
// from rocroller/solution_selection.cpp, without rocRoller code generation.
// The checked-in kernels are bound by the caller; a tile with no kernel is skipped.

#include "include/kfa_solution_selection.hpp"

#include "include/rocblaslt/utility.hpp"
#include "origami/hardware.hpp"
#include "origami/origami.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace
{
    constexpr int kWorkgroupMappingKSize = 4096;

    struct Tile
    {
        int m;
        int n;
        int k;
    };

    constexpr Tile kTiles[] = {
        {256, 256, 256}, {256, 256, 128}, {256, 192, 128}, {256, 128, 128}, {256, 64, 128},
        {256, 32, 128},  {256, 16, 128},  {192, 256, 128}, {192, 128, 128}, {192, 64, 128},
        {192, 32, 128},  {128, 256, 128}, {128, 192, 128}, {128, 128, 128}, {128, 64, 128},
        {128, 32, 128},  {64, 256, 128},  {64, 192, 128},  {64, 128, 128},  {64, 64, 128},
        {64, 32, 128},   {32, 256, 128},  {32, 192, 128},  {32, 128, 128},  {32, 64, 128},
        {32, 32, 128},   {32, 32, 64},    {16, 256, 128},  {64, 16, 128},   {16, 64, 128},
        {32, 16, 128},   {16, 32, 128},   {16, 16, 128},   {16, 16, 256},   {16, 64, 256}};

    constexpr Tile kSwizzleTiles[] = {
        {32, 32, 128},   {64, 32, 128},   {64, 64, 128},   {128, 32, 128},  {32, 128, 128},
        {32, 256, 128},  {32, 384, 128},  {32, 512, 128},  {32, 640, 128},  {32, 768, 128},
        {32, 896, 128},  {32, 1024, 128}, {64, 128, 128},  {64, 256, 128},  {64, 384, 128},
        {64, 512, 128},  {64, 640, 128},  {64, 768, 128},  {64, 896, 128},  {64, 1024, 128},
        {96, 128, 128},  {96, 256, 128},  {96, 384, 128},  {96, 512, 128},  {96, 640, 128},
        {128, 128, 128}, {128, 256, 128}, {128, 384, 128}, {160, 128, 128}, {160, 256, 128},
        {160, 384, 128}, {192, 128, 128}, {192, 256, 128}, {224, 128, 128}, {224, 256, 128},
        {256, 128, 128}, {256, 256, 128}};

    enum class MxType
    {
        Other,
        Half,
        Float,
        BF16,
        FP8,
        BF8,
        FP6,
        BF6,
        FP4
    };

    MxType classify(hipDataType type)
    {
        if(static_cast<int>(type) == HIP_R_6F_E2M3)
            return MxType::FP6;
        if(static_cast<int>(type) == HIP_R_6F_E3M2)
            return MxType::BF6;
        if(static_cast<int>(type) == HIP_R_4F_E2M1)
            return MxType::FP4;
        switch(type)
        {
        case HIP_R_16F:
            return MxType::Half;
        case HIP_R_32F:
            return MxType::Float;
        case HIP_R_16BF:
            return MxType::BF16;
        case HIP_R_8F_E4M3:
        case HIP_R_8F_E4M3_FNUZ:
            return MxType::FP8;
        case HIP_R_8F_E5M2:
        case HIP_R_8F_E5M2_FNUZ:
            return MxType::BF8;
        default:
            return MxType::Other;
        }
    }

    int elementBits(MxType type)
    {
        switch(type)
        {
        case MxType::FP4:
            return 4;
        case MxType::FP6:
        case MxType::BF6:
            return 6;
        case MxType::FP8:
        case MxType::BF8:
            return 8;
        case MxType::Half:
        case MxType::BF16:
            return 16;
        default:
            return 32;
        }
    }

    origami::data_type_t toOrigami(MxType type)
    {
        switch(type)
        {
        case MxType::Half:
            return origami::data_type_t::Half;
        case MxType::Float:
            return origami::data_type_t::Float;
        case MxType::BF16:
            return origami::data_type_t::BFloat16;
        case MxType::FP8:
            return origami::data_type_t::Float8;
        case MxType::BF8:
            return origami::data_type_t::BFloat8;
        case MxType::FP6:
            return origami::data_type_t::Float6;
        case MxType::BF6:
            return origami::data_type_t::BFloat6;
        case MxType::FP4:
            return origami::data_type_t::Float4;
        default:
            return origami::data_type_t::None;
        }
    }

    bool isF6(MxType type)
    {
        return type == MxType::FP6 || type == MxType::BF6;
    }

    bool isF8(MxType type)
    {
        return type == MxType::FP8 || type == MxType::BF8;
    }

    bool accumulationIsF32(rocblaslt_compute_type compute)
    {
        switch(compute)
        {
        case rocblaslt_compute_f32:
        case rocblaslt_compute_f32_fast_xf32:
        case rocblaslt_compute_f32_fast_f16:
        case rocblaslt_compute_f32_fast_bf16:
        case rocblaslt_compute_f32_fast_f8_fnuz:
        case rocblaslt_compute_f32_fast_bf8_fnuz:
        case rocblaslt_compute_f32_fast_f8bf8_fnuz:
        case rocblaslt_compute_f32_fast_bf8f8_fnuz:
        case rocblaslt_compute_f32_fast_f8:
        case rocblaslt_compute_f32_fast_bf8:
        case rocblaslt_compute_f32_fast_f8bf8:
        case rocblaslt_compute_f32_fast_bf8f8:
            return true;
        default:
            return false;
        }
    }

    bool scaleTypeAccepted(hipDataType scale)
    {
        MxType kind = classify(scale);
        return kind == MxType::Float || kind == MxType::Other;
    }

    struct Mi
    {
        int m;
        int n;
        int k;
    };

    Mi pickMi(MxType a, MxType b, Tile wgt, bool preSwizzle)
    {
        if(a == MxType::Half || a == MxType::BF16)
            return {32, 32, 8};
        if(a == MxType::Float)
            return {32, 32, 2};
        if(preSwizzle)
            return {16, 16, 128};
        if((isF6(a) || isF6(b)) && ((wgt.m == 256 && wgt.n == 64) || (wgt.m == 64 && wgt.n == 256)))
            return {32, 32, 64};
        if(wgt.k % 128 == 0)
            return {16, 16, 128};
        return {32, 32, 64};
    }

    int preferredUnrolling(MxType a, MxType b, Tile wgt, bool preSwizzle, bool preTile)
    {
        if(a == MxType::FP4 && b == MxType::FP4 && wgt.m > 32 && wgt.n > 32)
            return (preSwizzle && preTile) ? 2 : 4;
        return 2;
    }

    bool isPow2(int value)
    {
        return value > 0 && (value & (value - 1)) == 0;
    }

    template <size_t N>
    void appendTileList(const Tile (&tiles)[N],
                        MxType       a,
                        MxType       b,
                        bool         preSwizzle,
                        bool         preTile,
                        std::vector<origami::config_t>& out)
    {
        out.reserve(out.size() + N * 3);
        for(const Tile& wgt : tiles)
        {
            Mi  mi    = pickMi(a, b, wgt, preSwizzle);
            int wgtk  = wgt.k;
            if(a == MxType::Half || a == MxType::BF16 || a == MxType::Float)
                wgtk = 32;
            if(preSwizzle && preTile)
                wgtk = 256;
            int unroll = preferredUnrolling(a, b, wgt, preSwizzle, preTile);
            int hints[3][2] = {{0, 0}, {4, 0}, {0, 4}};
            for(const auto& hint : hints)
            {
                origami::config_t config;
                config.mt              = {static_cast<size_t>(wgt.m),
                             static_cast<size_t>(wgt.n),
                             static_cast<size_t>(wgtk * unroll)};
                config.mi              = {static_cast<size_t>(mi.m),
                             static_cast<size_t>(mi.n),
                             static_cast<size_t>(mi.k)};
                config.occupancy       = 1;
                config.cache_hints_a   = hint[0];
                config.cache_hints_b   = hint[1];
                out.push_back(config);
            }
        }
    }
} // namespace

rocblaslt_status rankKfaTiles(const RocblasltContractionProblem& prob, std::vector<KfaTile>& tiles)
{
    tiles.clear();

    if(prob.bias != nullptr)
        return rocblaslt_status_invalid_value;
    if(prob.batch_count != 1)
        return rocblaslt_status_invalid_value;
    if(!scaleTypeAccepted(prob.scale_type))
        return rocblaslt_status_invalid_value;
    if(!accumulationIsF32(prob.compute_type))
        return rocblaslt_status_invalid_value;

    const MxType typeA = classify(prob.a_type);
    const MxType typeB = classify(prob.b_type);
    const bool   preSwizzle
        = preSwizzleSizeForScale(prob.scaleAType).size() == 3
          && preSwizzleSizeForScale(prob.scaleBType).size() == 3;
    const bool preTile = preTileSizeForScaleA(prob.scaleAType).size() == 2
                         && preTileSizeForScaleB(prob.scaleBType).size() == 2;

    std::vector<origami::config_t> configs;
    if(prob.swizzleA && typeA == MxType::FP4 && typeB == MxType::FP4)
        appendTileList(kSwizzleTiles, typeA, typeB, preSwizzle, preTile, configs);
    else
        appendTileList(kTiles, typeA, typeB, preSwizzle, preTile, configs);

    const origami::hardware_t hardware = origami::hardware_t::get_hardware_for_device(0);
    origami::problem_t        problem;
    problem.size  = {static_cast<size_t>(prob.m),
                    static_cast<size_t>(prob.n),
                    static_cast<size_t>(prob.k)};
    problem.batch = static_cast<size_t>(prob.batch_count);
    problem.num_cus = static_cast<size_t>(prob.sm_count_target);
    problem.a_transpose
        = (prob.trans_a == HIPBLAS_OP_T) ? origami::transpose_t::T : origami::transpose_t::N;
    problem.b_transpose
        = (prob.trans_b == HIPBLAS_OP_T) ? origami::transpose_t::T : origami::transpose_t::N;
    problem.a_dtype = toOrigami(typeA);
    problem.b_dtype = toOrigami(typeB);
    problem.mi_dtype
        = elementBits(typeA) < elementBits(typeB) ? toOrigami(typeB) : toOrigami(typeA);
    // ScaleType defaults are blockRow 32 and blockCol 1 (and the swap on B).
    // genKernelType keeps those defaults, so the product passed to origami is 32
    // whether or not the problem is block-scaled.
    problem.a_mx_block_size = 32;
    problem.b_mx_block_size = 32;

    const int wgmForRank = static_cast<int>(
        std::ceil(std::sqrt(static_cast<double>(hardware.N_CU)
                            / static_cast<double>(std::max<size_t>(hardware.NUM_XCD, 1)))));
    for(auto& config : configs)
        config.workgroup_mapping = wgmForRank;

    const auto ranked = origami::rank_configs(problem, hardware, configs);

    std::vector<KfaTile> deferred;
    for(const auto& result : ranked)
    {
        Tile wgt{static_cast<int>(result.config.mt.m),
                 static_cast<int>(result.config.mt.n),
                 static_cast<int>(result.config.mt.k)};
        const int unroll = preferredUnrolling(typeA, typeB, wgt, preSwizzle, preTile);
        wgt.k /= unroll;

        if(isF8(typeA) || isF8(typeB))
        {
            if(wgt.m == 192 || wgt.n == 192)
                continue;
        }
        if((isF6(typeA) || isF6(typeB)) && (!isPow2(wgt.m) || !isPow2(wgt.n)))
            continue;
        if(preSwizzle)
        {
            if(typeA != MxType::FP4 || typeB != MxType::FP4)
                continue;
            if(wgt.m % 32 != 0 || wgt.n % 32 != 0)
                continue;
            if(wgt.k < 256)
                continue;
        }
        const bool is256 = wgt.m == 256 && wgt.n == 256 && wgt.k == 256;
        if(is256 && !(typeA == MxType::FP4 && typeB == MxType::FP4))
            continue;

        const bool useWgm = prob.k >= kWorkgroupMappingKSize;

        size_t numTiles = 0;
        if(wgt.m > 0 && wgt.n > 0)
            numTiles = (static_cast<size_t>(prob.m) / static_cast<size_t>(wgt.m))
                       * (static_cast<size_t>(prob.n) / static_cast<size_t>(wgt.n))
                       * static_cast<size_t>(prob.batch_count);
        const size_t itersPerTile
            = wgt.k > 0 ? static_cast<size_t>(prob.k) / static_cast<size_t>(wgt.k) : 0;
        const bool largeF8 = (isF8(typeA) || isF8(typeB)) && (wgt.m + wgt.n > 256);
        const int  cuMultiplier = prob.swizzleA ? 4 : 1;
        const size_t usableCUs
            = origami::resolve_num_cus(prob.sm_count_target, hardware.N_CU);
        bool useStreamK = false;
        if(numTiles * static_cast<size_t>(cuMultiplier) < usableCUs && itersPerTile >= 16
           && !isF6(typeA) && !isF6(typeB) && !largeF8)
            useStreamK = true;
        if(useStreamK && wgt.m == 64 && wgt.n == 256)
            continue;

        KfaTile tile;
        tile.m                = wgt.m;
        tile.n                = wgt.n;
        tile.k                = wgt.k;
        tile.workgroupMapping = useWgm;
        tile.streamK          = useStreamK;
        tile.nonTemporalA     = result.config.cache_hints_a != 0;
        tile.nonTemporalB     = result.config.cache_hints_b != 0;

        const bool defer = prob.swizzleA && !useStreamK
                           && ((wgt.m == 32 && wgt.n == 32) || (wgt.m == 64 && wgt.n == 32)
                               || (wgt.m == 64 && wgt.n == 64) || (wgt.m == 128 && wgt.n == 32));
        if(defer)
            deferred.push_back(tile);
        else
            tiles.push_back(tile);
    }
    tiles.insert(tiles.end(), deferred.begin(), deferred.end());
    return rocblaslt_status_success;
}
