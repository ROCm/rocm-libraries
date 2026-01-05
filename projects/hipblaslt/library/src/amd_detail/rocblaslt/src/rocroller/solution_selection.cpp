/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "solution_selection.hpp"
#include "analytical_utils.hpp"
#include "kernel_type.hpp"
#include "runtime_args_selection.hpp"

#include "origami/origami.hpp"

const int MAX_BITS_WORKGROUPTILE_M     = 8;
const int MAX_BITS_WORKGROUPTILE_N     = 8;
const int MAX_BITS_WORKGROUPTILE_K     = 7;
const int REQUIRED_MULTIPLE_M_N        = 16;
const int REQUIRED_MULTIPLE_K          = 32;
const int USE_WORKGROUP_MAPPING_K_SIZE = 4096;

/**
 **************************************************************************************************
 * This section defines the lists of macro-tile/matrix-instruction combinations so that they are
 * compile-time known.
 */

constexpr size_t possibleTileSizesCount = 35;

constexpr std::array<WorkGroupTileSize, possibleTileSizesCount> possibleTileSizes
    = {{{256, 256, 128}, {256, 192, 128}, {256, 128, 128}, {256, 64, 128}, {256, 32, 128},
        {256, 16, 128},  {192, 256, 128}, {192, 128, 128}, {192, 64, 128}, {192, 32, 128},
        {128, 256, 128}, {128, 192, 128}, {128, 128, 128}, {128, 64, 128}, {128, 32, 128},
        {64, 256, 128},  {64, 192, 128},  {64, 128, 128},  {64, 64, 128},  {64, 32, 128},
        {32, 256, 128},  {32, 192, 128},  {32, 128, 128},  {32, 64, 128},  {32, 32, 128},
        {32, 32, 64},    {16, 256, 128},  {64, 16, 128},   {16, 64, 128},  {32, 16, 128},
        {16, 32, 128},   {16, 16, 128},   {16, 16, 256},   {16, 64, 256},  {128, 128, 256}}};

template <rocRoller::DataType typeA, rocRoller::DataType typeB>
auto generateTileList()
{
    std::array<origami::config_t, possibleTileSizesCount> tileList{};

    for(size_t i = 0; i < possibleTileSizesCount; ++i)
    {
        const auto& wgt = possibleTileSizes[i];
        auto        MI  = pickMI(typeA, typeB, wgt);

        int wgtk = wgt.k;
        if(typeA == rocRoller::DataType::Half || typeA == rocRoller::DataType::BFloat16
           || typeA == rocRoller::DataType::Float)
        {
            wgtk = 32;
        }

        int unroll = preferredUnrolling(typeA, typeB, wgt);

        origami::config_t origami_config = {
            .mt = {static_cast<size_t>(wgt.m),
                   static_cast<size_t>(wgt.n),
                   static_cast<size_t>(wgtk * unroll)},
            .mi = {static_cast<size_t>(MI.m), static_cast<size_t>(MI.n), static_cast<size_t>(MI.k)},
            .occupancy     = 1,
            .cache_hints_a = 0,
            .cache_hints_b = 0,
        };

        tileList[i] = origami_config;
    }

    return tileList;
}

using TileListGeneratorFn = std::vector<origami::config_t> (*)();

template <rocRoller::DataType A, rocRoller::DataType B>
std::vector<origami::config_t> generateTileListWrapper()
{
    auto arr = generateTileList<A, B>();
    return {arr.begin(), arr.end()};
}

#define INSTANTIATE_TILE_LIST(A, B)                                                  \
    {                                                                                \
        {rocRoller::DataType::A, rocRoller::DataType::B},                            \
            &generateTileListWrapper<rocRoller::DataType::A, rocRoller::DataType::B> \
    }

#define INSTANTIATE_TILE_LIST_FOR(A)                                       \
    INSTANTIATE_TILE_LIST(A, Half), INSTANTIATE_TILE_LIST(A, Float),       \
        INSTANTIATE_TILE_LIST(A, BFloat16), INSTANTIATE_TILE_LIST(A, FP8), \
        INSTANTIATE_TILE_LIST(A, BF8), INSTANTIATE_TILE_LIST(A, FP4),      \
        INSTANTIATE_TILE_LIST(A, BF6), INSTANTIATE_TILE_LIST(A, FP6)

const std::map<std::pair<rocRoller::DataType, rocRoller::DataType>, TileListGeneratorFn>
    tileListGenerators = {INSTANTIATE_TILE_LIST_FOR(Half),
                          INSTANTIATE_TILE_LIST_FOR(Float),
                          INSTANTIATE_TILE_LIST_FOR(BFloat16),
                          INSTANTIATE_TILE_LIST_FOR(FP8),
                          INSTANTIATE_TILE_LIST_FOR(BF8),
                          INSTANTIATE_TILE_LIST_FOR(FP4),
                          INSTANTIATE_TILE_LIST_FOR(BF6),
                          INSTANTIATE_TILE_LIST_FOR(FP6)};

std::vector<origami::config_t> getTileListForKernelType(KernelType kernelType)
{
    auto key = std::make_pair(kernelType.typeA, kernelType.typeB);
    auto it  = tileListGenerators.find(key);
    if(it != tileListGenerators.end())
        return it->second();
    throw std::runtime_error("Unsupported DataType combination");
}

/**
 *
 **************************************************************************************************
 */

size_t maxNumberSolutions()
{
    return possibleTileSizes.size();
}

std::vector<SolutionIndexParameters> chooseSolutionIndexParameters(
    const KernelType& kernelType, const RocblasltContractionProblem& prob, int requestedAlgoCount)
{
    std::vector<SolutionIndexParameters> params;

    std::vector<origami::config_t> origami_config_list = getTileListForKernelType(kernelType);

    size_t elementSizeA_bits = rocRoller::DataTypeInfo::Get(kernelType.typeA).elementBits;
    size_t elementSizeB_bits = rocRoller::DataTypeInfo::Get(kernelType.typeB).elementBits;

    const origami::hardware_t analytical_hardware = origami::hardware_t::get_hardware_for_device(0);

    origami::problem_t origami_problem = {
        .size        = {prob.m, prob.n, prob.k},
        .batch       = prob.batch_count,
        .a_transpose = (prob.trans_a == hipblasOperation_t::HIPBLAS_OP_T) ? origami::transpose_t::T
                                                                          : origami::transpose_t::N,
        .b_transpose = (prob.trans_b == hipblasOperation_t::HIPBLAS_OP_T) ? origami::transpose_t::T
                                                                          : origami::transpose_t::N,
        .a_dtype     = rocroller_type_to_analytical_type(kernelType.typeA),
        .b_dtype     = rocroller_type_to_analytical_type(kernelType.typeB),
        .mi_dtype    = rocroller_type_to_analytical_type(
            elementSizeA_bits < elementSizeB_bits ? kernelType.typeB : kernelType.typeA),
        .a_mx_block_size = kernelType.scaleTypeA.blockRowSize * kernelType.scaleTypeA.blockColSize,
        .b_mx_block_size = kernelType.scaleTypeB.blockRowSize * kernelType.scaleTypeB.blockColSize,
    };

    int defaultWGM = std::ceil(std::sqrt(analytical_hardware.N_CU / analytical_hardware.NUM_XCD));
    for(auto& config : origami_config_list)
    {
        config.workgroup_mapping = defaultWGM;
    }

    auto prediction_result
        = origami::rank_configs(origami_problem, analytical_hardware, origami_config_list);

    for(auto const& result : prediction_result)
    {
        auto              mt_m = static_cast<int>(result.config.mt.m);
        auto              mt_n = static_cast<int>(result.config.mt.n);
        auto              mt_k = static_cast<int>(result.config.mt.k);
        WorkGroupTileSize wgt{mt_m, mt_n, mt_k};
        int unrollAmount = preferredUnrolling(kernelType.typeA, kernelType.typeB, wgt);
        wgt.k /= unrollAmount;

        if((requestedAlgoCount == -1)
           || (prob.m % wgt.m == 0 && prob.n % wgt.n == 0 && prob.k % wgt.k == 0))
        {
            // FP8 kernels run out of registers with larger tile sizes
            if((kernelType.typeA == rocRoller::DataType::FP8
                || kernelType.typeA == rocRoller::DataType::BF8
                || kernelType.typeB == rocRoller::DataType::FP8
                || kernelType.typeB == rocRoller::DataType::BF8)
               && wgt.m + wgt.n > 256)
                continue;

            // 6bit datatypes only work with power of 2 tile sizes
            if((kernelType.typeA == rocRoller::DataType::FP6
                || kernelType.typeA == rocRoller::DataType::BF6
                || kernelType.typeB == rocRoller::DataType::FP6
                || kernelType.typeB == rocRoller::DataType::BF6)
               && (!std::has_single_bit(static_cast<uint>(wgt.m))
                   || !std::has_single_bit(static_cast<uint>(wgt.n))))
                continue;

            // Pre-swizzled scald data requires the wgt.m >= 128 and wgt.n >= 128 to be able to turn on SwizzleScale
            if(kernelType.scaleTypeA.preSwizzleTile.size() == 3 && (wgt.m < 128))
                continue;
            if(kernelType.scaleTypeB.preSwizzleTile.size() == 3 && (wgt.n < 128))
                continue;
            // wgt.k has to be at least 256 when scale data is pre-swizzled
            if(kernelType.scaleTypeA.preSwizzleTile.size() == 3
               && kernelType.scaleTypeB.preSwizzleTile.size() == 3 && wgt.k < 256)
                continue;

            // Prevent selecting {256, 256, 256} as workgroup tile size due to VGPR register pressure
            if(wgt.m == 256 && wgt.n == 256 && wgt.k == 256)
                continue;

            params.push_back({wgt, true, false});

            if(prob.k < USE_WORKGROUP_MAPPING_K_SIZE)
            {
                params.back().workgroupMapping = false;
            }

            // Enable StreamK when number of output tiles < number of CUs and not f6 data type
            size_t numTilesM = prob.m / wgt.m;
            size_t numTilesN = prob.n / wgt.n;
            size_t numTiles  = numTilesM * numTilesN * prob.batch_count;
            auto   isF6      = (kernelType.typeA == rocRoller::DataType::FP6
                         || kernelType.typeA == rocRoller::DataType::BF6
                         || kernelType.typeB == rocRoller::DataType::FP6
                         || kernelType.typeB == rocRoller::DataType::BF6);
            if(numTiles < analytical_hardware.N_CU && !isF6)
            {
                params.back().streamK = true;
            }
        }
    }

    return params;
}

/**
 * Convert a solution index back into SolutionIndexParameters
 */
int parametersToIndex(const SolutionIndexParameters& params)
{
    int          result = params.workgroupTile.k / REQUIRED_MULTIPLE_K;
    unsigned int pos    = MAX_BITS_WORKGROUPTILE_K;
    result |= ((params.workgroupTile.n / REQUIRED_MULTIPLE_M_N) << pos);
    pos += MAX_BITS_WORKGROUPTILE_N;
    result |= ((params.workgroupTile.m / REQUIRED_MULTIPLE_M_N) << pos);
    pos += MAX_BITS_WORKGROUPTILE_M;
    result |= ((params.workgroupMapping ? 1 : 0) << pos);
    pos += 1;
    result |= ((params.streamK ? 1 : 0) << pos);

    // Set top bit indicating it is a rocRoller index
    result |= (1 << 31);
    return result;
}

inline unsigned int mask(unsigned int numBits)
{
    return (1 << numBits) - 1;
}

/**
 * Convert a solution index back into SolutionIndexParameters
 */
SolutionIndexParameters indexToParameters(int index)
{
    SolutionIndexParameters result;
    unsigned int            pos = 0;

    result.workgroupTile.k
        = ((index >> pos) & mask(MAX_BITS_WORKGROUPTILE_K)) * REQUIRED_MULTIPLE_K;
    pos += MAX_BITS_WORKGROUPTILE_K;
    result.workgroupTile.n
        = ((index >> pos) & mask(MAX_BITS_WORKGROUPTILE_N)) * REQUIRED_MULTIPLE_M_N;
    pos += MAX_BITS_WORKGROUPTILE_N;
    result.workgroupTile.m
        = ((index >> pos) & mask(MAX_BITS_WORKGROUPTILE_M)) * REQUIRED_MULTIPLE_M_N;
    pos += MAX_BITS_WORKGROUPTILE_M;
    result.workgroupMapping = (index >> pos) & 1;
    pos += 1;
    result.streamK = (index >> pos) & 1;

    return result;
}
