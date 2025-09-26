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

#include "analytical_utils.hpp"
#include "kernel_type.hpp"
#include "runtime_args_selection.hpp"
#include "solution_selection.hpp"

#include <origami/utils.hpp>

const int MAX_BITS_WORKGROUPTILE_M     = 8;
const int MAX_BITS_WORKGROUPTILE_N     = 8;
const int MAX_BITS_WORKGROUPTILE_K     = 7;
const int MAX_BITS_PREFETCH_IN_FLIGHT  = 4;
const int REQUIRED_MULTIPLE_M_N        = 16;
const int REQUIRED_MULTIPLE_K          = 32;
const int USE_WORKGROUP_MAPPING_K_SIZE = 4096;

/**
 **************************************************************************************************
 * This section defines the lists of macro-tile/matrix-instruction combinations so that they are
 * compile-time known.
 */

 constexpr std::array<WorkGroupTileSize, 34> possibleTileSizes = {{
    {256, 256, 128},
    {256, 192, 128},
    {256, 128, 128},
    {256, 64, 128},
    {256, 32, 128},
    {256, 16, 128},
    {192, 256, 128},
    {192, 128, 128},
    {192, 64, 128},
    {192, 32, 128},
    {128, 256, 128},
    {128, 192, 128},
    {128, 128, 128},
    {128, 64, 128},
    {128, 32, 128},
    {64, 256, 128},
    {64, 192, 128},
    {64, 128, 128},
    {64, 64, 128},
    {64, 32, 128},
    {32, 256, 128},
    {32, 192, 128},
    {32, 128, 128},
    {32, 64, 128},
    {32, 32, 128},
    {32, 32, 64},
    {16, 256, 128},
    {64, 16, 128},
    {16, 64, 128},
    {32, 16, 128},
    {16, 32, 128},
    {16, 16, 128},
    {16, 16, 256},
    {16, 64, 256}
}};

constexpr int preferredUnrolling(rocRoller::DataType typeA, rocRoller::DataType typeB, WorkGroupTileSize wgt) {
    // Other datatypes run out of registers when prefetchInFlight is too
    // large.
    // There is an error with smaller tile sizes and larger prefetchInFlight.
    if (typeA == rocRoller::DataType::FP4 && typeB == rocRoller::DataType::FP4 && wgt.m > 32 && wgt.n > 32)
        return 4;
    else
        return 2;
}

template <rocRoller::DataType typeA, rocRoller::DataType typeB>
constexpr auto generateTileList() {
    std::array<origami::tile_tuple, possibleTileSizes.size()> tileList{};

    for (size_t i = 0; i < possibleTileSizes.size(); ++i) {
        const auto& wgt = possibleTileSizes[i];
        auto MI = pickMI(typeA, typeB, wgt);

        int wgtk = wgt.k;
        if (typeA == rocRoller::DataType::Half || typeA == rocRoller::DataType::BFloat16 || typeA == rocRoller::DataType::Float) {
            wgtk = 32;
        }

        int unroll = preferredUnrolling(typeA, typeB, wgt);

        int non_temporal_a = 0;
        int non_temporal_b = 0;

        tileList[i] = std::make_tuple(
            wgt.m, wgt.n, wgtk * unroll,
            MI.m, MI.n, MI.k,
            1, // occupancy
            DEFAULT_WGM,
            non_temporal_a,
            non_temporal_b
        );
    }

    return tileList;
}

using TileListGeneratorFn = std::vector<origami::tile_tuple>(*)();

template <rocRoller::DataType A, rocRoller::DataType B>
std::vector<origami::tile_tuple> generateTileListWrapper() {
    constexpr auto arr = generateTileList<A, B>();
    return {arr.begin(), arr.end()};
}

#define INSTANTIATE_TILE_LIST(A, B) \
    { {rocRoller::DataType::A, rocRoller::DataType::B}, &generateTileListWrapper<rocRoller::DataType::A, rocRoller::DataType::B> }

#define INSTANTIATE_TILE_LIST_FOR(A) \
    INSTANTIATE_TILE_LIST(A, Half), \
    INSTANTIATE_TILE_LIST(A, Float), \
    INSTANTIATE_TILE_LIST(A, BFloat16), \
    INSTANTIATE_TILE_LIST(A, FP8), \
    INSTANTIATE_TILE_LIST(A, BF8), \
    INSTANTIATE_TILE_LIST(A, FP4), \
    INSTANTIATE_TILE_LIST(A, BF6), \
    INSTANTIATE_TILE_LIST(A, FP6)

const std::map<std::pair<rocRoller::DataType, rocRoller::DataType>, TileListGeneratorFn> tileListGenerators = {
    INSTANTIATE_TILE_LIST_FOR(Half),
    INSTANTIATE_TILE_LIST_FOR(Float),
    INSTANTIATE_TILE_LIST_FOR(BFloat16),
    INSTANTIATE_TILE_LIST_FOR(FP8),
    INSTANTIATE_TILE_LIST_FOR(BF8),
    INSTANTIATE_TILE_LIST_FOR(FP4),
    INSTANTIATE_TILE_LIST_FOR(BF6),
    INSTANTIATE_TILE_LIST_FOR(FP6)
};

std::vector<origami::tile_tuple> getTileListForKernelType(KernelType kernelType)
{
    auto key = std::make_pair(kernelType.typeA, kernelType.typeB);
    auto it = tileListGenerators.find(key);
    if (it != tileListGenerators.end())
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

    std::vector<origami::tile_tuple> tile_list = getTileListForKernelType(kernelType);

    size_t elementSizeA_bits = rocRoller::DataTypeInfo::Get(kernelType.typeA).elementBits;
    size_t elementSizeB_bits = rocRoller::DataTypeInfo::Get(kernelType.typeB).elementBits;
    size_t elementSizeC_bits = rocRoller::DataTypeInfo::Get(kernelType.typeC).elementBits;

    origami::data_type_t dataType;
    if (elementSizeA_bits < elementSizeB_bits)
        dataType = rocroller_type_to_analytical_type(kernelType.typeB);
    else
        dataType = rocroller_type_to_analytical_type(kernelType.typeA);

    const origami::hardware_t analaytical_hardware = origami::hardware_t::get_hardware_for_device(0);

    int WGM = std::sqrt(std::floor(analaytical_hardware.N_CU / analaytical_hardware.NUM_XCD));

    auto selected_tiles = origami::select_best_macro_tile_size(
        prob.m,
        prob.n,
        prob.k,
        prob.batch_count,
        prob.trans_a == hipblasOperation_t::HIPBLAS_OP_T,
        prob.trans_b == hipblasOperation_t::HIPBLAS_OP_T,
        analaytical_hardware,
        tile_list,
        elementSizeA_bits,
        elementSizeB_bits,
        elementSizeC_bits,
        dataType,
        kernelType.scaleABlockRowSize * kernelType.scaleABlockColSize, //Handle A vs B block size.
        0.8,
        false,
        WGM);

    for(auto const& selected_tile : selected_tiles)
    {
        WorkGroupTileSize wgt{(int)std::get<1>(selected_tile), (int)std::get<2>(selected_tile), (int)std::get<3>(selected_tile)};
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

            params.push_back({wgt, 1, true});
            while (unrollAmount > 1 && (prob.k % (wgt.k * unrollAmount) != 0))
            {
                unrollAmount = unrollAmount / 2;
            }

            params.back().prefetchInFlight = unrollAmount;

            if (prob.k < USE_WORKGROUP_MAPPING_K_SIZE)
            {
                params.back().workgroupMapping = false;
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
    result |= (params.prefetchInFlight << pos);
    pos += MAX_BITS_PREFETCH_IN_FLIGHT;
    result |= ((params.workgroupMapping ? 1 : 0) << pos);

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
    result.prefetchInFlight = (index >> pos) & mask(MAX_BITS_PREFETCH_IN_FLIGHT);
    pos += MAX_BITS_PREFETCH_IN_FLIGHT;
    result.workgroupMapping = (index >> pos) & 1;

    return result;
}
