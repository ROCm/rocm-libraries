// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "solution_selection.hpp"
#include "analytical_utils.hpp"
#include "kernel_type.hpp"
#include "runtime_args_selection.hpp"

#include "origami/origami.hpp"

#include <sstream>

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
    = {{{256, 256, 256}, {256, 256, 128}, {256, 192, 128}, {256, 128, 128}, {256, 64, 128},
        {256, 32, 128},  {256, 16, 128},  {192, 256, 128}, {192, 128, 128}, {192, 64, 128},
        {192, 32, 128},  {128, 256, 128}, {128, 192, 128}, {128, 128, 128}, {128, 64, 128},
        {128, 32, 128},  {64, 256, 128},  {64, 192, 128},  {64, 128, 128},  {64, 64, 128},
        {64, 32, 128},   {32, 256, 128},  {32, 192, 128},  {32, 128, 128},  {32, 64, 128},
        {32, 32, 128},   {32, 32, 64},    {16, 256, 128},  {64, 16, 128},   {16, 64, 128},
        {32, 16, 128},   {16, 32, 128},   {16, 16, 128},   {16, 16, 256},   {16, 64, 256}}};

constexpr size_t possibleSwizzleTileSizesCount = 37;

constexpr std::array<WorkGroupTileSize, possibleSwizzleTileSizesCount> possibleSwizzleTileSizes
    = {{{32,32,128}, {64, 32, 128}, {64, 64, 128}, {128, 32, 128},
        {32, 128, 128},  {32, 256, 128},  {32, 384, 128},  {32, 512, 128},  {32, 640, 128},
        {32, 768, 128},  {32, 896, 128},  {32, 1024, 128}, {64, 128, 128},  {64, 256, 128},
        {64, 384, 128},  {64, 512, 128},  {64, 640, 128},  {64, 768, 128},  {64, 896, 128},
        {64, 1024, 128}, {96, 128, 128},  {96, 256, 128},  {96, 384, 128},  {96, 512, 128},
        {96, 640, 128},  {128, 128, 128}, {128, 256, 128}, {128, 384, 128},
        {160, 128, 128}, {160, 256, 128}, {160, 384, 128}, {192, 128, 128}, {192, 256, 128},
        {224, 128, 128}, {224, 256, 128}, {256, 128, 128}, {256, 256, 128}}};

// Helper to generate tile list from a compile-time known tile array
// For each tile, generates 3 variants with different nontemporal settings:
// 1. Both A and B non-temporal disabled (cache_hints_a=0, cache_hints_b=0)
// 2. Only A non-temporal enabled (cache_hints_a=4, cache_hints_b=0)
// 3. Only B non-temporal enabled (cache_hints_a=0, cache_hints_b=4)
// Never generates configs where both are enabled simultaneously.
template <rocRoller::DataType typeA,
          rocRoller::DataType typeB,
          size_t              TileCount,
          const std::array<WorkGroupTileSize, TileCount>& TileArray>
std::vector<origami::config_t> generateTileListImpl(bool hasPreSwizzle, bool hasPreTile)
{
    // 3 variants per tile: (ntA=0,ntB=0), (ntA=1,ntB=0), (ntA=0,ntB=1)
    constexpr size_t numNonTemporalVariants = 3;
    std::vector<origami::config_t> tileList;
    tileList.reserve(TileCount * numNonTemporalVariants);

    for(size_t i = 0; i < TileCount; ++i)
    {
        const auto& wgt = TileArray[i];
        auto        MI  = pickMI(typeA, typeB, wgt);

        int wgtk = wgt.k;
        if constexpr(typeA == rocRoller::DataType::Half || typeA == rocRoller::DataType::BFloat16
                     || typeA == rocRoller::DataType::Float)
        {
            wgtk = 32;
        }

        if(hasPreSwizzle && hasPreTile)
        {
            wgtk = 256;
        }

        int unroll = preferredUnrolling(typeA, typeB, wgt, hasPreSwizzle, hasPreTile);

        // Generate 3 variants with different nontemporal settings
        // Variant 1: Both disabled
        origami::config_t config_both_off = {
            .mt = {static_cast<size_t>(wgt.m),
                   static_cast<size_t>(wgt.n),
                   static_cast<size_t>(wgtk * unroll)},
            .mi = {static_cast<size_t>(MI.m), static_cast<size_t>(MI.n), static_cast<size_t>(MI.k)},
            .occupancy     = 1,
            .cache_hints_a = 0,
            .cache_hints_b = 0,
        };
        tileList.push_back(config_both_off);

        // Variant 2: Only A non-temporal enabled
        origami::config_t config_a_on = {
            .mt = {static_cast<size_t>(wgt.m),
                   static_cast<size_t>(wgt.n),
                   static_cast<size_t>(wgtk * unroll)},
            .mi = {static_cast<size_t>(MI.m), static_cast<size_t>(MI.n), static_cast<size_t>(MI.k)},
            .occupancy     = 1,
            .cache_hints_a = 4,
            .cache_hints_b = 0,
        };
        tileList.push_back(config_a_on);

        // Variant 3: Only B non-temporal enabled
        origami::config_t config_b_on = {
            .mt = {static_cast<size_t>(wgt.m),
                   static_cast<size_t>(wgt.n),
                   static_cast<size_t>(wgtk * unroll)},
            .mi = {static_cast<size_t>(MI.m), static_cast<size_t>(MI.n), static_cast<size_t>(MI.k)},
            .occupancy     = 1,
            .cache_hints_a = 0,
            .cache_hints_b = 4,
        };
        tileList.push_back(config_b_on);
    }

    return tileList;
}

// Standard tile list generator using possibleTileSizes
template <rocRoller::DataType typeA, rocRoller::DataType typeB>
std::vector<origami::config_t> generateTileList(bool hasPreSwizzle, bool hasPreTile)
{
    return generateTileListImpl<typeA, typeB, possibleTileSizesCount, possibleTileSizes>(
        hasPreSwizzle, hasPreTile);
}

// Swizzle tile list generator using possibleSwizzleTileSizes (FP4 only)
template <rocRoller::DataType typeA, rocRoller::DataType typeB>
std::vector<origami::config_t> generateSwizzleTileList(bool hasPreSwizzle, bool hasPreTile)
{
    return generateTileListImpl<typeA, typeB, possibleSwizzleTileSizesCount, possibleSwizzleTileSizes>(
        hasPreSwizzle, hasPreTile);
}

using TileListGeneratorFn = std::vector<origami::config_t> (*)(bool, bool);

template <rocRoller::DataType A, rocRoller::DataType B>
std::vector<origami::config_t> generateTileListWrapper(bool hasPreSwizzle, bool hasPreTile)
{
    return generateTileList<A, B>(hasPreSwizzle, hasPreTile);
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

// Pre-instantiated swizzle tile generator for FP4 x FP4 (compile-time optimized)
static const TileListGeneratorFn fp4SwizzleTileGenerator
    = &generateSwizzleTileList<rocRoller::DataType::FP4, rocRoller::DataType::FP4>;

std::vector<origami::config_t> getTileListForKernelType(const KernelType& kernelType)
{
    // Compute hasPreSwizzle and hasPreTile from ScaleType
    bool hasPreSwizzle = (kernelType.scaleTypeA.preSwizzleTile.size() == 3
                          && kernelType.scaleTypeB.preSwizzleTile.size() == 3);
    bool hasPreTile    = (kernelType.scaleTypeA.preTile.size() == 2
                       && kernelType.scaleTypeB.preTile.size() == 2);

    // Use swizzle tile sizes only for FP4 x FP4 with swizzleA enabled
    if(kernelType.swizzleA && kernelType.typeA == rocRoller::DataType::FP4
       && kernelType.typeB == rocRoller::DataType::FP4)
    {
        return fp4SwizzleTileGenerator(hasPreSwizzle, hasPreTile);
    }

    // Standard path: look up generator in map
    auto key = std::make_pair(kernelType.typeA, kernelType.typeB);
    auto it  = tileListGenerators.find(key);
    if(it != tileListGenerators.end())
    {
        return it->second(hasPreSwizzle, hasPreTile);
    }
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
    std::vector<SolutionIndexParameters> lastParams;

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
        auto hasPreSwizzle = (kernelType.scaleTypeA.preSwizzleTile.size() == 3
                              && kernelType.scaleTypeB.preSwizzleTile.size() == 3);
        auto hasPreTile = (kernelType.scaleTypeA.preTile.size() == 2
                           && kernelType.scaleTypeB.preTile.size() == 2);
        int unrollAmount = preferredUnrolling(kernelType.typeA, kernelType.typeB, wgt, hasPreSwizzle, hasPreTile);
        wgt.k /= unrollAmount;

        // FP8 kernels run out of registers with larger tile sizes
        if((kernelType.typeA == rocRoller::DataType::FP8
            || kernelType.typeA == rocRoller::DataType::BF8
            || kernelType.typeB == rocRoller::DataType::FP8
            || kernelType.typeB == rocRoller::DataType::BF8)
            && (wgt.m == 192 || wgt.n == 192))
            continue;

        // 6bit datatypes only work with power of 2 tile sizes
        if((kernelType.typeA == rocRoller::DataType::FP6
            || kernelType.typeA == rocRoller::DataType::BF6
            || kernelType.typeB == rocRoller::DataType::FP6
            || kernelType.typeB == rocRoller::DataType::BF6)
            && (!std::has_single_bit(static_cast<uint>(wgt.m))
                || !std::has_single_bit(static_cast<uint>(wgt.n))))
            continue;

        // check if this size is valid for pre-swizzled data
        if (hasPreSwizzle)
        {
            if (kernelType.typeA != rocRoller::DataType::FP4 ||
                kernelType.typeB != rocRoller::DataType::FP4)
                continue;
            if (wgt.m % 32 != 0 || wgt.n % 32 != 0)
                continue;
        }

        // wgt.k has to be at least 256 when scale data is pre-swizzled
        if(kernelType.scaleTypeA.preSwizzleTile.size() == 3
            && kernelType.scaleTypeB.preSwizzleTile.size() == 3 && wgt.k < 256)
            continue;

        // {256, 256, 256} tile size is only supported for FP4 data types with preSwizzled and preTiled scale data
        bool isFP4 = (kernelType.typeA == rocRoller::DataType::FP4
                        && kernelType.typeB == rocRoller::DataType::FP4);

        bool is256Tile = (wgt.m == 256 && wgt.n == 256 && wgt.k == 256);

        // Only allow 256x256x256 for FP4 with preSwizzled and preTiled scale data
        if(is256Tile && !(isFP4))
            continue;

        bool useTailLoops = true;

        

        bool useWorkgroupMapping = true;
        if(prob.k < USE_WORKGROUP_MAPPING_K_SIZE)
        {
            useWorkgroupMapping = false;
        }

        // Enable StreamK when:
        // 1. Number of output tiles < number of CUs
        // 2. There are enough K iterations per tile (itersPerTile >= 16) to
        //    amortize StreamK overhead. Threshold is derived from origami's
        //    MinItersPerCU (8) applied to the smallest useful split factor (2).
        // 3. Data type is not f6 (unsupported) or large f8 (register pressure).
        // 4. Not the 256x256x256 FP4 pre-swizzled tile.
        bool useStreamK = false;
        size_t numTilesM    = prob.m / wgt.m;
        size_t numTilesN    = prob.n / wgt.n;
        size_t numTiles     = numTilesM * numTilesN * prob.batch_count;
        size_t itersPerTile = prob.k / wgt.k;
        auto   isF6         = (kernelType.typeA == rocRoller::DataType::FP6
                        || kernelType.typeA == rocRoller::DataType::BF6
                        || kernelType.typeB == rocRoller::DataType::FP6
                        || kernelType.typeB == rocRoller::DataType::BF6);
        auto isLargeF8 = ((kernelType.typeA == rocRoller::DataType::FP8
            || kernelType.typeA == rocRoller::DataType::BF8
            || kernelType.typeB == rocRoller::DataType::FP8
            || kernelType.typeB == rocRoller::DataType::BF8)
            && wgt.m + wgt.n > 256);
        int cu_multiplier = 1;
        if(kernelType.swizzleA)
            cu_multiplier = 4;
        if(numTiles * cu_multiplier < analytical_hardware.N_CU && itersPerTile >= 16
            && !isF6 && !isLargeF8)
        {
            useStreamK = true;
        }

        // Heuristics:
        // 64x256 performs poorly for StreamK
        if(useStreamK && wgt.m == 64 && wgt.n == 256)
            continue;

        // Extract nontemporal hints from the config
        bool useNonTemporalA = (result.config.cache_hints_a != 0);
        bool useNonTemporalB = (result.config.cache_hints_b != 0);

        // Prefer assembly kernels for swizzleA
        if(kernelType.swizzleA && !useStreamK && ((wgt.m == 32 && wgt.n == 32) ||
                                                    (wgt.m == 64 && wgt.n == 32) ||
                                                    (wgt.m == 64 && wgt.n == 64) ||
                                                    (wgt.m == 128 && wgt.n == 32)))
            lastParams.push_back({wgt, useWorkgroupMapping, useStreamK, useTailLoops, useNonTemporalA, useNonTemporalB});
        else
            params.push_back({wgt, useWorkgroupMapping, useStreamK, useTailLoops, useNonTemporalA, useNonTemporalB});
    }

    // Append lastParams to params so that assembly kernel tile sizes are included as fallback options
    params.insert(params.end(), lastParams.begin(), lastParams.end());

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
    pos += 1;
    result |= ((params.tailLoops ? 1 : 0) << pos);
    pos += 1;
    result |= ((params.nonTemporalA ? 1 : 0) << pos);
    pos += 1;
    result |= ((params.nonTemporalB ? 1 : 0) << pos);

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
    pos += 1;
    result.tailLoops = (index >> pos) & 1;
    pos += 1;
    result.nonTemporalA = (index >> pos) & 1;
    pos += 1;
    result.nonTemporalB = (index >> pos) & 1;

    return result;
}

std::string shortRocRollerKernelNameFromSolutionIndex(const SolutionIndexParameters& p)
{
    std::ostringstream o;
    o << "rr_" << p.workgroupTile.m << "x" << p.workgroupTile.n << "x" << p.workgroupTile.k;
    if(p.workgroupMapping)
        o << "_wgm";
    if(p.streamK)
        o << "_sk";
    if(p.nonTemporalA)
        o << "_ntA";
    if(p.nonTemporalB)
        o << "_ntB";
    return o.str();
}
