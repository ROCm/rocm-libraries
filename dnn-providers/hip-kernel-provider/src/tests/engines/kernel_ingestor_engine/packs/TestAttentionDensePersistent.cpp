// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/AttentionDensePersistent.hpp"

/**
 * @file TestAttentionDensePersistent.cpp
 * @brief The launch grids AttentionDensePersistent.hpp restates, against tables the rocKE
 *        Python generates (gen_attention_dense_persistent_goldens.py, which writes
 *        TestAttentionDensePersistentGoldens.inc).
 *
 * Nothing at run time compares the C++ grid with the Python one, and a mismatch does not
 * fault: a wrong fast-division magic number, divisor order or grid dimension makes the
 * persistent kernel compute some work items twice and others never. Each table here is
 * the Python's own answer: calculate_magic_numbers, attention_dense_signature's argument
 * names, and per shape whether spec construction or fastdiv_kernargs refuses the grid,
 * and otherwise attention_dense_grid and the fast-division kernargs.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

using attention_dense::AttentionDenseFastDivPair;
using attention_dense::AttentionDenseGrid;
using attention_dense::AttentionDenseGridOrder;
using attention_dense::AttentionDenseGridShape;
using attention_dense::AttentionDensePersistDecode;

struct MagicGolden
{
    int64_t divisor;
    int32_t magic;
    int32_t shift;
};

struct SignatureGolden
{
    const char* grid;
    bool lse;
    /// attention_dense_signature's argument names, space-separated.
    const char* names;
};

struct GridGolden
{
    /// A non-persistent order, or `persistent/<decode>`.
    const char* grid;
    int64_t batch;
    int64_t seqLenQ;
    int64_t seqLenKv;
    int64_t numQueryHeads;
    int64_t numKvHeads;
    int64_t blockM;
    bool causal;
    int64_t slidingWindow;
    /// Whether the Python builds the spec and its kernargs at this shape.
    bool legal;
    int64_t gridX;
    int64_t gridY;
    int64_t gridZ;
    int pairValues;
    std::array<int32_t, 6> pairs;
};

#include "TestAttentionDensePersistentGoldens.inc"

/// The num_persistent every golden persistent grid is built with.
constexpr int64_t GOLDEN_NUM_PERSISTENT = 304;

/// The grid a golden names.
AttentionDenseGrid gridOf(std::string_view name)
{
    constexpr std::string_view PERSISTENT_PREFIX = "persistent/";
    AttentionDenseGrid grid;
    if(name.substr(0, PERSISTENT_PREFIX.size()) == PERSISTENT_PREFIX)
    {
        const auto decode
            = attention_dense::parsePersistDecode(name.substr(PERSISTENT_PREFIX.size()));
        EXPECT_TRUE(decode.has_value()) << name;
        grid.order = AttentionDenseGridOrder::PERSISTENT;
        grid.decode = decode.value_or(AttentionDensePersistDecode::QB_MAJOR);
        grid.numPersistent = GOLDEN_NUM_PERSISTENT;
        return grid;
    }
    const auto order = attention_dense::parseNonpersistDecode(name);
    EXPECT_TRUE(order.has_value()) << name;
    grid.order = order.value_or(AttentionDenseGridOrder::QB_MINOR);
    return grid;
}

AttentionDenseGridShape shapeOf(const GridGolden& golden)
{
    AttentionDenseGridShape shape;
    shape.batch = golden.batch;
    shape.seqLenQ = golden.seqLenQ;
    shape.seqLenKv = golden.seqLenKv;
    shape.numQueryHeads = golden.numQueryHeads;
    shape.numKvHeads = golden.numKvHeads;
    shape.blockM = golden.blockM;
    shape.causal = golden.causal;
    shape.slidingWindow = golden.slidingWindow;
    return shape;
}

std::string describe(const GridGolden& golden)
{
    std::ostringstream out;
    out << golden.grid << " B" << golden.batch << " Sq" << golden.seqLenQ << " Skv"
        << golden.seqLenKv << " Hq" << golden.numQueryHeads << " Hkv" << golden.numKvHeads << " bm"
        << golden.blockM << " causal " << golden.causal << " window " << golden.slidingWindow;
    return out.str();
}

/// The argument names a grid's launch passes after the five shape arguments.
std::vector<std::string> fastDivNames(const AttentionDenseGrid& grid)
{
    std::vector<std::string> names;
    if(!grid.persistent())
    {
        return names;
    }
    for(const auto divisor : attention_dense::fastDivDivisors(grid.decode))
    {
        names.push_back(std::string(attention_dense::divisorName(divisor)) + "_magic");
        names.push_back(std::string(attention_dense::divisorName(divisor)) + "_shift");
    }
    return names;
}

} // namespace

TEST(TestAttentionDensePersistent, FastDivPairIsCalculateMagicNumbers)
{
    ASSERT_GE(std::size(MAGIC_GOLDENS), 4096U + 512U);
    for(const auto& golden : MAGIC_GOLDENS)
    {
        const auto pair = attention_dense::fastDivPair(golden.divisor);
        EXPECT_EQ(pair, (AttentionDenseFastDivPair{golden.magic, golden.shift}))
            << "divisor " << golden.divisor;
    }
}

/// The magic numbers divide exactly below the 2^30 dividend bound, as the kernel computes
/// it: (umulhi(n, magic) + n) >> shift.
TEST(TestAttentionDensePersistent, FastDivPairDividesExactlyBelowTheDividendBound)
{
    for(const int64_t divisor : {int64_t{1},
                                 int64_t{3},
                                 int64_t{7},
                                 int64_t{12},
                                 int64_t{40},
                                 int64_t{304},
                                 int64_t{4095},
                                 int64_t{32768},
                                 int64_t{1000003}})
    {
        const auto pair = attention_dense::fastDivPair(divisor);
        const auto magic = static_cast<uint64_t>(static_cast<uint32_t>(pair.magic));
        for(const int64_t n : {int64_t{0},
                               int64_t{1},
                               divisor - 1,
                               divisor,
                               divisor + 1,
                               int64_t{123456789},
                               attention_dense::FASTDIV_DIVIDEND_LIMIT - 1})
        {
            const auto dividend = static_cast<uint64_t>(n);
            const uint64_t quotient = (((dividend * magic) >> 32U) + dividend) >> pair.shift;
            EXPECT_EQ(quotient, dividend / static_cast<uint64_t>(divisor))
                << "n " << n << " / " << divisor;
        }
    }
}

/// Every name requireSignatureMatch compares: the pairs follow the shape in the decode's own
/// divisor order, and lse_ptr follows them.
TEST(TestAttentionDensePersistent, FastDivNamesFollowTheSignatureThePythonDeclares)
{
    ASSERT_EQ(std::size(SIGNATURE_GOLDENS), 14U);
    for(const auto& golden : SIGNATURE_GOLDENS)
    {
        std::string names = "q_ptr k_ptr v_ptr o_ptr scale batch seqlen_q seqlen_kv "
                            "num_query_heads num_kv_heads";
        for(const auto& name : fastDivNames(gridOf(golden.grid)))
        {
            names += " " + name;
        }
        if(golden.lse)
        {
            names += " lse_ptr";
        }
        EXPECT_EQ(names, golden.names) << golden.grid << " lse " << golden.lse;
    }
}

/// The legality verdict, the grid and the fast-division kernargs per (grid, shape), for
/// every grid the engine launches and over shapes that cover GQA (including non-power-of-2
/// groups), Hq % 8 != 0, odd and even query-block counts, windows shorter and longer than
/// the keys, W < NP and W >> NP, and both dividend bounds.
TEST(TestAttentionDensePersistent, LegalityGridAndKernargsMatchThePython)
{
    ASSERT_GT(std::size(GRID_GOLDENS), 1000U);
    std::size_t legal = 0;
    std::size_t illegal = 0;
    for(const auto& golden : GRID_GOLDENS)
    {
        const auto grid = gridOf(golden.grid);
        const auto shape = shapeOf(golden);
        const auto why = attention_dense::gridIllegality(grid, shape);
        ASSERT_EQ(!why.has_value(), golden.legal)
            << describe(golden) << ": " << why.value_or("legal here, refused by the Python");
        if(!golden.legal)
        {
            ++illegal;
            continue;
        }
        ++legal;
        const auto dims = attention_dense::gridDims(grid, shape);
        EXPECT_EQ(dims[0], golden.gridX) << describe(golden);
        EXPECT_EQ(dims[1], golden.gridY) << describe(golden);
        EXPECT_EQ(dims[2], golden.gridZ) << describe(golden);
        const auto pairs = attention_dense::fastDivKernargs(grid, shape);
        ASSERT_EQ(static_cast<int>(pairs.size() * 2), golden.pairValues) << describe(golden);
        for(std::size_t index = 0; index < pairs.size(); ++index)
        {
            EXPECT_EQ(pairs[index].magic, golden.pairs[2 * index]) << describe(golden);
            EXPECT_EQ(pairs[index].shift, golden.pairs[2 * index + 1]) << describe(golden);
        }
    }
    // Both verdicts occur, so neither direction of the comparison is vacuous.
    EXPECT_GT(legal, 0U);
    EXPECT_GT(illegal, 0U);
}

TEST(TestAttentionDensePersistent, ParsersRefuseEveryDecodeWithoutRuntimeKernargs)
{
    for(const char* refused : {"auto", "gqa_pair", "gqa_pair_2phase", "", "QB_MAJOR"})
    {
        EXPECT_FALSE(attention_dense::parsePersistDecode(refused).has_value()) << refused;
    }
    for(const char* refused : {"auto", "persistent", "", "qb_major"})
    {
        EXPECT_FALSE(attention_dense::parseNonpersistDecode(refused).has_value()) << refused;
    }
    EXPECT_EQ(attention_dense::parseGridOrder("persistent"), AttentionDenseGridOrder::PERSISTENT);
    EXPECT_FALSE(attention_dense::parseGridOrder("auto").has_value());
}

TEST(TestAttentionDensePersistent, APersistentGridWithoutCtasIsRefused)
{
    AttentionDenseGrid grid;
    grid.order = AttentionDenseGridOrder::PERSISTENT;
    grid.decode = AttentionDensePersistDecode::BT_HKV_MINOR;
    AttentionDenseGridShape shape;
    shape.batch = 1;
    shape.seqLenQ = 512;
    shape.seqLenKv = 512;
    shape.numQueryHeads = 8;
    shape.numKvHeads = 8;
    shape.blockM = 256;
    for(const int64_t numPersistent : {int64_t{0}, int64_t{-304}})
    {
        grid.numPersistent = numPersistent;
        EXPECT_TRUE(attention_dense::gridIllegality(grid, shape).has_value()) << numPersistent;
    }
    grid.numPersistent = GOLDEN_NUM_PERSISTENT;
    EXPECT_FALSE(attention_dense::gridIllegality(grid, shape).has_value());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
