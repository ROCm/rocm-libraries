// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "CaseGraph.hpp"

using sdpa_case_flow::BuiltCase;
using sdpa_case_flow::CaseContext;
using sdpa_case_flow::CaseRealisationError;
using sdpa_case_flow::ClassValues;
using sdpa_case_flow::Contents;
using sdpa_case_flow::DescaleForm;
using sdpa_case_flow::Row;
using sdpa_case_flow::SizeOverride;

namespace
{

constexpr const char* CLASS_VALUES_JSON = R"json({
  "schema_version": 3,
  "batch_size": {"one": 1, "many": 2},
  "q_heads": {"one": 1, "many": 8},
  "seq_lengths": [{"seq_q": "many", "seq_kv": "many", "seq_q_vs_seq_kv": "equal",
                   "S_q": 64, "S_kv": 64}],
  "head_dim": {"d64": 64, "d128": 128},
  "dtype_differs": {"by_dtype_q": {"bf16": "fp8_e4m3"},
                    "by_arch": {"gfx942": {"fp8_e4m3": "fp8_e4m3_fnuz"}}},
  "dim_order": {"bhsd": ["B", "H", "S", "D"], "bshd": ["B", "S", "H", "D"],
                "bsdh": ["B", "S", "D", "H"]},
  "scale_value": {"inverse_sqrt_d": ["1/sqrt(head_dim_qk)"]},
  "dropout_probability": {"zero": 0.0},
  "dropout_seed_offset": {"seed": 1234, "offset": 0},
  "dropout_mask_scale": {"mixed": 2.0},
  "dropout_mask_cycle": {"mixed": [1.0, 0.0]},
  "page_size": {"partial": 16, "whole_sequence": "S_kv"},
  "quantisation_factors": {"descale_q": 0.5, "descale_k": 0.25, "descale_v": 2.0, "other": 1.0},
  "descale_lists": {
    "per_head": {"descale_q": [0.5, 1.0], "descale_k": [0.25, 0.5], "descale_v": [2.0, 4.0]},
    "finer": {"descale_q": [0.5, 1.0, 2.0], "descale_k": [0.25, 0.125], "descale_v": [4.0, 2.0]}},
  "ragged_tensors": {"padded": [], "ragged": ["Q", "K", "V", "O"], "ragged_q_side": ["Q", "O"],
                     "ragged_kv_side": ["K", "V"]},
  "ragged_stats": {"dense": [], "packed_with_q": ["Stats"]},
  "ragged_offsets": {"contiguous": {"offsets": "prefix_sum_of_lengths"},
                     "gapped": {"offsets": "full_slots"},
                     "leading_gap": {"offsets": "prefix_sum_after_gap"}},
  "per_seq_len": {"all_full": ["S"], "all_equal_partial": [{"of": "S", "divide": 2}],
                  "mixed": ["S", {"of": "S", "divide": 2}], "some_zero": ["S", 0],
                  "all_zero": [0], "exceeds_storage": [{"of": "S", "add": 1}, "S"]},
  "paged_sides": {"present": ["K", "V"], "k_only": ["K"], "v_only": ["V"]},
  "v_page_multiplier": {"equal": 1, "differ": 2},
  "page_tables": {"separate": 2, "shared": 1},
  "packed_page_table": {"dense": false, "packed": true},
  "page_table_order": {"in_order": {"slot_of_page": "identity"},
                       "shuffled": {"slot_of_page": "seeded_permutation", "seed": 2893},
                       "shared_pages": {"slot_of_page": "identity",
                                        "shares": {"sequence": 1, "reuses_sequence": 0, "page": 0}}},
  "alibi_head_count": {"H_q": {"8": 6}, "heads": {"6": {"mha": 6, "gqa": 2, "mqa": 1,
                       "k_and_v_differ": {"H_k": 2, "H_v": 3}}}},
  "mma_fp8": {"otherwise": "fp8_e4m3", "by_arch": {"gfx942": {"fp8_e4m3": "fp8_e4m3_fnuz"}}}
})json";

// Element strides over the sequence dimension at B 4, H 8, D 128 (Stats: D 1).
constexpr int64_t BSHD_SEQUENCE_STRIDE = 8 * 128;
constexpr int64_t BHSD_SEQUENCE_STRIDE = 128;
constexpr int64_t BSHD_STATS_SEQUENCE_STRIDE = 8;

const ClassValues& classValues()
{
    static const ClassValues VALUES = sdpa_case_flow::parseClassValues(CLASS_VALUES_JSON);
    return VALUES;
}

SizeOverride sizes(int64_t b, int64_t sq, int64_t skv)
{
    SizeOverride z;
    z.b = b;
    z.hq = 8;
    z.hk = 8;
    z.hv = 8;
    z.sq = sq;
    z.skv = skv;
    z.dqk = 128;
    z.dv = 128;
    return z;
}

Row baseRow()
{
    Row row;
    for(const auto& axis : sdpa_case_flow::modelAxisIds())
    {
        row[axis] = sdpa_case_flow::sharedBaseValue(axis);
    }
    return row;
}

Row variableLengthRow(const std::string& qClass, const std::string& kvClass)
{
    auto row = baseRow();
    row["sequence_lengths"] = "variable";
    row["per_seq_len_q"] = qClass;
    row["per_seq_len_kv"] = kvClass;
    return row;
}

// Q, O and Stats token-major (bshd); K and V bhsd, so the two sides have different strides.
Row raggedRow(const std::string& packing,
              const std::string& offsets,
              const std::string& qClass,
              const std::string& kvClass)
{
    auto row = variableLengthRow(qClass, kvClass);
    row["batch_packing"] = packing;
    row["ragged_offsets"] = offsets;
    row["dim_order_q"] = "bshd";
    row["dim_order_o"] = "bshd";
    return row;
}

Row withStats(Row row, const std::string& packing)
{
    row["stats_output"] = "requested";
    row["dtype_stats"] = "fp32";
    row["dim_order_stats"] = "bshd";
    row["strides_stats"] = "packed";
    row["alignment_stats"] = "wide_aligned";
    row["stats_packing"] = packing;
    return row;
}

Row pagedRow(const std::string& storage)
{
    auto row = baseRow();
    row["paged_kv"] = "present";
    row["page_table_sharing"] = "separate";
    row["page_size_k_vs_v"] = "equal";
    row["page_table_storage"] = storage;
    row["page_size"] = "partial";
    row["page_table_order"] = "in_order";
    row["page_count"] = "exact";
    row["max_seq_len_kv"] = "equal";
    return row;
}

Row pagedRow(const std::string& storage, const std::string& order, const std::string& sharing)
{
    auto row = pagedRow(storage);
    row["page_table_order"] = order;
    row["page_table_sharing"] = sharing;
    return row;
}

BuiltCase build(const Row& row, const SizeOverride& z, const ClassValues& values)
{
    const CaseContext context{values, DescaleForm::BY_VALUE, "gfx942", z};
    return sdpa_case_flow::buildCaseGraph(row, {}, context);
}

BuiltCase build(const Row& row, const SizeOverride& z)
{
    return build(row, z, classValues());
}

const Contents& contentsOf(const BuiltCase& built, const std::string& name)
{
    const auto it = built.contents.find(name);
    if(it == built.contents.end())
    {
        throw std::runtime_error("no contents for '" + name + "'");
    }
    return it->second;
}

std::vector<int64_t> integersOf(const BuiltCase& built, const std::string& name)
{
    return std::get<std::vector<int64_t>>(contentsOf(built, name).values);
}

std::vector<std::string> tablesOf(const std::string& sharing)
{
    return sharing == "shared" ? std::vector<std::string>{"page_table_kv"}
                               : std::vector<std::string>{"page_table_k", "page_table_v"};
}

// Slot j of sequence b holds page b x owned + (j mod owned).
std::vector<int64_t> inOrderTable(int64_t sequences, int64_t width, int64_t owned)
{
    std::vector<int64_t> table;
    for(int64_t b = 0; b < sequences; ++b)
    {
        for(int64_t j = 0; j < width; ++j)
        {
            table.push_back(b * owned + j % owned);
        }
    }
    return table;
}

std::vector<int64_t> rowOf(const std::vector<int64_t>& table, int64_t width, int64_t b)
{
    const auto first = table.begin() + b * width;
    return {first, first + width};
}

std::vector<int64_t> stridesOf(const BuiltCase& built, const std::string& name)
{
    return built.graph->getTensorsByName().at(name)->get_stride();
}

// V is a view into K, or K a view into Q; the view's layout values are `inherited`.
Row viewRow(const std::string& view)
{
    auto row = baseRow();
    row["tensor_overlap"] = "inputs_overlap";
    row["strides_" + view] = "inherited";
    row["dim_order_" + view] = "inherited";
    return row;
}

std::vector<int64_t> scaled(std::vector<int64_t> tokens, int64_t stride)
{
    for(auto& token : tokens)
    {
        token *= stride;
    }
    return tokens;
}

struct LengthClass
{
    std::string id;
    // Lengths at stored length 64.
    std::vector<int64_t> cycle;
};

const std::vector<LengthClass>& lengthClasses()
{
    static const std::vector<LengthClass> CLASSES = {
        {"all_full", {64}},
        {"all_equal_partial", {32}},
        {"mixed", {64, 32}},
        {"some_zero", {64, 0}},
        {"all_zero", {0}},
        {"exceeds_storage", {65, 64}},
    };
    return CLASSES;
}

struct PackedClass
{
    std::string id;
    // Token boundaries of four sequences of stored length 64, packed end to end.
    std::vector<int64_t> prefixSums;
};

const std::vector<PackedClass>& packedClasses()
{
    static const std::vector<PackedClass> CLASSES = {
        {"all_full", {0, 64, 128, 192, 256}},
        {"all_equal_partial", {0, 32, 64, 96, 128}},
        {"mixed", {0, 64, 96, 160, 192}},
        {"some_zero", {0, 64, 64, 128, 128}},
        {"all_zero", {0, 0, 0, 0, 0}},
    };
    return CLASSES;
}

} // namespace

TEST(TestCaseGraph, SequenceLengthsHoldEachClassRecipe)
{
    for(const auto& lengthClass : lengthClasses())
    {
        SCOPED_TRACE(lengthClass.id);
        const auto built
            = build(variableLengthRow(lengthClass.id, lengthClass.id), sizes(4, 64, 64));
        for(const char* name : {"seq_len_q", "seq_len_kv"})
        {
            EXPECT_EQ(integersOf(built, name), lengthClass.cycle) << name;
            EXPECT_TRUE(contentsOf(built, name).repeat) << name;
        }
    }
}

TEST(TestCaseGraph, SizeOverrideSetsTheStoredLength)
{
    const auto built = build(variableLengthRow("mixed", "all_equal_partial"), sizes(4, 128, 96));
    EXPECT_EQ(integersOf(built, "seq_len_q"), (std::vector<int64_t>{128, 64}));
    EXPECT_EQ(integersOf(built, "seq_len_kv"), (std::vector<int64_t>{48}));
}

TEST(TestCaseGraph, LengthCycleLongerThanTheBatchIsCut)
{
    const auto built = build(variableLengthRow("mixed", "some_zero"), sizes(1, 64, 64));
    EXPECT_EQ(integersOf(built, "seq_len_q"), (std::vector<int64_t>{64}));
    EXPECT_EQ(integersOf(built, "seq_len_kv"), (std::vector<int64_t>{64}));
}

TEST(TestCaseGraph, QSideAndKeyValueSideLengthsDiffer)
{
    const auto built = build(variableLengthRow("mixed", "all_full"), sizes(4, 64, 64));
    EXPECT_EQ(integersOf(built, "seq_len_q"), (std::vector<int64_t>{64, 32}));
    EXPECT_EQ(integersOf(built, "seq_len_kv"), (std::vector<int64_t>{64}));

    const auto longerKeys = build(variableLengthRow("mixed", "mixed"), sizes(4, 64, 128));
    EXPECT_EQ(integersOf(longerKeys, "seq_len_q"), (std::vector<int64_t>{64, 32}));
    EXPECT_EQ(integersOf(longerKeys, "seq_len_kv"), (std::vector<int64_t>{128, 64}));
}

TEST(TestCaseGraph, UniformRowHasNoLengthContents)
{
    const auto built = build(baseRow(), sizes(4, 64, 64));
    EXPECT_EQ(built.contents.count("seq_len_q"), 0u);
    EXPECT_EQ(built.contents.count("seq_len_kv"), 0u);
}

TEST(TestCaseGraph, PagedUniformRowGetsFullLengths)
{
    const auto built = build(pagedRow("dense"), sizes(4, 64, 64));
    EXPECT_EQ(integersOf(built, "seq_len_q"), (std::vector<int64_t>{64}));
    EXPECT_EQ(integersOf(built, "seq_len_kv"), (std::vector<int64_t>{64}));
}

TEST(TestCaseGraph, InOrderPageTablesAddressEveryPoolPage)
{
    // Four sequences of 64 keys in pages of 16: a pool of 16 pages, four per sequence.
    const auto built = build(pagedRow("dense"), sizes(4, 64, 64));
    std::vector<int64_t> expected(16);
    for(size_t i = 0; i < expected.size(); ++i)
    {
        expected[i] = static_cast<int64_t>(i);
    }
    for(const char* table : {"page_table_k", "page_table_v"})
    {
        EXPECT_EQ(integersOf(built, table), expected) << table;
        EXPECT_FALSE(contentsOf(built, table).repeat) << table;
    }
    EXPECT_EQ(built.contents.count("page_table_k_offset"), 0u);
}

TEST(TestCaseGraph, PackedPageTableOffsetsCountTableEntries)
{
    const auto built = build(pagedRow("packed"), sizes(4, 64, 64));
    for(const char* offset : {"page_table_k_offset", "page_table_v_offset"})
    {
        EXPECT_EQ(integersOf(built, offset), (std::vector<int64_t>{0, 4, 8, 12, 16})) << offset;
        EXPECT_FALSE(contentsOf(built, offset).repeat) << offset;
    }
}

TEST(TestCaseGraph, ContiguousOffsetsArePrefixSumsOfEachClass)
{
    for(const auto& packedClass : packedClasses())
    {
        SCOPED_TRACE(packedClass.id);
        const auto built = build(raggedRow("ragged", "contiguous", packedClass.id, "all_full"),
                                 sizes(4, 64, 64));
        const auto expected = scaled(packedClass.prefixSums, BSHD_SEQUENCE_STRIDE);
        EXPECT_EQ(integersOf(built, "ragged_offset_q"), expected);
        EXPECT_EQ(integersOf(built, "ragged_offset_o"), expected);
        EXPECT_FALSE(contentsOf(built, "ragged_offset_q").repeat);
    }
}

TEST(TestCaseGraph, FullLengthOffsetsAreWholeSlots)
{
    // offset b = b x dims[2] x strides[2] on every ragged tensor.
    const auto built
        = build(raggedRow("ragged", "contiguous", "all_full", "all_full"), sizes(4, 64, 64));
    EXPECT_EQ(integersOf(built, "ragged_offset_q"),
              scaled({0, 64, 128, 192, 256}, BSHD_SEQUENCE_STRIDE));
    for(const char* offset : {"ragged_offset_k", "ragged_offset_v"})
    {
        EXPECT_EQ(integersOf(built, offset), scaled({0, 64, 128, 192, 256}, BHSD_SEQUENCE_STRIDE))
            << offset;
    }
}

TEST(TestCaseGraph, KeyValueOffsetsUseTheirOwnLengthsAndStrides)
{
    const auto built
        = build(raggedRow("ragged", "contiguous", "mixed", "some_zero"), sizes(4, 64, 64));
    EXPECT_EQ(integersOf(built, "ragged_offset_q"),
              scaled({0, 64, 96, 160, 192}, BSHD_SEQUENCE_STRIDE));
    EXPECT_EQ(integersOf(built, "ragged_offset_k"),
              scaled({0, 64, 64, 128, 128}, BHSD_SEQUENCE_STRIDE));
    EXPECT_EQ(integersOf(built, "ragged_offset_v"),
              scaled({0, 64, 64, 128, 128}, BHSD_SEQUENCE_STRIDE));
}

TEST(TestCaseGraph, OneSideRaggedRowsOffsetOnlyTheirSide)
{
    const auto qSide
        = build(raggedRow("ragged_q_side", "contiguous", "mixed", "all_full"), sizes(4, 64, 64));
    EXPECT_EQ(integersOf(qSide, "ragged_offset_q"),
              scaled({0, 64, 96, 160, 192}, BSHD_SEQUENCE_STRIDE));
    EXPECT_EQ(qSide.contents.count("ragged_offset_k"), 0u);

    const auto kvSide
        = build(raggedRow("ragged_kv_side", "contiguous", "all_full", "mixed"), sizes(4, 64, 64));
    EXPECT_EQ(integersOf(kvSide, "ragged_offset_k"),
              scaled({0, 64, 96, 160, 192}, BHSD_SEQUENCE_STRIDE));
    EXPECT_EQ(kvSide.contents.count("ragged_offset_q"), 0u);
}

TEST(TestCaseGraph, GappedOffsetsIgnoreTheLiveLengths)
{
    const auto built = build(raggedRow("ragged", "gapped", "mixed", "some_zero"), sizes(4, 64, 64));
    EXPECT_EQ(integersOf(built, "ragged_offset_q"),
              scaled({0, 64, 128, 192, 256}, BSHD_SEQUENCE_STRIDE));
    EXPECT_EQ(integersOf(built, "ragged_offset_k"),
              scaled({0, 64, 128, 192, 256}, BHSD_SEQUENCE_STRIDE));
    EXPECT_EQ(integersOf(built, "seq_len_q"), (std::vector<int64_t>{64, 32}));
}

TEST(TestCaseGraph, LeadingGapPutsAllSpareRoomFirst)
{
    // Slack = 4 x 64 - (64 + 32 + 64 + 32) = 64 tokens; the K/V side has no slack.
    const auto built
        = build(raggedRow("ragged", "leading_gap", "mixed", "all_full"), sizes(4, 64, 64));
    EXPECT_EQ(integersOf(built, "ragged_offset_q"),
              scaled({64, 128, 160, 224, 256}, BSHD_SEQUENCE_STRIDE));
    EXPECT_EQ(integersOf(built, "ragged_offset_o"),
              scaled({64, 128, 160, 224, 256}, BSHD_SEQUENCE_STRIDE));
    EXPECT_EQ(integersOf(built, "ragged_offset_k"),
              scaled({0, 64, 128, 192, 256}, BHSD_SEQUENCE_STRIDE));
}

TEST(TestCaseGraph, RaggedLengthLongerThanTheSlotThrows)
{
    for(const char* offsets : {"contiguous", "gapped", "leading_gap"})
    {
        SCOPED_TRACE(offsets);
        EXPECT_THROW(build(raggedRow("ragged_q_side", offsets, "exceeds_storage", "all_full"),
                           sizes(4, 64, 64)),
                     CaseRealisationError);
    }
}

TEST(TestCaseGraph, OffsetsFollowEachTensorsOwnSequenceExtent)
{
    // V is paged: its pool holds pages of 16 tokens. K is dense beside it and spans the declared
    // maximum of 128 keys. The length tensor keeps the key length of 64.
    auto row = raggedRow("ragged_kv_side", "contiguous", "all_full", "all_full");
    for(const auto& [axis, value] : pagedRow("dense"))
    {
        if(axis.rfind("page", 0) == 0 || axis == "max_seq_len_kv")
        {
            row[axis] = value;
        }
    }
    row["paged_kv"] = "v_only";
    row["max_seq_len_kv"] = "larger";
    const auto built = build(row, sizes(4, 64, 64));
    EXPECT_EQ(integersOf(built, "ragged_offset_k"),
              scaled({0, 128, 256, 384, 512}, BHSD_SEQUENCE_STRIDE));
    EXPECT_EQ(integersOf(built, "ragged_offset_v"),
              scaled({0, 16, 32, 48, 64}, BHSD_SEQUENCE_STRIDE));
    EXPECT_EQ(integersOf(built, "seq_len_kv"), (std::vector<int64_t>{64}));
}

TEST(TestCaseGraph, LengthsInOffsetsRowCarriesLengthsOnlyInOffsets)
{
    auto row = raggedRow("ragged", "contiguous", "mixed", "all_equal_partial");
    row["sequence_lengths"] = "lengths_in_offsets";
    row["dim_order_k"] = "bshd";
    row["dim_order_v"] = "bshd";
    const auto built = build(row, sizes(4, 64, 64));
    EXPECT_EQ(integersOf(built, "ragged_offset_q"),
              scaled({0, 64, 96, 160, 192}, BSHD_SEQUENCE_STRIDE));
    EXPECT_EQ(integersOf(built, "ragged_offset_k"),
              scaled({0, 32, 64, 96, 128}, BSHD_SEQUENCE_STRIDE));
    EXPECT_EQ(built.contents.count("seq_len_q"), 0u);
    EXPECT_EQ(built.contents.count("seq_len_kv"), 0u);
}

TEST(TestCaseGraph, LengthsInOffsetsNeedsARaggedTensor)
{
    auto row = variableLengthRow("mixed", "all_full");
    row["sequence_lengths"] = "lengths_in_offsets";
    EXPECT_THROW(build(row, sizes(4, 64, 64)), CaseRealisationError);
}

TEST(TestCaseGraph, PackedStatsUsesQBoundariesInStatsElements)
{
    const auto contiguous
        = build(withStats(raggedRow("ragged", "contiguous", "mixed", "all_full"), "packed_with_q"),
                sizes(4, 64, 64));
    EXPECT_EQ(integersOf(contiguous, "ragged_offset_stats"),
              scaled({0, 64, 96, 160, 192}, BSHD_STATS_SEQUENCE_STRIDE));
    EXPECT_FALSE(contentsOf(contiguous, "ragged_offset_stats").repeat);

    const auto leadingGap = build(
        withStats(raggedRow("ragged_q_side", "leading_gap", "mixed", "all_full"), "packed_with_q"),
        sizes(4, 64, 64));
    EXPECT_EQ(integersOf(leadingGap, "ragged_offset_stats"),
              scaled({64, 128, 160, 224, 256}, BSHD_STATS_SEQUENCE_STRIDE));
}

TEST(TestCaseGraph, DenseStatsHasNoOffset)
{
    const auto built
        = build(withStats(raggedRow("ragged", "contiguous", "mixed", "all_full"), "dense"),
                sizes(4, 64, 64));
    EXPECT_EQ(built.contents.count("ragged_offset_stats"), 0u);
}

TEST(TestCaseGraph, PackedStatsNeedsARaggedQ)
{
    const auto row = withStats(raggedRow("ragged_kv_side", "contiguous", "all_full", "mixed"),
                               "packed_with_q");
    EXPECT_THROW(build(row, sizes(4, 64, 64)), CaseRealisationError);
}

TEST(TestCaseGraph, OffsetAboveInt32Throws)
{
    // The last Q offset is 4 x 2^20 tokens x 1024 elements = 2^32.
    auto row = raggedRow("ragged_q_side", "contiguous", "all_full", "all_full");
    EXPECT_THROW(build(row, sizes(4, int64_t{1} << 20, 64)), CaseRealisationError);
}

TEST(TestCaseGraph, RepeatedListIsCutToItsTensor)
{
    auto row = baseRow();
    row["descale_k"] = "per_head";
    auto z = sizes(1, 64, 64);
    z.hq = 1;
    z.hk = 1;
    z.hv = 1;

    const auto oneElement = contentsOf(build(row, z), "descale_k");
    EXPECT_EQ(std::get<std::vector<double>>(oneElement.values), (std::vector<double>{0.25}));
    EXPECT_TRUE(oneElement.repeat);

    z.b = 2;
    const auto twoElements = contentsOf(build(row, z), "descale_k");
    EXPECT_EQ(std::get<std::vector<double>>(twoElements.values), (std::vector<double>{0.25, 0.5}));
}

// A per-head descale is [B, H_kv, 1, 1] (descale_q too); a finer one is [B, H, S, 1] of its own
// tensor.
TEST(TestCaseGraph, DescaleListsRepeatTheirFactorCycle)
{
    const std::map<std::string, std::map<std::string, std::vector<double>>> cycles = {
        {"per_head",
         {{"descale_q", {0.5, 1.0}}, {"descale_k", {0.25, 0.5}}, {"descale_v", {2.0, 4.0}}}},
        {"finer",
         {{"descale_q", {0.5, 1.0, 2.0}}, {"descale_k", {0.25, 0.125}}, {"descale_v", {4.0, 2.0}}}},
    };
    for(const int64_t heads : {2, 4})
    {
        auto z = sizes(2, 64, 32);
        z.hq = 2 * heads;
        z.hk = heads;
        z.hv = heads;
        for(const auto& [granularity, lists] : cycles)
        {
            auto row = baseRow();
            for(const auto& entry : lists)
            {
                row[entry.first] = granularity;
            }
            const auto built = build(row, z);
            for(const auto& [name, cycle] : lists)
            {
                SCOPED_TRACE(name + " " + granularity + " H_kv " + std::to_string(heads));
                const auto& contents = contentsOf(built, name);
                EXPECT_EQ(std::get<std::vector<double>>(contents.values), cycle);
                EXPECT_TRUE(contents.repeat);
                const bool perHead = granularity == "per_head";
                const bool query = name == "descale_q";
                const std::vector<int64_t> dims
                    = perHead
                          ? std::vector<int64_t>{2, heads, 1, 1}
                          : std::vector<int64_t>{2, query ? 2 * heads : heads, query ? 64 : 32, 1};
                EXPECT_EQ(built.graph->getTensorsByName().at(name)->get_dim(), dims);
            }
        }
    }
}

TEST(TestCaseGraph, CallerDropoutMaskRepeatsItsPatternCycle)
{
    auto row = baseRow();
    row["dropout"] = "caller_mask";
    row["dropout_mask_pattern"] = "mixed";
    row["dropout_pattern_output"] = "not_requested";
    for(const auto& [broadcast, dims] : std::vector<std::pair<std::string, std::vector<int64_t>>>{
            {"none", {2, 8, 64, 32}}, {"batch_and_heads", {1, 1, 64, 32}}})
    {
        SCOPED_TRACE(broadcast);
        row["dropout_mask_broadcast"] = broadcast;
        const auto built = build(row, sizes(2, 64, 32));
        const auto& contents = contentsOf(built, "dropout_mask");
        EXPECT_EQ(std::get<std::vector<double>>(contents.values), (std::vector<double>{1.0, 0.0}));
        EXPECT_TRUE(contents.repeat);
        EXPECT_EQ(built.graph->getTensorsByName().at("dropout_mask")->get_dim(), dims);
    }
}

// The mixed pattern allows key tile 0 and every tile whose row and key tile numbers sum to an
// even number.
TEST(TestCaseGraph, BlockMaskIsOneSliceRepeatedOverBatchAndHeads)
{
    struct Size
    {
        int64_t sq;
        int64_t skv;
        std::vector<int64_t> slice;
        std::vector<int64_t> dims;
    };
    const std::vector<Size> cases = {
        {128, 128, {1}, {2, 8, 1, 1}},
        {129, 129, {1, 3}, {2, 8, 2, 1}},
        {128, 1025, {85, 1}, {2, 8, 1, 2}},
    };
    auto row = baseRow();
    row["block_mask"] = "present";
    row["block_mask_pattern"] = "mixed";
    for(const auto& c : cases)
    {
        SCOPED_TRACE("Sq " + std::to_string(c.sq) + " Skv " + std::to_string(c.skv));
        const auto built = build(row, sizes(2, c.sq, c.skv));
        const auto& contents = contentsOf(built, "block_mask");
        EXPECT_EQ(integersOf(built, "block_mask"), c.slice);
        EXPECT_TRUE(contents.repeat);
        EXPECT_EQ(built.graph->getTensorsByName().at("block_mask")->get_dim(), c.dims);
    }
}

TEST(TestCaseGraph, BlockMaskOfOneBatchAndHeadIsTheFullList)
{
    auto row = baseRow();
    row["block_mask"] = "present";
    row["block_mask_pattern"] = "some_rows_blocked";
    auto z = sizes(1, 129, 129);
    z.hq = 1;
    z.hk = 1;
    z.hv = 1;
    const auto built = build(row, z);
    EXPECT_EQ(integersOf(built, "block_mask"), (std::vector<int64_t>{3, 0}));
    EXPECT_FALSE(contentsOf(built, "block_mask").repeat);
}

TEST(TestCaseGraph, BlockMaskDimsOverrideFillsSlicesThenOnes)
{
    auto row = baseRow();
    row["block_mask"] = "present";
    row["block_mask_pattern"] = "all_blocked";
    const CaseContext context{classValues(), DescaleForm::BY_VALUE, "gfx942", sizes(1, 128, 128)};
    const auto built = sdpa_case_flow::buildCaseGraph(row, {"block_mask_dims=1x8x1x2"}, context);
    const auto& contents = contentsOf(built, "block_mask");
    EXPECT_EQ(integersOf(built, "block_mask"),
              (std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1}));
    EXPECT_FALSE(contents.repeat);
}

TEST(TestCaseGraph, MaskPatternAxesAreRealised)
{
    EXPECT_TRUE(sdpa_case_flow::isRealisedContentAxis("block_mask_pattern"));
    EXPECT_TRUE(sdpa_case_flow::isRealisedContentAxis("dropout_mask_pattern"));
}

TEST(TestCaseGraph, LengthAndOffsetAxesAreRealised)
{
    for(const char* axis : {"per_seq_len_q", "per_seq_len_kv", "ragged_offsets", "stats_packing"})
    {
        EXPECT_TRUE(sdpa_case_flow::isRealisedContentAxis(axis)) << axis;
    }
    EXPECT_FALSE(sdpa_case_flow::isRealisedContentAxis("tensor_overlap"));
}

TEST(TestCaseGraph, ClassValuesRejectAnUnknownOffsetLayout)
{
    std::string text = CLASS_VALUES_JSON;
    const std::string layout = "\"full_slots\"";
    text.replace(text.find(layout), layout.size(), "\"full_slot\"");
    EXPECT_THROW(sdpa_case_flow::parseClassValues(text), sdpa_case_flow::ClassValuesError);
}

TEST(TestCaseGraph, ClassValuesRejectAnUnknownLengthTerm)
{
    std::string text = CLASS_VALUES_JSON;
    const std::string term = "\"divide\": 2";
    text.replace(text.find(term), term.size(), "\"halve\": 2");
    EXPECT_THROW(sdpa_case_flow::parseClassValues(text), sdpa_case_flow::ClassValuesError);
}

TEST(TestCaseGraph, ViewOfKReadsThroughKStrides)
{
    auto z = sizes(2, 32, 64);
    z.dqk = 192;
    const auto built = build(viewRow("v"), z);
    // K is packed bhsd with D 192, so a V row of 128 elements starts 192 elements after the last.
    EXPECT_EQ(stridesOf(built, "v"), (std::vector<int64_t>{8 * 64 * 192, 64 * 192, 192, 1}));
    EXPECT_EQ(stridesOf(built, "v"), stridesOf(built, "k"));
    EXPECT_NE(built.realised.find("v_view_of=k"), std::string::npos) << built.realised;
    EXPECT_EQ(built.realised.find("view_exceeds_owner"), std::string::npos) << built.realised;
}

TEST(TestCaseGraph, ViewOfQReadsThroughQStrides)
{
    auto row = viewRow("k");
    row["dim_order_q"] = "bshd";
    const auto built = build(row, sizes(2, 64, 32));
    // Q is packed bshd with S_q 64; K, with S_kv 32, takes Q's strides.
    EXPECT_EQ(stridesOf(built, "k"), (std::vector<int64_t>{64 * 8 * 128, 128, 8 * 128, 1}));
    EXPECT_EQ(stridesOf(built, "k"), stridesOf(built, "q"));
    EXPECT_NE(built.realised.find("k_view_of=q"), std::string::npos) << built.realised;
}

TEST(TestCaseGraph, ViewLargerThanItsOwnerIsMarked)
{
    auto row = viewRow("k");
    row["dim_order_q"] = "bshd";
    const auto built = build(row, sizes(2, 32, 64));
    EXPECT_EQ(stridesOf(built, "k"), stridesOf(built, "q"));
    EXPECT_NE(built.realised.find("k_view_exceeds_owner"), std::string::npos) << built.realised;
}

TEST(TestCaseGraph, OnlyKAndVCanBeViews)
{
    auto row = viewRow("v");
    row["strides_v"] = "packed";
    row["dim_order_v"] = "bhsd";
    row["strides_q"] = "inherited";
    row["dim_order_q"] = "inherited";
    EXPECT_THROW(build(row, sizes(2, 64, 64)), CaseRealisationError);
}

TEST(TestCaseGraph, ViewOwnerIsNotAView)
{
    auto row = viewRow("v");
    row["strides_k"] = "inherited";
    row["dim_order_k"] = "inherited";
    EXPECT_THROW(build(row, sizes(2, 64, 64)), CaseRealisationError);
}

TEST(TestCaseGraph, ViewNeedsInputsOverlap)
{
    auto row = viewRow("v");
    row["tensor_overlap"] = "separate";
    EXPECT_THROW(build(row, sizes(2, 64, 64)), CaseRealisationError);
}

TEST(TestCaseGraph, ViewInheritsOrderAndStridesTogether)
{
    auto row = viewRow("v");
    row["dim_order_v"] = "bhsd";
    EXPECT_THROW(build(row, sizes(2, 64, 64)), CaseRealisationError);
}

TEST(TestCaseGraph, HeadDimStrideDoublesTheHeadDimAndOuterStrides)
{
    auto row = baseRow();
    row["strides_q"] = "head_dim_strided";
    const auto bhsd = build(row, sizes(2, 64, 64));
    EXPECT_EQ(stridesOf(bhsd, "q"), (std::vector<int64_t>{2 * 8 * 64 * 128, 2 * 64 * 128, 256, 2}));

    // With the heads innermost, the heads keep stride 1 and D, S and B double.
    row["dim_order_q"] = "bsdh";
    const auto bsdh = build(row, sizes(2, 64, 64));
    EXPECT_EQ(stridesOf(bsdh, "q"),
              (std::vector<int64_t>{2 * 64 * 128 * 8, 1, 2 * 128 * 8, 2 * 8}));
}

TEST(TestCaseGraph, InOrderTablesHoldEachSequencesOwnPages)
{
    // Four sequences of 64 keys in pages of 16: each sequence owns four pages of the pool.
    for(const char* sharing : {"separate", "shared"})
    {
        SCOPED_TRACE(sharing);
        const auto built = build(pagedRow("dense", "in_order", sharing), sizes(4, 64, 64));
        for(const auto& table : tablesOf(sharing))
        {
            EXPECT_EQ(integersOf(built, table), inOrderTable(4, 4, 4)) << table;
            EXPECT_FALSE(contentsOf(built, table).repeat) << table;
        }
    }
}

TEST(TestCaseGraph, SurplusSlotsRepeatTheSequencesOwnPages)
{
    auto row = pagedRow("dense", "in_order", "separate");
    row["page_count"] = "surplus";
    const auto built = build(row, sizes(4, 64, 64));
    EXPECT_EQ(
        integersOf(built, "page_table_k"),
        (std::vector<int64_t>{0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 8, 9, 10, 11, 8, 12, 13, 14, 15, 12}));
}

TEST(TestCaseGraph, ShuffledTablesPermuteEachSequencesOwnPages)
{
    // Four sequences of 256 keys in pages of 16: sixteen pages per sequence.
    for(const char* sharing : {"separate", "shared"})
    {
        SCOPED_TRACE(sharing);
        const auto built = build(pagedRow("dense", "shuffled", sharing), sizes(4, 64, 256));
        const auto inOrder = inOrderTable(4, 16, 16);
        for(const auto& table : tablesOf(sharing))
        {
            const auto values = integersOf(built, table);
            ASSERT_EQ(values.size(), 64u) << table;
            EXPECT_FALSE(contentsOf(built, table).repeat) << table;
            for(int64_t b = 0; b < 4; ++b)
            {
                auto row = rowOf(values, 16, b);
                EXPECT_NE(row, rowOf(inOrder, 16, b)) << table << " sequence " << b;
                std::sort(row.begin(), row.end());
                EXPECT_EQ(row, rowOf(inOrder, 16, b)) << table << " sequence " << b;
            }
        }
    }
}

TEST(TestCaseGraph, ShuffledRowOfTwoPagesNeverKeepsItsOrder)
{
    // Sixteen sequences of 32 keys in pages of 16: every row is a swap of its two pages.
    const auto values = integersOf(
        build(pagedRow("dense", "shuffled", "separate"), sizes(16, 64, 32)), "page_table_k");
    ASSERT_EQ(values.size(), 32u);
    for(int64_t b = 0; b < 16; ++b)
    {
        EXPECT_EQ(rowOf(values, 2, b), (std::vector<int64_t>{2 * b + 1, 2 * b})) << b;
    }
}

TEST(TestCaseGraph, ShuffledTableFollowsTheModelSeed)
{
    const auto row = pagedRow("dense", "shuffled", "separate");
    const auto first = integersOf(build(row, sizes(4, 64, 256)), "page_table_k");
    EXPECT_EQ(integersOf(build(row, sizes(4, 64, 256)), "page_table_k"), first);

    std::string text = CLASS_VALUES_JSON;
    const std::string seed = "\"seed\": 2893";
    text.replace(text.find(seed), seed.size(), "\"seed\": 2894");
    const auto otherSeed = sdpa_case_flow::parseClassValues(text);
    EXPECT_NE(integersOf(build(row, sizes(4, 64, 256), otherSeed), "page_table_k"), first);
}

TEST(TestCaseGraph, SharedPagesReuseSequenceZerosFirstPage)
{
    for(const char* sharing : {"separate", "shared"})
    {
        SCOPED_TRACE(sharing);
        const auto built = build(pagedRow("dense", "shared_pages", sharing), sizes(4, 64, 64));
        auto expected = inOrderTable(4, 4, 4);
        expected[4] = expected[0];
        for(const auto& table : tablesOf(sharing))
        {
            EXPECT_EQ(integersOf(built, table), expected) << table;
            EXPECT_FALSE(contentsOf(built, table).repeat) << table;
        }
    }
}

TEST(TestCaseGraph, SharedPagesWorkWithTwoSequences)
{
    for(const char* sharing : {"separate", "shared"})
    {
        SCOPED_TRACE(sharing);
        const auto built = build(pagedRow("dense", "shared_pages", sharing), sizes(2, 64, 64));
        for(const auto& table : tablesOf(sharing))
        {
            EXPECT_EQ(integersOf(built, table), (std::vector<int64_t>{0, 1, 2, 3, 0, 5, 6, 7}))
                << table;
        }
    }
}

TEST(TestCaseGraph, SharedPagesNeedTwoSequences)
{
    EXPECT_THROW(build(pagedRow("dense", "shared_pages", "separate"), sizes(1, 64, 64)),
                 CaseRealisationError);
}

TEST(TestCaseGraph, PackedShuffledTableKeepsItsOffset)
{
    const auto built = build(pagedRow("packed", "shuffled", "separate"), sizes(4, 64, 64));
    EXPECT_EQ(integersOf(built, "page_table_k_offset"), (std::vector<int64_t>{0, 4, 8, 12, 16}));
    EXPECT_EQ(integersOf(built, "page_table_k").size(), 16u);
}

TEST(TestCaseGraph, PageTableOrderIsRealised)
{
    EXPECT_TRUE(sdpa_case_flow::isRealisedContentAxis("page_table_order"));
}

TEST(TestCaseGraph, ClassValuesRejectAnUnknownPageOrder)
{
    std::string text = CLASS_VALUES_JSON;
    const std::string order = "\"seeded_permutation\"";
    text.replace(text.find(order), order.size(), "\"random\"");
    EXPECT_THROW(sdpa_case_flow::parseClassValues(text), sdpa_case_flow::ClassValuesError);
}
