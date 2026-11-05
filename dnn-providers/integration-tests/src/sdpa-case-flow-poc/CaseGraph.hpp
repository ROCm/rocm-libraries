// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <hipdnn_frontend/Graph.hpp>

namespace sdpa_case_flow
{

// One case: model axis id -> model value id. A child axis whose parent is not active holds "-".
using Row = std::map<std::string, std::string>;

// Thrown when a row cannot become a graph: an unknown value id, a combination the
// class-to-value table does not cover, or a malformed override token.
class CaseRealisationError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

// Thrown when the class-to-value file is malformed or lacks a table the driver needs.
class ClassValuesError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

enum class DescaleForm
{
    BY_VALUE,
    DEVICE,
};

struct ScaleNumber
{
    bool inverseSqrtHeadDim = false;
    double value = 0.0;
};

// Head counts that replace a power-of-two query head count for ALiBi tests.
struct AlibiHeads
{
    int64_t hq = 0;
    int64_t mha = 0;
    int64_t gqa = 0;
    int64_t mqa = 0;
    int64_t differK = 0;
    int64_t differV = 0;
};

// How a page table orders each sequence's pages.
struct PageOrder
{
    // Seed of the per-sequence slot permutation; std::nullopt keeps the in_order slots.
    std::optional<int64_t> seed;
    // Slot `page` of sequence `sequence` takes the page of the same slot of `reusesSequence`.
    struct Share
    {
        int64_t sequence = 0;
        int64_t reusesSequence = 0;
        int64_t page = 0;
    };
    std::optional<Share> share;
};

// One per-sequence length: a constant, or the stored length S changed by one operation.
struct LengthTerm
{
    enum class Op
    {
        CONSTANT,
        STORED,
        DIVIDE,
        ADD,
        MULTIPLY,
    };
    Op op = Op::CONSTANT;
    int64_t operand = 0;
};

// How the B + 1 offsets of a ragged tensor follow from its per-sequence lengths.
enum class OffsetLayout
{
    // offset b = (sum of the lengths of sequences 0 .. b-1) x stride.
    PREFIX_SUM,
    // offset b = b x S x stride.
    FULL_SLOTS,
    // offset b = (g + sum of the lengths of sequences 0 .. b-1) x stride, g = B x S - all lengths.
    PREFIX_SUM_AFTER_GAP,
    // offset b = (b x S + S - length b) x stride for b < B, offset B = B x S x stride: each
    // sequence ends at the end of its full slot.
    SLOT_END_ALIGNED,
    // offset b = (B - 1 - b) x S x stride for b < B, offset B = B x S x stride: sequence b sits in
    // slot B - 1 - b, so the starts fall in reverse order. The length tensors give the lengths.
    REVERSED_SLOTS,
};

// Where an alignment class puts the data start, in bytes. std::nullopt means one element.
struct AlignmentEntry
{
    std::optional<int64_t> startOffset;
    std::optional<int64_t> declaredAlignment;
};

struct SequenceLengthEntry
{
    std::string seqQ;
    std::string seqKv;
    std::string relation;
    int64_t sq = 0;
    int64_t skv = 0;
};

// The concrete values of the model's classes, as written by the case generator.
struct ClassValues
{
    std::map<std::string, int64_t> batchSize;
    std::map<std::string, int64_t> qHeads;
    // Head sharing value (mha, gqa, mqa, non_divisor) -> q_heads class -> H_k or H_v.
    std::map<std::string, std::map<std::string, int64_t>> headCounts;
    // H_k and H_v when both sharings are gqa and k_heads_vs_v_heads is differ.
    int64_t differK = 0;
    int64_t differV = 0;
    std::vector<SequenceLengthEntry> sequenceLengths;
    std::map<std::string, int64_t> headDim;
    // dtype_q value -> format of a K or V whose relation to Q is "differs".
    std::map<std::string, std::string> differsFormat;
    // Base arch -> (format -> the format used on that arch).
    std::map<std::string, std::map<std::string, std::string>> archFormat;
    // Physical order, outermost first, as indices into the logical dims [B, H, S, D].
    std::map<std::string, std::vector<size_t>> dimOrder;
    // Alignment class -> data start offset and declared alignment.
    std::map<std::string, AlignmentEntry> alignment;
    // Representatives of each class, primary first. A case uses the primary.
    std::map<std::string, std::vector<ScaleNumber>> scaleValue;
    std::map<std::string, double> dropoutProbability;
    int64_t dropoutSeed = 0;
    int64_t dropoutOffset = 0;
    std::map<std::string, double> dropoutMaskScale;
    // dropout_mask_pattern value -> mask contents, repeated in element order.
    std::map<std::string, std::vector<double>> dropoutMaskCycle;
    // Tokens per page; std::nullopt means one page per sequence (S_kv tokens).
    std::map<std::string, std::optional<int64_t>> pageSize;
    // Keys descale_q, descale_k, descale_v and other.
    std::map<std::string, double> quantisationFactors;
    // Granularity (per_head, finer) -> descale name -> factors, repeated in element order.
    std::map<std::string, std::map<std::string, std::vector<double>>> descaleLists;
    // batch_packing value -> ragged tensors, as letters Q, K, V, O.
    std::map<std::string, std::set<std::string>> raggedTensors;
    // stats_packing value -> whether Stats is ragged with Q's token boundaries.
    std::map<std::string, bool> raggedStats;
    // per_seq_len class -> lengths of sequences 0 .. B-1, repeated in batch order.
    std::map<std::string, std::vector<LengthTerm>> perSeqLen;
    // ragged_offsets value -> how the offsets follow from the lengths.
    std::map<std::string, OffsetLayout> raggedOffsets;
    // paged_kv value -> paged sides, as letters K, V.
    std::map<std::string, std::set<std::string>> pagedSides;
    // page_size_k_vs_v value -> V page size as a multiple of K's.
    std::map<std::string, int64_t> vPageMultiplier;
    // page_table_sharing value -> number of page-table tensors (1 or 2).
    std::map<std::string, int64_t> pageTables;
    // page_table_storage value -> whether the table carries a ragged offset.
    std::map<std::string, bool> packedPageTable;
    // page_table_order value -> slot order of every page table.
    std::map<std::string, PageOrder> pageTableOrder;
    // Power-of-two query head count -> the counts that replace it.
    std::map<int64_t, AlibiHeads> alibiHeads;
    // fp8 mma_core_mode when Q is not fp8, and per-arch substitutions.
    std::string mmaFp8Otherwise;
    std::map<std::string, std::map<std::string, std::string>> mmaFp8ByArch;
};

// Parse the generator's class-to-value JSON. Throws ClassValuesError.
ClassValues parseClassValues(const std::string& jsonText);

// Concrete tensor extents that replace the class-to-value sizes of a row.
struct SizeOverride
{
    int64_t b = 0;
    int64_t hq = 0;
    int64_t hk = 0;
    int64_t hv = 0;
    int64_t sq = 0;
    int64_t skv = 0;
    int64_t dqk = 0;
    int64_t dv = 0;
};

// Parse "B2 Hq8 Hk1 Hv1 Sq64 Skv1 Dqk256 Dv128". Throws CaseRealisationError.
SizeOverride parseSizeOverride(const std::string& text);

struct CaseContext
{
    const ClassValues& values;
    DescaleForm descaleForm;
    // Base arch of the device (for example gfx942), or "none".
    std::string arch;
    // When set, these extents replace the class-to-value sizes of the row.
    std::optional<SizeOverride> sizes = std::nullopt;
};

// Exact contents of one input tensor. Element i in logical index order holds values[i % n]
// when repeat is true. When repeat is false, the list holds one value per element.
struct Contents
{
    std::variant<std::vector<int64_t>, std::vector<double>> values;
    bool repeat = true;
};

struct BuiltCase
{
    std::shared_ptr<hipdnn_frontend::graph::Graph> graph;
    // Concrete numbers and formats the row turned into, for the result record.
    std::string realised;
    // Contents that replace the random input fill, keyed by tensor name.
    std::map<std::string, Contents> contents;
    // Bytes between an aligned allocation start and the data start, keyed by the graph tensor's
    // name. Only tensors with a non-zero offset are listed.
    std::map<std::string, int64_t> startOffsets;
};

// The axis ids of the maintained forward SDPA model, in its column order.
const std::vector<std::string>& modelAxisIds();

// True for an axis whose value is decided by tensor contents or pointers at execute time.
bool isExecuteLevelAxis(const std::string& axis);

// True for an axis whose value buildCaseGraph turns into BuiltCase::contents.
bool isRealisedContentAxis(const std::string& axis);

// The shared base case value of an axis, or "-" when the base omits the axis.
std::string sharedBaseValue(const std::string& axis);

// Build one forward SDPA graph from a row and its override tokens. Nothing is lowered,
// validated or sent to a backend here.
BuiltCase buildCaseGraph(const Row& row,
                         const std::vector<std::string>& overrideTokens,
                         const CaseContext& context);

} // namespace sdpa_case_flow
