// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>

#include "tests/engines/kernel_ingestor_engine/packs/AttentionDenseTestArch.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/AttentionDenseTestGraphs.hpp"

/**
 * @file TestAttentionDenseSignature.cpp
 * @brief Pins the kernarg ABI hipkernel:Gfx942AttentionDense and
 *        hipkernel:Gfx950AttentionDense marshal against the ABI the rocKE builder declares.
 *
 * A wrong argument count, order, kind or width is not a launch error. The driver copies
 * whatever the host hands it into the kernarg segment and the kernel reads it as the
 * types it was compiled for, so the failure mode is misread memory: wrong numerics with
 * a success status. This is the silent-wrong-answer class the layout and stride gates in
 * the matcher exist to stop, one layer lower down.
 *
 * Both catalogs ship the 22-argument grid form -- four pointers, scale, then batch,
 * seqlen_q, seqlen_kv, num_query_heads and num_kv_heads, then the twelve i32 strides
 * (stride_{q,k,v,o}_{batch,head,token}) last -- and the TEST_P cases pin it on each. Every
 * variant in either catalog declares `use_sinks: false`, so the sink tail of
 * attention_dense_signature is unreachable. The 10-argument form of kernels built before
 * the strides is refused: only a plan saved under a legacy contract launches it.
 *
 * gfx942 only: a kernel on the persistent grid takes a magic/shift pair per fast-division
 * divisor of its decode after the shape, in the decode's own order, and an `emit_lse: true`
 * kernel then takes lse_ptr; the strides stay last (26, 27, 28 or 29 arguments). A graph
 * that binds a stats output prepares the LSE form and any other the form without it; the
 * candidate's own grid selects the pairs.
 *
 * gfx950 only: every variant declares `persistent: false`, `varlen: false` and
 * `paged: false`. The form of a baked persistent body (the pointers and scale, no shape,
 * then the strides) is covered only as a shape the pack must refuse.
 *
 * requireSignatureMatch compares `kind` and `size`, and `name` only when both sides carry
 * one. The name comparison is the only one that tells the four pointers apart, or `scale`
 * (f32) from the i32 shape and stride parameters -- all eighteen by-value slots are
 * `by_value` of 4 bytes -- so an exchange of two like slots, or a slot under another name,
 * is refused by name alone. `offset` is printed and never compared, which is why every expectation
 * below carries zero for it.
 *
 * Both arches also ship use_additive_bias kernels, which take the ten arguments (after any
 * lse_ptr), the bias pointer and its three strides, then the twelve strides (twenty-six); a
 * bias graph prepares that form and a plain graph the twenty-two-argument one. The saved plan
 * of a bias kernel (gfx942 dispatch v4, gfx950 v5) restores against the same. Those cases run
 * on the arches that serve a bias (biasServingArches()).
 *
 * The last section pins AttentionDenseArch::isConsistent(), the rule each engine static_asserts
 * on its descriptor, and isContractTable, the one it static_asserts on its contract table, as
 * compile-time checks on broken descriptors and tables.
 *
 * No device, no compile and no launch: the archive named here does not exist, and both
 * the signature refusal and the archive's absence are raised on the host.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

using namespace attention_dense_graphs;

// ---------------------------------------------------------------------------
// The ABI, as the Python declares it. pythonAbi() and pythonLseAbi() are
// AttentionDenseTestGraphs.hpp's, shared with the dispatch suite; the forms below derive
// from them.
// ---------------------------------------------------------------------------

constexpr std::size_t SCALE_INDEX = 4;

constexpr std::size_t NUM_KV_HEADS_INDEX = 9;

constexpr std::size_t LSE_INDEX = 10;

/// The divisors of each gfx942 persistent decode, in kernarg order
/// (PersistDecode.fastdiv_divisors).
const std::vector<std::pair<std::string, std::vector<std::string>>>& persistentDecodes()
{
    static const std::vector<std::pair<std::string, std::vector<std::string>>> s_decodes{
        {"qb_major", {"batch", "num_query_heads", "gqa"}},
        {"hkv_major", {"batch", "gqa", "nqb"}},
        {"bt_hkv_minor", {"batch", "num_query_heads"}},
        {"hq_minor_swz", {"nqb", "num_query_heads", "gqa"}},
    };
    return s_decodes;
}

/// The form the gfx942 persistent body takes on @p decode (qb_major by default): the same
/// ten arguments, then a magic/shift pair per fast-division divisor, so the pointers and
/// shape all still line up and only the count differs; lse_ptr for @p lse; the strides last.
std::vector<KernelArgument> persistentAbi(const std::string& decode = "qb_major", bool lse = false)
{
    auto signature = pythonShapeAbi();
    for(const auto& [name, divisors] : persistentDecodes())
    {
        if(name != decode)
        {
            continue;
        }
        for(const auto& divisor : divisors)
        {
            for(const char* part : {"_magic", "_shift"})
            {
                signature.push_back(KernelArgument{BY_VALUE_KIND, SCALAR_BYTES, 0, divisor + part});
            }
        }
    }
    if(lse)
    {
        signature.push_back(KernelArgument{BUFFER_KIND, POINTER_BYTES, 0, "lse_ptr"});
    }
    return withStrides(std::move(signature));
}

/// The form a gfx950 baked persistent body (ragged, or a gqa_pair order) takes: the four
/// pointers and scale with no shape tail, then the strides. None ships in that catalog.
std::vector<KernelArgument> bakedPersistentAbi()
{
    auto signature = pythonShapeAbi();
    signature.resize(SCALE_INDEX + 1);
    return withStrides(std::move(signature));
}

std::vector<KernelArgument> withArgument(std::size_t index, const KernelArgument& argument)
{
    auto signature = pythonAbi();
    signature.at(index) = argument;
    return signature;
}

std::vector<KernelArgument> withLseArgument(std::size_t index, const KernelArgument& argument)
{
    auto signature = pythonLseAbi();
    signature.at(index) = argument;
    return signature;
}

// ---------------------------------------------------------------------------
// The seam
// ---------------------------------------------------------------------------

/// Whether the graph binds a stats output, which selects the form prepare() checks.
enum class Stats
{
    NONE,
    BOUND
};

/// The fragment only the signature refusal carries. Kept short: the rest of that message
/// renders both argument lists and is not stable enough to match on.
constexpr const char* MISMATCH_MARKER = "is packaged with arguments";

/// A KPACK descriptor for @p arch's engine whose recorded argument list is @p recorded, at
/// the completed baseline tile prepare() reads before the comparison; the tile is not what
/// these cases are about, and TestAttentionDenseDispatch.cpp owns its refusal.
KernelDefinition makeKernel(const AttentionDenseTestArch& arch,
                            const std::vector<KernelArgument>& recorded,
                            const std::string& persistDecode = "")
{
    auto kernel = kpackKernel(arch, recorded);
    // A gfx942 persistent candidate at @p persistDecode, 304 workgroups.
    if(!persistDecode.empty())
    {
        kernel.metadata[std::string("persistent")] = int64_t{1};
        kernel.metadata[std::string("persist_decode")] = persistDecode;
    }
    return kernel;
}

/// Runs @p arch's prepare() over a descriptor recording @p recorded, for the default graph
/// (any graph the engine accepts would do) binding @p stats, and returns the message of
/// whatever it throws. Empty means it returned a prepared dispatch, which no case expects.
std::string prepareFailure(const AttentionDenseTestArch& arch,
                           const std::vector<KernelArgument>& recorded,
                           Stats stats = Stats::NONE,
                           const std::string& persistDecode = "")
{
    const Graph graph(arch, stats == Stats::BOUND ? withStats(GraphSpec{}) : GraphSpec{});
    const auto bound = graph.match();
    EXPECT_TRUE(bound.has_value()) << "graph_match declined the fixture graph, so prepare() "
                                      "never ran and no case below means anything";
    if(!bound.has_value())
    {
        return {};
    }
    return attention_dense_graphs::prepareFailure(
        graph, *bound, makeKernel(arch, recorded, persistDecode));
}

/// Asserts @p arch's pack refuses @p recorded on signature grounds rather than on any of
/// the other grounds prepare() can fail for.
void expectRefused(const AttentionDenseTestArch& arch,
                   const std::vector<KernelArgument>& recorded,
                   const char* what,
                   Stats stats = Stats::NONE,
                   const std::string& persistDecode = "")
{
    const std::string failure = prepareFailure(arch, recorded, stats, persistDecode);
    ASSERT_FALSE(failure.empty()) << what << ": prepare() did not fail at all";
    EXPECT_NE(failure.find(MISMATCH_MARKER), std::string::npos)
        << what << ", but the failure was: " << failure;
}

class TestAttentionDenseSignature : public ::testing::TestWithParam<const AttentionDenseTestArch*>
{
protected:
    static const AttentionDenseTestArch& arch()
    {
        return *GetParam();
    }
};

} // namespace

// =============================================================================
// The shipped grid form is accepted
// =============================================================================

/// The positive control every refusal below depends on. A pack that refused every
/// argument list would pass all of them and launch nothing.
///
/// Failure is still expected -- the archive is absent -- but it must arrive from the
/// loader, past the comparison, rather than from the comparison itself.
TEST_P(TestAttentionDenseSignature, AcceptsTheAbiThePythonDeclares)
{
    const std::string failure = prepareFailure(arch(), pythonAbi());
    ASSERT_FALSE(failure.empty()) << "the archive named by the fixture does not exist, so "
                                     "prepare() cannot have succeeded";
    EXPECT_EQ(failure.find(MISMATCH_MARKER), std::string::npos)
        << "the pack refused the argument list the rocKE builder declares: " << failure;
}

// =============================================================================
// Count
// =============================================================================

TEST_P(TestAttentionDenseSignature, RefusesADroppedTrailingArgument)
{
    // stride_o_token gone. 21 arguments against 22: the kernel would read the kernarg
    // segment one slot past what the host wrote.
    auto signature = pythonAbi();
    signature.pop_back();
    expectRefused(arch(), signature, "a 21-argument list is not this ABI");
}

TEST_P(TestAttentionDenseSignature, RefusesAnExtraTrailingArgument)
{
    // One i32 too many -- the shape a fifth stride, or another trailing scalar, would add.
    // Every argument the host does write still lands correctly, so nothing faults.
    auto signature = pythonAbi();
    signature.push_back(KernelArgument{BY_VALUE_KIND, SCALAR_BYTES, 0, ""});
    expectRefused(arch(), signature, "a 23-argument list is not this ABI");
}

TEST_P(TestAttentionDenseSignature, RefusesTheFormWithoutStrides)
{
    // A pack built before the strides: its kernels take the ten shape arguments only. This
    // pack would write twelve more that the kernel never reads, and the kernel would address
    // every operand as packed whatever the graph's strides are.
    expectRefused(arch(), pythonShapeAbi(), "a kernel without the strides is not this ABI");
}

TEST_P(TestAttentionDenseSignature, RefusesTheStridesAheadOfTheShape)
{
    // The same 22 entries, the strides moved in front of batch. Count, kinds and sizes all
    // agree slot by slot past the pointers and scale; only the names tell them apart.
    auto shape = pythonShapeAbi();
    const auto strides = pythonStrideAbi();
    shape.insert(shape.begin() + static_cast<std::ptrdiff_t>(SCALE_INDEX + 1),
                 strides.begin(),
                 strides.end());
    expectRefused(arch(), shape, "the strides come after the shape");
}

// =============================================================================
// Kind
// =============================================================================

TEST_P(TestAttentionDenseSignature, RefusesAScalarInAPointerSlot)
{
    // q_ptr as a by-value scalar. The count still agrees, so only the kind catches it.
    expectRefused(arch(),
                  withArgument(0, KernelArgument{BY_VALUE_KIND, SCALAR_BYTES, 0, ""}),
                  "q_ptr is a global buffer");
}

TEST_P(TestAttentionDenseSignature, RefusesAPointerInAScalarSlot)
{
    // scale as a buffer: the kernel would take the float's bit pattern for an address.
    expectRefused(arch(),
                  withArgument(SCALE_INDEX, KernelArgument{BUFFER_KIND, POINTER_BYTES, 0, ""}),
                  "scale is passed by value");
}

TEST_P(TestAttentionDenseSignature, RefusesBuffersAndScalarsTransposed)
{
    // The by-value tail ahead of the pointers. Same 22 entries, the same four buffers and
    // eighteen scalars, and the comparison is positional, so only the order distinguishes
    // this from the shipped ABI.
    const KernelArgument buffer{BUFFER_KIND, POINTER_BYTES, 0, ""};
    const KernelArgument scalar{BY_VALUE_KIND, SCALAR_BYTES, 0, ""};
    std::vector<KernelArgument> transposed(pythonAbi().size() - SCALE_INDEX, scalar);
    transposed.insert(transposed.end(), SCALE_INDEX, buffer);
    expectRefused(arch(), transposed, "the four pointers come first");
}

// =============================================================================
// Size
// =============================================================================

TEST_P(TestAttentionDenseSignature, RefusesANarrowedPointerSlot)
{
    // o_ptr as a 32-bit buffer. The kind still agrees; the width does not, and every
    // argument after it would be read from the wrong offset.
    expectRefused(arch(),
                  withArgument(3, KernelArgument{BUFFER_KIND, SCALAR_BYTES, 0, ""}),
                  "a device pointer is eight bytes");
}

TEST_P(TestAttentionDenseSignature, RefusesAWidenedScalarSlot)
{
    // num_kv_heads as an i64. The shape parameters are i32 and the pack marshals int32_t,
    // so a 64-bit slot would take four bytes the host never wrote as its high half.
    expectRefused(
        arch(),
        withArgument(NUM_KV_HEADS_INDEX, KernelArgument{BY_VALUE_KIND, POINTER_BYTES, 0, ""}),
        "the shape parameters are 32-bit");
    // The same for a stride: the pack marshals each as int32_t too.
    expectRefused(
        arch(),
        withArgument(pythonAbi().size() - 1, KernelArgument{BY_VALUE_KIND, POINTER_BYTES, 0, ""}),
        "the strides are 32-bit");
}

// =============================================================================
// Name
// =============================================================================

TEST_P(TestAttentionDenseSignature, RefusesEachSlotUnderAnotherName)
{
    // Kind, size, count and order all agree; only the one slot's name differs.
    const auto abi = pythonAbi();
    for(std::size_t index = 0; index < abi.size(); ++index)
    {
        auto renamed = abi;
        renamed.at(index).name += "_renamed";
        expectRefused(arch(), renamed, abi.at(index).name.c_str());
    }
}

TEST_P(TestAttentionDenseSignature, RefusesEachExchangeOfTwoLikeSlots)
{
    // Kind and size agree position by position, so the names are the only thing that
    // separates each of these from the shipped ABI; accepted, the kernel reads keys as
    // queries, or a batch count as a sequence length, with no fault and no status.
    const auto abi = pythonAbi();
    for(std::size_t first = 0; first < abi.size(); ++first)
    {
        for(std::size_t second = first + 1; second < abi.size(); ++second)
        {
            if(abi.at(first).kind != abi.at(second).kind
               || abi.at(first).size != abi.at(second).size)
            {
                continue;
            }
            auto exchanged = abi;
            std::swap(exchanged.at(first), exchanged.at(second));
            const std::string what = abi.at(first).name + " <-> " + abi.at(second).name;
            expectRefused(arch(), exchanged, what.c_str());
        }
    }
}

INSTANTIATE_TEST_SUITE_P(,
                         TestAttentionDenseSignature,
                         ::testing::Values(&GFX942_ATTENTION_DENSE, &GFX950_ATTENTION_DENSE),
                         attentionDenseArchName);

// =============================================================================
// gfx942: the persistent fast-division form, selected by the candidate's own decode
// =============================================================================

TEST(TestGfx942AttentionDenseSignature, RefusesThePersistentFastDivisionFormForAGridKernel)
{
    expectRefused(
        GFX942_ATTENTION_DENSE, persistentAbi(), "a grid kernel takes no fast-division pairs");
}

/// A persistent candidate prepares its own decode's form, with and without lse_ptr; the
/// grid form and every other decode's pairs are refused.
TEST(TestGfx942AttentionDenseSignature, AcceptsThePersistentFormOfTheCandidatesOwnDecode)
{
    for(const auto& [decode, divisors] : persistentDecodes())
    {
        for(const auto stats : {Stats::NONE, Stats::BOUND})
        {
            const bool lse = stats == Stats::BOUND;
            const std::string failure
                = prepareFailure(GFX942_ATTENTION_DENSE, persistentAbi(decode, lse), stats, decode);
            ASSERT_FALSE(failure.empty()) << decode;
            EXPECT_EQ(failure.find(MISMATCH_MARKER), std::string::npos)
                << decode << " lse " << lse << ": " << failure;
        }
    }
}

TEST(TestGfx942AttentionDenseSignature, RefusesTheGridFormForAPersistentKernel)
{
    for(const auto& [decode, divisors] : persistentDecodes())
    {
        expectRefused(GFX942_ATTENTION_DENSE,
                      pythonAbi(),
                      "a persistent kernel takes its pairs",
                      Stats::NONE,
                      decode);
        expectRefused(GFX942_ATTENTION_DENSE,
                      pythonLseAbi(),
                      "a persistent LSE kernel takes its pairs before lse_ptr",
                      Stats::BOUND,
                      decode);
    }
}

TEST(TestGfx942AttentionDenseSignature, RefusesAnotherDecodesPairs)
{
    for(const auto& [decode, divisors] : persistentDecodes())
    {
        for(const auto& [other, otherDivisors] : persistentDecodes())
        {
            if(other != decode)
            {
                expectRefused(GFX942_ATTENTION_DENSE,
                              persistentAbi(other),
                              "another decode's pairs name other divisors",
                              Stats::NONE,
                              decode);
            }
        }
    }
}

// =============================================================================
// gfx942: the emit_lse form, lse_ptr last, for a graph that binds a stats output
// =============================================================================

/// The positive control for the cases below, as AcceptsTheAbiThePythonDeclares is for the
/// 10-argument form.
TEST(TestGfx942AttentionDenseSignature, AcceptsTheLseAbiForAGraphThatBindsStats)
{
    const std::string failure
        = prepareFailure(GFX942_ATTENTION_DENSE, pythonLseAbi(), Stats::BOUND);
    ASSERT_FALSE(failure.empty()) << "the archive named by the fixture does not exist, so "
                                     "prepare() cannot have succeeded";
    EXPECT_EQ(failure.find(MISMATCH_MARKER), std::string::npos)
        << "the pack refused the emit_lse argument list the rocKE builder declares: " << failure;
}

TEST(TestGfx942AttentionDenseSignature, RefusesTheFormWithoutLseForAGraphThatBindsStats)
{
    // A non-LSE kernel would never write the stats output the graph asked for.
    expectRefused(
        GFX942_ATTENTION_DENSE, pythonAbi(), "a stats graph launches lse_ptr", Stats::BOUND);
}

TEST(TestGfx942AttentionDenseSignature, RefusesTheLseFormForAGraphWithoutStats)
{
    // The launch would pass no lse_ptr, and the kernel would store the LSE through whatever
    // its lse_ptr slot then holds: the first two strides.
    expectRefused(
        GFX942_ATTENTION_DENSE, pythonLseAbi(), "a graph without stats launches no lse_ptr");
}

TEST(TestGfx942AttentionDenseSignature, RefusesTheLseFormWithLsePtrAfterTheStrides)
{
    // lse_ptr belongs before the strides, which stay the constant tail of every form.
    auto late = pythonAbi();
    late.push_back(KernelArgument{BUFFER_KIND, POINTER_BYTES, 0, "lse_ptr"});
    expectRefused(GFX942_ATTENTION_DENSE, late, "lse_ptr precedes the strides", Stats::BOUND);
}

TEST(TestGfx942AttentionDenseSignature, RefusesAnLseSlotThatIsNotAPointer)
{
    expectRefused(GFX942_ATTENTION_DENSE,
                  withLseArgument(LSE_INDEX, KernelArgument{BY_VALUE_KIND, SCALAR_BYTES, 0, ""}),
                  "lse_ptr is a global buffer",
                  Stats::BOUND);
    expectRefused(GFX942_ATTENTION_DENSE,
                  withLseArgument(LSE_INDEX, KernelArgument{BUFFER_KIND, SCALAR_BYTES, 0, ""}),
                  "a device pointer is eight bytes",
                  Stats::BOUND);
}

TEST(TestGfx942AttentionDenseSignature, RefusesTheLseFormWithAnyPointerRenamedOrExchanged)
{
    // lse_ptr is a fifth like slot beside the four operand pointers; only its name keeps
    // a kernel that writes the LSE over O, or O over the LSE, from passing.
    const auto abi = pythonLseAbi();
    auto renamed = abi;
    renamed.at(LSE_INDEX).name += "_renamed";
    expectRefused(GFX942_ATTENTION_DENSE, renamed, "lse_ptr", Stats::BOUND);

    for(std::size_t pointer = 0; pointer < SCALE_INDEX; ++pointer)
    {
        auto exchanged = abi;
        std::swap(exchanged.at(pointer), exchanged.at(LSE_INDEX));
        const std::string what = abi.at(pointer).name + " <-> lse_ptr";
        expectRefused(GFX942_ATTENTION_DENSE, exchanged, what.c_str(), Stats::BOUND);
    }
}

TEST(TestGfx942AttentionDenseSignature, RefusesAnLseFormWithASlotDroppedOrAdded)
{
    auto dropped = pythonLseAbi();
    dropped.erase(dropped.begin() + static_cast<std::ptrdiff_t>(NUM_KV_HEADS_INDEX));
    expectRefused(
        GFX942_ATTENTION_DENSE, dropped, "num_kv_heads still precedes lse_ptr", Stats::BOUND);

    auto extra = pythonLseAbi();
    extra.push_back(KernelArgument{BY_VALUE_KIND, SCALAR_BYTES, 0, ""});
    expectRefused(
        GFX942_ATTENTION_DENSE, extra, "stride_o_token is the last argument", Stats::BOUND);
}

// =============================================================================
// gfx950: no persistent form ships
// =============================================================================

TEST(TestGfx950AttentionDenseSignature, RefusesTheBakedPersistentForm)
{
    expectRefused(GFX950_ATTENTION_DENSE,
                  bakedPersistentAbi(),
                  "the persistent form does not ship in this catalog");
}

// =============================================================================
// The additive-bias form
// =============================================================================

namespace
{

/// graph_match over the default graph, or over it with a [B, 1, Sq, Skv] fp32 attn_mask when
/// @p bias, then prepare() over a descriptor of @p arch recording @p recorded whose
/// use_additive_bias is @p bias too. The message prepare() throws; empty if it returned.
std::string biasPrepareFailure(const AttentionDenseTestArch& arch,
                               const std::vector<KernelArgument>& recorded,
                               bool bias)
{
    const Graph graph(arch, bias ? withBias({BATCH, 1, SEQ, SEQ}) : GraphSpec{});
    const auto bound = graph.match();
    EXPECT_TRUE(bound.has_value()) << "graph_match declined the fixture graph, so prepare() "
                                      "never ran and no case below means anything";
    if(!bound.has_value())
    {
        return {};
    }
    auto kernel = makeKernel(arch, recorded);
    kernel.metadata[std::string("use_additive_bias")] = int64_t{bias ? 1 : 0};
    return attention_dense_graphs::prepareFailure(graph, *bound, kernel);
}

/// Asserts the pack refuses @p recorded on signature grounds. @p bias makes both the graph
/// and the candidate a bias one.
void expectBiasRefused(const AttentionDenseTestArch& arch,
                       const std::vector<KernelArgument>& recorded,
                       const char* what,
                       bool bias)
{
    const std::string failure = biasPrepareFailure(arch, recorded, bias);
    ASSERT_FALSE(failure.empty()) << what << ": prepare() did not fail at all";
    EXPECT_NE(failure.find(MISMATCH_MARKER), std::string::npos)
        << what << ", but the failure was: " << failure;
}

/// A tile the kernel is built with, so the saved block_m reads back.
constexpr int64_t SAVED_BLOCK_M = 256;
constexpr int64_t SAVED_BLOCK_N = 64;

/// No device. A restore that passes the signature comparison and the launch values stops
/// at the device load, which refuses a negative ordinal on the host.
constexpr int NO_DEVICE = -1;

/// The fragment only the restore's signature refusal carries, ahead of MISMATCH_MARKER.
constexpr const char* RESTORE_MISMATCH_MARKER
    = "the saved kernel's arguments do not match what this provider launches";

/// The fragment of the device-ordinal refusal that follows every restore check.
constexpr const char* NO_DEVICE_MARKER = "which names no device";

/// Valid launch values for the fixture graph under @p arch's plain contract, or its bias
/// contract with the fixture's full [B, Hq, Sq, Skv] bias when @p withBias, and with the
/// stats output @p statsUid where the arch saves one. Every value reads back, so only the
/// recorded signature can stop a restore before the device load.
hipdnn_plugin_sdk::ingestor::SavedLaunchInputs savedLaunchInputs(const AttentionDenseTestArch& arch,
                                                                 bool withBias,
                                                                 int64_t statsUid
                                                                 = ATTENTION_DENSE_NO_STATS)
{
    AttentionDenseBinding binding;
    binding.q = Q_UID;
    binding.k = K_UID;
    binding.v = V_UID;
    binding.o = O_UID;
    binding.scale = SCALE;
    binding.stats = statsUid;
    // The strided contracts save the kernel's own block_n and the strides it addresses with:
    // here the packed BSHD ones of Q, K, V and O alike, which a kernel that does not read them
    // (every bias kernel) is saved with.
    binding.blockN = SAVED_BLOCK_N;
    binding.readsStrides = false;
    {
        const auto batch = static_cast<int32_t>(SEQ * HEADS * HEAD_SIZE);
        const auto head = static_cast<int32_t>(HEAD_SIZE);
        const auto token = static_cast<int32_t>(HEADS * HEAD_SIZE);
        binding.strides
            = {batch, head, token, batch, head, token, batch, head, token, batch, head, token};
    }

    AttentionDenseProblem problem;
    problem.batch = BATCH;
    problem.seqLenQ = SEQ;
    problem.seqLenKv = SEQ;
    problem.numQueryHeads = HEADS;
    problem.numKvHeads = HEADS;
    problem.headSize = HEAD_SIZE;
    problem.dataType = data_objects::DataType::BFLOAT16;

    if(withBias)
    {
        binding.bias = AttentionDenseBias{BIAS_UID,
                                          {static_cast<int32_t>(HEADS * SEQ * SEQ),
                                           static_cast<int32_t>(SEQ * SEQ),
                                           static_cast<int32_t>(SEQ)}};
    }
    const auto symbol = withBias ? arch.biasContractSymbol : arch.plainContractSymbol;
    return {std::string(symbol),
            launchValuesUnder(symbol, arch.engineArch(), binding, problem, SAVED_BLOCK_M)};
}

/// What restoreLaunch() threw. SUCCESS with no message means it returned, which no case
/// expects.
struct RestoreFailure
{
    hipdnnPluginStatus_t status = HIPDNN_PLUGIN_STATUS_SUCCESS;
    std::string message;
};

/// Runs the restoreLaunch() of the handler registered under the saved contract over valid
/// plain (or, @p withBias, bias) launch values and a saved kernel whose recorded argument
/// list is @p recorded, and returns what it throws.
RestoreFailure restoreFailure(const AttentionDenseTestArch& arch,
                              const std::vector<KernelArgument>& recorded,
                              bool withBias,
                              int64_t statsUid = ATTENTION_DENSE_NO_STATS)
{
    registerNativeIngestorSymbols();

    const auto saved = savedLaunchInputs(arch, withBias, statsUid);
    const auto* handler
        = hipdnn_plugin_sdk::ingestor::DispatchRegistry<Handle>::resolve(saved.dispatchSymbol);
    EXPECT_NE(handler, nullptr);
    if(handler == nullptr)
    {
        return {};
    }

    hipdnn_plugin_sdk::ingestor::SavedKernelCode code;
    code.kernelId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-0000000051a1");
    code.sourceKind = hipdnn_plugin_sdk::ingestor::KernelSourceKind::KPACK;
    code.symbol = "attention_dense";
    code.target = std::string(arch.gcnArchName);
    code.recordedSignature = recorded;

    try
    {
        static_cast<void>(handler->restoreLaunch(saved, std::move(code), NO_DEVICE));
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        return {error.getStatus(), error.getMessage()};
    }
    return {};
}

/// The arches that serve an additive bias.
class TestAttentionDenseBiasSignature
    : public ::testing::TestWithParam<const AttentionDenseTestArch*>
{
protected:
    static const AttentionDenseTestArch& arch()
    {
        return *GetParam();
    }
};

} // namespace

/// The positive control for a bias variant: a bias graph on a bias candidate recording the
/// twenty-six arguments the builder declares gets past the comparison to the loader.
TEST_P(TestAttentionDenseBiasSignature, AcceptsTheBiasAbiThePythonDeclares)
{
    const std::string failure = biasPrepareFailure(arch(), pythonBiasAbi(), /*bias=*/true);
    ASSERT_FALSE(failure.empty()) << "the archive named by the fixture does not exist, so "
                                     "prepare() cannot have succeeded";
    EXPECT_EQ(failure.find(MISMATCH_MARKER), std::string::npos)
        << "the pack refused the argument list the rocKE builder declares: " << failure;
}

TEST_P(TestAttentionDenseBiasSignature, RefusesThePlainFormForABiasCandidate)
{
    // A bias candidate whose object records no bias tail: the launch passes four arguments
    // the kernel does not declare.
    expectBiasRefused(arch(), pythonAbi(), "a bias variant takes the bias tail", /*bias=*/true);
}

TEST_P(TestAttentionDenseBiasSignature, RefusesTheBiasFormWithoutStrides)
{
    // A bias kernel built before the strides: the ten shape arguments and the bias tail, 14 in
    // all, the likeliest skew of a bias catalog. The strided bias contract's launch would write
    // twelve arguments the kernel never declares.
    expectBiasRefused(arch(),
                      withBiasTail(pythonShapeAbi()),
                      "a bias kernel without the strides is not this ABI",
                      /*bias=*/true);
}

TEST_P(TestAttentionDenseBiasSignature, RefusesTheBiasFormForAPlainCandidate)
{
    expectBiasRefused(
        arch(), pythonBiasAbi(), "a plain variant takes no bias tail", /*bias=*/false);
}

TEST_P(TestAttentionDenseBiasSignature, RefusesEachExchangeOfTwoBiasStrides)
{
    // Same kind and size; accepted, the kernel would stride over heads by the batch pitch.
    const auto abi = pythonBiasAbi();
    const auto batchStride = static_cast<std::size_t>(
        std::find_if(abi.begin(),
                     abi.end(),
                     [](const KernelArgument& argument) {
                         return argument.name == "additive_bias_batch_stride";
                     })
        - abi.begin());
    ASSERT_LT(batchStride + 2, abi.size());
    for(std::size_t first = batchStride; first < batchStride + 3; ++first)
    {
        for(std::size_t second = first + 1; second < batchStride + 3; ++second)
        {
            auto exchanged = abi;
            std::swap(exchanged.at(first), exchanged.at(second));
            const std::string what = abi.at(first).name + " <-> " + abi.at(second).name;
            expectBiasRefused(arch(), exchanged, what.c_str(), /*bias=*/true);
        }
    }
}

// =============================================================================
// Restoring a saved plan
// =============================================================================

/// The positive control for the refusal below. The twenty-two-argument form that the plain
/// contract restores passes the signature comparison and the launch values and reaches the
/// device load.
TEST_P(TestAttentionDenseBiasSignature, RestoresThePlainFormPastTheComparison)
{
    const auto failure = restoreFailure(arch(), pythonAbi(), /*withBias=*/false);
    EXPECT_EQ(failure.status, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR) << failure.message;
    EXPECT_NE(failure.message.find(NO_DEVICE_MARKER), std::string::npos) << failure.message;
    EXPECT_EQ(failure.message.find(MISMATCH_MARKER), std::string::npos) << failure.message;
}

/// The plain contract holds no bias, so a bias plan saves under the bias contract. A plain
/// payload that names a bias kernel anyway (forged, or edited) records the twenty-six-argument
/// list. The restore must refuse it as incompatible, and must not launch a bias kernel with
/// twenty-two arguments.
TEST_P(TestAttentionDenseBiasSignature, RefusesToRestoreAPlainPlanThatRecordsTheBiasForm)
{
    const auto failure = restoreFailure(arch(), pythonBiasAbi(), /*withBias=*/false);
    EXPECT_EQ(failure.status, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE) << failure.message;
    EXPECT_NE(failure.message.find(RESTORE_MISMATCH_MARKER), std::string::npos) << failure.message;
    EXPECT_NE(failure.message.find(MISMATCH_MARKER), std::string::npos) << failure.message;
}

/// The positive control for the refusal below. The twenty-six-argument form that the bias
/// contract restores passes the signature comparison and the bias values and reaches the
/// device load.
TEST_P(TestAttentionDenseBiasSignature, RestoresTheBiasFormPastTheComparison)
{
    const auto failure = restoreFailure(arch(), pythonBiasAbi(), /*withBias=*/true);
    EXPECT_EQ(failure.status, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR) << failure.message;
    EXPECT_NE(failure.message.find(NO_DEVICE_MARKER), std::string::npos) << failure.message;
    EXPECT_EQ(failure.message.find(MISMATCH_MARKER), std::string::npos) << failure.message;
}

/// A bias payload that names a plain kernel records the twenty-two-argument list. The restore must
/// refuse it as incompatible, and must not pass four bias arguments the kernel does not
/// declare.
TEST_P(TestAttentionDenseBiasSignature, RefusesToRestoreABiasPlanThatRecordsThePlainForm)
{
    const auto failure = restoreFailure(arch(), pythonAbi(), /*withBias=*/true);
    EXPECT_EQ(failure.status, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE) << failure.message;
    EXPECT_NE(failure.message.find(RESTORE_MISMATCH_MARKER), std::string::npos) << failure.message;
    EXPECT_NE(failure.message.find(MISMATCH_MARKER), std::string::npos) << failure.message;
}

/// The same pre-stride bias kernel in a plan saved under the strided bias contract (edited, or
/// paired with a catalog rebuilt between save and load): the restore refuses it as
/// incompatible rather than launch twelve arguments the kernel never declares.
TEST_P(TestAttentionDenseBiasSignature, RefusesToRestoreABiasPlanThatRecordsTheFormWithoutStrides)
{
    const auto failure = restoreFailure(arch(), withBiasTail(pythonShapeAbi()), /*withBias=*/true);
    EXPECT_EQ(failure.status, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE) << failure.message;
    EXPECT_NE(failure.message.find(RESTORE_MISMATCH_MARKER), std::string::npos) << failure.message;
}

INSTANTIATE_TEST_SUITE_P(,
                         TestAttentionDenseBiasSignature,
                         ::testing::ValuesIn(biasServingArches()),
                         attentionDenseArchName);

// =============================================================================
// gfx942: the bias beside the stats output
// =============================================================================

/// A bias kernel that writes LSE records lse_ptr, the bias tail and then the twelve strides,
/// and a graph that binds a stats output and a bias is prepared on it. lse_ptr after the
/// strides, or no tail, is refused.
TEST(TestGfx942AttentionDenseSignature, AcceptsTheLseBiasFormOnlyInItsDeclaredOrder)
{
    const auto& arch = GFX942_ATTENTION_DENSE;
    const Graph graph(arch, withStats(withBias({BATCH, 1, SEQ, SEQ})));
    const auto bound = graph.match();
    ASSERT_TRUE(bound.has_value()) << "graph_match declined the fixture graph";

    const auto failureOf = [&](const std::vector<KernelArgument>& recorded) {
        auto kernel = makeKernel(arch, recorded);
        kernel.metadata[std::string("use_additive_bias")] = int64_t{1};
        return attention_dense_graphs::prepareFailure(graph, *bound, kernel);
    };

    const std::string accepted = failureOf(pythonLseBiasAbi());
    ASSERT_FALSE(accepted.empty()) << "prepare() cannot succeed without the archive";
    EXPECT_EQ(accepted.find(MISMATCH_MARKER), std::string::npos) << accepted;

    auto lseLast = pythonBiasAbi();
    lseLast.push_back(KernelArgument{BUFFER_KIND, POINTER_BYTES, 0, "lse_ptr"});
    EXPECT_NE(failureOf(lseLast).find(MISMATCH_MARKER), std::string::npos);
    EXPECT_NE(failureOf(pythonLseAbi()).find(MISMATCH_MARKER), std::string::npos)
        << "a graph with a bias needs the bias tail";
    EXPECT_NE(failureOf(pythonBiasAbi()).find(MISMATCH_MARKER), std::string::npos)
        << "a graph with stats needs lse_ptr";
}

/// The same for a saved plan: a bias plan with a stats output restores the LSE form with the
/// tail before the strides, and refuses the form without the tail.
TEST(TestGfx942AttentionDenseSignature, RestoresTheLseBiasFormOfABiasPlanWithStats)
{
    const auto& arch = GFX942_ATTENTION_DENSE;
    constexpr int64_t STATS = 7;
    const auto restored = restoreFailure(arch, pythonLseBiasAbi(), /*withBias=*/true, STATS);
    EXPECT_EQ(restored.status, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR) << restored.message;
    EXPECT_NE(restored.message.find(NO_DEVICE_MARKER), std::string::npos) << restored.message;

    const auto tailless = restoreFailure(arch, pythonLseAbi(), /*withBias=*/true, STATS);
    EXPECT_EQ(tailless.status, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE) << tailless.message;
    EXPECT_NE(tailless.message.find(RESTORE_MISMATCH_MARKER), std::string::npos)
        << tailless.message;
}

/// The LSE form of a bias kernel built before the strides (lse_ptr, then the bias tail, and no
/// strides) is refused for a graph with a bias and a stats output, at prepare() and at restore.
TEST(TestGfx942AttentionDenseSignature, RefusesTheLseBiasFormWithoutStrides)
{
    const auto& arch = GFX942_ATTENTION_DENSE;
    const auto preStride = withBiasTail(pythonLseShapeAbi());

    const Graph graph(arch, withStats(withBias({BATCH, 1, SEQ, SEQ})));
    const auto bound = graph.match();
    ASSERT_TRUE(bound.has_value()) << "graph_match declined the fixture graph";
    auto kernel = makeKernel(arch, preStride);
    kernel.metadata[std::string("use_additive_bias")] = int64_t{1};
    const std::string prepared = attention_dense_graphs::prepareFailure(graph, *bound, kernel);
    EXPECT_NE(prepared.find(MISMATCH_MARKER), std::string::npos) << prepared;

    constexpr int64_t STATS = 7;
    const auto restored = restoreFailure(arch, preStride, /*withBias=*/true, STATS);
    EXPECT_EQ(restored.status, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE) << restored.message;
    EXPECT_NE(restored.message.find(RESTORE_MISMATCH_MARKER), std::string::npos)
        << restored.message;
}

// =============================================================================
// The arch descriptor's coherence rule
// =============================================================================
//
// Each engine static_asserts AttentionDenseArch::isConsistent() on its own descriptor, so a
// shipped descriptor is checked where it is defined. That says nothing about a BROKEN one: a
// branch of isConsistent() could be loosened and both engines would still build. These are
// compile-time checks too, so a loosened rule fails this file's build. Each descriptor below is
// a coherent one (stats and a bias served) with one rule broken, and must read as inconsistent.

namespace
{

using attention_dense::AttentionDenseAbi;
using attention_dense::AttentionDenseArch;

/// Hooks the descriptors below carry: isConsistent() reads whether one is set, never calls it.
std::optional<bool> unreachedHook(const hipdnn_plugin_sdk::ingestor::KernelDefinition& /*kernel*/)
{
    return std::nullopt;
}

bool unreachedStrideRule(const hipdnn_plugin_sdk::ingestor::KernelDefinition& /*kernel*/)
{
    return false;
}

/// A descriptor that serves stats and a bias, with every rule of isConsistent() met.
constexpr AttentionDenseArch coherentArch()
{
    AttentionDenseArch arch{};
    arch.tokens.stats = "arch.stats";
    arch.tokens.bias = "arch.bias";
    arch.tokens.biasUid = "arch.bias.uid";
    arch.tokens.biasStrideBatch = "arch.bias.stride_batch";
    arch.tokens.biasStrideHead = "arch.bias.stride_head";
    arch.tokens.biasStrideQuery = "arch.bias.stride_query";
    arch.candidateWritesLse = &unreachedHook;
    arch.candidateReadsStrides = &unreachedStrideRule;
    arch.candidateAddsBias = &unreachedHook;
    arch.currentContract = {"arch.dispatch.v4", AttentionDenseAbi::RUNTIME_STRIDES};
    arch.legacyContracts[0] = {"arch.dispatch.v1", AttentionDenseAbi::BAKED_HEAD_COUNTS};
    arch.legacyContracts[1] = {"arch.dispatch.v2", AttentionDenseAbi::RUNTIME_HEAD_COUNTS};
    arch.legacyContracts[2]
        = {"arch.dispatch.v3", AttentionDenseAbi::RUNTIME_HEAD_COUNTS_WITH_BIAS};
    arch.biasContract = {"arch.dispatch.v5", AttentionDenseAbi::RUNTIME_STRIDES_WITH_BIAS};
    return arch;
}

/// Whether the coherent descriptor, after @p edit, still reads as consistent.
template <typename Edit>
constexpr bool consistentAfter(Edit edit)
{
    AttentionDenseArch arch = coherentArch();
    edit(arch);
    return arch.isConsistent();
}

} // namespace

// The rule is not simply "refuse everything": a coherent descriptor, a plain one that serves
// neither stats nor a bias (gfx950 serves no stats) and one with no superseded contract are all
// accepted.
static_assert(consistentAfter([](AttentionDenseArch&) {}),
              "the coherent baseline must be consistent, or no refusal below means anything");
static_assert(consistentAfter([](AttentionDenseArch& arch) {
                  arch.candidateWritesLse = nullptr;
                  arch.candidateAddsBias = nullptr;
                  arch.biasContract = {};
              }),
              "a descriptor that serves neither stats nor a bias is consistent");
static_assert(consistentAfter([](AttentionDenseArch& arch) { arch.legacyContracts = {}; }),
              "a descriptor with no superseded contract is consistent");

// Every arch has its stride rule, and a named current contract whose kernels take the strides
// and no bias: the graph match admits strided layouts on every arch.
static_assert(!consistentAfter([](AttentionDenseArch& arch) {
    arch.candidateReadsStrides = nullptr;
}),
              "an arch without a stride rule must be refused");
static_assert(!consistentAfter([](AttentionDenseArch& arch) { arch.currentContract.symbol = {}; }),
              "a current contract without a name must be refused");
static_assert(!consistentAfter([](AttentionDenseArch& arch) {
    arch.candidateAddsBias = nullptr;
    arch.biasContract = {};
    arch.currentContract.abi = AttentionDenseAbi::RUNTIME_HEAD_COUNTS;
}),
              "a current contract that takes no strides must be refused on an arch without a bias");
static_assert(!consistentAfter([](AttentionDenseArch& arch) {
    arch.candidateAddsBias = nullptr;
    arch.biasContract = {};
    arch.currentContract.abi = AttentionDenseAbi::RUNTIME_STRIDES_WITH_BIAS;
}),
              "a current contract that launches the bias form must be refused");

// No two contracts share a name, so a saved name selects exactly one.
static_assert(!consistentAfter([](AttentionDenseArch& arch) {
    arch.legacyContracts[1].symbol = arch.legacyContracts[0].symbol;
}),
              "two legacy contracts under one name must be refused");
static_assert(!consistentAfter([](AttentionDenseArch& arch) {
    arch.legacyContracts[2].symbol = arch.currentContract.symbol;
}),
              "a legacy contract under the current contract's name must be refused");

// A served stats output needs its token spelled.
static_assert(!consistentAfter([](AttentionDenseArch& arch) { arch.tokens.stats = {}; }),
              "stats served with no stats token must be refused");

// A bias hook and a bias contract come together.
static_assert(!consistentAfter([](AttentionDenseArch& arch) { arch.biasContract = {}; }),
              "a bias hook without a bias contract must be refused");
static_assert(!consistentAfter([](AttentionDenseArch& arch) { arch.candidateAddsBias = nullptr; }),
              "a bias contract without a bias hook must be refused");

// A served bias spells all five of its tokens.
static_assert(!consistentAfter([](AttentionDenseArch& arch) { arch.tokens.bias = {}; }),
              "a served bias with no bias token must be refused");
static_assert(!consistentAfter([](AttentionDenseArch& arch) { arch.tokens.biasUid = {}; }),
              "a served bias with no bias uid token must be refused");
static_assert(!consistentAfter([](AttentionDenseArch& arch) { arch.tokens.biasStrideBatch = {}; }),
              "a served bias with no batch stride token must be refused");
static_assert(!consistentAfter([](AttentionDenseArch& arch) { arch.tokens.biasStrideHead = {}; }),
              "a served bias with no head stride token must be refused");
static_assert(!consistentAfter([](AttentionDenseArch& arch) { arch.tokens.biasStrideQuery = {}; }),
              "a served bias with no query stride token must be refused");

// The bias contract's kernel arguments follow the current contract's, and its name is its own.
static_assert(
    !consistentAfter([](AttentionDenseArch& arch) {
        arch.currentContract.abi = AttentionDenseAbi::BAKED_HEAD_COUNTS;
    }),
    "a bias contract after a current contract that bakes the head counts must be refused");
static_assert(!consistentAfter([](AttentionDenseArch& arch) {
    arch.currentContract.abi = AttentionDenseAbi::RUNTIME_HEAD_COUNTS;
}),
              "a bias contract after a current contract that takes no strides must be refused");
static_assert(!consistentAfter([](AttentionDenseArch& arch) {
    arch.biasContract.abi = AttentionDenseAbi::RUNTIME_STRIDES;
}),
              "a bias contract that launches the form without the bias must be refused");
static_assert(!consistentAfter([](AttentionDenseArch& arch) {
    arch.biasContract.abi = AttentionDenseAbi::RUNTIME_HEAD_COUNTS_WITH_BIAS;
}),
              "a bias contract that launches the form without the strides must be refused");
static_assert(!consistentAfter([](AttentionDenseArch& arch) {
    arch.biasContract.symbol = arch.currentContract.symbol;
}),
              "a bias contract under the current contract's name must be refused");
static_assert(!consistentAfter([](AttentionDenseArch& arch) {
    arch.biasContract.symbol = arch.legacyContracts[0].symbol;
}),
              "a bias contract under a legacy contract's name must be refused");
static_assert(!consistentAfter([](AttentionDenseArch& arch) {
    arch.biasContract.symbol = arch.legacyContracts[2].symbol;
}),
              "a bias contract under the legacy bias contract's name must be refused");

// The contract table each engine static_asserts (isContractTable): one current row, and the
// bias row after it, since a new current version needs a new bias version. A bump that adds a
// current row but leaves the old bias row in place is refused; a new bias version alone is not.
namespace
{

using attention_dense::AttentionDenseContractRole;
using attention_dense::AttentionDenseContractRow;

constexpr AttentionDenseContractRow tableRow(AttentionDenseContractRole role)
{
    return {{"arch.dispatch", AttentionDenseAbi::RUNTIME_STRIDES}, role};
}

constexpr auto CURRENT_ROW = tableRow(AttentionDenseContractRole::CURRENT);
constexpr auto BIAS_ROW = tableRow(AttentionDenseContractRole::BIAS);
constexpr auto LEGACY_ROW = tableRow(AttentionDenseContractRole::LEGACY);

} // namespace

static_assert(attention_dense::isContractTable(std::array{
                  LEGACY_ROW, LEGACY_ROW, LEGACY_ROW, CURRENT_ROW, BIAS_ROW}),
              "today's tables (legacy rows, then current, then bias) are contract tables");
static_assert(attention_dense::isContractTable(std::array{LEGACY_ROW, CURRENT_ROW}),
              "a table without a bias row is a contract table");
static_assert(attention_dense::isContractTable(std::array{CURRENT_ROW, LEGACY_ROW, BIAS_ROW}),
              "a new bias version alone leaves the bias row after the current one");
static_assert(!attention_dense::isContractTable(std::array{
                  LEGACY_ROW, LEGACY_ROW, LEGACY_ROW, LEGACY_ROW, BIAS_ROW, CURRENT_ROW}),
              "a new current version that leaves the old bias row in place must be refused");
static_assert(!attention_dense::isContractTable(std::array{CURRENT_ROW, BIAS_ROW, CURRENT_ROW}),
              "two current rows must be refused");
static_assert(!attention_dense::isContractTable(std::array{LEGACY_ROW, BIAS_ROW}),
              "a table without a current row must be refused");
static_assert(!attention_dense::isContractTable(std::array{CURRENT_ROW, BIAS_ROW, BIAS_ROW}),
              "two bias rows must be refused");

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
