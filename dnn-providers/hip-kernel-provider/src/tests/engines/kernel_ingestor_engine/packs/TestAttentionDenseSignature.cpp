// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>

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
 * Both catalogs ship the 10-argument grid form -- four pointers, scale, then batch,
 * seqlen_q, seqlen_kv, num_query_heads and num_kv_heads -- and the TEST_P cases pin it on
 * each. Every variant in either catalog declares `use_sinks: false`, so the sink tail of
 * attention_dense_signature is unreachable.
 *
 * gfx942 only: a kernel on the persistent grid then takes a magic/shift pair per
 * fast-division divisor of its decode, in the decode's own order (14 or 16 arguments), and
 * an `emit_lse: true` kernel takes lse_ptr last. A graph that binds a stats output prepares
 * the LSE form and any other the form without it; the candidate's own grid selects the
 * pairs.
 *
 * gfx950 only: every variant declares `persistent: false`, `varlen: false` and
 * `paged: false`. The five-argument form of a baked persistent body (the pointers and
 * scale, no shape) is covered only as a shape the pack must refuse.
 *
 * requireSignatureMatch compares `kind` and `size`, and `name` only when both sides carry
 * one. The name comparison is the only one that tells the four pointers apart, or `scale`
 * (f32) from the i32 shape parameters -- all six by-value slots are `by_value` of 4
 * bytes -- so an exchange of two like slots, or a slot under another name, is refused by
 * name alone. `offset` is printed and never compared, which is why every expectation
 * below carries zero for it.
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

constexpr std::size_t LAST_INDEX = 9;

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
/// shape all still line up and only the count differs; lse_ptr last for @p lse.
std::vector<KernelArgument> persistentAbi(const std::string& decode = "qb_major", bool lse = false)
{
    auto signature = pythonAbi();
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
    return signature;
}

/// The form a gfx950 baked persistent body (ragged, or a gqa_pair order) takes: the four
/// pointers and scale with no shape tail. None ships in that catalog.
std::vector<KernelArgument> bakedPersistentAbi()
{
    auto signature = pythonAbi();
    signature.resize(SCALE_INDEX + 1);
    return signature;
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
    // A gfx942 persistent candidate at @p persistDecode, 304 CTAs.
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
    // num_kv_heads gone. Nine arguments against ten: the kernel would read the kernarg
    // segment one slot past what the host wrote.
    auto signature = pythonAbi();
    signature.pop_back();
    expectRefused(arch(), signature, "a nine-argument list is not this ABI");
}

TEST_P(TestAttentionDenseSignature, RefusesAnExtraTrailingArgument)
{
    // One i32 too many -- the shape a fourth shape parameter, or a stride argument, would
    // add. Every argument the host does write still lands correctly, so nothing faults.
    auto signature = pythonAbi();
    signature.push_back(KernelArgument{BY_VALUE_KIND, SCALAR_BYTES, 0, ""});
    expectRefused(arch(), signature, "an eleven-argument list is not this ABI");
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
    // The by-value tail ahead of the pointers. Same ten entries, the same four buffers
    // and six scalars, and the comparison is positional, so only the order distinguishes
    // this from the shipped ABI.
    const KernelArgument buffer{BUFFER_KIND, POINTER_BYTES, 0, ""};
    const KernelArgument scalar{BY_VALUE_KIND, SCALAR_BYTES, 0, ""};
    const std::vector<KernelArgument> transposed{
        scalar, scalar, scalar, scalar, scalar, scalar, buffer, buffer, buffer, buffer};
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
    expectRefused(arch(),
                  withArgument(LAST_INDEX, KernelArgument{BY_VALUE_KIND, POINTER_BYTES, 0, ""}),
                  "the shape parameters are 32-bit");
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

TEST(TestGfx942AttentionDenseSignature, RefusesTheTenArgumentFormForAGraphThatBindsStats)
{
    // A non-LSE kernel would never write the stats output the graph asked for.
    expectRefused(
        GFX942_ATTENTION_DENSE, pythonAbi(), "a stats graph launches lse_ptr", Stats::BOUND);
}

TEST(TestGfx942AttentionDenseSignature, RefusesTheLseFormForAGraphWithoutStats)
{
    // The launch would pass ten arguments, and the kernel would store the LSE through
    // whatever the eleventh kernarg slot holds.
    expectRefused(
        GFX942_ATTENTION_DENSE, pythonLseAbi(), "a graph without stats launches ten arguments");
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
    dropped.erase(dropped.begin() + static_cast<std::ptrdiff_t>(LAST_INDEX));
    expectRefused(
        GFX942_ATTENTION_DENSE, dropped, "num_kv_heads still precedes lse_ptr", Stats::BOUND);

    auto extra = pythonLseAbi();
    extra.push_back(KernelArgument{BY_VALUE_KIND, SCALAR_BYTES, 0, ""});
    expectRefused(GFX942_ATTENTION_DENSE, extra, "lse_ptr is the last argument", Stats::BOUND);
}

// =============================================================================
// gfx950: no persistent form ships
// =============================================================================

TEST(TestGfx950AttentionDenseSignature, RefusesThePersistentFiveArgumentForm)
{
    expectRefused(GFX950_ATTENTION_DENSE,
                  bakedPersistentAbi(),
                  "the persistent form does not ship in this catalog");
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
