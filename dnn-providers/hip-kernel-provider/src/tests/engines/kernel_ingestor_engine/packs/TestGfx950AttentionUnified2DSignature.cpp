// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "tests/engines/kernel_ingestor_engine/packs/Gfx950AttentionUnified2DTestGraphs.hpp"

/**
 * @file TestGfx950AttentionUnified2DSignature.cpp
 * @brief Pins the kernarg ABI hipkernel:Gfx950AttentionUnified2D marshals against the ABI the
 *        rocKE builder declares.
 *
 * A wrong argument count, order, kind or width is not a launch error. The driver copies
 * whatever the host hands it into the kernarg segment and the kernel reads it as the types it
 * was compiled for, so the failure mode is misread memory: wrong numerics with a success
 * status. This kernel has eighteen arguments, ten of them pointers the engine mostly passes
 * null, so a shifted slot is easy to write and invisible to a smoke test.
 *
 * requireSignatureMatch compares `kind` and `size`, and `name` only when both sides carry one.
 * The name comparison is the only one that tells the ten pointers apart, or the six f32
 * scalars from the i32 ones -- every pointer is `global_buffer` of 8 bytes and every
 * scalar `by_value` of 4 bytes -- so an exchange of two like slots is refused by name alone.
 * `offset` is printed and never compared, which is why every expectation carries zero for it.
 *
 * No device, no compile and no launch: the archive named here does not exist, and both the
 * signature refusal and the archive's absence are raised on the host.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

using namespace unified2d;

constexpr const char* BUFFER_KIND = "global_buffer";
constexpr const char* BY_VALUE_KIND = "by_value";
constexpr uint32_t POINTER_BYTES = 8;
constexpr uint32_t SCALAR_BYTES = 4;

constexpr std::size_t ARGUMENT_COUNT = 18;
constexpr std::size_t FIRST_SCALAR_INDEX = 10;

/// The fragment only the signature refusal carries. The rest of that message renders both
/// argument lists and is not stable enough to match on.
constexpr const char* MISMATCH_MARKER = "is packaged with arguments";

std::vector<KernelArgument> withArgument(std::size_t index, const KernelArgument& argument)
{
    auto signature = pythonAbi();
    signature.at(index) = argument;
    return signature;
}

/// Runs the pack's prepare() over the default graph and a descriptor recording @p recorded,
/// and returns the message of whatever it throws. Empty means it returned a prepared
/// dispatch, which no case expects.
std::string prepareFailureFor(const std::vector<KernelArgument>& recorded)
{
    const Graph graph;
    const auto bound = matchGraph(graph);
    EXPECT_TRUE(bound.has_value()) << "graph_match declined the fixture graph, so prepare() "
                                      "never ran and no case below means anything";
    if(!bound.has_value())
    {
        return {};
    }
    return prepareFailure(graph, *bound, makeKpackKernel(KernelSpec{}, recorded));
}

/// Asserts the pack refuses @p recorded on signature grounds rather than on any of the other
/// grounds prepare() can fail for.
void expectRefused(const std::vector<KernelArgument>& recorded, const std::string& what)
{
    const std::string failure = prepareFailureFor(recorded);
    ASSERT_FALSE(failure.empty()) << what << ": prepare() did not fail at all";
    EXPECT_NE(failure.find(MISMATCH_MARKER), std::string::npos)
        << what << ", but the failure was: " << failure;
}

} // namespace

// =============================================================================
// The shipped form is accepted
// =============================================================================

/// The positive control every refusal below depends on: a pack that refused every argument
/// list would pass all of them and launch nothing. Failure is still expected -- the archive is
/// absent -- but it must arrive from the loader, past the comparison.
TEST(TestGfx950AttentionUnified2DSignature, AcceptsTheAbiThePythonDeclares)
{
    ASSERT_EQ(pythonAbi().size(), ARGUMENT_COUNT);

    const std::string failure = prepareFailureFor(pythonAbi());
    ASSERT_FALSE(failure.empty()) << "the archive named by the fixture does not exist, so "
                                     "prepare() cannot have succeeded";
    EXPECT_EQ(failure.find(MISMATCH_MARKER), std::string::npos)
        << "the pack refused the argument list the rocKE builder declares: " << failure;
}

// =============================================================================
// Count
// =============================================================================

TEST(TestGfx950AttentionUnified2DSignature, RefusesADroppedTrailingArgument)
{
    // qq_bias_stride_0 gone: 17 arguments against 18, so the kernel would read one slot past
    // what the host wrote.
    auto signature = pythonAbi();
    signature.pop_back();
    expectRefused(signature, "a seventeen-argument list is not this ABI");
}

TEST(TestGfx950AttentionUnified2DSignature, RefusesAnExtraTrailingArgument)
{
    auto signature = pythonAbi();
    signature.push_back(KernelArgument{BY_VALUE_KIND, SCALAR_BYTES, 0, ""});
    expectRefused(signature, "a nineteen-argument list is not this ABI");
}

/// The form without the two trailing strides, which `_attn_signature` emits when its
/// include_bt_stride / include_qq_bias_stride switches are off. The pack hands the kernel both,
/// so a build without them is one it must not launch.
TEST(TestGfx950AttentionUnified2DSignature, RefusesTheFormWithoutTheTrailingStrides)
{
    auto signature = pythonAbi();
    signature.resize(ARGUMENT_COUNT - 2);
    expectRefused(signature, "the stride-less form is a different ABI");
}

// =============================================================================
// Kind
// =============================================================================

TEST(TestGfx950AttentionUnified2DSignature, RefusesAScalarInAPointerSlot)
{
    expectRefused(withArgument(1, KernelArgument{BY_VALUE_KIND, SCALAR_BYTES, 0, ""}),
                  "query_ptr is a global buffer");
}

TEST(TestGfx950AttentionUnified2DSignature, RefusesAPointerInAScalarSlot)
{
    // scale as a buffer: the kernel would take the float's bit pattern for an address.
    expectRefused(
        withArgument(FIRST_SCALAR_INDEX, KernelArgument{BUFFER_KIND, POINTER_BYTES, 0, ""}),
        "scale is passed by value");
}

TEST(TestGfx950AttentionUnified2DSignature, RefusesBuffersAndScalarsTransposed)
{
    // The eight scalars ahead of the ten pointers: the same eighteen entries and the same
    // counts of each kind, and the comparison is positional.
    const KernelArgument buffer{BUFFER_KIND, POINTER_BYTES, 0, ""};
    const KernelArgument scalar{BY_VALUE_KIND, SCALAR_BYTES, 0, ""};
    std::vector<KernelArgument> transposed(8, scalar);
    transposed.insert(transposed.end(), 10, buffer);
    expectRefused(transposed, "the ten pointers come first");
}

// =============================================================================
// Size
// =============================================================================

TEST(TestGfx950AttentionUnified2DSignature, RefusesANarrowedPointerSlot)
{
    // block_tables_ptr as a 32-bit buffer: every argument after it would be read from the
    // wrong offset.
    expectRefused(withArgument(5, KernelArgument{BUFFER_KIND, SCALAR_BYTES, 0, ""}),
                  "a device pointer is eight bytes");
}

TEST(TestGfx950AttentionUnified2DSignature, RefusesAWidenedScalarSlot)
{
    // block_table_stride as an i64: the next four bytes of the segment would be its high half.
    expectRefused(withArgument(16, KernelArgument{BY_VALUE_KIND, POINTER_BYTES, 0, ""}),
                  "the strides are 32-bit");
}

// =============================================================================
// Name
// =============================================================================

TEST(TestGfx950AttentionUnified2DSignature, RefusesEachSlotUnderAnotherName)
{
    const auto abi = pythonAbi();
    for(std::size_t index = 0; index < abi.size(); ++index)
    {
        auto renamed = abi;
        renamed.at(index).name += "_renamed";
        expectRefused(renamed, abi.at(index).name);
    }
}

/// Kind and size agree position by position, so the names are the only thing that separates
/// each of these from the shipped ABI. Accepted, the kernel reads keys as queries, or the
/// block table as the sequence lengths, with no fault and no status.
TEST(TestGfx950AttentionUnified2DSignature, RefusesEachExchangeOfTwoLikeSlots)
{
    const auto abi = pythonAbi();
    std::size_t exchanged = 0;
    for(std::size_t first = 0; first < abi.size(); ++first)
    {
        for(std::size_t second = first + 1; second < abi.size(); ++second)
        {
            if(abi.at(first).kind != abi.at(second).kind
               || abi.at(first).size != abi.at(second).size)
            {
                continue;
            }
            auto swapped = abi;
            std::swap(swapped.at(first), swapped.at(second));
            expectRefused(swapped, abi.at(first).name + " <-> " + abi.at(second).name);
            ++exchanged;
        }
    }
    // Ten pointers and eight scalars, every one of each group the same kind and size:
    // C(10, 2) + C(8, 2) pairs. A signature that grew or lost a slot changes this count.
    constexpr std::size_t POINTER_PAIRS = 45;
    constexpr std::size_t SCALAR_PAIRS = 28;
    EXPECT_EQ(exchanged, POINTER_PAIRS + SCALAR_PAIRS);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
