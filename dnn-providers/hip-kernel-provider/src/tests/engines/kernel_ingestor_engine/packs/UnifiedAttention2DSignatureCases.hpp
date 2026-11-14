// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>

/**
 * @file UnifiedAttention2DSignatureCases.hpp
 * @brief The kernarg-ABI cases the unified-attention 2D packs share: pins the ABI each pack
 *        marshals (unified2DKernelSignature in UnifiedAttention2DCommon.hpp) against the ABI
 *        the rocKE builder declares.
 *
 * A wrong argument count, order, kind or width is not a launch error. The driver copies
 * whatever the host hands it into the kernarg segment and the kernel reads it as the types it
 * was compiled for, so the failure mode is misread memory: wrong numerics with a success
 * status. This kernel has twenty-one arguments, ten of them pointers the engine mostly passes
 * null, so a shifted slot is easy to write and invisible to a smoke test.
 *
 * requireSignatureMatch compares `kind` and `size`, and `name` only when both sides carry one.
 * The name comparison is the only one that tells the ten pointers apart, or the f32
 * scalars from the i32 ones (every pointer is `global_buffer` of 8 bytes and every scalar
 * `by_value` of 4 bytes), so an exchange of two like slots is refused by name alone.
 * `offset` is printed and never compared, which is why every expectation carries zero for it.
 *
 * No device, no compile and no launch: the archive named by a pack's fixture does not exist,
 * and both the signature refusal and the archive's absence are raised on the host.
 *
 * Each case is a function template over an `Arch`, which supplies
 *   - `static std::vector<KernelArgument> pythonAbi()`: the ABI the pack's rocKE builder
 *     declares;
 *   - `static std::string prepareFailureFor(const std::vector<KernelArgument>&)`: the message
 *     of what the pack's prepare() throws over its default graph and a descriptor recording
 *     the argument list (empty if it returned a prepared dispatch).
 * Each pack's test file runs every case under its own suite name, so a pack that regressed
 * fails under its own name.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing::unified2d_signature
{

using hipdnn_plugin_sdk::ingestor::KernelArgument;

inline constexpr const char* BUFFER_KIND = "global_buffer";
inline constexpr const char* BY_VALUE_KIND = "by_value";
inline constexpr uint32_t POINTER_BYTES = 8;
inline constexpr uint32_t SCALAR_BYTES = 4;

inline constexpr std::size_t ARGUMENT_COUNT = 21;
inline constexpr std::size_t FIRST_SCALAR_INDEX = 10;

/// The fragment only the signature refusal carries. The rest of that message renders both
/// argument lists and is not stable enough to match on.
inline constexpr const char* MISMATCH_MARKER = "is packaged with arguments";

template <typename Arch>
std::vector<KernelArgument> withArgument(std::size_t index, const KernelArgument& argument)
{
    auto signature = Arch::pythonAbi();
    signature.at(index) = argument;
    return signature;
}

/// Asserts the pack refuses @p recorded on signature grounds rather than on any of the other
/// grounds prepare() can fail for.
template <typename Arch>
void expectRefused(const std::vector<KernelArgument>& recorded, const std::string& what)
{
    const std::string failure = Arch::prepareFailureFor(recorded);
    ASSERT_FALSE(failure.empty()) << what << ": prepare() did not fail at all";
    EXPECT_NE(failure.find(MISMATCH_MARKER), std::string::npos)
        << what << ", but the failure was: " << failure;
}

/// The positive control every refusal below depends on: a pack that refused every argument
/// list would pass all of them and launch nothing. Failure is still expected: the archive is
/// absent, but it must arrive from the loader, past the comparison.
template <typename Arch>
void acceptsTheAbiThePythonDeclares()
{
    ASSERT_EQ(Arch::pythonAbi().size(), ARGUMENT_COUNT);

    const std::string failure = Arch::prepareFailureFor(Arch::pythonAbi());
    ASSERT_FALSE(failure.empty()) << "the archive named by the fixture does not exist, so "
                                     "prepare() cannot have succeeded";
    EXPECT_EQ(failure.find(MISMATCH_MARKER), std::string::npos)
        << "the pack refused the argument list the rocKE builder declares: " << failure;
}

template <typename Arch>
void refusesADroppedTrailingArgument()
{
    // sliding_window gone: 20 arguments against 21, so the kernel would read one slot past
    // what the host wrote.
    auto signature = Arch::pythonAbi();
    signature.pop_back();
    expectRefused<Arch>(signature, "a twenty-argument list is not this ABI");
}

template <typename Arch>
void refusesAnExtraTrailingArgument()
{
    auto signature = Arch::pythonAbi();
    signature.push_back(KernelArgument{BY_VALUE_KIND, SCALAR_BYTES, 0, ""});
    expectRefused<Arch>(signature, "a twenty-two-argument list is not this ABI");
}

/// The form without the two trailing strides, which `_attn_signature` emits when its
/// include_bt_stride / include_qq_bias_stride switches are off. The pack hands the kernel both,
/// so a build without them is one it must not launch.
template <typename Arch>
void refusesTheFormWithoutTheTrailingStrides()
{
    // block_table_stride and qq_bias_stride_0 sit between num_seqs and the runtime shape
    // kernargs.
    auto signature = Arch::pythonAbi();
    signature.erase(signature.begin() + 16, signature.begin() + 18);
    expectRefused<Arch>(signature, "the stride-less form is a different ABI");
}

template <typename Arch>
void refusesAScalarInAPointerSlot()
{
    expectRefused<Arch>(withArgument<Arch>(1, KernelArgument{BY_VALUE_KIND, SCALAR_BYTES, 0, ""}),
                        "query_ptr is a global buffer");
}

template <typename Arch>
void refusesAPointerInAScalarSlot()
{
    // scale as a buffer: the kernel would take the float's bit pattern for an address.
    expectRefused<Arch>(
        withArgument<Arch>(FIRST_SCALAR_INDEX, KernelArgument{BUFFER_KIND, POINTER_BYTES, 0, ""}),
        "scale is passed by value");
}

template <typename Arch>
void refusesBuffersAndScalarsTransposed()
{
    // The eleven scalars ahead of the ten pointers: the same twenty-one entries and the same
    // counts of each kind, and the comparison is positional.
    const KernelArgument buffer{BUFFER_KIND, POINTER_BYTES, 0, ""};
    const KernelArgument scalar{BY_VALUE_KIND, SCALAR_BYTES, 0, ""};
    std::vector<KernelArgument> transposed(11, scalar);
    transposed.insert(transposed.end(), 10, buffer);
    expectRefused<Arch>(transposed, "the ten pointers come first");
}

template <typename Arch>
void refusesANarrowedPointerSlot()
{
    // block_tables_ptr as a 32-bit buffer: every argument after it would be read from the
    // wrong offset.
    expectRefused<Arch>(withArgument<Arch>(5, KernelArgument{BUFFER_KIND, SCALAR_BYTES, 0, ""}),
                        "a device pointer is eight bytes");
}

template <typename Arch>
void refusesAWidenedScalarSlot()
{
    // block_table_stride as an i64: the next four bytes of the segment would be its high half.
    expectRefused<Arch>(withArgument<Arch>(16, KernelArgument{BY_VALUE_KIND, POINTER_BYTES, 0, ""}),
                        "the strides are 32-bit");
}

template <typename Arch>
void refusesEachSlotUnderAnotherName()
{
    const auto abi = Arch::pythonAbi();
    for(std::size_t index = 0; index < abi.size(); ++index)
    {
        auto renamed = abi;
        renamed.at(index).name += "_renamed";
        expectRefused<Arch>(renamed, abi.at(index).name);
    }
}

/// Kind and size agree position by position, so the names are the only thing that separates
/// each of these from the shipped ABI. Accepted, the kernel reads keys as queries, or the
/// block table as the sequence lengths, with no fault and no status.
template <typename Arch>
void refusesEachExchangeOfTwoLikeSlots()
{
    const auto abi = Arch::pythonAbi();
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
            expectRefused<Arch>(swapped, abi.at(first).name + " <-> " + abi.at(second).name);
            ++exchanged;
        }
    }
    // Ten pointers and eleven scalars, every one of each group the same kind and size:
    // C(10, 2) + C(11, 2) pairs. A signature that grew or lost a slot changes this count.
    constexpr std::size_t POINTER_PAIRS = 45;
    constexpr std::size_t SCALAR_PAIRS = 55;
    EXPECT_EQ(exchanged, POINTER_PAIRS + SCALAR_PAIRS);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing::unified2d_signature

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
