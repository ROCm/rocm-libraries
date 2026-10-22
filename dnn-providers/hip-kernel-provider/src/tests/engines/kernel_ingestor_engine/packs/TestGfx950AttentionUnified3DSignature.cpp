// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "tests/engines/kernel_ingestor_engine/packs/Gfx950AttentionUnified3DTestGraphs.hpp"

/**
 * @file TestGfx950AttentionUnified3DSignature.cpp
 * @brief Pins the kernarg ABIs hipkernel:Gfx950AttentionUnified3D marshals for its two
 *        kernels against the ABIs the rocKE builders declare.
 *
 * A wrong argument count, order, kind or width is not a launch error: the kernel reads the
 * kernarg segment as the types it was compiled for, so the failure is misread memory with a
 * success status. Two kernels share three scratch pointers here, in different positions, so
 * a slot carried over from one list to the other is easy to write.
 *
 * Names are the only thing that tells two pointers apart (every pointer is an 8-byte
 * `global_buffer`), so the exchanges below are refused by name alone. No device, no compile
 * and no launch: the archive does not exist, and every refusal is raised on the host.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

using namespace unified3d;

constexpr std::size_t SEGMENT_ARGUMENTS = 19;
constexpr std::size_t REDUCE_ARGUMENTS = 5;
constexpr const char* MISMATCH_MARKER = "is packaged with arguments";

std::string prepareFailureFor(const std::vector<KernelArgument>& segment,
                              const std::vector<KernelArgument>& reduce)
{
    const Graph graph;
    const auto bound = matchGraph(graph);
    EXPECT_TRUE(bound.has_value());
    if(!bound.has_value())
    {
        return {};
    }
    return prepareFailure(graph, *bound, makeKpackKernel(KernelSpec{}, segment, reduce));
}

void expectRefused(const std::vector<KernelArgument>& segment,
                   const std::vector<KernelArgument>& reduce,
                   const std::string& what)
{
    const std::string failure = prepareFailureFor(segment, reduce);
    ASSERT_FALSE(failure.empty()) << what << ": prepare() did not fail at all";
    EXPECT_NE(failure.find(MISMATCH_MARKER), std::string::npos)
        << what << ", but the failure was: " << failure;
}

std::vector<KernelArgument>
    swapped(std::vector<KernelArgument> signature, std::size_t a, std::size_t b)
{
    std::swap(signature.at(a), signature.at(b));
    return signature;
}

} // namespace

/// The positive control: both lists as the builders declare them are accepted, and prepare()
/// fails only at the loader, past both comparisons.
TEST(TestGfx950AttentionUnified3DSignature, AcceptsTheAbisThePythonDeclares)
{
    ASSERT_EQ(pythonSegmentAbi().size(), SEGMENT_ARGUMENTS);
    ASSERT_EQ(pythonReduceAbi().size(), REDUCE_ARGUMENTS);
    const std::string failure = prepareFailureFor(pythonSegmentAbi(), pythonReduceAbi());
    ASSERT_FALSE(failure.empty());
    EXPECT_EQ(failure.find(MISMATCH_MARKER), std::string::npos) << failure;
    EXPECT_NE(failure.find("does not exist"), std::string::npos) << failure;
}

TEST(TestGfx950AttentionUnified3DSignature, RefusesASegmentListOfTheWrongLength)
{
    auto shorter = pythonSegmentAbi();
    shorter.pop_back();
    expectRefused(shorter, pythonReduceAbi(), "eighteen arguments is not the segment ABI");

    auto longer = pythonSegmentAbi();
    longer.push_back(KernelArgument{"by_value", 4, 0, ""});
    expectRefused(longer, pythonReduceAbi(), "twenty arguments is not the segment ABI");
}

/// Unified2D's 18-argument list leads with output_ptr; the segment kernel leads with the
/// three scratch pointers and has no out_scale. A descriptor naming a 2D kernel must not be
/// launched as a segment kernel.
TEST(TestGfx950AttentionUnified3DSignature, RefusesTheUnified2DListAsASegmentKernel)
{
    const auto buffer
        = [](const char* name) { return KernelArgument{"global_buffer", 8, 0, name}; };
    const auto scalar = [](const char* name) { return KernelArgument{"by_value", 4, 0, name}; };
    const std::vector<KernelArgument> unified2d{buffer("output_ptr"),
                                                buffer("query_ptr"),
                                                buffer("key_cache_ptr"),
                                                buffer("value_cache_ptr"),
                                                buffer("sink_ptr"),
                                                buffer("block_tables_ptr"),
                                                buffer("seq_lens_ptr"),
                                                buffer("alibi_slopes_ptr"),
                                                buffer("qq_bias_ptr"),
                                                buffer("query_start_len_ptr"),
                                                scalar("scale"),
                                                scalar("k_scale"),
                                                scalar("v_scale"),
                                                scalar("out_scale"),
                                                scalar("softcap"),
                                                scalar("num_seqs"),
                                                scalar("block_table_stride"),
                                                scalar("qq_bias_stride_0")};
    expectRefused(unified2d, pythonReduceAbi(), "the 2D kernel is not a segment kernel");
}

/// segm_max and segm_expsum are both f32 buffers of one size; exchanged, the reduce would
/// weight by the sums and divide by the maxima.
TEST(TestGfx950AttentionUnified3DSignature, RefusesExchangedSegmentScratchPointers)
{
    expectRefused(swapped(pythonSegmentAbi(), 1, 2), pythonReduceAbi(), "segm_max/segm_expsum");
}

TEST(TestGfx950AttentionUnified3DSignature, RefusesExchangedKeyAndValueCaches)
{
    expectRefused(swapped(pythonSegmentAbi(), 4, 5), pythonReduceAbi(), "key/value cache");
}

/// num_seqs and block_table_stride are both i32 scalars: exchanged, the kernel would stride
/// the table by the batch.
TEST(TestGfx950AttentionUnified3DSignature, RefusesExchangedIntegerScalars)
{
    expectRefused(swapped(pythonSegmentAbi(), 16, 17), pythonReduceAbi(), "num_seqs/stride");
}

/// The reduce kernel's list is checked too, before either archive is opened: a companion
/// that disagrees is refused even though the segment kernel's list is right.
TEST(TestGfx950AttentionUnified3DSignature, RefusesAReduceListOfTheWrongLength)
{
    auto shorter = pythonReduceAbi();
    shorter.pop_back();
    expectRefused(pythonSegmentAbi(), shorter, "four arguments is not the reduce ABI");
}

TEST(TestGfx950AttentionUnified3DSignature, RefusesTheReduceOutputMovedOffItsSlot)
{
    expectRefused(pythonSegmentAbi(), swapped(pythonReduceAbi(), 0, 1), "output/segm_output");
}

TEST(TestGfx950AttentionUnified3DSignature, RefusesTheSegmentListAsTheCompanion)
{
    expectRefused(pythonSegmentAbi(), pythonSegmentAbi(), "the segment kernel is not a reduce");
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
