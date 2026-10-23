// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>

#include "Gfx942AttentionUnified3dTestGraphs.hpp"

/**
 * @file TestGfx942AttentionUnified3dSignature.cpp
 * @brief Pins the two kernarg ABIs hipkernel:Gfx942AttentionUnified3d marshals against the
 *        ABIs the rocKE builder declares.
 *
 * segment = `_3d_signature(dtype, strided_kv=True)` and reduce = `_reduce_signature(dtype)`
 * (`rocke/library/kernels/common/attention_unified.py`): 19 base arguments then 8
 * strided-K/V arguments, and 5 pointers. The lists below are written out independently of
 * the pack; hkp_pack records the builder's own parameter names
 * (descriptor-packaging/tests/rocke/test_hkp_pack_gfx942_unified_3d_rocke.py), and
 * prepare() compares both launches' recorded lists before loading either.
 *
 * A wrong count, order, kind or width is not a launch error: the kernel reads the kernarg
 * segment as the types it was compiled for. requireSignatureMatch compares kind and size,
 * and name where both sides carry one; offsets are never compared (zero below).
 *
 * No device, no compile, no launch: the archive the fixture names does not exist, so an
 * accepted list fails later, at the loader, with a different message.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing::gfx942_unified_3d
{
namespace
{

using hipdnn_plugin_sdk::ingestor::KernelArgument;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::KernelSource;
using hipdnn_plugin_sdk::ingestor::KernelSourceKind;

constexpr const char* BUFFER = "global_buffer";
constexpr const char* BY_VALUE = "by_value";

/// `_3d_signature(dtype, strided_kv=True)`, by name and width.
std::vector<KernelArgument> pythonSegmentAbi()
{
    std::vector<KernelArgument> abi;
    for(const char* name : {"segm_output_ptr",
                            "segm_max_ptr",
                            "segm_expsum_ptr",
                            "query_ptr",
                            "key_cache_ptr",
                            "value_cache_ptr",
                            "sink_ptr",
                            "block_tables_ptr",
                            "seq_lens_ptr",
                            "alibi_slopes_ptr",
                            "qq_bias_ptr",
                            "query_start_len_ptr"})
    {
        abi.push_back(KernelArgument{BUFFER, 8, 0, name});
    }
    for(const char* name : {"scale",
                            "k_scale",
                            "v_scale",
                            "softcap",
                            "num_seqs",
                            "block_table_stride",
                            "qq_bias_stride_0"})
    {
        abi.push_back(KernelArgument{BY_VALUE, 4, 0, name});
    }
    for(const std::string prefix : {"k", "v"})
    {
        abi.push_back(KernelArgument{BY_VALUE, 8, 0, prefix + "_stride_batch_bytes"});
        abi.push_back(KernelArgument{BY_VALUE, 8, 0, prefix + "_stride_head_bytes"});
        abi.push_back(KernelArgument{BY_VALUE, 4, 0, prefix + "_stride_token_bytes"});
        abi.push_back(KernelArgument{BY_VALUE, 4, 0, prefix + "_span_bytes"});
    }
    return abi;
}

/// `_reduce_signature(dtype)` without LSE.
std::vector<KernelArgument> pythonReduceAbi()
{
    return {KernelArgument{BUFFER, 8, 0, "output_ptr"},
            KernelArgument{BUFFER, 8, 0, "segm_output_ptr"},
            KernelArgument{BUFFER, 8, 0, "segm_max_ptr"},
            KernelArgument{BUFFER, 8, 0, "segm_expsum_ptr"},
            KernelArgument{BUFFER, 8, 0, "seq_lens_ptr"}};
}

constexpr const char* SEGMENT_SYMBOL = "attention_unified_3d_segment_test";
constexpr const char* REDUCE_SYMBOL = "attention_unified_3d_reduce_test";

KernelSource launchSource(const char* symbol, std::vector<KernelArgument> signature)
{
    KernelSource source;
    source.kind = KernelSourceKind::KPACK;
    source.library = "there-is-no-archive-here.kpack";
    source.tocKey = "toc#0";
    source.symbol = symbol;
    source.signature = std::move(signature);
    return source;
}

/// A two-launch record of the base graph's cell recording @p segment and @p reduce.
KernelDefinition makeTwoLaunchKernel(std::vector<KernelArgument> segment,
                                     std::vector<KernelArgument> reduce)
{
    KernelDefinition kernel = makeKernel(KernelSpec{});
    kernel.sources = {{"segment", launchSource(SEGMENT_SYMBOL, std::move(segment))},
                      {"reduce", launchSource(REDUCE_SYMBOL, std::move(reduce))}};
    kernel.originDirectory = "/nonexistent";
    kernel.treeRoot = "/nonexistent";
    return kernel;
}

/// The message prepare() throws for this record; empty if it returned.
std::string prepareFailure(const KernelDefinition& kernel)
{
    const auto graph = fixture(GraphSpec{});
    const auto bound = matchesGraph(PACK, graph.context());
    EXPECT_TRUE(bound.has_value());
    if(!bound.has_value())
    {
        return {};
    }
    try
    {
        static_cast<void>(dispatchHandler(PACK).prepare(graph.context(), *bound, kernel));
    }
    catch(const std::exception& error)
    {
        return error.what();
    }
    return {};
}

constexpr const char* MISMATCH_MARKER = "is packaged with arguments";

void expectRefused(const KernelDefinition& kernel, const char* symbol, const std::string& what)
{
    const std::string failure = prepareFailure(kernel);
    ASSERT_FALSE(failure.empty()) << what << ": prepare() did not fail at all";
    EXPECT_NE(failure.find(MISMATCH_MARKER), std::string::npos) << what << ": " << failure;
    EXPECT_NE(failure.find(std::string("symbol '") + symbol + "'"), std::string::npos)
        << what << " was refused under the wrong launch: " << failure;
}

void expectSegmentRefused(const std::vector<KernelArgument>& segment, const std::string& what)
{
    expectRefused(makeTwoLaunchKernel(segment, pythonReduceAbi()), SEGMENT_SYMBOL, what);
}

void expectReduceRefused(const std::vector<KernelArgument>& reduce, const std::string& what)
{
    expectRefused(makeTwoLaunchKernel(pythonSegmentAbi(), reduce), REDUCE_SYMBOL, what);
}

TEST(TestGfx942AttentionUnified3dSignature, TheAbisHaveTheDeclaredShape)
{
    EXPECT_EQ(pythonSegmentAbi().size(), 27U); // 19 base + 8 strided-KV
    EXPECT_EQ(pythonReduceAbi().size(), 5U);
}

/// The positive control: both lists accepted, so prepare() reaches the (absent) archive.
TEST(TestGfx942AttentionUnified3dSignature, AcceptsTheAbisThePythonDeclares)
{
    const std::string failure
        = prepareFailure(makeTwoLaunchKernel(pythonSegmentAbi(), pythonReduceAbi()));
    ASSERT_FALSE(failure.empty()) << "the archive does not exist, so prepare() cannot succeed";
    EXPECT_EQ(failure.find(MISMATCH_MARKER), std::string::npos)
        << "the pack refused the argument lists the rocKE builder declares: " << failure;
}

TEST(TestGfx942AttentionUnified3dSignature, RefusesTheNonStridedSegmentAbi)
{
    // `_3d_signature(dtype)` without strided_kv: the paged kernel's 19 arguments.
    auto paged = pythonSegmentAbi();
    paged.resize(19);
    expectSegmentRefused(paged, "the paged (non-strided) segment ABI");
}

TEST(TestGfx942AttentionUnified3dSignature, RefusesTheLseReduceAbi)
{
    auto withLse = pythonReduceAbi();
    withLse.push_back(KernelArgument{BUFFER, 8, 0, "lse_ptr"});
    expectReduceRefused(withLse, "the LSE reduce ABI");
}

TEST(TestGfx942AttentionUnified3dSignature, RefusesEachSegmentSlotWidthChange)
{
    const auto abi = pythonSegmentAbi();
    for(std::size_t index = 0; index < abi.size(); ++index)
    {
        auto changed = abi;
        changed.at(index).size = changed.at(index).size == 8 ? 4 : 8;
        expectSegmentRefused(changed, abi.at(index).name + " width");
    }
}

TEST(TestGfx942AttentionUnified3dSignature, RefusesEachSlotUnderAnotherName)
{
    const auto segment = pythonSegmentAbi();
    for(std::size_t index = 0; index < segment.size(); ++index)
    {
        auto renamed = segment;
        renamed.at(index).name += "_renamed";
        expectSegmentRefused(renamed, segment.at(index).name);
    }
    const auto reduce = pythonReduceAbi();
    for(std::size_t index = 0; index < reduce.size(); ++index)
    {
        auto renamed = reduce;
        renamed.at(index).name += "_renamed";
        expectReduceRefused(renamed, reduce.at(index).name);
    }
}

TEST(TestGfx942AttentionUnified3dSignature, RefusesEachExchangeOfTwoLikeSlots)
{
    const auto check = [](const std::vector<KernelArgument>& abi, bool segment) {
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
                if(segment)
                {
                    expectSegmentRefused(exchanged, what);
                }
                else
                {
                    expectReduceRefused(exchanged, what);
                }
            }
        }
    };
    check(pythonSegmentAbi(), true);
    check(pythonReduceAbi(), false);
}

TEST(TestGfx942AttentionUnified3dSignature, RefusesTheTwoLaunchesSwapped)
{
    // The reduce's list recorded for the segment launch and vice versa.
    expectRefused(makeTwoLaunchKernel(pythonReduceAbi(), pythonSegmentAbi()),
                  SEGMENT_SYMBOL,
                  "launch lists swapped");
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine::testing::gfx942_unified_3d

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
