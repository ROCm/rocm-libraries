// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>

#include "engines/kernel_ingestor_engine/CompanionKernel.hpp"

/**
 * @file TestCompanionKernel.cpp
 * @brief companionKernelDefinition: the descriptor of a kernel packaged in another kernel's
 *        code object, as `<role>_symbol` / `<role>_signature` metadata.
 *
 * The metadata strings below are byte-for-byte what hkp_pack.companions writes (compact
 * JSON in kernel_source.signature's shape), so a drift on either side turns these red.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using hipdnn_plugin_sdk::HipdnnPluginException;

constexpr const char* MERGE_SIGNATURE
    = R"([{"kind":"global_buffer","size":8,"offset":0,"name":"partial_o"},)"
      R"({"kind":"global_buffer","size":8,"offset":8,"name":"o_ptr"},)"
      R"({"kind":"by_value","size":4,"offset":16,"name":"num_splits"}])";

KernelDefinition makeDecodeKernel()
{
    KernelDefinition kernel;
    kernel.kernelId.fill(0x31);
    kernel.packId.fill(0x32);
    kernel.dispatchId.fill(0x33);
    kernel.name = "decode_with_merge";
    kernel.source.kind = KernelSourceKind::KPACK;
    kernel.source.library = "../kpack/hip_kernel_provider_gfx1100.kpack";
    kernel.source.tocKey = "wmma_fmha_decode_0123456789abcdef";
    kernel.source.symbol = "rocke_decode";
    kernel.source.sha256 = std::string(64, 'a');
    kernel.source.signature = {KernelArgument{"global_buffer", 8, 0, "q_ptr"},
                               KernelArgument{"by_value", 4, 8, "seqlen_k"}};
    kernel.metadata["dtype"] = std::string("FP16");
    kernel.metadata["merge_symbol"] = std::string("rocke_decode_merge");
    kernel.metadata["merge_signature"] = std::string(MERGE_SIGNATURE);
    kernel.priority = 7;
    kernel.arch = {"gfx1100"};
    kernel.originDirectory = "/tree/gfx1100/rocKE/decode";
    kernel.treeRoot = "/tree";
    return kernel;
}

void expectInvalidValue(const KernelDefinition& kernel,
                        const std::string& role,
                        const std::string& fragment)
{
    try
    {
        (void)companionKernelDefinition(kernel, role);
        FAIL() << "expected companionKernelDefinition to refuse role '" << role << "'";
    }
    catch(const HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        EXPECT_NE(std::string(error.what()).find(fragment), std::string::npos) << error.what();
    }
}

TEST(TestCompanionKernel, NamesTheRoleKeyedMetadataFields)
{
    EXPECT_EQ(companionSymbolField("merge"), "merge_symbol");
    EXPECT_EQ(companionSignatureField("merge"), "merge_signature");
}

TEST(TestCompanionKernel, SwapsOnlyTheSymbolAndSignature)
{
    const KernelDefinition decode = makeDecodeKernel();
    const KernelDefinition merge = companionKernelDefinition(decode, "merge");

    EXPECT_EQ(merge.source.symbol, "rocke_decode_merge");
    ASSERT_EQ(merge.source.signature.size(), 3u);
    EXPECT_EQ(merge.source.signature[0].kind, "global_buffer");
    EXPECT_EQ(merge.source.signature[0].size, 8u);
    EXPECT_EQ(merge.source.signature[0].offset, 0u);
    EXPECT_EQ(merge.source.signature[0].name, "partial_o");
    EXPECT_EQ(merge.source.signature[1].name, "o_ptr");
    EXPECT_EQ(merge.source.signature[2].kind, "by_value");
    EXPECT_EQ(merge.source.signature[2].size, 4u);
    EXPECT_EQ(merge.source.signature[2].offset, 16u);
    EXPECT_EQ(merge.source.signature[2].name, "num_splits");

    // Same code object: the companion resolves out of the module the primary loaded.
    EXPECT_EQ(merge.source.kind, KernelSourceKind::KPACK);
    EXPECT_EQ(merge.source.library, decode.source.library);
    EXPECT_EQ(merge.source.tocKey, decode.source.tocKey);
    EXPECT_EQ(merge.source.sha256, decode.source.sha256);
    EXPECT_EQ(merge.kernelId, decode.kernelId);
    EXPECT_EQ(merge.name, decode.name);
    EXPECT_EQ(merge.originDirectory, decode.originDirectory);
    EXPECT_EQ(merge.treeRoot, decode.treeRoot);
    EXPECT_TRUE(merge.metadata == decode.metadata);

    // The primary is untouched.
    EXPECT_EQ(decode.source.symbol, "rocke_decode");
    EXPECT_EQ(decode.source.signature.size(), 2u);
}

TEST(TestCompanionKernel, AcceptsAnArgumentWithoutAName)
{
    KernelDefinition decode = makeDecodeKernel();
    decode.metadata["merge_signature"]
        = std::string(R"([{"kind":"global_buffer","size":8,"offset":0}])");
    const KernelDefinition merge = companionKernelDefinition(decode, "merge");
    ASSERT_EQ(merge.source.signature.size(), 1u);
    EXPECT_TRUE(merge.source.signature[0].name.empty());
}

TEST(TestCompanionKernel, AcceptsAnEmptyArgumentList)
{
    KernelDefinition decode = makeDecodeKernel();
    decode.metadata["merge_signature"] = std::string("[]");
    EXPECT_TRUE(companionKernelDefinition(decode, "merge").source.signature.empty());
}

TEST(TestCompanionKernel, RefusesARoleTheObjectWasNotPackagedWith)
{
    expectInvalidValue(makeDecodeKernel(), "reduce", "reduce_symbol");
}

TEST(TestCompanionKernel, RefusesAnEmptySymbol)
{
    KernelDefinition decode = makeDecodeKernel();
    decode.metadata["merge_symbol"] = std::string();
    expectInvalidValue(decode, "merge", "merge_symbol");
}

TEST(TestCompanionKernel, RefusesANonStringSymbol)
{
    KernelDefinition decode = makeDecodeKernel();
    decode.metadata["merge_symbol"] = int64_t{1};
    expectInvalidValue(decode, "merge", "merge_symbol");
}

TEST(TestCompanionKernel, RefusesAMissingSignature)
{
    KernelDefinition decode = makeDecodeKernel();
    decode.metadata.erase("merge_signature");
    expectInvalidValue(decode, "merge", "merge_signature");
}

TEST(TestCompanionKernel, RefusesASignatureThatIsNotJson)
{
    KernelDefinition decode = makeDecodeKernel();
    decode.metadata["merge_signature"] = std::string("global_buffer:8@0");
    expectInvalidValue(decode, "merge", "not valid JSON");
}

TEST(TestCompanionKernel, RefusesASignatureOfTheWrongShape)
{
    KernelDefinition decode = makeDecodeKernel();
    decode.metadata["merge_signature"]
        = std::string(R"([{"kind":"global_buffer","size":-8,"offset":0}])");
    expectInvalidValue(decode, "merge", "merge_signature");

    decode.metadata["merge_signature"] = std::string(R"({"kind":"global_buffer"})");
    expectInvalidValue(decode, "merge", "merge_signature");

    decode.metadata["merge_signature"]
        = std::string(R"([{"kind":"global_buffer","size":8,"offset":0,"align":8}])");
    expectInvalidValue(decode, "merge", "merge_signature");
}

TEST(TestCompanionKernel, RefusesAKernelWithoutACodeObject)
{
    KernelDefinition decode = makeDecodeKernel();
    decode.source.kind = KernelSourceKind::EMBEDDED_SOURCE;
    expectInvalidValue(decode, "merge", "not a kpack kernel");
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
