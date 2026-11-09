// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "PackedKernelSource.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/LaunchKernelDefinition.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/ConvFwdTestGraphs.hpp"

/**
 * @file TestLaunchKernelDefinition.cpp
 * @brief launchKernelDefinition: the definition of one launch of a kernel whose UKD carries a
 *        `sources` map.
 *
 * The first suite is pure data: what the copy takes from the launch's entry, what it keeps
 * and what it refuses. The second takes the result through buildIngestorKernelCode against a
 * code object this build actually packed, which is what shows two launches resolving out of
 * one loaded module.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hipdnn_plugin_sdk::ingestor;
using hipdnn_plugin_sdk::HipdnnPluginException;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

// ---------------------------------------------------------------------------
// The definition as data
// ---------------------------------------------------------------------------

KernelSource makeLaunchSource(const std::string& symbol, std::vector<KernelArgument> signature)
{
    KernelSource source;
    source.kind = KernelSourceKind::KPACK;
    source.library = "../kpack/hip_kernel_provider_gfx1100.kpack";
    source.tocKey = "wmma_fmha_decode_0123456789abcdef";
    source.symbol = symbol;
    source.sha256 = std::string(64, 'a');
    source.signature = std::move(signature);
    return source;
}

KernelDefinition makeDecodeKernel()
{
    KernelDefinition kernel;
    kernel.kernelId.fill(0x31);
    kernel.packId.fill(0x32);
    kernel.dispatchId.fill(0x33);
    kernel.name = "decode_with_merge";
    kernel.sources.emplace("decode",
                           makeLaunchSource("rocke_decode",
                                            {KernelArgument{"global_buffer", 8, 0, "q_ptr"},
                                             KernelArgument{"by_value", 4, 8, "seqlen_k"}}));
    kernel.sources.emplace("merge",
                           makeLaunchSource("rocke_decode_merge",
                                            {KernelArgument{"global_buffer", 8, 0, "partial_o"},
                                             KernelArgument{"global_buffer", 8, 8, "o_ptr"},
                                             KernelArgument{"by_value", 4, 16, "num_splits"}}));
    kernel.sources.emplace("reduce",
                           makeLaunchSource("rocke_decode_reduce",
                                            {KernelArgument{"global_buffer", 8, 0, "lse_ptr"}}));
    kernel.metadata["dtype"] = std::string("FP16");
    kernel.priority = 7;
    kernel.arch = {"gfx1100"};
    kernel.originDirectory = "/tree/gfx1100/rocKE/decode";
    kernel.treeRoot = "/tree";
    return kernel;
}

/// The message of the INVALID_VALUE @p kernel's launch @p launch is refused with.
std::string refusalOf(const KernelDefinition& kernel, const std::string& launch)
{
    try
    {
        (void)launchKernelDefinition(kernel, launch);
    }
    catch(const HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        return error.what();
    }
    ADD_FAILURE() << "expected launchKernelDefinition to refuse launch '" << launch << "'";
    return {};
}

TEST(TestLaunchKernelDefinition, TakesTheLaunchsEntryAsItsSource)
{
    const KernelDefinition decode = makeDecodeKernel();
    const KernelDefinition merge = launchKernelDefinition(decode, "merge");

    EXPECT_EQ(merge.source.kind, KernelSourceKind::KPACK);
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
    EXPECT_EQ(merge.source.library, decode.sources.at("merge").library);
    EXPECT_EQ(merge.source.tocKey, decode.sources.at("merge").tocKey);
    EXPECT_EQ(merge.source.sha256, decode.sources.at("merge").sha256);

    // The rest is the kernel's own.
    EXPECT_EQ(merge.kernelId, decode.kernelId);
    EXPECT_EQ(merge.packId, decode.packId);
    EXPECT_EQ(merge.dispatchId, decode.dispatchId);
    EXPECT_EQ(merge.name, decode.name);
    EXPECT_EQ(merge.priority, decode.priority);
    EXPECT_EQ(merge.arch, decode.arch);
    EXPECT_EQ(merge.originDirectory, decode.originDirectory);
    EXPECT_EQ(merge.treeRoot, decode.treeRoot);
    EXPECT_TRUE(merge.metadata == decode.metadata);

    // The kernel is untouched.
    EXPECT_EQ(decode.sources.size(), 3u);
    EXPECT_TRUE(decode.source.symbol.empty());
}

TEST(TestLaunchKernelDefinition, ResolvesTheLaunchItWasAskedFor)
{
    const KernelDefinition reduce = launchKernelDefinition(makeDecodeKernel(), "reduce");
    const KernelDefinition decode = launchKernelDefinition(makeDecodeKernel(), "decode");

    EXPECT_EQ(reduce.source.symbol, "rocke_decode_reduce");
    ASSERT_EQ(reduce.source.signature.size(), 1u);
    EXPECT_EQ(reduce.source.signature[0].name, "lse_ptr");
    EXPECT_EQ(decode.source.symbol, "rocke_decode");
    EXPECT_EQ(decode.source.signature.size(), 2u);
}

TEST(TestLaunchKernelDefinition, TwoLaunchNamesMayShareOneEntryAndEachGetsItsOwnCopy)
{
    KernelDefinition kernel = makeDecodeKernel();
    const KernelSource shared = kernel.sources.at("decode");
    kernel.sources.at("reduce") = shared;

    for(const char* launch : {"decode", "reduce"})
    {
        SCOPED_TRACE(launch);
        KernelDefinition definition;
        ASSERT_NO_THROW(definition = launchKernelDefinition(kernel, launch));

        EXPECT_EQ(definition.source.symbol, shared.symbol);
        EXPECT_EQ(definition.source.sha256, shared.sha256);
        ASSERT_EQ(definition.source.signature.size(), shared.signature.size());
        for(size_t i = 0; i < shared.signature.size(); ++i)
        {
            EXPECT_EQ(definition.source.signature[i].kind, shared.signature[i].kind) << i;
            EXPECT_EQ(definition.source.signature[i].size, shared.signature[i].size) << i;
            EXPECT_EQ(definition.source.signature[i].offset, shared.signature[i].offset) << i;
            EXPECT_EQ(definition.source.signature[i].name, shared.signature[i].name) << i;
        }
    }

    // The kernel's own entries are untouched by either resolution.
    EXPECT_EQ(kernel.sources.size(), 3u);
    EXPECT_EQ(kernel.sources.at("decode").symbol, shared.symbol);
    EXPECT_EQ(kernel.sources.at("reduce").symbol, shared.symbol);
}

TEST(TestLaunchKernelDefinition, ClearsTheLaunchesOnTheCopy)
{
    const KernelDefinition merge = launchKernelDefinition(makeDecodeKernel(), "merge");

    // One kernel to load, not a kernel with launches of its own: the copy cannot be asked for
    // a launch in turn.
    EXPECT_TRUE(merge.sources.empty());
    EXPECT_FALSE(refusalOf(merge, "reduce").empty());
}

TEST(TestLaunchKernelDefinition, RefusesALaunchTheKernelDoesNotHaveNamingTheOnesItDoes)
{
    const std::string what = refusalOf(makeDecodeKernel(), "split");

    EXPECT_NE(what.find("'split'"), std::string::npos) << what;
    EXPECT_NE(what.find("'decode'"), std::string::npos) << what;
    EXPECT_NE(what.find("'merge'"), std::string::npos) << what;
    EXPECT_NE(what.find("'reduce'"), std::string::npos) << what;
}

TEST(TestLaunchKernelDefinition, RefusesASingleLaunchKernel)
{
    KernelDefinition single;
    single.name = "single";
    single.source = makeLaunchSource("rocke_single", {});

    const std::string what = refusalOf(single, "merge");

    EXPECT_NE(what.find("single-launch"), std::string::npos) << what;
    EXPECT_NE(what.find("'merge'"), std::string::npos) << what;
}

/// A library that does not exist is enough: the signature comparison runs before the load,
/// so a launch that reaches the loader has already failed to be refused.
TEST(TestLaunchKernelDefinition, ChecksTheLaunchsOwnSignatureBeforeLoadingItsArchive)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory scratch
        = hipdnn_test_sdk::utilities::claimScratchDirectory("launchkerneldefinition");
    hip_kernel_provider::compilation::KpackModuleCache cache;
    const hip_kernel_provider::compilation::KpackKernelLoader loader{cache};
    const GraphFixture graph(buildConvFwdGraph(data_objects::DataType::HALF), DeviceProperties{});

    KernelDefinition kernel = makeDecodeKernel();
    kernel.originDirectory = scratch.path();
    kernel.treeRoot = scratch.path();
    for(auto& entry : kernel.sources)
    {
        entry.second.library = "absent.kpack";
    }

    // What the pack marshals for either launch: the decode entry's recorded list. The merge
    // entry records three arguments, so only it disagrees.
    const std::vector<KernelArgument> expected = kernel.sources.at("decode").signature;

    try
    {
        static_cast<void>(buildIngestorKernelCode(
            loader, graph.context(), launchKernelDefinition(kernel, "merge"), expected));
        FAIL() << "expected the merge launch's disagreeing signature to be refused";
    }
    catch(const HipdnnPluginException& error)
    {
        const std::string what = error.what();
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        EXPECT_NE(what.find("rocke_decode_merge"), std::string::npos) << what;
        EXPECT_NE(what.find("this pack launches it with"), std::string::npos) << what;
        EXPECT_EQ(what.find("does not exist"), std::string::npos) << what;
    }

    try
    {
        static_cast<void>(buildIngestorKernelCode(
            loader, graph.context(), launchKernelDefinition(kernel, "decode"), expected));
        FAIL() << "expected the absent archive to be what fails, not the signature";
    }
    catch(const HipdnnPluginException& error)
    {
        // Reaching the loader is the pass: the agreeing launch got past the comparison.
        EXPECT_NE(std::string(error.what()).find("does not exist"), std::string::npos)
            << error.what();
    }
}

// ---------------------------------------------------------------------------
// Against a code object this build packed
// ---------------------------------------------------------------------------

/// The descriptor the packed conv set stages for the kernel the test takes its archive,
/// toc_key and digest from. Read out of the built file rather than written here: a copy
/// would silently decouple this test from the artifact it exists to read.
constexpr const char* PACKED_UKD_DESCRIPTOR = "conv_fwd_f16_block64.ukd.json";

/// The two entry points that descriptor's translation unit exports. Both must match
/// test_descriptors/shared/conv_fwd/kernels/ConvFwd.cpp. Only the first is named by a
/// descriptor; the second exists so one code object carries two symbols.
constexpr const char* FIRST_SYMBOL = "ConvFwd";
constexpr const char* SECOND_SYMBOL = "ConvFwdSecondSymbol";

/// The launch names the test gives the two entry points.
constexpr const char* FIRST_LAUNCH = "first";
constexpr const char* SECOND_LAUNCH = "second";

/// The argument list the conv pack marshals: three device pointers followed by the seven
/// int extents. Mirrored here from TestConvFwdPack.cpp's convSignature() rather than read
/// from the pack (convFwdKernelSignature() is not visible to tests), so the list the
/// packer recorded is checked against an independent one and never against itself.
std::vector<KernelArgument> packSignature()
{
    const KernelArgument buffer{"global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
    const KernelArgument extent{"by_value", static_cast<uint32_t>(sizeof(int)), 0, ""};
    return {buffer, buffer, buffer, extent, extent, extent, extent, extent, extent, extent};
}

/// The `kernel_source.signature` the packer recorded in @p descriptor, parsed directly
/// as readPackedKernelSource parses the rest of that object. Fatal assertions: call
/// through ASSERT_NO_FATAL_FAILURE.
void readShippedSignature(const std::filesystem::path& descriptor,
                          std::vector<KernelArgument>& signature)
{
    std::ifstream in(descriptor);
    ASSERT_TRUE(in.good()) << "could not open " << descriptor;

    nlohmann::json document;
    ASSERT_NO_THROW(document = nlohmann::json::parse(in)) << descriptor;
    ASSERT_TRUE(document.contains("kernel_source")) << descriptor;
    const nlohmann::json& source = document["kernel_source"];
    ASSERT_TRUE(source.contains("signature")) << descriptor;
    ASSERT_TRUE(source["signature"].is_array()) << descriptor;

    signature.clear();
    for(const auto& entry : source["signature"])
    {
        KernelArgument argument;
        argument.kind = entry.at("kind").get<std::string>();
        argument.size = entry.at("size").get<uint32_t>();
        argument.offset = entry.at("offset").get<uint32_t>();
        if(entry.contains("name"))
        {
            argument.name = entry.at("name").get<std::string>();
        }
        signature.push_back(std::move(argument));
    }
    ASSERT_FALSE(signature.empty()) << descriptor << " records no arguments for its kernel";
}

/// One packed conv code object, one module cache and one loader per case. A shared cache
/// would let an earlier case's load answer a later case's lookup and hide the growth the
/// assertions below measure.
class TestLaunchKernelDefinitionKpack : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        std::filesystem::path packed;
        ASSERT_NO_FATAL_FAILURE(
            hip_kernel_provider::testing::findPackedArchDirectory(_properties, _arch, packed));
        if(packed.empty())
        {
            GTEST_SKIP() << "nothing was packed for this device (" << _arch
                         << "): " << hip_kernel_provider::testing::unitKpackRoot() / _arch
                         << " does not exist. Environmental -- the build packs per arch and this "
                            "device is outside GPU_TARGETS.";
        }

        hip_kernel_provider::testing::PackedKernelSource source;
        ASSERT_NO_FATAL_FAILURE(hip_kernel_provider::testing::readPackedKernelSource(
            packed, PACKED_UKD_DESCRIPTOR, source));
        ASSERT_NO_FATAL_FAILURE(
            readShippedSignature(source.originDirectory / PACKED_UKD_DESCRIPTOR, _shipped));

        // Built from the stripped arch rather than currentDeviceProperties(), which keeps
        // the feature flags hipGetDeviceProperties reports ("gfx1152:xnack-"). The packer
        // names shards with the bare arch, so the archive lookup has to be asked in that
        // spelling.
        _deviceProperties.gcnArchName = _arch;
        _deviceProperties.warpSize = _properties.warpSize;

        // Set as makeKpackConvKernel does in TestConvFwdPack.cpp: originDirectory is the
        // descriptor's own (nested) folder, and the arch root is the tree the archive sits
        // under. ConvFwdSecondSymbol has no packed descriptor of its own, and both entry
        // points take the same ten arguments, so the first's recorded list is the second's.
        KernelSource first;
        first.kind = KernelSourceKind::KPACK;
        first.library = source.library;
        first.tocKey = source.tocKey;
        first.symbol = FIRST_SYMBOL;
        first.sha256 = source.sha256;
        first.signature = _shipped;
        KernelSource second = first;
        second.symbol = SECOND_SYMBOL;

        _kernel.name = "conv_fwd_f16_kpack";
        _kernel.sources = {{FIRST_LAUNCH, first}, {SECOND_LAUNCH, second}};
        _kernel.originDirectory = source.originDirectory;
        _kernel.treeRoot = packed;
    }

    GraphFixture makeGraph() const
    {
        return GraphFixture(buildConvFwdGraph(data_objects::DataType::HALF), _deviceProperties);
    }

    hipDeviceProp_t _properties{};
    std::string _arch;
    DeviceProperties _deviceProperties;
    std::vector<KernelArgument> _shipped;
    KernelDefinition _kernel;
    hip_kernel_provider::compilation::KpackModuleCache _cache;
    hip_kernel_provider::compilation::KpackKernelLoader _loader{_cache};
};

TEST_F(TestLaunchKernelDefinitionKpack, BuildsTwoLaunchesOutOfOneModule)
{
    const GraphFixture graph = makeGraph();
    const KernelDefinition first = launchKernelDefinition(_kernel, FIRST_LAUNCH);
    const KernelDefinition second = launchKernelDefinition(_kernel, SECOND_LAUNCH);
    EXPECT_EQ(first.source.symbol, FIRST_SYMBOL);
    EXPECT_EQ(second.source.symbol, SECOND_SYMBOL);

    const size_t before = _cache.size();

    // The independent list, not the packed one: each build compares what the packer
    // recorded against what the pack marshals.
    static_cast<void>(buildIngestorKernelCode(_loader, graph.context(), first, packSignature()));
    EXPECT_EQ(_cache.size(), before + 1);
    static_cast<void>(buildIngestorKernelCode(_loader, graph.context(), second, packSignature()));

    // Two symbols, one (archive, toc_key, arch): the second launch came out of the module the
    // first loaded rather than a second load of the same blob.
    EXPECT_EQ(_cache.size(), before + 1);
}

TEST_F(TestLaunchKernelDefinitionKpack, RefusesALaunchWhoseRecordedSignatureDisagreesWithThePack)
{
    const GraphFixture graph = makeGraph();
    ASSERT_GT(_shipped.size(), 3u);

    // requireSignatureMatch compares kind and size, and names only where both sides carry
    // one, so each of these is a disagreement it can see.
    std::vector<std::vector<KernelArgument>> wrong(3, _shipped);
    wrong[0][3].kind = "global_buffer";
    wrong[1][3].size = static_cast<uint32_t>(sizeof(int64_t));
    wrong[2].pop_back();

    for(const auto& signature : wrong)
    {
        KernelDefinition kernel = _kernel;
        kernel.sources.at(SECOND_LAUNCH).signature = signature;
        const KernelDefinition first = launchKernelDefinition(kernel, FIRST_LAUNCH);
        const KernelDefinition second = launchKernelDefinition(kernel, SECOND_LAUNCH);

        // Only the second launch's entry is wrong: the first still builds.
        static_cast<void>(
            buildIngestorKernelCode(_loader, graph.context(), first, packSignature()));

        try
        {
            static_cast<void>(
                buildIngestorKernelCode(_loader, graph.context(), second, packSignature()));
            ADD_FAILURE() << "expected the second launch's wrong signature to be refused";
        }
        catch(const HipdnnPluginException& error)
        {
            const std::string what = error.what();
            EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
            EXPECT_NE(what.find(std::string("'") + SECOND_SYMBOL + "'"), std::string::npos) << what;
            EXPECT_EQ(what.find(std::string("symbol '") + FIRST_SYMBOL + "'"), std::string::npos)
                << what;
        }

        // The refusal's ordering against the load is pinned device-free by
        // ChecksTheLaunchsOwnSignatureBeforeLoadingItsArchive: here the module is cached by
        // the first launch, so the cache could not show it.
    }
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
