// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose: a future edit that puts this file back on
// an ungated source list then yields an empty translation unit rather than a suite that
// reads a staged shard nothing produced.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdNativeShared.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdNoSerialization.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdSchemaGate.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestFlydslSdpaBwdPacks.cpp
 * @brief A census of what hipkernel:flydsl_sdpa_bwd_merged actually loads, read from the shard
 *        the census entry names rather than from anything built in-process.
 *
 * The merged table is 309 objects: 189 without a bias and 120 with one. The 44 bias objects
 * that store dB carry the dBias reduce kernel as a second symbol of the same code object; a
 * store_db=0 object never writes dB and carries no reduce kernel. Every object is a
 * multi-launch pack: its `sources` hold the merged kernel and the tail-zero kernel, one of
 * three tail-zero objects shared by dtype, and a packed object records a resolved dK/dV query
 * tile within the pack bound. Every case opens by reading loadedSet(), none skips and none
 * touches a device, so an empty shard fails here instead of passing silently. The seam suite,
 * over rows written in the test, is TestFlydslSdpaBwdEngine.cpp.
 *
 * A descriptor carries the argument lists of the merged, reduce and tail-zero symbols. The reduce
 * kernel's 88-byte list (`_PARAMS["dbias_reduce"]`) is checked when the object is generated
 * and, as the engine launches it, by the seam suite; here every dB-storing bias row names the
 * reduce symbol and launches it as the `dbias_reduce` entry, which names the row's own object.
 *
 * Registered with hkp_register_census_tests() in src/tests/CMakeLists.txt, over the
 * architectures the op ships for.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

constexpr const char* ENGINE_NAME = "hipkernel:flydsl_sdpa_bwd_merged";
constexpr const char* GRAPH_MATCHER = "hipkernel.flydsl_sdpa_bwd_merged.graph_match";
constexpr const char* REDUCE_SYMBOL = "fmha_bwd_dbias_reduce_gfx950_kernel_1";
constexpr const char* TAILZERO_SYMBOL = "fmha_bwd_tailzero_gfx950_kernel_0";
constexpr const char* MERGED_LAUNCH = "merged";
constexpr const char* REDUCE_LAUNCH = "dbias_reduce";
constexpr const char* TAILZERO_LAUNCH = "tailzero";

constexpr size_t MERGED_ROWS = 309;
constexpr size_t BIAS_ROWS = 120;
constexpr size_t REDUCE_ROWS = 44;

/// A metadata field of type @p T, or nullptr when absent or of another type.
template <typename T>
const T* tryGetMetadataField(const MetadataValues& values, const char* field)
{
    const auto it = values.find(std::string(field));
    return it == values.end() ? nullptr : std::get_if<T>(&it->second);
}

TEST(TestFlydslSdpaBwdPacks, IsOnePackWithAGraphMatchAndOneKernelScopedMatcher)
{
    const auto& set = loadedSet(ENGINE_NAME);

    // One engine, two KDPs: the rows that launch the dBias reduce share a launch set, so they are a pack of their own.
    EXPECT_EQ(distinctPackIdCount(set), 2U);
    EXPECT_EQ(set.engine.graphMatchNativeSymbol, GRAPH_MATCHER);
    size_t graphScoped = 0;
    size_t kernelScoped = 0;
    for(const auto& matcher : set.matchers)
    {
        (matcher.scope == MatchScope::GRAPH ? graphScoped : kernelScoped) += 1;
    }
    EXPECT_EQ(graphScoped, 0U);
    EXPECT_EQ(kernelScoped, 1U);
}

TEST(TestFlydslSdpaBwdPacks, ShipsEveryMergedRowOnce)
{
    const auto& set = loadedSet(ENGINE_NAME);

    const std::set<int64_t> tiles{32, 64, 96, 128, 160, 192, 224, 256, 384, 512};
    const std::set<std::string> dtypes{"f16", "bf16"};
    const std::set<std::string> dforms{"rt", "bk"};
    const std::set<std::string> roleOrders{"auto", "concat", "lpt"};

    std::set<std::string> names;
    size_t kernels = 0;
    size_t biasRows = 0;
    size_t reduceRows = 0;
    std::map<std::string, std::set<std::string>> tailzeroByDtype;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            ++kernels;
            names.insert(kernel.name);
            // A multi-launch pack: `source` is left default beside `sources`. A row that names
            // a reduce symbol launches the reduce between the merged kernel and the tail-zero.
            const auto* reduceSymbol
                = tryGetMetadataField<std::string>(kernel.metadata, "reduce_symbol");
            const bool launchesReduce = reduceSymbol != nullptr && !reduceSymbol->empty();
            EXPECT_EQ(kernel.sources.size(), launchesReduce ? 3U : 2U) << kernel.name;
            if(launchesReduce)
            {
                const auto reduce = kernel.sources.find(REDUCE_LAUNCH);
                ASSERT_NE(reduce, kernel.sources.end())
                    << kernel.name << " lacks the reduce launch";
                EXPECT_EQ(reduce->second.symbol, *reduceSymbol) << kernel.name;
                // The reduce is a second kernel of the row's own object.
                const auto merged = kernel.sources.find(MERGED_LAUNCH);
                ASSERT_NE(merged, kernel.sources.end());
                EXPECT_EQ(reduce->second.sha256, merged->second.sha256) << kernel.name;
                EXPECT_EQ(reduce->second.tocKey, merged->second.tocKey) << kernel.name;
            }
            for(const char* launch : {MERGED_LAUNCH, TAILZERO_LAUNCH})
            {
                const auto entry = kernel.sources.find(launch);
                ASSERT_NE(entry, kernel.sources.end())
                    << kernel.name << " lacks the launch " << launch;
                EXPECT_EQ(entry->second.kind, KernelSourceKind::KPACK)
                    << kernel.name << " " << launch;
                EXPECT_FALSE(entry->second.sha256.empty()) << kernel.name << " " << launch;
            }
            std::string tailzeroSha;
            if(const auto zero = kernel.sources.find(TAILZERO_LAUNCH); zero != kernel.sources.end())
            {
                EXPECT_EQ(zero->second.symbol, TAILZERO_SYMBOL) << kernel.name;
                tailzeroSha = zero->second.sha256;
            }

            const auto integer = [&](const char* field) {
                const auto* value = tryGetMetadataField<int64_t>(kernel.metadata, field);
                EXPECT_NE(value, nullptr) << kernel.name << " lacks the integer field " << field;
                return value == nullptr ? int64_t{0} : *value;
            };
            const auto text = [&](const char* field) {
                const auto* value = tryGetMetadataField<std::string>(kernel.metadata, field);
                EXPECT_NE(value, nullptr) << kernel.name << " lacks the string field " << field;
                return value == nullptr ? std::string() : *value;
            };
            EXPECT_EQ(dtypes.count(text("dtype")), 1U) << kernel.name;
            tailzeroByDtype[text("dtype")].insert(tailzeroSha);
            EXPECT_EQ(tiles.count(integer("head_dim")), 1U) << kernel.name;
            EXPECT_GT(integer("head_dim_v"), 0) << kernel.name;
            EXPECT_EQ(dforms.count(text("dform")), 1U) << kernel.name;
            EXPECT_EQ(roleOrders.count(text("role_order")), 1U) << kernel.name;
            EXPECT_FALSE(text("floor").empty()) << kernel.name;
            for(const char* flag : {"causal", "has_bias", "has_dropout", "padded", "hints"})
            {
                const auto value = integer(flag);
                EXPECT_TRUE(value == 0 || value == 1) << kernel.name << " " << flag;
            }
            const auto storeDb = integer("store_db");
            EXPECT_TRUE(storeDb == -1 || storeDb == 0) << kernel.name;
            const auto gqaPack = integer("gqa_pack");
            EXPECT_TRUE(gqaPack == -1 || gqaPack == 1) << kernel.name;
            // A packed row records the resolved tile the pack bound applies to (C2).
            const auto blockQ = integer("block_q");
            EXPECT_TRUE(gqaPack != 1
                        || (blockQ > 0 && blockQ <= flydsl_sdpa_bwd_native::PACK_MAX_BLOCK_Q))
                << kernel.name << " block_q " << blockQ;
            for(const char* geometry : {"block_threads", "block_kv", "block_m"})
            {
                EXPECT_GT(integer(geometry), 0) << kernel.name << " " << geometry;
            }

            // A dB-storing bias object carries the dBias reduce kernel; the class is exclusive
            // of causal.
            const bool hasBias = integer("has_bias") == 1;
            const bool storesDb = hasBias && storeDb != 0;
            EXPECT_FALSE(hasBias && integer("causal") == 1) << kernel.name;
            EXPECT_EQ(text("reduce_symbol"), storesDb ? std::string(REDUCE_SYMBOL) : std::string())
                << kernel.name;
            biasRows += hasBias ? 1 : 0;
            reduceRows += storesDb ? 1 : 0;
        }
    }

    EXPECT_EQ(kernels, MERGED_ROWS);
    EXPECT_EQ(names.size(), MERGED_ROWS) << "a name is shipped twice";
    EXPECT_EQ(biasRows, BIAS_ROWS);
    EXPECT_EQ(reduceRows, REDUCE_ROWS);
    EXPECT_EQ(tailzeroByDtype.size(), dtypes.size()) << "one tail-zero entry per row dtype";
    for(const auto& [dtype, objects] : tailzeroByDtype)
    {
        // The f16 and bf16 objects are byte-identical (they store 16-bit zeros): entries of
        // two dtypes may resolve to one object, so objects are not counted.
        EXPECT_EQ(objects.size(), 1U) << dtype << ": one tail-zero object per dtype entry";
        EXPECT_EQ(objects.count(std::string()), 0U) << dtype << ": the entry names no object";
    }
}

TEST(TestFlydslSdpaBwdPacks, EveryKernelDeclaresTheMergedSignature)
{
    const auto& set = loadedSet(ENGINE_NAME);

    struct Run
    {
        const char* kind;
        uint32_t size;
        size_t count;
    };
    // Each argument at its natural offset.
    using Layout = std::pair<std::vector<std::pair<const char*, uint32_t>>, std::vector<uint32_t>>;
    const auto layoutOf = [](const std::vector<Run>& runs) {
        Layout layout;
        uint32_t offset = 0;
        for(const auto& run : runs)
        {
            for(size_t item = 0; item < run.count; ++item)
            {
                offset = (offset + run.size - 1) / run.size * run.size;
                layout.first.emplace_back(run.kind, run.size);
                layout.second.push_back(offset);
                offset += run.size;
            }
        }
        return std::pair{layout, offset};
    };
    // `_PARAMS["merged"]`: eleven pointers (Q K V B O DO DK DV DQ DB LSE), batch_size, four
    // seqinfo pointers, the i64 multipliers of Q and K, an (offset pointer, i64 multiplier)
    // pair for each of V, O, dO, dQ, dK and dV, the stats' offset pointer, multiplier and three
    // i64 strides, six i32 (varlen_bits .. window_right), the two philox pointers, the i64
    // offset, i32 threshold, f32 scale, four i32 (heads, head dims), f32 sm_scale, thirty i64
    // strides and i32 rt_flags: 82 arguments, 604 bytes.
    std::vector<Run> merged{
        {"global_buffer", 8, 11}, {"by_value", 4, 1}, {"global_buffer", 8, 4}, {"by_value", 8, 2}};
    for(size_t pair = 0; pair < 6; ++pair)
    {
        merged.push_back({"global_buffer", 8, 1});
        merged.push_back({"by_value", 8, 1});
    }
    for(const Run& run : {Run{"global_buffer", 8, 1},
                          Run{"by_value", 8, 4},
                          Run{"by_value", 4, 6},
                          Run{"global_buffer", 8, 2},
                          Run{"by_value", 8, 1},
                          Run{"by_value", 4, 1},
                          Run{"by_value", 4, 1},
                          Run{"by_value", 4, 4},
                          Run{"by_value", 4, 1},
                          Run{"by_value", 8, 30},
                          Run{"by_value", 4, 1}})
    {
        merged.push_back(run);
    }
    const auto [mergedLayout, mergedBytes] = layoutOf(merged);
    ASSERT_EQ(mergedLayout.first.size(), 82U);
    ASSERT_EQ(mergedBytes, 604U);
    // `_PARAMS["tailzero"]`: the dQ, dK and dV pointers, the two length pointers, seven i32
    // extents, nine i64 strides and the i32 mask: 22 arguments, 148 bytes.
    const auto [tailzeroLayout, tailzeroBytes] = layoutOf(
        {{"global_buffer", 8, 5}, {"by_value", 4, 7}, {"by_value", 8, 9}, {"by_value", 4, 1}});
    ASSERT_EQ(tailzeroLayout.first.size(), 22U);
    ASSERT_EQ(tailzeroBytes, 148U);

    const auto expectLayout = [](const std::string& name,
                                 const std::vector<KernelArgument>& signature,
                                 const Layout& layout) {
        ASSERT_EQ(signature.size(), layout.first.size()) << name;
        for(size_t slot = 0; slot < layout.first.size(); ++slot)
        {
            EXPECT_EQ(signature[slot].kind, layout.first[slot].first) << name << " slot " << slot;
            EXPECT_EQ(signature[slot].size, layout.first[slot].second) << name << " slot " << slot;
            EXPECT_EQ(signature[slot].offset, layout.second[slot]) << name << " slot " << slot;
        }
    };

    size_t kernels = 0;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            ++kernels;
            expectLayout(kernel.name, kernel.sources.at(MERGED_LAUNCH).signature, mergedLayout);
            expectLayout(kernel.name + " tailzero",
                         kernel.sources.at(TAILZERO_LAUNCH).signature,
                         tailzeroLayout);
        }
    }
    EXPECT_EQ(kernels, MERGED_ROWS);
}

// The ingestor declines a graph whose schema floor is above the engine's `sdk_version` before
// the matcher runs. The matcher serves ragged tensors, non-default alignment and ragged offset
// multipliers, so the engine declares the schema that introduced the last of them: an SDK that
// raises the multiplier floor again fails here until the matcher has been reviewed for it.
TEST(TestFlydslSdpaBwdPacks, DeclaresASchemaAtLeastTheRaggedOffsetMultiplierFloor)
{
    const auto& set = loadedSet(ENGINE_NAME);
    const auto& floor
        = hipdnn_plugin_sdk::computeMinimumEnginePluginApiVersion(false, false, false, false, true);
    EXPECT_FALSE(set.engine.sdkVersion < floor) << set.engine.sdkVersion.str();
}

// Graphs stamped as the backend stamps them reach the matcher through GenericPlanBuilder and
// through the provider's ApplicabilityManager, the engine registered at its UED's
// `sdk_version` as Container does, and the matcher's verdicts hold on both: ragged tensors with
// and without a multiplier and an aligned operand are served; a misaligned operand, ragged
// stats beside a dense Q, a ragged bias and a multiplier below 1 are declined by the matcher,
// not by a schema gate.
TEST(TestFlydslSdpaBwdPacks, ServesStampedRaggedGraphsAndDeclinesTheFieldsTheMatcherRejects)
{
    const auto& set = loadedSet(ENGINE_NAME);
    const auto failures = schema_gate::failures(
        set, hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_EXPECTED_ARCH"));
    std::string report;
    for(const auto& failure : failures)
    {
        report += "\n  " + failure;
    }
    EXPECT_TRUE(failures.empty()) << report;
    // The cases only matter if they can see each gate: an engine of the schema these engines
    // shipped with, as descriptor and as registration, declines the same graph.
    const auto invisible = schema_gate::gateIsInvisible(
        set, hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_EXPECTED_ARCH"));
    EXPECT_FALSE(invisible.has_value()) << invisible.value_or("");
}

// Pins that neither engine advertises or performs plan saving. Remove or invert when RFC 0021
// slot-program saving lands (split) or the merged dispatch handler implements save/restore.
TEST(TestFlydslSdpaBwdPacks, RefusesPlanSerialization)
{
    expectEngineRefusesPlanSerialization(loadedSet(ENGINE_NAME));
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
