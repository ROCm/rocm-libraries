// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn_corpus_gen/DeclaredOracle.hpp>
#include <hipdnn_corpus_gen/GraphSize.hpp>
#include <hipdnn_corpus_gen/KernelCatalogSource.hpp>
#include <hipdnn_corpus_gen/OperationMetadata.hpp>
#include <hipdnn_corpus_gen/PoolAssembly.hpp>

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>

/// @file TestKernelCatalogSource.cpp
/// @brief What a descriptor pack contributes to a corpus, and what it is allowed to lose.
///
/// A deterministic pack (one kernel per geometry) must contribute all its geometries; the
/// legitimate losses are shapes this tool cannot name, cannot build, or cannot time.

using namespace hipdnn_corpus_gen;

namespace
{

OperationMetadata shippedSdpa()
{
    const std::string path
        = std::string(HIPDNN_CORPUS_GEN_OPERATIONS_DIR) + "/sdpa_fwd.opmeta.json";
    std::ifstream file(path);
    EXPECT_TRUE(file.is_open()) << path;
    const auto parsed = parseOperationMetadata(nlohmann::json::parse(file));
    EXPECT_TRUE(parsed.ok()) << (parsed.errors.empty() ? "" : parsed.errors.front());
    return parsed.ok() ? *parsed.metadata : OperationMetadata{};
}

/// One descriptor, spelled as a real pack spells it (`num_query_heads`, `seqlen_kv`,
/// `head_size`), so the declaration's `kernel_catalog` block must map these names.
nlohmann::json descriptor(int64_t batch,
                          int64_t heads,
                          int64_t headsKv,
                          int64_t seqlenQ,
                          int64_t seqlenK,
                          int64_t headDim,
                          const nlohmann::json& causal,
                          const std::string& dtype,
                          int64_t blockM = 128)
{
    return nlohmann::json{{"metadata",
                           {{"batch", batch},
                            {"num_query_heads", heads},
                            {"num_kv_heads", headsKv},
                            {"seqlen_q", seqlenQ},
                            {"seqlen_kv", seqlenK},
                            {"head_size", headDim},
                            {"causal", causal},
                            {"dtype", dtype},
                            {"block_m", blockM}}}};
}

/// A scratch tree under the working directory: /tmp is not always writable in containers.
class TempTree
{
public:
    explicit TempTree(const std::string& name)
        : _root(std::filesystem::current_path() / ("kdp_test_" + name))
    {
        std::error_code ignored;
        std::filesystem::remove_all(_root, ignored);
        std::filesystem::create_directories(_root, ignored);
    }
    TempTree(const TempTree&) = delete;
    TempTree& operator=(const TempTree&) = delete;
    ~TempTree()
    {
        std::error_code ignored;
        std::filesystem::remove_all(_root, ignored);
    }

    const std::filesystem::path& root() const
    {
        return _root;
    }

    /// @brief Writes @p descriptors as a pack at @p relative, creating any parent directories.
    std::filesystem::path pack(const std::string& relative, const nlohmann::json& descriptors)
    {
        const auto path = _root / relative;
        std::error_code ignored;
        std::filesystem::create_directories(path.parent_path(), ignored);
        std::ofstream file(path);
        file << nlohmann::json{{"kernelDescriptors", descriptors}}.dump(2);
        return path;
    }

private:
    std::filesystem::path _root;
};

} // namespace

TEST(TestKernelCatalogSource, ADeterministicPackContributesEveryGeometryItCarries)
{
    // One kernel per geometry: exactly the engine `predict_engine` exists for.
    TempTree tree("deterministic");
    const auto path = tree.pack("dense.kdp.json",
                                nlohmann::json::array({
                                    descriptor(1, 32, 8, 1, 512, 128, 1, "BF16"),
                                    descriptor(1, 32, 8, 1, 2048, 128, 1, "BF16"),
                                    descriptor(1, 32, 8, 512, 512, 128, 1, "BF16"),
                                    descriptor(1, 32, 32, 2048, 2048, 128, 0, "FP16"),
                                }));

    const auto harvest = fromPack(shippedSdpa(), path);

    EXPECT_EQ(harvest.report.kernels, 4);
    EXPECT_EQ(harvest.report.geometries, 4);
    EXPECT_EQ(harvest.report.eligible, 4);
    EXPECT_EQ(harvest.entries.size(), 4u);
    EXPECT_TRUE(harvest.report.shutOut.empty()) << harvest.report.shutOut;

    // Reported only; no geometry is dropped for it.
    EXPECT_TRUE(harvest.report.deterministic);
    EXPECT_EQ(harvest.report.maxCandidates, 1);
}

TEST(TestKernelCatalogSource, TheDeclarationSuppliesTheFieldNamesAndTheEnumSpellings)
{
    // `BF16` maps through the `enums` table and `causal: 1` becomes a bool.
    TempTree tree("vocabulary");
    const auto path = tree.pack(
        "dense.kdp.json", nlohmann::json::array({descriptor(2, 16, 4, 128, 4096, 64, 1, "BF16")}));

    const auto harvest = fromPack(shippedSdpa(), path);

    ASSERT_EQ(harvest.entries.size(), 1u);
    const auto& point = harvest.entries.front().point;
    EXPECT_EQ(std::get<int64_t>(point.at("batch")), 2);
    EXPECT_EQ(std::get<int64_t>(point.at("heads")), 16);
    EXPECT_EQ(std::get<int64_t>(point.at("heads_kv")), 4);
    EXPECT_EQ(std::get<int64_t>(point.at("seqlen_q")), 128);
    EXPECT_EQ(std::get<int64_t>(point.at("seqlen_k")), 4096);
    EXPECT_EQ(std::get<int64_t>(point.at("head_dim")), 64);
    EXPECT_TRUE(std::get<bool>(point.at("is_causal")));
    EXPECT_EQ(std::get<std::string>(point.at("dtype")), "bf16");

    // Both are manifest columns.
    EXPECT_EQ(harvest.entries.front().source, "kernel");
    EXPECT_NE(harvest.entries.front().origin.find("dense.kdp.json"), std::string::npos);
    EXPECT_EQ(harvest.entries.front().regime, "append_long_gqa");
}

TEST(TestKernelCatalogSource, AParameterThePackNeverMentionsComesFromTheDeclaredConstant)
{
    // Packs never mention `alignment`; it comes from `kernel_catalog.constants`, without which
    // every geometry would be unbuildable.
    TempTree tree("constants");
    const auto path = tree.pack(
        "dense.kdp.json", nlohmann::json::array({descriptor(1, 32, 8, 128, 4096, 128, 1, "BF16")}));

    const auto harvest = fromPack(shippedSdpa(), path);

    ASSERT_EQ(harvest.entries.size(), 1u) << harvest.report.shutOut;
    EXPECT_EQ(harvest.report.unbuildable, 0) << harvest.report.firstBuildError;
    EXPECT_EQ(std::get<std::string>(harvest.entries.front().point.at("alignment")), "top_left");
}

TEST(TestKernelCatalogSource, TwoSpellingsOfOneShapeAreOneProblemToMeasure)
{
    // `causal: 1` and `causal: true` are the same graph and must not be measured twice.
    TempTree tree("spellings");
    const auto path = tree.pack("dense.kdp.json",
                                nlohmann::json::array({
                                    descriptor(1, 8, 8, 512, 512, 64, 1, "BF16", 128),
                                    descriptor(1, 8, 8, 512, 512, 64, true, "BF16", 64),
                                }));

    const auto harvest = fromPack(shippedSdpa(), path);

    EXPECT_EQ(harvest.report.kernels, 2);
    EXPECT_EQ(harvest.report.geometries, 1);
    EXPECT_EQ(harvest.entries.size(), 1u);

    EXPECT_EQ(harvest.report.maxCandidates, 2);
    EXPECT_FALSE(harvest.report.deterministic);
}

TEST(TestKernelCatalogSource, APackDescribingNoShapesIsReadAndStaysSilent)
{
    // `pointwise_add` and `tiled_attention` carry no shape fields; no flag would help, so
    // nothing is reported.
    TempTree tree("shapeless");
    const auto path = tree.pack("pointwise_add.kdp.json",
                                nlohmann::json::array({
                                    {{"metadata", {{"block_size", 256}, {"dtype", "BF16"}}}},
                                    {{"metadata", {{"block_size", 512}, {"dtype", "BF16"}}}},
                                    {{"metadata", {{"block_size", 1024}, {"dtype", "FP16"}}}},
                                }));

    const auto harvest = fromPack(shippedSdpa(), path);

    EXPECT_EQ(harvest.report.kernels, 3);
    EXPECT_EQ(harvest.report.geometries, 1);
    EXPECT_EQ(harvest.report.noGeometry, 1);
    EXPECT_EQ(harvest.report.eligible, 0);
    EXPECT_TRUE(harvest.entries.empty());
    EXPECT_TRUE(harvest.report.shutOut.empty()) << harvest.report.shutOut;
}

TEST(TestKernelCatalogSource, AValueTheDeclarationDoesNotMapIsCountedAndNamedRatherThanGuessed)
{
    // Guessing a mapping would mislabel every row.
    TempTree tree("unmapped");
    const auto path = tree.pack(
        "fp8.kdp.json", nlohmann::json::array({descriptor(1, 8, 8, 512, 512, 64, 1, "FP8")}));

    const auto harvest = fromPack(shippedSdpa(), path);

    EXPECT_EQ(harvest.report.geometries, 1);
    EXPECT_EQ(harvest.report.unmappedValue, 1);
    EXPECT_EQ(harvest.report.eligible, 0);
    EXPECT_NE(harvest.report.shutOut.find("kernel_catalog.enums"), std::string::npos)
        << harvest.report.shutOut;
}

TEST(TestKernelCatalogSource, AGeometryTooLargeToBenchmarkIsCountedApartFromOneThatCannotBuild)
{
    // Both drop the graph, but only the ceiling is fixable by a flag, so they are reported
    // separately.
    const auto metadata = shippedSdpa();

    const ProblemPoint point{{"batch", int64_t{1}},
                             {"heads", int64_t{8}},
                             {"heads_kv", int64_t{8}},
                             {"seqlen_q", int64_t{512}},
                             {"seqlen_k", int64_t{512}},
                             {"head_dim", int64_t{64}},
                             {"is_causal", true},
                             // Built directly; a pack gets this from `kernel_catalog.constants`.
                             {"alignment", std::string("top_left")},
                             {"generate_stats", false},
                             {"dtype", std::string("bf16")}};
    const auto built = buildAdmissible(metadata, point);
    ASSERT_TRUE(built.has_value());
    const auto footprint = graphBytes(*built);
    ASSERT_GT(footprint, 0);

    TempTree tree("budget");
    const auto path = tree.pack(
        "dense.kdp.json", nlohmann::json::array({descriptor(1, 8, 8, 512, 512, 64, 1, "BF16")}));

    const auto tight = fromPack(metadata, path, footprint - 1);
    EXPECT_EQ(tight.report.overByteBudget, 1);
    EXPECT_EQ(tight.report.unbuildable, 0);
    EXPECT_EQ(tight.report.eligible, 0);
    EXPECT_NE(tight.report.shutOut.find("--max-bytes"), std::string::npos) << tight.report.shutOut;

    // Proves the refusal was the ceiling.
    const auto roomy = fromPack(metadata, path, footprint);
    EXPECT_EQ(roomy.report.eligible, 1);
    EXPECT_EQ(roomy.report.overByteBudget, 0);
    EXPECT_TRUE(roomy.report.shutOut.empty()) << roomy.report.shutOut;
}

TEST(TestKernelCatalogSource, AGeometryTheDeclarationCannotBuildIsCountedAndTheFirstErrorKept)
{
    // A catalog missing a field the builder needs is a declaration defect that costs every
    // geometry, so it gets a message.
    auto metadata = shippedSdpa();
    ASSERT_EQ(metadata.kernelCatalog.fields.erase("head_dim"), 1u);

    TempTree tree("unbuildable");
    const auto path = tree.pack(
        "dense.kdp.json", nlohmann::json::array({descriptor(1, 8, 8, 512, 512, 64, 1, "BF16")}));

    const auto harvest = fromPack(metadata, path);

    EXPECT_EQ(harvest.report.geometries, 1);
    EXPECT_EQ(harvest.report.unbuildable, 1);
    EXPECT_EQ(harvest.report.eligible, 0);
    EXPECT_FALSE(harvest.report.firstBuildError.empty());
    EXPECT_NE(harvest.report.shutOut.find("builds a graph"), std::string::npos)
        << harvest.report.shutOut;
}

TEST(TestKernelCatalogSource, AnOperationDeclaringNoKernelCatalogHasNoKernelPool)
{
    // Reading packs is opt-in per declaration; there is no operation list to maintain.
    const auto load = parseOperationMetadata(nlohmann::json::parse(R"({
      "schema_version": "0.1",
      "operation": "toy",
      "stratification_axis": "arithmetic_intensity",
      "graph_builder": {
        "function": "createValidLayernormFpropGraph",
        "source": "hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp",
        "arguments": []
      },
      "parameters": {"groups": {"type": "int64"}}
    })"));
    ASSERT_TRUE(load.ok()) << (load.errors.empty() ? "" : load.errors.front());

    TempTree tree("nocatalog");
    const auto path = tree.pack(
        "dense.kdp.json", nlohmann::json::array({descriptor(1, 8, 8, 512, 512, 64, 1, "BF16")}));

    const auto harvest = fromPack(*load.metadata, path);
    EXPECT_TRUE(harvest.entries.empty());
    EXPECT_EQ(harvest.report.kernels, 0);
    EXPECT_EQ(harvest.report.geometries, 0);
    EXPECT_TRUE(harvest.report.shutOut.empty());
}

TEST(TestKernelCatalogSource, PacksAreFoundByStructureAndInAStableOrder)
{
    // Discovered by extension, not name; order is fixed so corpora reproduce across machines.
    TempTree tree("discovery");
    const auto gfx942 = tree.pack("gfx942/attention.kdp.json", nlohmann::json::array());
    const auto gfx950 = tree.pack("gfx950/attention.kdp.json", nlohmann::json::array());
    tree.pack("gfx950/notes.json", nlohmann::json::array());

    const auto found = discoverPacks({tree.root()});
    ASSERT_EQ(found.size(), 2u);
    EXPECT_EQ(found[0].filename().string(), "attention.kdp.json");
    EXPECT_NE(found[0].string().find("gfx942"), std::string::npos);
    EXPECT_NE(found[1].string().find("gfx950"), std::string::npos);
    EXPECT_EQ(found, discoverPacks({tree.root()}));

    // Naming a file twice (directly and via its root) contributes it once.
    EXPECT_EQ(discoverPacks({gfx950}).size(), 1u);
    EXPECT_EQ(discoverPacks({tree.root(), gfx942}).size(), 2u);
}

TEST(TestKernelCatalogSource, ACollectedPoolIsSpreadAcrossRegimesRatherThanLeftInPackOrder)
{
    // Packs are written one arch/dtype/head size at a time, so pack order would bias any
    // prefix cut.
    TempTree tree("spread");
    tree.pack("a/decode.kdp.json",
              nlohmann::json::array({
                  descriptor(1, 32, 8, 1, 4096, 128, 1, "BF16"),
                  descriptor(2, 32, 8, 1, 4096, 128, 1, "BF16"),
                  descriptor(4, 32, 8, 1, 4096, 128, 1, "BF16"),
              }));
    tree.pack("b/prefill.kdp.json",
              nlohmann::json::array({
                  descriptor(1, 32, 32, 512, 512, 128, 1, "BF16"),
              }));

    const auto collected = collectPacks(shippedSdpa(), discoverPacks({tree.root()}));
    const auto& entries = collected.first;
    const auto& reports = collected.second;

    ASSERT_EQ(entries.size(), 4u);
    EXPECT_EQ(reports.size(), 2u);
    EXPECT_NE(reports[0].pack.find("decode.kdp.json"), std::string::npos);
    EXPECT_NE(reports[1].pack.find("prefill.kdp.json"), std::string::npos);

    // The lone prefill geometry is not last: the smaller population is interleaved, so a prefix
    // of this pool still contains it.
    EXPECT_NE(entries.back().regime, "prefill_short_mha");
    EXPECT_EQ(
        std::count_if(entries.begin(),
                      entries.end(),
                      [](const PoolEntry& entry) { return entry.regime == "prefill_short_mha"; }),
        1);
}
