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
/// The behaviour under test is the one the retired Python kernel pool got wrong:
/// a pack whose every geometry is claimed by exactly one kernel used to contribute nothing, so
/// the engine that most needs measuring -- a deterministic one, whose matcher pins every field
/// -- got a corpus with none of its own shapes in it. The first test here is that regression,
/// and the rest are the losses that ARE real: a shape this tool cannot name, one it cannot
/// build, one too large to time.

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

/// One descriptor, spelled the way a real pack spells it -- `num_query_heads`, `seqlen_kv`,
/// `head_size` -- so the declaration's `kernel_catalog` block is what makes this readable and a
/// change to those names is a test failure rather than a silent empty pool.
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

/// A scratch tree under the test's own working directory rather than under TMPDIR: these runs
/// happen inside containers where /tmp is not always writable, and a test that cannot write is
/// indistinguishable from one that failed.
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

    const std::filesystem::path& root() const { return _root; }

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
    // The regression. Four geometries, one kernel each: the matcher pins every distinguishing
    // field, so kernel identity is a total function of the problem. That is precisely the engine
    // `predict_engine_tflops` exists for, and it cannot be measured on a corpus that excludes it.
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

    // Reported so an operator learns here rather than eight hours into a sweep, and acted on
    // nowhere: not one geometry was dropped for it.
    EXPECT_TRUE(harvest.report.deterministic);
    EXPECT_EQ(harvest.report.maxCandidates, 1);
}

TEST(TestKernelCatalogSource, TheDeclarationSuppliesTheFieldNamesAndTheEnumSpellings)
{
    // The point read back is the pack's geometry in the declaration's vocabulary: `BF16` became
    // `bf16` through the `enums` table, and `causal: 1` became a bool. Without that the graph
    // would be built for a shape nobody asked for.
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

    // The pool entry says where it came from and which population it joins, because both are
    // manifest columns and a `kernel` row with neither cannot be reconciled against the pack.
    EXPECT_EQ(harvest.entries.front().source, "kernel");
    EXPECT_NE(harvest.entries.front().origin.find("dense.kdp.json"), std::string::npos);
    EXPECT_EQ(harvest.entries.front().regime, "append_long_gqa");
}

TEST(TestKernelCatalogSource, AParameterThePackNeverMentionsComesFromTheDeclaredConstant)
{
    // A pack records what its kernels were compiled for, and `alignment` is not in that
    // vocabulary -- these kernels all anchor the causal diagonal top-left, so the pack simply
    // says `causal: 1`. The shipped declaration says so under `kernel_catalog.constants`, and
    // that is what lets the builder resolve an argument no descriptor supplies.
    //
    // Without it the failure is not an error but an absence: `$q.alignment` resolves for no
    // geometry, so every one of them is unbuildable and the kernel pool is empty.
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
    // `causal: 1` and `causal: true` are the same graph. Keying geometries on the raw metadata
    // would make them two corpus entries with identical bytes -- a duplicated measurement that
    // reads as two independent samples in the training set.
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

    // Two kernels on one geometry: a choice exists, so the pack is not deterministic.
    EXPECT_EQ(harvest.report.maxCandidates, 2);
    EXPECT_FALSE(harvest.report.deterministic);
}

TEST(TestKernelCatalogSource, APackDescribingNoShapesIsReadAndStaysSilent)
{
    // `pointwise_add` and the shipped `tiled_attention` carry `block_size`, `dtype`, `operation`
    // and nothing else. There is no flag that admits them, so naming one would be advice that
    // does not work -- and the descriptors are one population, not one geometry each.
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
    // A pack in fp8 read as bf16 would produce a corpus of graphs the engine does not serve,
    // labelled as ones it does -- every row of it wrong, and nothing downstream able to tell.
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
    // Two refusals that read identically in a corpus -- the graph is absent either way -- and
    // mean opposite things: one is a ceiling the operator can raise, the other is a declaration
    // that cannot express the shape. Reporting them together advises raising a flag that will
    // not help.
    const auto metadata = shippedSdpa();

    const ProblemPoint point{{"batch", int64_t{1}},
                             {"heads", int64_t{8}},
                             {"heads_kv", int64_t{8}},
                             {"seqlen_q", int64_t{512}},
                             {"seqlen_k", int64_t{512}},
                             {"head_dim", int64_t{64}},
                             {"is_causal", true},
                             // Supplied here because this point is built directly rather than
                             // harvested; a pack gets it from `kernel_catalog.constants`.
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

    // The same geometry at a ceiling it fits, so the refusal was the ceiling and nothing else.
    const auto roomy = fromPack(metadata, path, footprint);
    EXPECT_EQ(roomy.report.eligible, 1);
    EXPECT_EQ(roomy.report.overByteBudget, 0);
    EXPECT_TRUE(roomy.report.shutOut.empty()) << roomy.report.shutOut;
}

TEST(TestKernelCatalogSource, AGeometryTheDeclarationCannotBuildIsCountedAndTheFirstErrorKept)
{
    // A catalog mapping fewer fields than the builder needs: the geometry is read, and the graph
    // it should become references a parameter no descriptor supplied. That is an authoring
    // defect in the declaration rather than anything about the pack, and it is worth a message
    // because it costs the pack every geometry it has -- which is what distinguishes it from the
    // one-off refusals above.
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
    // How coverage stays "whatever has a declaration": an operation that has not been taught a
    // pack vocabulary reads no pack at all, rather than there being an operation list somewhere
    // that has to be kept in step with the declarations.
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
    // Packs are laid out one arch per directory and are named by whoever wrote them, so the
    // discovery is over the extension rather than over a list of expected names. Order is fixed
    // because a corpus has to reproduce across machines.
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

    // A path naming a file is that file, and naming it twice -- directly and through its root --
    // contributes it once.
    EXPECT_EQ(discoverPacks({gfx950}).size(), 1u);
    EXPECT_EQ(discoverPacks({tree.root(), gfx942}).size(), 2u);
}

TEST(TestKernelCatalogSource, ACollectedPoolIsSpreadAcrossRegimesRatherThanLeftInPackOrder)
{
    // Packs are written one arch, one dtype, one head size at a time, so a pool left in pack
    // order and cut to a budget is a corpus of whatever the first pack listed. The first few
    // entries have to be a sample of the whole.
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
    EXPECT_EQ(std::count_if(entries.begin(),
                            entries.end(),
                            [](const PoolEntry& entry) {
                                return entry.regime == "prefill_short_mha";
                            }),
              1);
}
