// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <hipdnn_data_sdk/logging/LogLevel.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_plugin_sdk/BehaviorNote.h>
#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

/**
 * @file TestDescriptorLoader.cpp
 * @brief The descriptor loader against real files on disk.
 *
 * Every case writes its own descriptor files into a scoped directory and loads them, so
 * what is under test is the path from bytes to DescriptorSet -- not descriptor structs
 * built in memory, which is what KernelIngestorTestFixtures.hpp already covers.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;

/// Distinct from TestKernelIngestor.cpp's `int` handle: DispatchRegistry is keyed on the
/// handle type, so a private one keeps these registrations out of that suite's registry.
struct LoaderHandle
{
};

const std::string GRAPH_SYMBOL = "descriptorloader.graph_match";
const std::string KERNEL_SYMBOL = "descriptorloader.kernel_match";
const std::string SCORE_SYMBOL = "descriptorloader.score";
const std::string DISPATCH_SYMBOL = "descriptorloader.dispatch";

bool matchGraph(const MatchContext& /*context*/, BoundTokens& /*bound*/)
{
    return true;
}

bool matchKernel(const MatchContext& /*context*/, const KernelDefinition& /*kernel*/)
{
    return true;
}

double score(const KernelDefinition& /*kernel*/, const MatchContext& /*context*/)
{
    return 0.0;
}

class NoopDispatchHandler : public IKernelDispatchHandler<LoaderHandle>
{
public:
    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        return 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& /*context*/,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& /*kernel*/) const override
    {
        return std::make_unique<PreparedDispatch>();
    }

    void launch(const LoaderHandle& /*handle*/,
                const PreparedDispatch& /*prepared*/,
                const hipdnnPluginDeviceBuffer_t* /*deviceBuffers*/,
                uint32_t /*numDeviceBuffers*/,
                void* /*workspace*/) const override
    {
    }
};

/// Registers this suite's native symbols for one test's duration. NativeRegistry has
/// unregisterSymbol() for exactly this: the registry is process-wide, so a test that left
/// its symbols behind would decide the next test's answer.
class ScopedSymbols
{
public:
    ScopedSymbols()
    {
        GraphMatcherRegistry::registerSymbol(GRAPH_SYMBOL, &matchGraph);
        KernelMatcherRegistry::registerSymbol(KERNEL_SYMBOL, &matchKernel);
        ScoreRegistry::registerSymbol(SCORE_SYMBOL, &score);
        DispatchRegistry<LoaderHandle>::registerSymbol(DISPATCH_SYMBOL, &_handler);
    }

    ~ScopedSymbols()
    {
        GraphMatcherRegistry::unregisterSymbol(GRAPH_SYMBOL);
        KernelMatcherRegistry::unregisterSymbol(KERNEL_SYMBOL);
        ScoreRegistry::unregisterSymbol(SCORE_SYMBOL);
        DispatchRegistry<LoaderHandle>::unregisterSymbol(DISPATCH_SYMBOL);
    }

    ScopedSymbols(const ScopedSymbols&) = delete;
    ScopedSymbols& operator=(const ScopedSymbols&) = delete;

private:
    NoopDispatchHandler _handler;
};

/// A well-formed UUID per (set, role) pair. The loader only cares that ids differ and
/// parse, so generating them beats pasting a page of literals.
std::string testUuid(char setTag, char roleTag)
{
    std::string id = "00000000-0000-4000-8000-000000000000";
    id[0] = setTag;
    id[1] = roleTag;
    return id;
}

constexpr char ROLE_SCHEMA = '1';
constexpr char ROLE_HEURISTIC = '2';
constexpr char ROLE_ENGINE = '3';
constexpr char ROLE_GRAPH_MATCHER = '4';
constexpr char ROLE_KERNEL_MATCHER = '5';
constexpr char ROLE_DISPATCH = '6';
constexpr char ROLE_PACK = '7';
constexpr char ROLE_STANDALONE_KERNEL = 'd';

/// A document plus the type the loader will read it as. The type is no longer inside the
/// body: it comes from the filename writeDocument() builds from `suffix`.
struct TestDocument
{
    std::string_view suffix; ///< ".ued.json" etc; selects the type the loader will read
    nlohmann::json body;
};
using Documents = std::vector<TestDocument>;

/// The complete seven-file set one engine needs: a KMD, a UHD, a UED, two UMDs, a UDD,
/// and one KDP over three kernels.
Documents makeSetDocuments(char tag, const std::string& engineName)
{
    const auto schemaId = testUuid(tag, ROLE_SCHEMA);
    const auto heuristicId = testUuid(tag, ROLE_HEURISTIC);
    const auto engineId = testUuid(tag, ROLE_ENGINE);
    const auto graphMatcherId = testUuid(tag, ROLE_GRAPH_MATCHER);
    const auto kernelMatcherId = testUuid(tag, ROLE_KERNEL_MATCHER);
    const auto dispatchId = testUuid(tag, ROLE_DISPATCH);

    const auto kernel = [tag](char slot, int64_t blockSize, const std::string& dtype) {
        return nlohmann::json{{"id", testUuid(tag, slot)},
                              {"name", std::string("kernel_") + slot},
                              {"kernel_source",
                               {{"kind", "embedded_source"},
                                {"source_file", "Kernel.cpp"},
                                {"entry_point", "Entry"}}},
                              {"metadata", {{"block_size", blockSize}, {"dtype", dtype}}},
                              {"priority", 0}};
    };

    return {
        {".kmd.json",
         {{"version", "1.0"},
          {"id", schemaId},
          {"name", "variant fields"},
          {"fields",
           {{{"name", "block_size"}, {"type", "int"}, {"default_value", 64}},
            {{"name", "dtype"}, {"type", "string"}}}}}},
        {".uhd.json",
         {{"version", "1.0"},
          {"id", heuristicId},
          {"name", "selector"},
          {"kind", "native"},
          {"payload", SCORE_SYMBOL}}},
        {".ued.json",
         {{"version", "1.0"},
          {"id", engineId},
          {"name", engineName},
          {"heuristic", heuristicId},
          {"metadata", schemaId},
          {"knobs", {"block_size"}},
          {"behavior_notes", {"runtime_compilation"}}}},
        {".umd.json",
         {{"version", "1.0"},
          {"id", graphMatcherId},
          {"name", "graph shape"},
          {"scope", "graph"},
          {"match_symbol", GRAPH_SYMBOL}}},
        {".umd.json",
         {{"version", "1.0"},
          {"id", kernelMatcherId},
          {"name", "kernel dtype"},
          {"scope", "kernel"},
          {"match_symbol", KERNEL_SYMBOL}}},
        {".udd.json",
         {{"version", "1.0"},
          {"id", dispatchId},
          {"name", "dispatch"},
          {"dispatch_symbol", DISPATCH_SYMBOL}}},
        {".kdp.json",
         {{"version", "1.0"},
          {"id", testUuid(tag, ROLE_PACK)},
          {"name", "pack"},
          {"matchers", {graphMatcherId, kernelMatcherId}},
          {"engine", engineId},
          {"dispatch", dispatchId},
          {"kernelDescriptors",
           {kernel('8', 64, "FLOAT"), kernel('9', 256, "FLOAT"), kernel('a', 64, "HALF")}}}},
    };
}

void writeDocument(const std::filesystem::path& directory, const TestDocument& document)
{
    std::filesystem::create_directories(directory);
    // Stem is the id purely to keep names unique here. The loader never parses the stem,
    // which is what makes an arbitrary one the right choice for a fixture.
    std::ofstream file(
        directory / (document.body.at("id").get<std::string>() + std::string(document.suffix)),
        std::ios::binary);
    file << document.body.dump(2) << '\n';
}

void writeDocuments(const std::filesystem::path& directory, const Documents& documents)
{
    for(const auto& document : documents)
    {
        writeDocument(directory, document);
    }
}

/// The body of the first document in @p documents of type @p suffix, for a case that
/// corrupts it.
nlohmann::json& documentOfType(Documents& documents, std::string_view suffix)
{
    for(auto& document : documents)
    {
        if(document.suffix == suffix)
        {
            return document.body;
        }
    }
    throw std::runtime_error("no document of type " + std::string(suffix));
}

/// Moves the pack's last inline kernel into its own `.ukd.json`, leaving a bare-id
/// reference behind: the same corpus, authored the other way.
void referenceLastKernel(Documents& documents)
{
    auto& kernels = documentOfType(documents, ".kdp.json").at("kernelDescriptors");
    // Scans from the end rather than trusting kernels.back(): a second call (building an
    // all-references pack) must find the next remaining inline entry, not a string left
    // behind by the first call.
    for(auto i = kernels.size(); i-- > 0;)
    {
        if(kernels[i].is_object())
        {
            nlohmann::json kernel = kernels[i];
            kernel["version"] = "1.0";
            kernels[i] = kernel.at("id").get<std::string>();
            documents.push_back(TestDocument{".ukd.json", std::move(kernel)});
            return;
        }
    }
    throw std::runtime_error("no inline kernel left to reference");
}

/// The body of the *second* document in @p documents of type @p suffix. makeSetDocuments
/// emits two `.umd.json` documents -- graph scope, then kernel scope -- and
/// documentOfType always returns the first, so this is the only way to corrupt the
/// kernel-scope matcher specifically.
nlohmann::json& secondDocumentOfType(Documents& documents, std::string_view suffix)
{
    bool sawFirst = false;
    for(auto& document : documents)
    {
        if(document.suffix == suffix)
        {
            if(sawFirst)
            {
                return document.body;
            }
            sawFirst = true;
        }
    }
    throw std::runtime_error("no second document of type " + std::string(suffix));
}

/// Distinct per call, so ScopedDirectory below -- which creates the directory atomically
/// and throws if the name is taken -- always has a free one. The per-process stamp keeps
/// two suites running this binary at once out of each other's tree, and remove_all clears
/// a leftover from a killed run. std::filesystem only: mkdtemp is POSIX, and MSVC ships
/// no <unistd.h>.
std::filesystem::path uniqueDirectory(const std::string& name)
{
    static const std::string s_session
        = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    static unsigned s_counter = 0;
    const auto path
        = std::filesystem::temp_directory_path()
          / ("descriptor_loader_" + name + "_" + s_session + "_" + std::to_string(s_counter++));
    std::filesystem::remove_all(path);
    return path;
}

std::vector<DescriptorSet> loadFrom(const std::filesystem::path& root)
{
    return resolveDescriptorSets(loadDescriptorCatalog(root));
}

std::vector<DescriptorSet> loadFromRoots(const std::vector<std::filesystem::path>& roots)
{
    return resolveDescriptorSets(loadDescriptorCatalog(roots));
}

} // namespace

TEST(TestDescriptorLoader, ResolvesACompleteSetIntoOneEngine)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("complete"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:complete"));

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    const auto& set = sets.front();
    EXPECT_EQ(set.engine.name, "test:complete");
    EXPECT_EQ(set.schema.fields.size(), 2u);
    ASSERT_TRUE(set.heuristic.has_value());
    EXPECT_EQ(set.heuristic->payload, SCORE_SYMBOL);
    EXPECT_EQ(set.matchers.size(), 2u);
    EXPECT_EQ(set.dispatches.size(), 1u);
    ASSERT_EQ(set.packs.size(), 1u);
    EXPECT_EQ(set.packs.front().kernels.size(), 3u);
    ASSERT_EQ(set.engine.behaviorNotes.size(), 1u);
    EXPECT_EQ(set.engine.behaviorNotes.front(), HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION);
}

/// A UED naming no UHD ranks on priority then id instead of failing, so the SDK can adopt a
/// model later without the engine being unloadable until it does. The UHD is removed with
/// the reference: a descriptor no engine names is the orphan case, tested separately.
TEST(TestDescriptorLoader, LoadsAnEngineThatShipsNoHeuristic)
{
    const ScopedSymbols symbols;
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("no_heuristic"));
    auto documents = makeSetDocuments('1', "test:orderly");
    documentOfType(documents, ".ued.json").erase("heuristic");
    documents.erase(
        std::remove_if(documents.begin(),
                       documents.end(),
                       [](const TestDocument& document) { return document.suffix == ".uhd.json"; }),
        documents.end());
    writeDocuments(dir.path(), documents);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_FALSE(sets.front().engine.heuristicId.has_value());
    EXPECT_FALSE(sets.front().heuristic.has_value());
}

TEST(TestDescriptorLoader, CollapsesIdenticalDuplicatesAcrossArchDirectories)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("arch_dupes"));
    const auto documents = makeSetDocuments('1', "test:duplicated");
    writeDocuments(dir.path() / "gfx942", documents);
    writeDocuments(dir.path() / "gfx950", documents);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:duplicated");
}

/// The headline per-arch case: a real two-shard install ships one KDP id per arch, each
/// stamped with its own arch and built against its own shard's kernels. Keyed by id alone
/// these collide on content and both drop, taking the engine with them.
TEST(TestDescriptorLoader, KeepsPerArchPacksSharingOneIdAcrossArchDirectories)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("arch_packs"));
    for(const std::string arch : {"gfx90a", "gfx942"})
    {
        auto documents = makeSetDocuments('1', "test:sharded");
        auto& pack = documentOfType(documents, ".kdp.json");
        pack["arch"] = nlohmann::json::array({arch});
        pack.at("kernelDescriptors")[0]["kernel_source"]["source_file"] = "Kernel_" + arch + ".cpp";
        writeDocuments(dir.path() / arch, documents);
    }

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    const auto& packs = sets.front().packs;
    ASSERT_EQ(packs.size(), 2u);
    EXPECT_EQ(packs[0].arch, std::vector<std::string>{"gfx90a"});
    EXPECT_EQ(packs[1].arch, std::vector<std::string>{"gfx942"});
    EXPECT_EQ(packs[0].kernels.front().source.sourceFile, "Kernel_gfx90a.cpp");
    EXPECT_EQ(packs[1].kernels.front().source.sourceFile, "Kernel_gfx942.cpp");
}

/// Same shape one file down: each shard's `.ukd.json` carries that shard's arch, so a
/// pack resolves the copy built for its own arch rather than whichever file loaded first.
TEST(TestDescriptorLoader, KeepsPerArchStandaloneKernelsSharingOneId)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("arch_kernels"));
    for(const std::string arch : {"gfx90a", "gfx942"})
    {
        auto documents = makeSetDocuments('1', "test:sharded_kernel");
        referenceLastKernel(documents);
        documentOfType(documents, ".kdp.json")["arch"] = nlohmann::json::array({arch});
        auto& kernel = documentOfType(documents, ".ukd.json");
        kernel["arch"] = nlohmann::json::array({arch});
        kernel["kernel_source"]["source_file"] = "Kernel_" + arch + ".cpp";
        writeDocuments(dir.path() / arch, documents);
    }

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    const auto& packs = sets.front().packs;
    ASSERT_EQ(packs.size(), 2u);
    // Referenced kernels are appended after the inline ones, so the shard's own UKD is
    // last in each pack.
    ASSERT_EQ(packs[0].kernels.size(), 3u);
    ASSERT_EQ(packs[1].kernels.size(), 3u);
    EXPECT_EQ(packs[0].kernels.back().source.sourceFile, "Kernel_gfx90a.cpp");
    EXPECT_EQ(packs[1].kernels.back().source.sourceFile, "Kernel_gfx942.cpp");
}

/// The empty-arch fallback: a UKD declaring no arch is the shared definition every pack
/// reaches, which is how one kernel file serves packs of several engines.
TEST(TestDescriptorLoader, ResolvesAnArchIndependentKernelFromAnArchSpecificPack)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("shared_arch_kernel"));
    auto documents = makeSetDocuments('1', "test:shared_kernel");
    referenceLastKernel(documents);
    documentOfType(documents, ".kdp.json")["arch"] = nlohmann::json::array({"gfx942"});
    writeDocuments(dir.path(), documents);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    ASSERT_EQ(sets.front().packs.size(), 1u);
    EXPECT_EQ(sets.front().packs.front().kernels.size(), 3u);
}

/// Arch widens the key; it does not weaken the collision rule. Two files claiming one id
/// *and* one arch with different contents still poison each other.
TEST(TestDescriptorLoader, DropsBothWhenOneIdAndArchIsDefinedTwiceWithDifferentContents)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("arch_conflict"));
    auto documents = makeSetDocuments('1', "test:arch_conflict");
    documentOfType(documents, ".kdp.json")["arch"] = nlohmann::json::array({"gfx942"});
    writeDocuments(dir.path(), documents);

    auto second = documentOfType(documents, ".kdp.json");
    second["name"] = "a different pack with the same id and arch";
    std::ofstream(dir.path() / "second-claim.kdp.json", std::ios::binary) << second.dump(2);

    EXPECT_TRUE(loadFrom(dir.path()).empty());
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "both define"));
}

/// The key sorts arch, so two spellings of one target list are one identity -- and then
/// collide on content like any other repeated key. Unsorted, these would be two packs.
TEST(TestDescriptorLoader, TreatsArchOrderAsOneKey)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("arch_order"));
    auto documents = makeSetDocuments('1', "test:arch_order");
    documentOfType(documents, ".kdp.json")["arch"] = nlohmann::json::array({"gfx942", "gfx950"});
    writeDocuments(dir.path(), documents);

    auto reordered = documentOfType(documents, ".kdp.json");
    reordered["arch"] = nlohmann::json::array({"gfx950", "gfx942"});
    reordered["name"] = "the same pack, targets listed the other way";
    std::ofstream(dir.path() / "reordered.kdp.json", std::ios::binary) << reordered.dump(2);

    EXPECT_TRUE(loadFrom(dir.path()).empty());
    // Named rather than inferred from the empty result: two entries under two keys would
    // also leave nothing loadable, for an entirely different reason.
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "arch=[gfx942,gfx950]"))
        << recorder.getRecordedLogsAsString();
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "both define"));
}

/// Both spellings of one kernel id present at once: the pack's own arch is the answer,
/// and the arch-independent entry is the fallback for a pack that finds nothing else --
/// tested together, since each resolves on its own and only the order between them can
/// regress.
TEST(TestDescriptorLoader, PrefersTheKernelBuiltForThePacksArchOverTheArchIndependentOne)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("arch_precedence"));
    auto documents = makeSetDocuments('1', "test:arch_precedence");
    referenceLastKernel(documents);
    documentOfType(documents, ".kdp.json")["arch"] = nlohmann::json::array({"gfx942"});

    // The referenced kernel, a second time: same id, stamped for this pack's arch, and
    // named differently so which one resolved is visible in the result.
    auto pinned = documentOfType(documents, ".ukd.json");
    pinned["arch"] = nlohmann::json::array({"gfx942"});
    pinned["name"] = "the gfx942 build";
    documents.push_back(TestDocument{".ukd.json", pinned});
    writeDocuments(dir.path(), documents);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    ASSERT_EQ(sets.front().packs.size(), 1u);
    const auto& kernels = sets.front().packs.front().kernels;
    ASSERT_EQ(kernels.size(), 3u);
    EXPECT_EQ(kernels.back().name, "the gfx942 build");
}

/// A pack missing its shard's arch stamp names a kernel that exists -- just not under any
/// arch it claims. Reporting that as "no descriptor defines it" sends the reader hunting
/// for a missing file.
TEST(TestDescriptorLoader, DropsAPackWhoseKernelIsDefinedOnlyForAnotherArch)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("kernel_other_arch"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:valid"));

    auto broken = makeSetDocuments('2', "test:broken");
    referenceLastKernel(broken);
    documentOfType(broken, ".ukd.json")["arch"] = nlohmann::json::array({"gfx942"});
    writeDocuments(dir.path(), broken);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:valid");
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR,
                                          "names kernel " + testUuid('2', 'a')
                                              + ", which is defined only for another arch"));
}

/// The whole point of the multi-root change: a set's seven files split across two roots
/// still resolve as cross-root id references into one catalog.
TEST(TestDescriptorLoader, ResolvesADescriptorSetSplitAcrossTwoRootsIntoOneEngine)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("split_roots"));
    auto documents = makeSetDocuments('1', "test:split");
    Documents rootADocuments;
    Documents rootBDocuments;
    for(auto& document : documents)
    {
        // UED/UHD/KMD in root A, the rest -- both UMDs, the UDD, and the KDP -- in root
        // B: the engine can only be assembled by resolving ids across both roots.
        auto& target = (document.suffix == ".ued.json" || document.suffix == ".uhd.json"
                        || document.suffix == ".kmd.json")
                           ? rootADocuments
                           : rootBDocuments;
        target.push_back(std::move(document));
    }
    const auto rootA = dir.path() / "a";
    const auto rootB = dir.path() / "b";
    writeDocuments(rootA, rootADocuments);
    writeDocuments(rootB, rootBDocuments);

    const auto sets = loadFromRoots({rootA, rootB});

    ASSERT_EQ(sets.size(), 1u);
    const auto& set = sets.front();
    EXPECT_EQ(set.engine.name, "test:split");
    EXPECT_EQ(set.matchers.size(), 2u);
    EXPECT_EQ(set.dispatches.size(), 1u);
    ASSERT_EQ(set.packs.size(), 1u);
    EXPECT_EQ(set.packs.front().kernels.size(), 3u);
}

TEST(TestDescriptorLoader, CollapsesAnIdenticalDescriptorSetPresentInBothRoots)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("identical_roots"));
    const auto documents = makeSetDocuments('1', "test:mirrored");
    writeDocuments(dir.path() / "a", documents);
    writeDocuments(dir.path() / "b", documents);

    const auto sets = loadFromRoots({dir.path() / "a", dir.path() / "b"});

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:mirrored");
}

/// The drop-in rule: a later root may add descriptors, never replace one. Two files
/// disagreeing inside one root drop both, because nothing ranks them; across roots the
/// installed tree is the answer, so the newcomer is refused and the incumbent survives
/// intact -- an operator-controlled root that could delete a shipped engine would do it
/// at a severity the default log level never shows.
TEST(TestDescriptorLoader, RefusesADropInRedefiningAnInstalledId)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("cross_root_conflict"));
    const auto installed = dir.path() / "a";
    const auto dropIn = dir.path() / "b";

    auto documents = makeSetDocuments('1', "test:installed");
    writeDocuments(installed, documents);

    // Same UED id, different content, filed under the later root.
    auto& engine = documentOfType(documents, ".ued.json");
    engine["name"] = "test:redefined";
    writeDocument(dropIn, TestDocument{".ued.json", engine});

    const auto sets = loadFromRoots({installed, dropIn});

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:installed");
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "under an earlier root"))
        << recorder.getRecordedLogsAsString();
}

/// The other half: a drop-in that claims no installed id is loaded like any other file,
/// so the tree is genuinely additive rather than decorative.
TEST(TestDescriptorLoader, LoadsAnEngineThatOnlyTheDropInRootDefines)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("cross_root_add"));
    const auto installed = dir.path() / "a";
    const auto dropIn = dir.path() / "b";

    writeDocuments(installed, makeSetDocuments('1', "test:installed"));
    writeDocuments(dropIn, makeSetDocuments('2', "test:dropped_in"));

    const auto sets = loadFromRoots({installed, dropIn});

    ASSERT_EQ(sets.size(), 2u);
    EXPECT_EQ(sets.front().engine.name, "test:installed");
    EXPECT_EQ(sets.back().engine.name, "test:dropped_in");
}

/// The multi-root overload the provider actually calls, rather than the catalog helper
/// the tests above use: a drop-in descriptor has to survive symbol validation, the
/// engine-name claim, and the state-manager probe before it is an engine anyone can
/// serve. Registering the engine name is process-wide, so the names here are unique to
/// this test.
TEST(TestDescriptorLoader, ValidatesAnEngineComingOnlyFromTheDropInRoot)
{
    const ScopedSymbols symbols;
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("cross_root_validated"));
    const auto installed = dir.path() / "a";
    const auto dropIn = dir.path() / "b";

    writeDocuments(installed, makeSetDocuments('1', "test:validated_installed"));
    writeDocuments(dropIn, makeSetDocuments('2', "test:validated_drop_in"));

    const auto sets = loadValidatedDescriptorSets<LoaderHandle>({installed, dropIn});

    ASSERT_EQ(sets.size(), 2u);
    EXPECT_EQ(sets.front().engine.name, "test:validated_installed");
    EXPECT_EQ(sets.back().engine.name, "test:validated_drop_in");
}

TEST(TestDescriptorLoader, AMissingRootContributesNothingButTheOtherRootStillLoads)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("missing_root"));
    const auto goodRoot = dir.path() / "good";
    writeDocuments(goodRoot, makeSetDocuments('1', "test:present"));

    const auto sets = loadFromRoots({dir.path() / "does-not-exist", goodRoot});

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:present");
}

TEST(TestDescriptorLoader, DropsAnIdTwoFilesDisagreeAbout)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("conflict"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:survivor"));

    auto conflicted = makeSetDocuments('2', "test:conflicted");
    writeDocuments(dir.path(), conflicted);
    // Same id, different content, different filename: the file's own id is what claims
    // the entry, so this is a second definition rather than a second descriptor.
    auto& engine = documentOfType(conflicted, ".ued.json");
    engine["name"] = "test:conflicted_other";
    std::ofstream(dir.path() / "second-claim.ued.json", std::ios::binary) << engine.dump(2);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:survivor");
}

TEST(TestDescriptorLoader, LoadsNothingFromAnEmptyDirectory)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("empty"));

    EXPECT_TRUE(loadFrom(dir.path()).empty());
}

TEST(TestDescriptorLoader, LoadsNothingFromAMissingDirectory)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("missing"));

    EXPECT_TRUE(loadFrom(dir.path() / "not-there").empty());
}

TEST(TestDescriptorLoader, MalformedJsonDoesNotCostTheOtherEngine)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("malformed"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:intact"));
    // Named with a real suffix so it reaches the parser: a bare `broken.json` would be
    // skipped at the filename stage and prove nothing about malformed bodies.
    std::ofstream(dir.path() / "broken.ued.json", std::ios::binary) << "not json";

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:intact");
}

TEST(TestDescriptorLoader, IgnoresANonJsonFile)
{
    // A file whose name matches no descriptor suffix and has no `.json`/`.jsonc`
    // extension costs nothing -- distinct from SkipsAJsonFileThatNamesNoDescriptorType,
    // which is a near-miss that does warn.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("non_json"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:intact"));
    std::ofstream(dir.path() / "README.txt", std::ios::binary) << "not a descriptor";

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:intact");
}

TEST(TestDescriptorLoader, RejectsADescriptorWhoseRootIsNotAnObject)
{
    // Valid JSON, so this reaches requireObject rather than nlohmann::json::parse --
    // a different rejection path than MalformedJsonDoesNotCostTheOtherEngine above.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("non_object_root"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:intact"));
    std::ofstream(dir.path() / "not-an-object.ued.json", std::ios::binary)
        << nlohmann::json::array({1, 2, 3}).dump();

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:intact");
}

namespace
{

struct ViolationCase
{
    std::string name;
    std::function<void(Documents&)> corrupt;
};

class TestDescriptorLoaderViolation : public ::testing::TestWithParam<ViolationCase>
{
};

} // namespace

/// Every authored-format violation is rejected file by file: the engine whose descriptor
/// broke is dropped, and the valid engine sharing the directory still loads.
TEST_P(TestDescriptorLoaderViolation, RejectsTheOffenderAndKeepsTheSibling)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory(GetParam().name));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:valid"));

    auto broken = makeSetDocuments('2', "test:broken");
    GetParam().corrupt(broken);
    writeDocuments(dir.path(), broken);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:valid");
}

INSTANTIATE_TEST_SUITE_P(
    Format,
    TestDescriptorLoaderViolation,
    ::testing::Values(
        // `metadata`, not `heuristic`: a UED may now omit its UHD and rank by declared
        // order, so erasing that one no longer violates anything. Every engine still
        // names a KMD, because a pack's kernels are checked against its field list.
        ViolationCase{
            "missing_required_key",
            [](Documents& documents) { documentOfType(documents, ".ued.json").erase("metadata"); }},
        // RFC 0017 §4 names fields Descriptors.hpp does not model yet, so an authored one
        // is either a typo or a field arriving before its parsed form -- both are load
        // errors rather than something to ignore, unless they carry an extension prefix
        // (LoadsADescriptorCarryingTrackingFields covers that half).
        ViolationCase{"unknown_key",
                      [](Documents& documents) {
                          documentOfType(documents, ".ued.json")["features_signature"]
                              = nlohmann::json::array({"tensor_core"});
                      }},
        // A file's type comes from its filename alone. A `schema` member would be a second
        // spelling of that fact, so it is rejected outright rather than tolerated: two
        // sources of truth have no correct reading when they disagree.
        ViolationCase{"schema_key_is_not_a_member",
                      [](Documents& documents) {
                          documentOfType(documents, ".ued.json")["schema"] = "hipdnn.ued/v1";
                      }},
        ViolationCase{"unknown_behavior_note",
                      [](Documents& documents) {
                          documentOfType(documents, ".ued.json")["behavior_notes"]
                              = nlohmann::json::array({"teleportation"});
                      }},
        ViolationCase{"default_value_contradicts_type",
                      [](Documents& documents) {
                          documentOfType(documents, ".kmd.json").at("fields")[1]["default_value"]
                              = 5;
                      }},
        ViolationCase{"unparsable_id",
                      [](Documents& documents) {
                          documentOfType(documents, ".ued.json")["id"] = "not-a-uuid";
                      }},
        // Required on every type, with no absence-safe default: a type carrying no
        // version cannot be gated by the RFC 0020 §11.1 accept rule at all.
        ViolationCase{
            "ued_missing_version",
            [](Documents& documents) { documentOfType(documents, ".ued.json").erase("version"); }},
        // Pinned as per-type rather than UED-only.
        ViolationCase{
            "udd_missing_version",
            [](Documents& documents) { documentOfType(documents, ".udd.json").erase("version"); }},
        // RFC 0020 §11.1: `file.minor <= provider.minor`. A newer minor may carry fields
        // this build has no reader for.
        ViolationCase{"version_newer_minor",
                      [](Documents& documents) {
                          documentOfType(documents, ".ued.json")["version"] = "1.1";
                      }},
        // RFC 0020 §11.1: a major mismatch is a hard break in either direction.
        ViolationCase{"version_newer_major",
                      [](Documents& documents) {
                          documentOfType(documents, ".ued.json")["version"] = "2.0";
                      }},
        ViolationCase{"version_older_major",
                      [](Documents& documents) {
                          documentOfType(documents, ".ued.json")["version"] = "0.9";
                      }},
        // `major.minor`, exactly two numeric halves: a three-part version is the SDK's
        // `Version` spelling, not this field's, and reading it as 1.0 would accept a file
        // stamped for a generation this build never saw.
        ViolationCase{"version_three_components",
                      [](Documents& documents) {
                          documentOfType(documents, ".ued.json")["version"] = "1.0.0";
                      }},
        ViolationCase{"version_not_numeric",
                      [](Documents& documents) {
                          documentOfType(documents, ".ued.json")["version"] = "1.x";
                      }},
        // The gate is per file type, not UED-only: a KMD this build cannot read is
        // skipped, and the engine whose `metadata` named it drops with it.
        ViolationCase{"non_ued_newer_minor",
                      [](Documents& documents) {
                          documentOfType(documents, ".kmd.json")["version"] = "1.1";
                      }},
        // A JSON number where `version` must be a string.
        ViolationCase{
            "version_is_a_number",
            [](Documents& documents) { documentOfType(documents, ".ued.json")["version"] = 1.0; }},
        // `arch` must be an array; a bare string is rejected rather than treated as a
        // one-element list.
        ViolationCase{"arch_is_not_an_array",
                      [](Documents& documents) {
                          documentOfType(documents, ".kdp.json")["arch"] = "gfx942";
                      }},
        // Every `arch` entry must be a string, not just the field as a whole.
        ViolationCase{"arch_holds_a_non_string",
                      [](Documents& documents) {
                          documentOfType(documents, ".kdp.json")["arch"]
                              = nlohmann::json::array({123});
                      }},
        // Empty means arch-independent only as the whole list; an empty entry inside it
        // is an authoring mistake, not a value.
        ViolationCase{"arch_holds_an_empty_string",
                      [](Documents& documents) {
                          documentOfType(documents, ".kdp.json")["arch"]
                              = nlohmann::json::array({""});
                      }},
        // archSupports (DeviceProperties.hpp) is a case-sensitive exact compare, so a
        // typo here would otherwise silently disable the pack everywhere.
        ViolationCase{"arch_holds_a_malformed_id",
                      [](Documents& documents) {
                          documentOfType(documents, ".kdp.json")["arch"]
                              = nlohmann::json::array({"x86_64"});
                      }},
        // Only 'embedded_source' has an implementation the dispatch handler can call; any
        // other kind would pass validation and only throw inside getKernelSrc("") at
        // plan-build time, after applicability already promised the graph.
        ViolationCase{"kernel_source_kind_not_dispatchable",
                      [](Documents& documents) {
                          documentOfType(documents, ".kdp.json")
                              .at("kernelDescriptors")[0]["kernel_source"]["kind"]
                              = "hsaco_file";
                      }},
        ViolationCase{"arch_has_duplicate_entries",
                      [](Documents& documents) {
                          documentOfType(documents, ".kdp.json")["arch"]
                              = nlohmann::json::array({"gfx942", "gfx942"});
                      }}),
    [](const ::testing::TestParamInfo<ViolationCase>& info) { return info.param.name; });

TEST(TestDescriptorLoader, DropsOnlyThePackWhoseMatcherIsMissing)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("dangling_matcher"));
    auto documents = makeSetDocuments('1', "test:two_packs");

    auto danglingPack = documentOfType(documents, ".kdp.json");
    danglingPack["id"] = testUuid('1', 'b');
    danglingPack["name"] = "pack with a dangling matcher";
    danglingPack["matchers"] = nlohmann::json::array({testUuid('f', 'f')});
    documents.push_back(TestDocument{".kdp.json", danglingPack});
    writeDocuments(dir.path(), documents);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().packs.size(), 1u);
}

TEST(TestDescriptorLoader, DropsAnEngineWhoseOnlyPackIsUnresolvable)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("no_pack"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:valid"));

    auto broken = makeSetDocuments('2', "test:packless");
    documentOfType(broken, ".kdp.json")["dispatch"] = testUuid('f', 'f');
    writeDocuments(dir.path(), broken);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:valid");
}

TEST(TestDescriptorLoader, DropsAnEngineWhoseOnlyPackDeclaresNoKernels)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("empty_pack"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:valid"));

    auto broken = makeSetDocuments('2', "test:empty_pack");
    documentOfType(broken, ".kdp.json")["kernelDescriptors"] = nlohmann::json::array();
    writeDocuments(dir.path(), broken);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:valid");
}

TEST(TestDescriptorLoader, DropsAPackWhoseEngineIdNamesNoLoadedEngine)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("orphan_pack"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:valid"));

    // Reached by no per-engine scan at all, since nothing in the catalog claims this id --
    // the one place resolveDescriptorSets logs a pack rather than losing it with no trace.
    // test:orphaned's own UED is left with no pack of its own and is dropped along with it.
    auto orphaned = makeSetDocuments('2', "test:orphaned");
    documentOfType(orphaned, ".kdp.json")["engine"] = testUuid('f', 'f');
    writeDocuments(dir.path(), orphaned);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:valid");
    for(const auto& set : sets)
    {
        EXPECT_TRUE(std::none_of(set.packs.begin(), set.packs.end(), [](const auto& pack) {
            return toString(pack.id) == testUuid('2', ROLE_PACK);
        }));
    }
    // The logging block is diagnostics-only: an orphan pack is absent from every set
    // whether or not it exists, so the assertions above hold either way. This is what
    // actually pins it -- without the block, the pack vanishes with nothing to say why.
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, testUuid('2', ROLE_PACK)));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "no descriptor defines"));
}

/// findDescriptor's third outcome, distinct from missing: an id present but conflicted.
/// Losing the `|| it->second.conflicted` check there would hand the engine back
/// whichever of the two disagreeing KMDs the walk happened to insert first, so the
/// engine loads and runs against an arbitrary schema instead of dropping.
TEST(TestDescriptorLoader, DropsAnEngineWhoseMetadataSchemaIsConflicted)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("conflicted_schema"));
    auto documents = makeSetDocuments('1', "test:conflicted_schema");
    auto schema = documentOfType(documents, ".kmd.json");
    writeDocuments(dir.path(), documents);

    // Same id, different content, different filename -- a second KMD claiming the id
    // conflicts it rather than colliding on filename.
    schema["name"] = "a different schema name";
    std::ofstream(dir.path() / "second-claim.kmd.json", std::ios::binary) << schema.dump(2);

    EXPECT_TRUE(loadFrom(dir.path()).empty());
}

/// The same findDescriptor outcome, reached through the pack loop's matcher lookup
/// instead of the engine's schema lookup, so a regression to that check's other call
/// site is covered too.
TEST(TestDescriptorLoader, DropsAPackWhoseMatcherIsConflicted)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("conflicted_matcher"));
    auto documents = makeSetDocuments('1', "test:conflicted_matcher");
    auto matcher = documentOfType(documents, ".umd.json");
    writeDocuments(dir.path(), documents);

    matcher["name"] = "a different matcher name";
    std::ofstream(dir.path() / "second-claim.umd.json", std::ios::binary) << matcher.dump(2);

    EXPECT_TRUE(loadFrom(dir.path()).empty());
}

/// The nameClaims loop behind the drop-all-on-shared-name rule skips conflicted entries
/// before counting; DropsAnIdTwoFilesDisagreeAbout never puts a shared name in play,
/// since its two engines are named differently. Without that skip, a healthy engine is
/// silently taken down for merely sharing a name with a broken shard.
TEST(TestDescriptorLoader, ConflictedEngineDoesNotClaimANameItsHealthySiblingUses)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("conflicted_name_claim"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:shared_name"));

    // Two files disagree about tag '2''s UED id, so it is conflicted -- but both copies
    // also claim the same name as tag '1''s healthy engine, which is what the nameClaims
    // guard has to see through.
    auto conflicted = makeSetDocuments('2', "test:shared_name");
    writeDocuments(dir.path(), conflicted);
    auto& engine = documentOfType(conflicted, ".ued.json");
    engine["heuristic"] = testUuid('1', ROLE_HEURISTIC);
    std::ofstream(dir.path() / "second-claim.ued.json", std::ios::binary) << engine.dump(2);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(toString(sets.front().engine.id), testUuid('1', ROLE_ENGINE));
}

TEST(TestDescriptorLoader, DropsAnEngineWhoseMetadataSchemaIsMissing)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("dangling_schema"));
    auto documents = makeSetDocuments('1', "test:schemaless");
    documentOfType(documents, ".ued.json")["metadata"] = testUuid('f', 'f');
    writeDocuments(dir.path(), documents);

    EXPECT_TRUE(loadFrom(dir.path()).empty());
}

/// The applicability-descriptor analogue of DropsAnEngineWhoseMetadataSchemaIsMissing
/// above. Naming a UHD no file defines is a broken install, and stays a drop; naming none
/// at all is deliberate, and loads -- LoadsAnEngineThatShipsNoHeuristic covers that half.
TEST(TestDescriptorLoader, DropsAnEngineWhoseHeuristicIsMissing)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("dangling_heuristic"));
    auto documents = makeSetDocuments('1', "test:heuristicless");
    documentOfType(documents, ".ued.json")["heuristic"] = testUuid('f', 'f');
    writeDocuments(dir.path(), documents);

    EXPECT_TRUE(loadFrom(dir.path()).empty());
}

TEST(TestDescriptorLoader, DropsEveryEngineClaimingTheSameEngineId)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("id_collision"));
    // Two independent descriptor sets whose engine names hash to the same hipDNN engine
    // id. Two distinct names colliding under FNV-1a is not something a test can
    // construct, so the same name in two sets stands in: the check is on the hashed id,
    // and both reach it the same way.
    writeDocuments(dir.path(), makeSetDocuments('1', "test:same_name"));
    writeDocuments(dir.path(), makeSetDocuments('2', "test:same_name"));

    // RFC 0020 §10.2.1: not keep-the-first. Directory order decides which set is seen
    // first, so keeping one would make the surviving definition a property of the
    // filesystem.
    EXPECT_TRUE(loadFrom(dir.path()).empty());
}

TEST(TestDescriptorLoader, DisablingOneOfTwoCollidingEnginesLetsTheOtherLoad)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("collision_recovery"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:same_name"));
    writeDocuments(dir.path(), makeSetDocuments('2', "test:same_name"));

    // The disabled UED is skipped before it claims the name, which frees it for the
    // survivor -- the recovery lever RFC 0020 §12 names for the drop-all rule above.
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter disabled(
        "HIPDNN_DISABLE_ENGINES", testUuid('2', ROLE_ENGINE));

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(toString(sets.front().engine.id), testUuid('1', ROLE_ENGINE));
}

TEST(TestDescriptorLoader, CoercesAnIntegerValueForAFloatField)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("float_coercion"));
    auto documents = makeSetDocuments('1', "test:coerced");
    documentOfType(documents, ".kmd.json")
        .at("fields")
        .push_back({{"name", "scale"}, {"type", "float"}, {"default_value", 1}});
    for(auto& kernel : documentOfType(documents, ".kdp.json").at("kernelDescriptors"))
    {
        kernel["metadata"]["scale"] = 2;
    }
    writeDocuments(dir.path(), documents);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    const auto& schemaFields = sets.front().schema.fields;
    ASSERT_EQ(schemaFields.size(), 3u);
    ASSERT_TRUE(schemaFields[2].defaultValue.has_value());
    EXPECT_DOUBLE_EQ(std::get<double>(*schemaFields[2].defaultValue), 1.0);

    const auto& metadata = sets.front().packs.front().kernels.front().metadata;
    EXPECT_DOUBLE_EQ(std::get<double>(metadata.at("scale")), 2.0);
}

TEST(TestDescriptorLoader, DropsAPackWhoseMetadataContradictsTheSchema)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("bad_metadata"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:valid"));

    auto broken = makeSetDocuments('2', "test:bad_metadata");
    for(auto& kernel : documentOfType(broken, ".kdp.json").at("kernelDescriptors"))
    {
        kernel["metadata"]["block_size"] = "sixty-four";
    }
    writeDocuments(dir.path(), broken);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:valid");
}

/// coerceKernelMetadata's omit arm, at engine scope so a regression there is
/// distinguishable from the wrong-type arm the test above covers: without it, the
/// incomplete kernel is not caught here at all, and instead reaches the probe in
/// loadValidatedDescriptorSets, where KernelIngestorStateManager::completeMetadata throws
/// -- which costs the WHOLE engine via that catch, not just the one pack that named it.
TEST(TestDescriptorLoader, DropsAPackWhoseKernelOmitsAnUndefaultedMetadataField)
{
    const ScopedSymbols symbols;
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(
        uniqueDirectory("missing_metadata_field"));
    auto documents = makeSetDocuments('1', "test:omitted_field");

    auto brokenPack = documentOfType(documents, ".kdp.json");
    brokenPack["id"] = testUuid('1', 'b');
    brokenPack["name"] = "pack whose kernel omits dtype";
    for(auto& kernel : brokenPack.at("kernelDescriptors"))
    {
        kernel["metadata"].erase("dtype"); // declares no default_value in the KMD
    }
    documents.push_back(TestDocument{".kdp.json", brokenPack});
    writeDocuments(dir.path(), documents);

    const auto sets = loadValidatedDescriptorSets<LoaderHandle>(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().packs.size(), 1u);
}

/// coerceKernelMetadata's other arm: an undeclared key must drop the pack rather than
/// survive into the completed tuple, where it would make two otherwise-identical kernels
/// present as distinct catalog entries -- a silent selection change, not a drop, so
/// nothing here throws either with or without the guard.
TEST(TestDescriptorLoader, DropsAPackWhoseKernelSuppliesAnUndeclaredMetadataField)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(
        uniqueDirectory("undeclared_metadata_field"));
    auto documents = makeSetDocuments('1', "test:undeclared_field");

    auto brokenPack = documentOfType(documents, ".kdp.json");
    brokenPack["id"] = testUuid('1', 'b');
    brokenPack["name"] = "pack whose kernel supplies an undeclared field";
    for(auto& kernel : brokenPack.at("kernelDescriptors"))
    {
        kernel["metadata"]["undeclared_flag"] = true;
    }
    documents.push_back(TestDocument{".kdp.json", brokenPack});
    writeDocuments(dir.path(), documents);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().packs.size(), 1u);
}

/// GenericEngine's constructor throws on this, and by then copyEngineIds has advertised
/// the id -- so the engine has to be gone before it is ever counted.
TEST(TestDescriptorLoader, DropsAnEngineWhoseKnobNamesNoSchemaField)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("bad_knob"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:valid"));

    auto broken = makeSetDocuments('2', "test:bad_knob");
    documentOfType(broken, ".ued.json")["knobs"] = {"block_sizes"};
    writeDocuments(dir.path(), broken);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:valid");
}

TEST(TestDescriptorLoader, ValidationDropsAnEngineNamingAnUnregisteredSymbol)
{
    const ScopedSymbols symbols;
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("unregistered"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:symbol_check_sibling"));

    auto unregistered = makeSetDocuments('2', "test:unregistered");
    documentOfType(unregistered, ".umd.json")["match_symbol"] = "descriptorloader.absent";
    writeDocuments(dir.path(), unregistered);

    const auto sets = loadValidatedDescriptorSets<LoaderHandle>(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:symbol_check_sibling");
}

/// The kernel-scope arm of the match-symbol pre-flight: the test above only corrupts the
/// first `.umd.json`, always the graph-scope matcher, so KernelMatcherRegistry's branch
/// was never taken. Pointing the kernel-scope matcher at a symbol registered only for
/// graph scope still gets the engine dropped even with the ternary collapsed onto one
/// registry for both scopes -- the state manager's constructor would also reject it --
/// so the pre-flight's specific diagnostic is the only thing that distinguishes the two.
TEST(TestDescriptorLoader, ValidationDropsAnEngineNamingAGraphSymbolAsItsKernelScopeMatcher)
{
    const ScopedSymbols symbols;
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(
        uniqueDirectory("kernel_scope_wrong_registry"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:kernel_scope_check_sibling"));

    auto misrouted = makeSetDocuments('2', "test:kernel_scope_misrouted");
    secondDocumentOfType(misrouted, ".umd.json")["match_symbol"] = GRAPH_SYMBOL;
    writeDocuments(dir.path(), misrouted);

    const auto sets = loadValidatedDescriptorSets<LoaderHandle>(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:kernel_scope_check_sibling");
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "names unregistered match symbol"));
}

/// The dispatch-symbol pre-flight, independent of the match-symbol arm above and until
/// now untested. The state manager's constructor also rejects an unregistered dispatch
/// symbol, but with a different, generic message -- that difference is what pins this
/// loop rather than the probe's fallback catching it instead.
TEST(TestDescriptorLoader, ValidationDropsAnEngineNamingAnUnregisteredDispatchSymbol)
{
    const ScopedSymbols symbols;
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("unregistered_dispatch"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:dispatch_check_sibling"));

    auto unregistered = makeSetDocuments('2', "test:unregistered_dispatch");
    documentOfType(unregistered, ".udd.json")["dispatch_symbol"] = "descriptorloader.absent";
    writeDocuments(dir.path(), unregistered);

    const auto sets = loadValidatedDescriptorSets<LoaderHandle>(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:dispatch_check_sibling");
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "names unregistered dispatch symbol"));
}

/// The score-symbol pre-flight: the third and last of the three independently-pre-flighted
/// symbol families, also untested until now and also redundant with the probe on the
/// drop/survive outcome alone -- NativeKernelHeuristic's constructor resolves the score
/// symbol eagerly too. Same reasoning as the dispatch test above.
TEST(TestDescriptorLoader, ValidationDropsAnEngineNamingAnUnregisteredScoreSymbol)
{
    const ScopedSymbols symbols;
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("unregistered_score"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:score_check_sibling"));

    auto unregistered = makeSetDocuments('2', "test:unregistered_score");
    documentOfType(unregistered, ".uhd.json")["payload"] = "descriptorloader.absent";
    writeDocuments(dir.path(), unregistered);

    const auto sets = loadValidatedDescriptorSets<LoaderHandle>(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:score_check_sibling");
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "names unregistered score symbol"));
}

/// The probe's catch: two kernels completing to the same metadata tuple make the state
/// manager's constructor throw, which must cost that engine and nothing else.
TEST(TestDescriptorLoader, ValidationDropsAnEngineWhoseKernelsShareAMetadataTuple)
{
    const ScopedSymbols symbols;
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("duplicate_tuple"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:tuple_check_sibling"));

    auto duplicated = makeSetDocuments('2', "test:duplicate_tuple");
    auto& kernels = documentOfType(duplicated, ".kdp.json").at("kernelDescriptors");
    kernels[1]["metadata"] = kernels[0]["metadata"];
    writeDocuments(dir.path(), duplicated);

    const auto sets = loadValidatedDescriptorSets<LoaderHandle>(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:tuple_check_sibling");
}

/// A name hashing onto an engine registered elsewhere in the process: EngineManager would
/// emplace-drop the loser while its id stayed advertised.
TEST(TestDescriptorLoader, ValidationDropsAnEngineCollidingWithARegisteredName)
{
    const ScopedSymbols symbols;
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("collision"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:collision_check_sibling"));

    static const std::string s_claimed = "test:already_claimed";
    static const hipdnn_data_sdk::utilities::EngineRegistrar s_registrar{s_claimed};
    writeDocuments(dir.path(), makeSetDocuments('2', s_claimed));

    const auto sets = loadValidatedDescriptorSets<LoaderHandle>(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:collision_check_sibling");
}

/// The loader registers the names it accepts, so a second load of the same directory has
/// to recognise its own registrations rather than reject them as collisions.
TEST(TestDescriptorLoader, ValidationIsIdempotentAcrossReloads)
{
    const ScopedSymbols symbols;
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("reload"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:reloaded"));

    ASSERT_EQ(loadValidatedDescriptorSets<LoaderHandle>(dir.path()).size(), 1u);

    const auto reloaded = loadValidatedDescriptorSets<LoaderHandle>(dir.path());

    ASSERT_EQ(reloaded.size(), 1u);
    EXPECT_EQ(reloaded.front().engine.name, "test:reloaded");
}

// ---------------------------------------------------------------------------
// RFC 0020: UED format, collision handling, disable lever
// ---------------------------------------------------------------------------

/// The suffix is the only thing consulted. Nothing infers a type from a file's contents,
/// and the stem is free-form documentation, so renaming every stem changes nothing.
TEST(TestDescriptorLoader, ReadsTheTypeFromTheSuffixNotTheStem)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("stem_ignored"));
    auto documents = makeSetDocuments('1', "test:stems");
    referenceLastKernel(documents); // proves .ukd.json is read by suffix too

    // Deliberately not the id, and deliberately not descriptive: a stem carrying no
    // information at all must still load.
    int index = 0;
    for(const auto& document : documents)
    {
        std::ofstream file(
            dir.path() / ("descriptor" + std::to_string(index++) + std::string(document.suffix)),
            std::ios::binary);
        file << document.body.dump(2) << '\n';
    }

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:stems");
    ASSERT_EQ(sets.front().packs.size(), 1u);
    EXPECT_EQ(sets.front().packs.front().kernels.size(), 3u);
}

/// The name is hashed into a global id space, so an unscoped one is the name two vendors
/// both pick. Rejected at parse rather than left to collide at registration.
TEST(TestDescriptorLoader, DropsAnEngineWhoseNameIsNotScoped)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("unscoped_name"));
    writeDocuments(dir.path(), makeSetDocuments('1', "unscoped"));

    EXPECT_TRUE(loadFrom(dir.path()).empty());
}

/// Optional per RFC 0020 §4.2 and mapped to no hipDNN enum, so what is under test is that
/// a conforming UED carrying them still loads -- the field used to be an unknown key.
TEST(TestDescriptorLoader, AcceptsAnEngineDeclaringNumericalNotes)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("numerical_notes"));
    auto documents = makeSetDocuments('1', "test:numerical");
    documentOfType(documents, ".ued.json")["numerical_notes"]
        = nlohmann::json::array({"tensor_core", "reduced_precision_reduction"});
    writeDocuments(dir.path(), documents);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.numericalNotes,
              (std::vector<std::string>{"tensor_core", "reduced_precision_reduction"}));
}

/// A note repeated is reported twice downstream, so it is an authoring mistake rather
/// than a redundancy the loader should quietly collapse.
TEST(TestDescriptorLoader, DropsAnEngineRepeatingANumericalNote)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("repeated_note"));
    auto documents = makeSetDocuments('1', "test:repeated");
    documentOfType(documents, ".ued.json")["numerical_notes"]
        = nlohmann::json::array({"tensor_core", "tensor_core"});
    writeDocuments(dir.path(), documents);

    EXPECT_TRUE(loadFrom(dir.path()).empty());
}

const std::string DISABLED_ENGINE_NAME = "test:disabled";
const int64_t DISABLED_ENGINE_ID = hipdnn_data_sdk::utilities::engineNameToId(DISABLED_ENGINE_NAME);

/// All three spellings RFC 0020 §12 admits reach the same engine. Parameterised over the
/// identifier rather than repeated, since the matcher is one list walk for all three.
class TestDisabledEngineIdentifier : public ::testing::TestWithParam<std::string>
{
};

TEST_P(TestDisabledEngineIdentifier, SkipsTheEngineBeforeItIsRegistered)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("disabled"));
    writeDocuments(dir.path(), makeSetDocuments('a', DISABLED_ENGINE_NAME));

    // Surrounded by an unmatched entry and stray whitespace: one list is meant to span
    // providers, so entries naming someone else's engine are skipped, not errors.
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter disabled(
        "HIPDNN_DISABLE_ENGINES", "other:engine, " + GetParam() + " ,");

    EXPECT_TRUE(loadFrom(dir.path()).empty());
}

// Declared out here because the preprocessor splits macro arguments on every comma it
// sees, including the ones inside a braced initializer.
const std::array<std::string, 6> DISABLED_SPELLINGS{
    "ByName", "ByUuid", "ByHexId", "ByDecimalId", "ByLowercaseHexId", "ByUppercaseUuid"};

INSTANTIATE_TEST_SUITE_P(
    Spelling,
    TestDisabledEngineIdentifier,
    ::testing::Values(
        DISABLED_ENGINE_NAME,
        testUuid('a', ROLE_ENGINE),
        hipdnn_data_sdk::utilities::formatEngineIdHex(DISABLED_ENGINE_ID),
        std::to_string(DISABLED_ENGINE_ID),
        // formatEngineIdHex's canonical spelling is uppercase; only equalsIgnoringCase,
        // not ==, accepts this one.
        [] {
            auto hex = hipdnn_data_sdk::utilities::formatEngineIdHex(DISABLED_ENGINE_ID);
            std::transform(hex.begin(), hex.end(), hex.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            return hex;
        }(),
        // testUuid's canonical spelling is lowercase; same reason in the other direction.
        [] {
            auto uuid = testUuid('a', ROLE_ENGINE);
            std::transform(uuid.begin(), uuid.end(), uuid.begin(), [](unsigned char c) {
                return static_cast<char>(std::toupper(c));
            });
            return uuid;
        }()),
    [](const ::testing::TestParamInfo<std::string>& info) {
        return DISABLED_SPELLINGS.at(info.index);
    });

/// An entry naming nothing must not disable everything -- the shared-list case again,
/// from the other side.
TEST(TestDescriptorLoader, IgnoresADisableEntryThatNamesNoLoadedEngine)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("disabled_other"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:kept"));

    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter disabled(
        "HIPDNN_DISABLE_ENGINES", "somebody:else");

    EXPECT_EQ(loadFrom(dir.path()).size(), 1u);
}

/// Absent leaves the baseline the struct defaults to, which is what keeps a UED authored
/// before the field existed loading unchanged.
TEST(TestDescriptorLoader, DefaultsAnEngineWithNoSdkVersionToTheBaseline)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("no_sdk_version"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:baseline"));

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.sdkVersion,
              hipdnn_data_sdk::utilities::Version{
                  hipdnn_plugin_sdk::K_ENGINE_PLUGIN_API_VERSION_BASELINE});
}

/// Carried as authored: the loader does not gate on it, since the floor it is compared
/// against is a property of each graph and only known at match time.
TEST(TestDescriptorLoader, CarriesTheEnginesDeclaredSdkVersion)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("sdk_version"));
    auto documents = makeSetDocuments('1', "test:versioned");
    documentOfType(documents, ".ued.json")["sdk_version"] = "1.2.3";
    writeDocuments(dir.path(), documents);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.sdkVersion, (hipdnn_data_sdk::utilities::Version{1, 2, 3}));
}

/// A version that cannot be parsed is an authoring mistake, not a zero: silently reading
/// it as the baseline would let an engine claim a schema it does not understand.
TEST(TestDescriptorLoader, DropsAnEngineWhoseSdkVersionIsMalformed)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("bad_sdk_version"));
    auto documents = makeSetDocuments('1', "test:bad_version");
    documentOfType(documents, ".ued.json")["sdk_version"] = "1.2";
    writeDocuments(dir.path(), documents);

    EXPECT_TRUE(loadFrom(dir.path()).empty());
}

/// The inverse of the UED case: `version` is required on every type, with no absence-safe
/// default. A KMD with no version drops, and the engine whose `metadata` named it goes
/// with it -- which is what makes the rule enforceable rather than advisory.
TEST(TestDescriptorLoader, RejectsANonUedDescriptorWithNoVersion)
{
    const ScopedSymbols symbols;
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("version_all"));
    auto documents = makeSetDocuments('1', "test:versioned");
    documentOfType(documents, ".kmd.json").erase("version");
    writeDocuments(dir.path(), documents);

    EXPECT_TRUE(loadFrom(dir.path()).empty());
}

/// RFC 0020 §10.2.1: the version check runs before duplicate detection, so a UED the
/// runtime cannot read is dropped for its version alone and the descriptor it would have
/// collided with is retained. Ordered the other way, an unreadable file would take a
/// perfectly good engine down with it.
TEST(TestDescriptorLoader, AnUnsupportedVersionDropsBeforeItCanCollideByName)
{
    const ScopedSymbols symbols;
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("version_first"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:contested"));

    auto newer = makeSetDocuments('2', "test:contested");
    documentOfType(newer, ".ued.json")["version"] = "2.0";
    writeDocuments(dir.path(), newer);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:contested");
    EXPECT_EQ(toString(sets.front().engine.id), testUuid('1', ROLE_ENGINE));
}

/// The same ordering for the `id` invariant: two UEDs share an id and differ in content,
/// which is normally a drop-all collision, but one is unreadable so it never participates.
TEST(TestDescriptorLoader, AnUnsupportedVersionDropsBeforeItCanCollideById)
{
    const ScopedSymbols symbols;
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("version_first_id"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:survivor"));

    // Written to a second directory rather than beside the first: files are named for the
    // id they carry, so writing this into one directory would overwrite the descriptor it
    // is supposed to collide with and prove nothing. Without the version bump this is the
    // drop-all case DropsAnIdTwoFilesDisagreeAbout covers.
    auto casualtySet = makeSetDocuments('2', "test:casualty");
    auto casualty = documentOfType(casualtySet, ".ued.json");
    casualty["id"] = testUuid('1', ROLE_ENGINE);
    casualty["version"] = "2.0";
    writeDocument(dir.path() / "gfx950", TestDocument{".ued.json", casualty});

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:survivor");
}

/// RFC 0020 §4.3: the authored form is JSONC. Comments are the parser's business only --
/// they must not reach the duplicate check, which compares parsed documents, so the same
/// descriptor commented and uncommented is one definition rather than a collision.
TEST(TestDescriptorLoader, ReadsCommentedDescriptorsAndIgnoresCommentsWhenComparing)
{
    const ScopedSymbols symbols;
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("jsonc"));
    const auto documents = makeSetDocuments('1', "test:commented");
    writeDocuments(dir.path(), documents);

    // The same set again under a second arch directory, this time with a comment on every
    // file: RFC 0020 §10.2.1's content-identical exception has to see through it.
    const auto commented = dir.path() / "gfx950";
    std::filesystem::create_directories(commented);
    for(const auto& document : documents)
    {
        std::ofstream file(
            commented / (document.body.at("id").get<std::string>() + std::string(document.suffix)),
            std::ios::binary);
        file << "// authored with a comment, per RFC 0020 §4.3\n" << document.body.dump(2) << "\n";
    }

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:commented");
}

/// RFC 0020 §4.3's authored form strips comments only; a trailing comma is a hard
/// nlohmann parse_error.101, not the broader "permit trailing commas" many mean by
/// "JSONC" (VS Code, tsconfig). The rejection is right -- pinned here so the label
/// staying wrong in a docblock doesn't quietly become the behavior.
TEST(TestDescriptorLoader, RejectsATrailingComma)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("trailing_comma"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:intact"));
    std::ofstream(dir.path() / "broken.ued.json", std::ios::binary)
        << R"({"version": "1.0", "id": "x",})";

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:intact");
}

/// A `.json` naming no descriptor type is skipped before it is opened, so an unrelated
/// JSON file under the descriptor root costs nothing. Distinct from IgnoresANonJsonFile,
/// which never had a `.json` extension to begin with; the WARN is what distinguishes
/// "skipped before opening" from "opened and rejected as malformed".
TEST(TestDescriptorLoader, SkipsAJsonFileThatNamesNoDescriptorType)
{
    const ScopedSymbols symbols;
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_WARN);
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("stray_json"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:intact"));
    std::ofstream(dir.path() / "notes.json", std::ios::binary) << R"({"id":"x"})";

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:intact");
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, "notes.json"));
}

/// findFileType() requires a non-empty stem; a filename that is nothing but the suffix
/// has none, so it names no type despite ending in one.
TEST(TestDescriptorLoader, IgnoresAFileWhoseWholeNameIsASuffix)
{
    const ScopedSymbols symbols;
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_WARN);
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("bare_suffix"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:intact"));
    std::ofstream(dir.path() / ".ued.json", std::ios::binary) << "{}";

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:intact");
    // The mechanism, not just the outcome: widening the stem guard types this file as a
    // UED, which opens it and rejects `{}` with a per-file ERROR instead. The sibling
    // engine survives either way, so only this WARN separates them.
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, "is not a descriptor filename"));
}

/// findFileType() is case-sensitive on purpose; an uppercased suffix must still warn
/// instead of vanishing the way it did before the WARN check was widened to catch it.
TEST(TestDescriptorLoader, IgnoresAnUppercasedSuffix)
{
    const ScopedSymbols symbols;
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_WARN);
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("uppercase_suffix"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:intact"));
    std::ofstream(dir.path() / "pointwise.KDP.JSON", std::ios::binary) << "{}";

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:intact");
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, "pointwise.KDP.JSON"));
}

/// `arch` is optional and empty means arch-independent, but it has to survive the parse:
/// KernelIngestorStateManager drops a pack whose arch excludes the calling device, so a
/// field the loader silently discarded would leave that gate permanently open, and one
/// the allow-list omitted would reject the whole pack as an unknown key.
TEST(TestDescriptorLoader, CarriesAPacksDeclaredArchitectures)
{
    const ScopedSymbols symbols;
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("pack_arch"));
    auto documents = makeSetDocuments('1', "test:arch");
    documentOfType(documents, ".kdp.json")["arch"] = nlohmann::json::array({"gfx90a", "gfx942"});
    writeDocuments(dir.path(), documents);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    ASSERT_EQ(sets.front().packs.size(), 1u);
    EXPECT_EQ(sets.front().packs.front().arch, (std::vector<std::string>{"gfx90a", "gfx942"}));
}

/// The validator must admit exactly what archMatches admits. PREFIX mode terminates the
/// candidate on ':' or end-of-string, so a candidate carrying its own feature suffix
/// matches a device reporting more of them, and LLVM generic targets are real gcnArchName
/// values. A stricter shape check would make both unauthorable while the matcher still
/// handled them.
TEST(TestDescriptorLoader, AcceptsArchIdsCarryingFeaturesAndGenericTargets)
{
    const ScopedSymbols symbols;
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("pack_arch_suffix"));
    auto documents = makeSetDocuments('1', "test:arch_suffix");
    documentOfType(documents, ".kdp.json")["arch"]
        = nlohmann::json::array({"gfx942:sramecc+", "gfx90a:sramecc+:xnack-", "gfx9-4-generic"});
    writeDocuments(dir.path(), documents);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    ASSERT_EQ(sets.front().packs.size(), 1u);
    EXPECT_EQ(
        sets.front().packs.front().arch,
        (std::vector<std::string>{"gfx942:sramecc+", "gfx90a:sramecc+:xnack-", "gfx9-4-generic"}));
}

/// The default: a pack naming no architecture applies everywhere, so absence must parse
/// as empty rather than as a constraint nothing satisfies.
TEST(TestDescriptorLoader, APackWithNoDeclaredArchIsArchIndependent)
{
    const ScopedSymbols symbols;
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("pack_no_arch"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:no_arch"));

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    ASSERT_EQ(sets.front().packs.size(), 1u);
    EXPECT_TRUE(sets.front().packs.front().arch.empty());
}

/// Explicit `"arch": []` and an absent `arch` key both mean arch-independent; the
/// validation added for arch entries must not reject the empty list itself.
TEST(TestDescriptorLoader, AnExplicitlyEmptyArchIsArchIndependent)
{
    const ScopedSymbols symbols;
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(
        uniqueDirectory("pack_arch_explicit_empty"));
    auto documents = makeSetDocuments('1', "test:explicit_empty_arch");
    documentOfType(documents, ".kdp.json")["arch"] = nlohmann::json::array();
    writeDocuments(dir.path(), documents);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    ASSERT_EQ(sets.front().packs.size(), 1u);
    EXPECT_TRUE(sets.front().packs.front().arch.empty());
}

/// Folders are organizational only: the walk is recursive and a file's directory means
/// nothing to the loader, so a set split across a subdirectory resolves as one engine.
TEST(TestDescriptorLoader, LoadsDescriptorsFromNestedFolders)
{
    const ScopedSymbols symbols;
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("nested"));
    auto documents = makeSetDocuments('1', "test:nested");

    // One descriptor a level down, the rest at the root: the cross-references that bind
    // them carry ids, not paths.
    writeDocument(dir.path() / "pointwise", documents.front());
    documents.erase(documents.begin());
    writeDocuments(dir.path(), documents);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:nested");
}

// ---------------------------------------------------------------------------
// UKD: standalone kernel files and by-id references from a KDP
// ---------------------------------------------------------------------------

TEST(TestDescriptorLoader, LoadsAPackMixingInlineAndReferencedKernels)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("ukd_mixed"));
    auto documents = makeSetDocuments('1', "test:mixed");
    referenceLastKernel(documents);
    writeDocuments(dir.path(), documents);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    ASSERT_EQ(sets.front().packs.size(), 1u);
    const auto& kernels = sets.front().packs.front().kernels;
    ASSERT_EQ(kernels.size(), 3u);
    const auto referencedId = testUuid('1', 'a');
    const auto it = std::find_if(kernels.begin(), kernels.end(), [&](const auto& kernel) {
        return toString(kernel.id) == referencedId;
    });
    ASSERT_NE(it, kernels.end());
    EXPECT_EQ(std::get<std::string>(it->metadata.at("dtype")), "HALF");
}

/// Guards resolveDescriptorSets' ordering: kernelIds must resolve into `kernels` before
/// the "declares no kernels" check runs, or an all-references pack reads as empty.
TEST(TestDescriptorLoader, LoadsAPackWhoseKernelsAreAllReferences)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("ukd_all_refs"));
    auto documents = makeSetDocuments('1', "test:all_refs");
    referenceLastKernel(documents);
    referenceLastKernel(documents);
    referenceLastKernel(documents);
    writeDocuments(dir.path(), documents);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    ASSERT_EQ(sets.front().packs.size(), 1u);
    EXPECT_EQ(sets.front().packs.front().kernels.size(), 3u);
}

TEST(TestDescriptorLoader, DropsAPackReferencingAKernelNoFileDefines)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("ukd_dangling"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:valid"));

    auto broken = makeSetDocuments('2', "test:broken");
    documentOfType(broken, ".kdp.json")["kernelDescriptors"].push_back(testUuid('f', 'f'));
    writeDocuments(dir.path(), broken);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:valid");
    EXPECT_TRUE(recorder.hasLogContaining(
        HIPDNN_SEV_ERROR,
        "descriptor loader: pack 'pack' id=" + testUuid('2', ROLE_PACK) + " names kernel "
            + testUuid('f', 'f') + ", which no descriptor defines; dropping the pack"));
}

TEST(TestDescriptorLoader, DropsAPackNamingTheSameKernelTwice)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("ukd_duplicate_ref"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:valid"));

    auto broken = makeSetDocuments('2', "test:broken");
    referenceLastKernel(broken);
    auto& kernelRefs = documentOfType(broken, ".kdp.json").at("kernelDescriptors");
    const nlohmann::json duplicateRef = kernelRefs.back();
    kernelRefs.push_back(duplicateRef);
    writeDocuments(dir.path(), broken);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:valid");
    EXPECT_TRUE(recorder.hasLogContaining(
        HIPDNN_SEV_ERROR,
        "descriptor loader: pack 'pack' id=" + testUuid('2', ROLE_PACK) + " names kernel "
            + testUuid('2', 'a') + " more than once; dropping the pack"));
}

/// The inline+reference spelling of the same duplicate, rather than ref+ref above.
TEST(TestDescriptorLoader, DropsAPackWhoseReferencedKernelIsAlsoInline)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("ukd_ref_and_inline"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:valid"));

    auto broken = makeSetDocuments('2', "test:broken");
    referenceLastKernel(broken);
    auto inlineAgain = documentOfType(broken, ".ukd.json");
    inlineAgain.erase("version"); // else this is RejectsAnInlineKernelCarryingAVersion instead
    documentOfType(broken, ".kdp.json")["kernelDescriptors"].push_back(inlineAgain);
    writeDocuments(dir.path(), broken);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:valid");
    EXPECT_TRUE(recorder.hasLogContaining(
        HIPDNN_SEV_ERROR,
        "descriptor loader: pack 'pack' id=" + testUuid('2', ROLE_PACK) + " names kernel "
            + testUuid('2', 'a') + " more than once; dropping the pack"));
}

TEST(TestDescriptorLoader, WarnsAboutAStandaloneKernelNoPackReferences)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_WARN);
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("ukd_orphan"));
    auto documents = makeSetDocuments('1', "test:orphan_kernel");
    documents.push_back(TestDocument{".ukd.json",
                                     {{"version", "1.0"},
                                      {"id", testUuid('1', ROLE_STANDALONE_KERNEL)},
                                      {"name", "unreferenced kernel"},
                                      {"kernel_source",
                                       {{"kind", "embedded_source"},
                                        {"source_file", "Kernel.cpp"},
                                        {"entry_point", "Entry"}}}}});
    writeDocuments(dir.path(), documents);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:orphan_kernel");
    ASSERT_EQ(sets.front().packs.size(), 1u);
    EXPECT_EQ(sets.front().packs.front().kernels.size(), 3u);
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, testUuid('1', ROLE_STANDALONE_KERNEL)));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, "no pack references it"));
}

TEST(TestDescriptorLoader, SharesAStandaloneKernelBetweenTwoEngines)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("ukd_shared"));
    auto first = makeSetDocuments('1', "test:first");
    referenceLastKernel(first);
    const auto sharedId = testUuid('1', 'a');

    // A second, unrelated engine names the same standalone kernel: the two-packs-of-one-
    // -engine collision is pre-existing behavior for identical metadata tuples, not this.
    auto second = makeSetDocuments('2', "test:second");
    documentOfType(second, ".kdp.json")["kernelDescriptors"].push_back(sharedId);

    writeDocuments(dir.path(), first);
    writeDocuments(dir.path(), second);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 2u);
    for(const auto& set : sets)
    {
        ASSERT_EQ(set.packs.size(), 1u);
        const auto& kernels = set.packs.front().kernels;
        EXPECT_TRUE(std::any_of(kernels.begin(), kernels.end(), [&](const auto& kernel) {
            return toString(kernel.id) == sharedId;
        }));
    }
}

/// Two files claiming the same kernel id are both ignored (DropsAnIdTwoFilesDisagreeAbout's
/// pattern, one catalog map over), so the pack naming that id sees a dangling reference.
TEST(TestDescriptorLoader, DropsAPackWhoseReferencedKernelConflicts)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("ukd_conflict"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:valid"));

    auto broken = makeSetDocuments('2', "test:broken");
    referenceLastKernel(broken);
    writeDocuments(dir.path(), broken);
    auto& conflicting = documentOfType(broken, ".ukd.json");
    conflicting["name"] = "a different kernel with the same id";
    std::ofstream(dir.path() / "second-claim.ukd.json", std::ios::binary) << conflicting.dump(2);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:valid");
    EXPECT_TRUE(recorder.hasLogContaining(
        HIPDNN_SEV_ERROR,
        "descriptor loader: pack 'pack' id=" + testUuid('2', ROLE_PACK) + " names kernel "
            + testUuid('2', 'a') + ", which no descriptor defines; dropping the pack"));
}

TEST(TestDescriptorLoader, RejectsAStandaloneKernelWithNoVersion)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("ukd_missing_version"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:valid"));

    auto broken = makeSetDocuments('2', "test:broken");
    referenceLastKernel(broken);
    documentOfType(broken, ".ukd.json").erase("version");
    writeDocuments(dir.path(), broken);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:valid");
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "missing required key 'version'"));
}

/// Pins the asymmetry the contract draws: `version` is a file-level key, so the field
/// required standalone is not merely unused on the inline spelling of the very same
/// kernel -- it is a key that spelling has no place for, and the pack carrying it fails.
TEST(TestDescriptorLoader, RejectsAnInlineKernelCarryingAVersion)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("inline_with_version"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:valid"));

    auto broken = makeSetDocuments('2', "test:broken");
    documentOfType(broken, ".kdp.json").at("kernelDescriptors").front()["version"] = "1.0";
    writeDocuments(dir.path(), broken);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:valid");
    // The locator is the point: "a 'kernelDescriptors' entry" alone names no file, and a
    // shard layout ships the same filename under every arch.
    EXPECT_TRUE(recorder.hasLogContaining(
        HIPDNN_SEV_ERROR, "unknown key 'version' in a 'kernelDescriptors' entry in "))
        << recorder.getRecordedLogsAsString();
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, ".kdp.json"));
}

/// A typo in an OPTIONAL key is the case the extension rule exists to catch: `heuristik`
/// leaves a UED with no heuristic at all, which is legal, so warning about it would let
/// the engine load and rank by the fallback forever with the default log level off.
TEST(TestDescriptorLoader, RejectsAMisspelledOptionalKey)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("misspelled_optional"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:valid"));

    auto broken = makeSetDocuments('2', "test:broken");
    auto& engine = documentOfType(broken, ".ued.json");
    engine["heuristik"] = engine.at("heuristic");
    engine.erase("heuristic");
    writeDocuments(dir.path(), broken);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:valid");
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "unknown key 'heuristik'"))
        << recorder.getRecordedLogsAsString();
}

/// The point of the policy: a descriptor may carry tracking fields the loader has no
/// reader for -- what the build-time packager stamps onto its output -- and still load,
/// as long as they announce themselves. `provenance` is the packager's one unprefixed
/// block; anything else has to wear the `x-`/`_` prefix.
TEST(TestDescriptorLoader, LoadsADescriptorCarryingTrackingFields)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_WARN);
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("tracking_fields"));
    auto documents = makeSetDocuments('1', "test:tracked");
    auto& engine = documentOfType(documents, ".ued.json");
    engine["provenance"] = {{"origin_kind", "hip"}};
    engine["x-build-id"] = "abc123";
    engine["_internal"] = 7;
    writeDocuments(dir.path(), documents);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:tracked");
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, "extension key 'provenance'"));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, "extension key 'x-build-id'"));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, "extension key '_internal'"));
}

/// Follows AnUnsupportedVersionDropsBeforeItCanCollideByName's shape: an unreadable file
/// is skipped, not treated as absent-and-defaulted -- the seventh type is gated the same.
TEST(TestDescriptorLoader, SkipsAStandaloneKernelOnAnUnsupportedMajor)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(uniqueDirectory("ukd_bad_major"));
    writeDocuments(dir.path(), makeSetDocuments('1', "test:valid"));

    auto broken = makeSetDocuments('2', "test:broken");
    referenceLastKernel(broken);
    documentOfType(broken, ".ukd.json")["version"] = "2.0";
    writeDocuments(dir.path(), broken);

    const auto sets = loadFrom(dir.path());

    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets.front().engine.name, "test:valid");
}

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
