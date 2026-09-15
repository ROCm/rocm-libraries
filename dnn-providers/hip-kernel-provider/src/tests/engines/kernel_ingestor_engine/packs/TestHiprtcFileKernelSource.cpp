// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include "compilation/KernelCompileOptions.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "engines/hip_mlops_engine/HipMlopsModuleCache.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"
#include "mocks/MockCompiledProgram.hpp"
#include "mocks/MockKernelCompiler.hpp"
#include "mocks/MockRunnableKernel.hpp"

/**
 * @file TestHiprtcFileKernelSource.cpp
 * @brief The `hiprtc_file` kernel source: a drop-in bundle of hipRTC sources compiled at
 *        prepare() with defines bound from the kernel's own metadata.
 *
 * Every case here runs on MockKernelCompiler, so the whole adapter -- resolution,
 * containment, substitution, header assembly and the cache key -- is observable with no
 * device. What a real hipRTC compile does with the text is the end-to-end tier's
 * question; what this provider hands hipRTC is this one's.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using hip_kernel_provider::kernel_ingestor_engine::testing::BLOCK_SIZE_FIELD;
using hip_kernel_provider::kernel_ingestor_engine::testing::buildPointwiseGraph;
using hip_kernel_provider::kernel_ingestor_engine::testing::DTYPE_FIELD;
using hip_kernel_provider::kernel_ingestor_engine::testing::GraphFixture;
using hip_kernel_provider::kernel_ingestor_engine::testing::INPUT_A_UID;
using ::testing::_;
using ::testing::Contains;
using ::testing::HasSubstr;
using ::testing::Not;

constexpr const char* BUNDLE_NAME = "sources";
constexpr const char* SOURCE_FILE = "attention.hip";
constexpr const char* ENTRY_POINT = "attention_fwd";
/// Deliberately not compilable HIP: no case here reaches a compiler, and text that only
/// this suite's assertions read says so.
constexpr const char* SOURCE_TEXT = "// drop-in attention source\n";

DescriptorId id(uint8_t seed)
{
    DescriptorId value{};
    value.fill(seed);
    return value;
}

/// Distinct per call, mirroring TestDescriptorLoader's helper: ScopedDirectory creates
/// atomically and throws if the name is taken.
std::filesystem::path uniqueDirectory(const std::string& name)
{
    static const std::string s_session
        = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    static unsigned s_counter = 0;
    const auto path
        = std::filesystem::temp_directory_path()
          / ("hiprtc_file_" + name + "_" + s_session + "_" + std::to_string(s_counter++));
    std::filesystem::remove_all(path);
    return path;
}

void writeFile(const std::filesystem::path& path, const std::string& text)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path, std::ios::binary);
    file << text;
}

/// A descriptor tree holding one bundle: `<root>/sources/attention.hip`. The descriptor
/// is taken to live at the root, so `bundle` is the plain directory name an author would
/// write.
class BundleTree
{
public:
    explicit BundleTree(const std::string& name)
        : _directory(uniqueDirectory(name))
    {
        writeFile(sourcePath(), SOURCE_TEXT);
    }

    const std::filesystem::path& root() const
    {
        return _directory.path();
    }

    std::filesystem::path sourcePath() const
    {
        return _directory.path() / BUNDLE_NAME / SOURCE_FILE;
    }

    std::filesystem::path bundlePath() const
    {
        return _directory.path() / BUNDLE_NAME;
    }

    /// Adds a file to the bundle, so a case can pin which of them become headers.
    void writeIntoBundle(const std::filesystem::path& relative, const std::string& text) const
    {
        writeFile(bundlePath() / relative, text);
    }

private:
    hipdnn_test_sdk::utilities::ScopedDirectory _directory;
};

/// A KernelDefinition of kind HIPRTC_FILE, in the shape the state manager builds one:
/// metadata already completed, originDirectory and treeRoot already stamped by the
/// loader.
KernelDefinition makeHiprtcKernel(const std::filesystem::path& root,
                                  const std::string& bundle,
                                  const std::string& dtype,
                                  std::map<std::string, std::string> defines,
                                  uint8_t seed)
{
    KernelDefinition kernel;
    kernel.kernelId = id(seed);
    kernel.packId = id(static_cast<uint8_t>(seed + 1));
    kernel.dispatchId = id(static_cast<uint8_t>(seed + 2));
    kernel.name = "attention_dropin";
    kernel.source.kind = KernelSourceKind::HIPRTC_FILE;
    kernel.source.bundle = bundle;
    kernel.source.sourceFile = SOURCE_FILE;
    kernel.source.entryPoint = ENTRY_POINT;
    kernel.source.defines = std::move(defines);
    kernel.originDirectory = root;
    kernel.treeRoot = root;
    kernel.metadata
        = {{std::string(BLOCK_SIZE_FIELD), int64_t{256}}, {std::string(DTYPE_FIELD), dtype}};
    return kernel;
}

/// What the adapter handed the compiler, captured off the mock.
struct CapturedCompile
{
    std::string sourceText;
    std::string programName;
    std::vector<compilation::KernelHeader> headers;
    std::vector<std::string> options;
};

/// The loader half of buildIngestorKernelCode's two arguments. Never consulted on this
/// path -- a hiprtc_file kernel names no archive -- but the signature requires one.
const compilation::KpackKernelLoader& unusedKpackLoader()
{
    static const compilation::KpackKernelLoader s_loader(pointwiseKpackModuleCache());
    return s_loader;
}

/// Arms @p compiler to record one compileSource call and answer it with a program whose
/// getKernel succeeds for @p entryPoint and fails for anything else -- so a case that
/// asserts the entry point resolved is asserting the adapter asked for the right one.
void expectOneCompile(MockKernelCompiler& compiler,
                      CapturedCompile& captured,
                      const std::string& entryPoint = ENTRY_POINT)
{
    EXPECT_CALL(compiler, compileSource(_, _, _, _))
        .WillOnce([&captured, entryPoint](const std::string& sourceText,
                                          const std::string& programName,
                                          const std::vector<compilation::KernelHeader>& headers,
                                          const std::vector<std::string>& options) {
            captured.sourceText = sourceText;
            captured.programName = programName;
            captured.headers = headers;
            captured.options = options;

            auto program = std::make_unique<MockCompiledProgram>();
            EXPECT_CALL(*program, getKernel(entryPoint)).WillOnce([]() {
                return std::make_unique<MockRunnableKernel>();
            });
            return program;
        });
}

/// Compile options in the shape the pointwise handler builds them, so a bound define is
/// added to a populated set rather than an empty one.
compilation::KernelCompileOptions makeOptions(const GraphFixture& fixture)
{
    const auto& tensors = fixture.context().graph.getTensorMap();
    return compilation::KernelCompileOptions(tensors.at(INPUT_A_UID), std::string("gfx942"));
}

// ---------------------------------------------------------------------------
// (a) A bundle-sourced kernel compiles and resolves its entry point
// ---------------------------------------------------------------------------

TEST(TestHiprtcFileKernelSource, CompilesABundleSourcedKernelAndResolvesItsEntryPoint)
{
    const BundleTree tree("compiles");
    const GraphFixture fixture(buildPointwiseGraph());
    auto options = makeOptions(fixture);

    MockKernelCompiler compiler;
    CapturedCompile captured;
    expectOneCompile(compiler, captured);

    const auto kernel = makeHiprtcKernel(tree.root(), BUNDLE_NAME, "bfloat16", {}, 0x10);
    auto code = buildIngestorKernelCode(
        compiler, unusedKpackLoader(), fixture.context(), kernel, options);

    EXPECT_NE(code.program, nullptr);
    // The entry point resolved: the mock answers getKernel only for ENTRY_POINT.
    EXPECT_NE(code.kernel, nullptr);
    // The text came off disk, not from the embedded table.
    EXPECT_EQ(captured.sourceText, SOURCE_TEXT);
    EXPECT_EQ(captured.programName, std::filesystem::weakly_canonical(tree.sourcePath()).string());
}

// ---------------------------------------------------------------------------
// (b) One source file, two variants, two different -D flags
// ---------------------------------------------------------------------------

/// The feature itself: two kernels differing only in `metadata.dtype` share a source file
/// and compile with different bound defines. Asserted on the exact flag text, because two
/// variants whose flags render identically compile to one binary and silently become one
/// kernel.
TEST(TestHiprtcFileKernelSource, BindsEachVariantsMetadataIntoItsDefines)
{
    const BundleTree tree("variants");
    const GraphFixture fixture(buildPointwiseGraph());
    const std::map<std::string, std::string> defines{{"DTYPE", "$kernel.dtype"},
                                                     {"BLOCK_M", "$kernel.block_size"}};

    MockKernelCompiler compiler;
    CapturedCompile first;
    CapturedCompile second;
    {
        ::testing::InSequence order;
        expectOneCompile(compiler, first);
        expectOneCompile(compiler, second);
    }

    auto firstOptions = makeOptions(fixture);
    const auto bfloat = makeHiprtcKernel(tree.root(), BUNDLE_NAME, "bfloat16", defines, 0x20);
    buildIngestorKernelCode(compiler, unusedKpackLoader(), fixture.context(), bfloat, firstOptions);

    auto secondOptions = makeOptions(fixture);
    const auto half = makeHiprtcKernel(tree.root(), BUNDLE_NAME, "float16", defines, 0x30);
    buildIngestorKernelCode(compiler, unusedKpackLoader(), fixture.context(), half, secondOptions);

    EXPECT_THAT(first.options, Contains("-DDTYPE=bfloat16"));
    EXPECT_THAT(first.options, Not(Contains("-DDTYPE=float16")));
    EXPECT_THAT(second.options, Contains("-DDTYPE=float16"));
    EXPECT_THAT(second.options, Not(Contains("-DDTYPE=bfloat16")));

    // An int field renders decimal, so a second bound define is not a second dtype.
    EXPECT_THAT(first.options, Contains("-DBLOCK_M=256"));
}

// ---------------------------------------------------------------------------
// (c) A source outside the tree is refused
// ---------------------------------------------------------------------------

/// A descriptor may name a bundle inside the tree it was loaded from and nowhere else.
/// Without this, a dropped-in descriptor could name any readable file on the machine and
/// have it compiled as device code.
TEST(TestHiprtcFileKernelSource, RefusesABundleThatResolvesOutsideTheDescriptorTree)
{
    const BundleTree tree("escape");
    const GraphFixture fixture(buildPointwiseGraph());
    auto options = makeOptions(fixture);

    MockKernelCompiler compiler;
    // Nothing is compiled: the refusal happens before the file is even opened.
    EXPECT_CALL(compiler, compileSource(_, _, _, _)).Times(0);

    // `..` is normalised rather than rejected by weakly_canonical, which is exactly why
    // the containment check compares canonical forms.
    const std::string escaping = std::string("..") + "/" + BUNDLE_NAME;
    const auto kernel = makeHiprtcKernel(tree.root(), escaping, "bfloat16", {}, 0x40);

    try
    {
        buildIngestorKernelCode(compiler, unusedKpackLoader(), fixture.context(), kernel, options);
        FAIL() << "a bundle outside the descriptor tree must not be compiled";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        // Named, not just refused: the author has to be told which field is wrong.
        EXPECT_THAT(error.what(), HasSubstr(escaping));
        EXPECT_THAT(error.what(), HasSubstr("outside the descriptor tree"));
        EXPECT_THAT(error.what(), HasSubstr(tree.root().string()));
    }
}

// ---------------------------------------------------------------------------
// (d) A define naming an undeclared field drops the pack at load
// ---------------------------------------------------------------------------

/// Every `$kernel.<field>` is checked against the engine's KMD once, at set resolution.
/// A token naming a field the schema does not declare can never resolve, and catching it
/// here turns what would otherwise be a hipRTC failure on the first graph that selected
/// the kernel into an authoring message at load. Load-time failure in this codebase is a
/// drop with a log, not a throw, so the assertion is on the log.
TEST(TestHiprtcFileKernelSource, DropsAPackWhoseDefineNamesAnUndeclaredField)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory descriptors(uniqueDirectory("undeclared"));

    const std::string schemaId = "aa000000-0000-4000-8000-000000000001";
    const std::string engineId = "aa000000-0000-4000-8000-000000000002";
    const std::string matcherId = "aa000000-0000-4000-8000-000000000003";
    const std::string dispatchId = "aa000000-0000-4000-8000-000000000004";
    const std::string packId = "aa000000-0000-4000-8000-000000000005";
    const std::string kernelId = "aa000000-0000-4000-8000-000000000006";

    const auto write = [&descriptors](const std::string& stem, const nlohmann::json& body) {
        std::ofstream file(descriptors.path() / stem, std::ios::binary);
        file << body.dump(2) << '\n';
    };

    write("schema.kmd.json",
          {{"version", "1.0"},
           {"id", schemaId},
           {"name", "attention fields"},
           {"fields", {{{"name", "dtype"}, {"type", "string"}}}}});
    write("engine.ued.json",
          {{"version", "1.0"},
           {"id", engineId},
           {"name", "hiprtc:UndeclaredDefine"},
           {"metadata", schemaId}});
    write("matcher.umd.json",
          {{"version", "1.0"},
           {"id", matcherId},
           {"name", "graph shape"},
           {"scope", "graph"},
           {"match_symbol", "hiprtc.undeclared.match"}});
    write("dispatch.udd.json",
          {{"version", "1.0"},
           {"id", dispatchId},
           {"name", "dispatch"},
           {"dispatch_symbol", "hiprtc.undeclared.dispatch"}});
    write("pack.kdp.json",
          {{"version", "1.0"},
           {"id", packId},
           {"name", "the undeclared-define pack"},
           {"matchers", {matcherId}},
           {"engine", engineId},
           {"dispatch", dispatchId},
           {"kernelDescriptors",
            {{{"version", "1.0"},
              {"id", kernelId},
              {"name", "attention_dropin"},
              {"kernel_source",
               {{"kind", "hiprtc_file"},
                {"bundle", BUNDLE_NAME},
                {"source_file", SOURCE_FILE},
                {"entry_point", ENTRY_POINT},
                // `nonesuch` is not in the KMD above.
                {"defines", {{"DTYPE", "$kernel.nonesuch"}}}}},
              {"metadata", {{"dtype", "bfloat16"}}},
              {"priority", 0}}}}});

    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    const auto sets = resolveDescriptorSets(loadDescriptorCatalog(descriptors.path()));

    // The pack was the engine's only one, so dropping it drops the engine.
    EXPECT_TRUE(sets.empty());
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "nonesuch"));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "dropping the pack"));
}

// ---------------------------------------------------------------------------
// (e) Two bundles, one file name: two cache entries
// ---------------------------------------------------------------------------

/// The compile cache is keyed on (program name, options). If a hiprtc_file kernel were
/// keyed on the bare `source_file`, two bundles each shipping `attention.hip` with
/// identical defines would collide and the second would silently be served the first's
/// binary -- two different kernels, one code object, no diagnostic.
TEST(TestHiprtcFileKernelSource, KeysTwoBundlesSharingASourceFileNameApart)
{
    const BundleTree firstTree("cache_a");
    const BundleTree secondTree("cache_b");
    const GraphFixture fixture(buildPointwiseGraph());
    const std::map<std::string, std::string> defines{{"DTYPE", "$kernel.dtype"}};

    MockKernelCompiler compiler;
    CapturedCompile first;
    CapturedCompile second;
    {
        ::testing::InSequence order;
        expectOneCompile(compiler, first);
        expectOneCompile(compiler, second);
    }

    auto firstOptions = makeOptions(fixture);
    const auto fromFirst
        = makeHiprtcKernel(firstTree.root(), BUNDLE_NAME, "bfloat16", defines, 0x50);
    buildIngestorKernelCode(
        compiler, unusedKpackLoader(), fixture.context(), fromFirst, firstOptions);

    auto secondOptions = makeOptions(fixture);
    const auto fromSecond
        = makeHiprtcKernel(secondTree.root(), BUNDLE_NAME, "bfloat16", defines, 0x60);
    buildIngestorKernelCode(
        compiler, unusedKpackLoader(), fixture.context(), fromSecond, secondOptions);

    // Same file name, same defines -- and therefore identical options.
    ASSERT_EQ(first.options, second.options);
    EXPECT_NE(first.programName, second.programName);

    // The distinction survives into the key the cache actually builds, which is the only
    // thing that decides whether the second compile is served the first's binary.
    EXPECT_NE(HipMlopsSourceModuleCache::makeKey(
                  first.sourceText, first.programName, first.headers, first.options),
              HipMlopsSourceModuleCache::makeKey(
                  second.sourceText, second.programName, second.headers, second.options));
}

// ---------------------------------------------------------------------------
// (f) A bundle header that resolves outside the tree is refused
// ---------------------------------------------------------------------------

/// The containment rule governs every file the adapter reads, not only the two the
/// descriptor names. is_regular_file follows a symlink, so without this a link dropped
/// into an otherwise contained bundle is read whole and handed to hipRTC as a virtual
/// header -- and the author controls the `.hip` that #includes it, so the contents come
/// back in the compile log.
TEST(TestHiprtcFileKernelSource, RefusesABundleHeaderThatResolvesOutsideTheDescriptorTree)
{
    const BundleTree tree("header_escape");
    const hipdnn_test_sdk::utilities::ScopedDirectory outside(uniqueDirectory("header_outside"));
    const auto secret = outside.path() / "secret.txt";
    writeFile(secret, "// not the author's to read\n");

    // A symlink INSIDE the bundle, pointing out of the tree: the shape a blanket
    // is_symlink() refusal would catch and a lexical check would miss.
    std::filesystem::create_symlink(secret, tree.bundlePath() / "helper.h");

    const GraphFixture fixture(buildPointwiseGraph());
    auto options = makeOptions(fixture);

    MockKernelCompiler compiler;
    EXPECT_CALL(compiler, compileSource(_, _, _, _)).Times(0);

    const auto kernel = makeHiprtcKernel(tree.root(), BUNDLE_NAME, "bfloat16", {}, 0x70);

    try
    {
        buildIngestorKernelCode(compiler, unusedKpackLoader(), fixture.context(), kernel, options);
        FAIL() << "a bundle header outside the descriptor tree must not be compiled against";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        // The offending header is named, and the refusal reads the same as the `bundle`
        // one: one rule, one message shape, whichever file crossed the boundary.
        EXPECT_THAT(error.what(), HasSubstr("bundle header 'helper.h'"));
        EXPECT_THAT(error.what(), HasSubstr("outside the descriptor tree"));
        EXPECT_THAT(error.what(), HasSubstr(tree.root().string()));
    }
}

// ---------------------------------------------------------------------------
// (g) The bundle's header list: which files, and in what order
// ---------------------------------------------------------------------------

/// The whole virtual-header contract in one assertion. A header in a subdirectory or
/// under an unlisted extension is silently absent from the list and surfaces only as a
/// hipRTC "file not found" inside someone else's source; ordering is asserted because the
/// list is part of the compile cache key, so a directory-order-dependent list would make
/// the same kernel key differently between runs.
TEST(TestHiprtcFileKernelSource, CollectsOnlyTopLevelHeadersByExtensionInNameOrder)
{
    const BundleTree tree("headers");
    tree.writeIntoBundle("b.h", "// b.h\n");
    tree.writeIntoBundle("a.hpp", "// a.hpp\n");
    tree.writeIntoBundle("z.txt", "// z.txt\n");
    tree.writeIntoBundle("helper.hip", "// helper.hip\n");
    tree.writeIntoBundle(std::filesystem::path("nested") / "c.h", "// nested/c.h\n");

    const GraphFixture fixture(buildPointwiseGraph());
    auto options = makeOptions(fixture);

    MockKernelCompiler compiler;
    CapturedCompile captured;
    expectOneCompile(compiler, captured);

    const auto kernel = makeHiprtcKernel(tree.root(), BUNDLE_NAME, "bfloat16", {}, 0x80);
    buildIngestorKernelCode(compiler, unusedKpackLoader(), fixture.context(), kernel, options);

    // Exactly these, in this order, at the END of the list: the extension filter kept
    // `.txt` and `.hip` out, the one-level rule kept `nested/c.h` out, and the name sort
    // put `a.hpp` before `b.h` though the directory holds them the other way round.
    //
    // The bundle's contribution is checked as the tail rather than as the whole vector.
    // What precedes it is the provider's embedded header list, which is whatever this
    // binary happens to embed -- any engine adding a kernel header changes it, and that
    // is not this test's subject. Pinning the whole vector made an unrelated pack's new
    // header fail here instead of wherever it was wrong.
    const std::vector<compilation::KernelHeader> expected{{"a.hpp", "// a.hpp\n"},
                                                          {"b.h", "// b.h\n"}};
    ASSERT_GE(captured.headers.size(), expected.size());
    EXPECT_EQ(std::vector<compilation::KernelHeader>(
                  captured.headers.end() - static_cast<std::ptrdiff_t>(expected.size()),
                  captured.headers.end()),
              expected);

    // And the excluded files are absent from the list ENTIRELY, not merely out of order:
    // a `.txt`, a second `.hip` or a nested header reaching hipRTC at all is the defect.
    for(const auto* excluded : {"z.txt", "helper.hip", "c.h"})
    {
        EXPECT_EQ(
            std::count_if(captured.headers.begin(),
                          captured.headers.end(),
                          [excluded](const auto& header) { return header.first == excluded; }),
            0)
            << excluded;
    }
}

// ---------------------------------------------------------------------------
// (h) A bundle header shadowing an embedded one is refused
// ---------------------------------------------------------------------------

/// hipRTC resolves the first match, so a collision would be broken invisibly and the
/// author's mental model -- "my file wins" or "theirs does" -- would be right only by
/// luck. Reached by calling the collector with an injected embedded list: this binary
/// embeds no headers, which is precisely why the parameter exists.
TEST(TestHiprtcFileKernelSource, RefusesABundleHeaderNamedLikeAnEmbeddedHeader)
{
    const BundleTree tree("collision");
    tree.writeIntoBundle("helper.h", "// the bundle's helper\n");

    const auto kernel = makeHiprtcKernel(tree.root(), BUNDLE_NAME, "bfloat16", {}, 0x90);
    const std::vector<const char*> embeddedNames{"helper.h"};
    const std::vector<std::string_view> embeddedTexts{"// the provider's helper\n"};

    try
    {
        detail::collectKernelHeaders(
            kernel, tree.bundlePath(), "kernel 'attention_dropin'", embeddedNames, embeddedTexts);
        FAIL() << "a bundle header shadowing an embedded one must not be assembled";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        EXPECT_THAT(error.what(), HasSubstr("bundle header 'helper.h'"));
        EXPECT_THAT(error.what(), HasSubstr("embedded headers"));
    }
}

// ---------------------------------------------------------------------------
// (i) A source_file escaping from inside a contained bundle is refused
// ---------------------------------------------------------------------------

/// `source_file` is authored too, so containment is checked on it separately rather than
/// inherited from the bundle: `../../escape.hip` inside a bundle that is itself perfectly
/// contained would otherwise be compiled as device code.
TEST(TestHiprtcFileKernelSource, RefusesASourceFileThatEscapesFromInsideAContainedBundle)
{
    const BundleTree tree("source_escape");
    const GraphFixture fixture(buildPointwiseGraph());
    auto options = makeOptions(fixture);

    MockKernelCompiler compiler;
    EXPECT_CALL(compiler, compileSource(_, _, _, _)).Times(0);

    auto kernel = makeHiprtcKernel(tree.root(), BUNDLE_NAME, "bfloat16", {}, 0xA0);
    const std::string escaping = "../../escape.hip";
    kernel.source.sourceFile = escaping;

    try
    {
        buildIngestorKernelCode(compiler, unusedKpackLoader(), fixture.context(), kernel, options);
        FAIL() << "a source_file outside the descriptor tree must not be compiled";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        // The bundle is contained, so naming the field is the only way the author learns
        // which half of the pair is wrong.
        EXPECT_THAT(error.what(), HasSubstr("source_file '" + escaping + "'"));
        EXPECT_THAT(error.what(), HasSubstr("outside the descriptor tree"));
    }
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
