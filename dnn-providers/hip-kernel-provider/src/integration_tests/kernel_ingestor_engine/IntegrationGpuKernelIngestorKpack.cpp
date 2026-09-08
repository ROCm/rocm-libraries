// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Logging.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/knob/Knob.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hip_kernel_provider::test_utilities;

/**
 * @file IntegrationGpuKernelIngestorKpack.cpp
 * @brief A kernel that was compiled and packed at build time, executed end to end through
 *        the public frontend API. Nothing here names a loader type or a build path: the
 *        descriptor set reaches the runtime because the packaging rule staged it beside
 *        the plugin, and the kernel binary reaches the device because the KPACK kernel
 *        source resolved its archive relative to the descriptor that declared it.
 *
 * The suite deliberately sets no HIPDNN_DESCRIPTOR_DIR and its CTest registration carries
 * no ENVIRONMENT entry -- the module-relative walk from the loaded plugin is the thing
 * under test, and overriding it would test the override instead.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

namespace
{

/// The packaged fixture declares its own engine. Reusing the shipped pointwise engine's
/// identity would collide on the completed metadata tuple and take that engine down with
/// it, so this name appears nowhere in src/engines.
constexpr const char* PACKED_ENGINE_NAME = "hipkernel:pointwise_packed";

/// The same packaged pointwise catalog, ranked by a trained UHD instead of a
/// compiled score function.
///
/// Its descriptors sit beside the native-ranked engine's and share their
/// matchers and dispatch handler, so the two differ in exactly one thing: how
/// the catalog is ordered. That is what lets a test attribute an ordering to the
/// model rather than to anything else about the pack.
constexpr const char* MODEL_ENGINE_NAME = "hipkernel:pointwise_model";

/// The one knob both packaged pointwise engines expose. Its default is the
/// top-ranked kernel, which is how an ordering becomes observable through the
/// public API rather than by reaching into the engine.
constexpr const char* BLOCK_SIZE_KNOB = "block_size";

/// The engine that ships with the provider. Its kernels are `embedded_source`, so it is
/// reachable whatever state the packaged archive is in, and it claims the same single-node
/// FLOAT add. That makes it the fallback ...SurvivesABrokenArchive requires to still serve.
constexpr const char* SHIPPED_POINTWISE_ENGINE_NAME = "hipkernel:Pointwise";

/// Distinguishes the packaged FIXTURE's archive from every other root's in the shared
/// descriptor tree. Each source root packs under its own archive group, so the group name
/// is what names an archive to a particular root; this is the group the fixture root is
/// wired with in HkpPackaging.cmake (HKP_GROUP_TESTFIXTURE).
constexpr const char* PACKED_FIXTURE_ARCHIVE_STEM = "testfixture";

/// Header-length garbage: long enough that the file exists and is readable, short enough
/// that no table of contents can be parsed out of it.
constexpr size_t CORRUPTION_BYTE_COUNT = 64;

/// Holds the pristine archive while ...SurvivesABrokenArchive breaks the staged one.
constexpr const char* BACKUP_DIR_NAME = "kpack-fixture-backup";

/// Appended to an archive's filename to name its backup. The backup must not end in
/// .kpack, so that nothing can mistake it for a staged archive.
constexpr const char* PRISTINE_SUFFIX = ".pristine";

/// A single-node FLOAT add: the one graph shape the packaged descriptor set claims.
std::shared_ptr<TensorAttributes> makeScalarTensor(int64_t uid, const std::string& name)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_uid(uid)
        .set_name(name)
        .set_dim({1, 1, 1, 1})
        .set_stride({1, 1, 1, 1})
        .set_data_type(DataType::FLOAT);
    return tensor;
}

std::shared_ptr<Graph> buildPointwiseAddGraph()
{
    auto graph = std::make_shared<Graph>();
    graph->set_name("packed_pointwise")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto a = makeScalarTensor(1, "A");
    auto b = makeScalarTensor(2, "B");

    PointwiseAttributes attrs;
    attrs.set_name("packed_pointwise").set_mode(PointwiseMode::ADD);
    auto c = graph->pointwise(a, b, attrs);
    c->set_uid(3).set_name("C").set_output(true).set_data_type(DataType::FLOAT);

    return graph;
}

/// The directory the loader walks, derived the same way the loader derives it: from the
/// plugin module, not from a path compiled in at configure time. HIPDNN_PACKAGED_FIXTURE_SUBDIR
/// is the single spelling of the arch_content layout, forwarded from the provider's
/// CMakeLists so a rename cannot leave a stale copy here.
std::filesystem::path packagedDescriptorRoot()
{
    const std::filesystem::path pluginTarget(PLUGIN_PATH);
    return std::filesystem::weakly_canonical(getCurrentExecutableDirectory()
                                             / pluginTarget.parent_path()
                                             / HIPDNN_PACKAGED_FIXTURE_SUBDIR);
}

/// Every .kpack under `root`, sorted so the choice of "first" is stable across runs.
std::vector<std::filesystem::path> findKpackArchives(const std::filesystem::path& root)
{
    std::vector<std::filesystem::path> archives;

    std::error_code ec;
    if(!std::filesystem::is_directory(root, ec))
    {
        return archives;
    }

    for(const auto& entry : std::filesystem::recursive_directory_iterator(root, ec))
    {
        if(entry.is_regular_file(ec) && entry.path().extension() == ".kpack")
        {
            archives.push_back(entry.path());
        }
    }

    std::sort(archives.begin(), archives.end());
    return archives;
}

/// The archives packed for `arch` specifically, which is not the same question as which
/// archives exist.
///
/// The packer emits one shard per arch and the ingestor drops a pack whose arch the
/// device does not satisfy, so on a device outside GPU_TARGETS the tree is full of
/// archives that no engine here can ever claim. Asking only "was anything packed" then
/// runs every case against an engine that cannot appear, and they fail for a reason that
/// has nothing to do with what they test -- which cost real debugging time on a gfx90a
/// box holding a gfx942 tree.
std::vector<std::filesystem::path> findKpackArchivesForArch(const std::filesystem::path& root,
                                                            const std::string& arch)
{
    std::vector<std::filesystem::path> matching;
    for(const auto& archive : findKpackArchives(root))
    {
        // The shard directory is named for its arch: <root>/<arch>/kpack/<file>.kpack.
        const auto shard = archive.parent_path().parent_path().filename().string();
        if(shard == arch)
        {
            matching.push_back(archive);
        }
    }
    return matching;
}

/// The directory holding the pristine archive. It sits beside the descriptor tree rather
/// than in the working directory, so the same backup is found again whatever directory the
/// binary was launched from, and outside the tree findKpackArchives() walks, so it can never
/// be mistaken for a staged archive.
std::filesystem::path backupRootPath()
{
    return packagedDescriptorRoot().parent_path() / BACKUP_DIR_NAME;
}

std::vector<char> readWholeFile(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

[[nodiscard]] bool writeWholeFile(const std::filesystem::path& path, const std::vector<char>& bytes)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if(!bytes.empty())
    {
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
    out.close();
    return out.good();
}

/// Nothing the packager emits is all zeroes, and the corruption below is. The fixture uses
/// this to recognise its own damage on disk rather than accepting it as pristine bytes.
bool isAllZero(const std::vector<char>& bytes)
{
    return !bytes.empty()
           && std::all_of(bytes.begin(), bytes.end(), [](char byte) { return byte == '\0'; });
}

/// Puts back whatever an abandoned backup still holds. The backup directory is removed in
/// teardown, so one surviving here means a run was killed while an archive was deliberately
/// corrupt, and these are the last known good bytes.
///
/// Returns a description of the first failure, or an empty string.
std::string recoverAbandonedBackups(const std::vector<std::filesystem::path>& archives)
{
    const auto backupRoot = backupRootPath();

    std::error_code ec;
    if(!std::filesystem::is_directory(backupRoot, ec))
    {
        return {};
    }

    for(const auto& archive : archives)
    {
        const auto pristine = backupRoot / (archive.filename().string() + PRISTINE_SUFFIX);
        if(!std::filesystem::is_regular_file(pristine, ec))
        {
            continue;
        }

        const auto bytes = readWholeFile(pristine);
        if(bytes.empty())
        {
            continue;
        }

        if(!writeWholeFile(archive, bytes))
        {
            return "could not restore " + archive.string() + " from " + pristine.string();
        }
    }

    return {};
}

/// Drops the provider's resident kpack modules, so the next dispatch re-reads its
/// archive from disk.
///
/// The module cache is process-lifetime by design -- one hipModule_t per
/// (archive, toc_key, arch), deliberately outliving every Container. That is correct
/// for the product and fatal for ...SurvivesABrokenArchive: if any earlier case has
/// already executed the packaged kernel, a resident module serves the plan, the corrupt
/// bytes are read by nothing, and the diagnostics this suite asserts on never fire.
/// This suite used to depend on being FIRST in the file to avoid that, which
/// --gtest_shuffle destroys.
///
/// Reached by dlsym rather than a direct call because this binary links only the SDKs;
/// the provider arrives via dlopen. Same route as
/// IntegrationGpuKernelIngestorDirectAbi.SelfRegistersAllEngineIds, except that this
/// takes the RTLD_NOLOAD form: the harness has already loaded the plugin, and the point
/// is to reach the statics in THAT copy. openLibrary() would refcount the same image
/// rather than produce a second one, but asking for a load at all would misstate the
/// intent -- if the plugin is somehow not resident, resetting a freshly loaded copy's
/// empty caches would be a silent no-op rather than the error it should be.
///
/// Returns a description of the failure, or an empty string.
std::string resetProviderModuleCaches()
{
    const std::filesystem::path pluginTarget(PLUGIN_PATH);
    const auto pluginFile = hipdnn_data_sdk::utilities::LIB_PREFIX
                            + pluginTarget.filename().string()
                            + hipdnn_data_sdk::utilities::SHARED_LIB_EXT;
    const auto pluginPath = std::filesystem::weakly_canonical(
        getCurrentExecutableDirectory() / pluginTarget.parent_path() / pluginFile);

    // RTLD_NOLOAD: returns null rather than throwing when the image is not already
    // resident, unlike openLibrary().
    auto* library = hipdnn_data_sdk::utilities::openLoadedLibrary(pluginPath);
    if(library == nullptr)
    {
        return "the provider at " + pluginPath.string()
               + " is not loaded, so there are no resident modules to reset";
    }

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto* reset = reinterpret_cast<void (*)()>(hipdnn_data_sdk::utilities::getSymbol(
        library, "hipdnnEnginePluginResetKpackModuleCacheForTesting"));
    if(reset == nullptr)
    {
        // Balances the RTLD_NOLOAD reference taken above before bailing out.
        hipdnn_data_sdk::utilities::closeLibrary(library);
        return "the provider at " + pluginPath.string()
               + " exports no hipdnnEnginePluginResetKpackModuleCacheForTesting; it was "
                 "built without HIPDNN_ENABLE_KERNEL_INGESTOR, or the test-only reset "
                 "hook was removed";
    }

    reset();

    // Drops only the reference this call took. The harness holds its own, so the plugin
    // stays loaded and the statics just reset are the ones the next dispatch will use.
    hipdnn_data_sdk::utilities::closeLibrary(library);
    return {};
}

} // namespace

class IntegrationGpuKernelIngestorKpack
    : public hip_kernel_provider::test_utilities::IntegrationGraphVerificationHarness<float, int>
{
protected:
    void SetUp() override
    {
        IntegrationGraphVerificationHarness<float, int>::SetUp();
        if(IsSkipped() || HasFatalFailure())
        {
            return;
        }

        const auto allArchives = findKpackArchives(packagedDescriptorRoot());
        if(allArchives.empty())
        {
            GTEST_SKIP() << "packaged artifact absent -- packaging did not run. Looked for "
                            "*.kpack under "
                         << packagedDescriptorRoot()
                         << ". Configure with -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON and a hipcc "
                            "the packaging rule can find.";
        }

        // "Something was packed" is not "something was packed for THIS device". The
        // packer emits one shard per arch and the ingestor drops a pack whose arch the
        // device does not satisfy, so on a device outside GPU_TARGETS the engine below
        // can never be a candidate and every case would fail asserting about it. That is
        // environmental -- the build packed for other arches -- not a defect.
        const auto arch = hip_kernel_provider_common::getDeviceString(_stream);
        _archives = findKpackArchivesForArch(packagedDescriptorRoot(), arch);
        if(_archives.empty())
        {
            GTEST_SKIP() << "nothing was packaged for this device (" << arch
                         << "); the staged tree holds shards for other arches only. "
                            "Add "
                         << arch << " to GPU_TARGETS to exercise this suite here.";
        }

        // Before any case reads the staged tree, not just the one that damages it: gtest runs
        // the cases in declaration order, so recovering here means a killed run is undone
        // whichever case happens to go first.
        const auto recoveryError = recoverAbandonedBackups(_archives);
        ASSERT_TRUE(recoveryError.empty()) << recoveryError;
    }

    static int64_t packedEngineId()
    {
        return hipdnn_data_sdk::utilities::engineNameToId(PACKED_ENGINE_NAME);
    }

    static int64_t shippedPointwiseEngineId()
    {
        return hipdnn_data_sdk::utilities::engineNameToId(SHIPPED_POINTWISE_ENGINE_NAME);
    }

    static int64_t modelEngineId()
    {
        return hipdnn_data_sdk::utilities::engineNameToId(MODEL_ENGINE_NAME);
    }

    /// Pins the packaged engine and compiles. Every step asserts: the artifact is on disk,
    /// so a failure here is the regression this suite exists to catch, never a skip.
    void buildAndCompilePacked(Graph& graph)
    {
        graph.set_preferred_engine_id_ext(packedEngineId());

        auto result = graph.build_operation_graph(_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        // The packaged engine is an addition to the catalog, not a replacement: the shipped
        // pointwise engine claims this graph too. Membership plus the pin above is what
        // makes the execution below attributable to the packaged descriptors.
        std::vector<int64_t> rankedEngineIds;
        result = graph.get_ranked_engine_ids(rankedEngineIds);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_NE(std::find(rankedEngineIds.begin(), rankedEngineIds.end(), packedEngineId()),
                  rankedEngineIds.end())
            << "the packaged engine did not offer itself for a single-node FLOAT add";

        result = graph.create_execution_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph.check_support();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph.build_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    }

    std::vector<std::filesystem::path> _archives;
};

// ---------------------------------------------------------------------------
// The artifact fails without taking the process with it
//
// Position-independent, deliberately. This suite used to rely on being FIRST in the
// file -- gtest registers suites in definition order within a translation unit, so it
// ran before IntegrationGpuKernelIngestorKpack below and therefore before any case had
// executed the packaged kernel. That mattered because the module cache is
// process-lifetime: once the kernel has run, a resident hipModule_t serves the plan and
// the corrupt bytes on disk are read by nothing, so every diagnostic below silently
// stops firing. --gtest_shuffle destroys that ordering, and the suite failed
// deterministically whenever the shuffle put the executing case first.
//
// SetUp() now drops the resident modules explicitly, so the corrupt archive is re-read
// whatever ran before. Nothing here depends on file position any more.
// ---------------------------------------------------------------------------

/// A truncated archive must produce a diagnosable failure, never a crash, and must leave
/// nothing behind for the next test in this binary.
///
/// The staged tree is shared process state: descriptor discovery memoizes into a
/// function-local static, so the engine can only ever read the one tree it found first,
/// and this suite is forbidden from redirecting it. Breaking a private copy would
/// therefore corrupt bytes nothing reads. A ScopedDirectory holds the pristine archive
/// instead, and TearDown puts it back unconditionally (not at the end of the body, which
/// an assertion failure would skip).
///
/// The staged archive is also durable state that outlives the process. TearDown always
/// removes the backup directory, so a leftover backup means an earlier run was killed while
/// the archive was corrupt; recoverAbandonedBackups(), run from the base fixture's SetUp,
/// puts those last known good bytes back, costing a re-run rather than cementing the
/// damage into every later build.
class IntegrationGpuKernelIngestorKpackBroken : public IntegrationGpuKernelIngestorKpack
{
protected:
    void SetUp() override
    {
        IntegrationGpuKernelIngestorKpack::SetUp();
        if(IsSkipped() || HasFatalFailure())
        {
            return;
        }

        // The archive THIS fixture's engine reads, chosen by name rather than by taking
        // the first of a sorted list. Every source root stages into one descriptor tree
        // and is told apart by its archive group, so the tree holds one .kpack per root;
        // `front()` picked whichever sorted first, which is the product archive, and
        // corrupting that leaves the packaged fixture engine perfectly loadable while the
        // assertions below wait for a failure that never comes.
        const auto victim = std::find_if(
            _archives.begin(), _archives.end(), [](const std::filesystem::path& archive) {
                return archive.filename().string().find(PACKED_FIXTURE_ARCHIVE_STEM)
                       != std::string::npos;
            });
        ASSERT_NE(victim, _archives.end())
            << "no staged archive is named for the packaged fixture ('"
            << PACKED_FIXTURE_ARCHIVE_STEM << "'), so there is nothing this suite can break "
            << "that its own engine would read. Staged archives: " << [this] {
                   std::string names;
                   for(const auto& archive : _archives)
                   {
                       names += archive.filename().string() + " ";
                   }
                   return names;
               }();
        _victim = *victim;

        const auto backupRoot = backupRootPath();
        const auto backupName = _victim.filename().string() + PRISTINE_SUFFIX;

        _pristine = readWholeFile(_victim);
        ASSERT_FALSE(_pristine.empty()) << "staged archive is empty: " << _victim;
        ASSERT_FALSE(isAllZero(_pristine))
            << "the staged archive at " << _victim << " holds nothing but zero bytes and no "
            << "backup was available to restore it. Reconfigure to make the packaging rule "
               "stage it again.";

        // ScopedDirectory throws when the directory already exists, and the recovery in the
        // base fixture's SetUp has already taken everything of value out of it.
        std::error_code ec;
        std::filesystem::remove_all(backupRoot, ec);
        _backup = std::make_unique<ScopedDirectory>(backupRoot);
        _backupFile = _backup->path() / backupName;
        ASSERT_TRUE(writeWholeFile(_backupFile, _pristine))
            << "could not write the backup at " << _backupFile;
        _corrupted = true;
    }

    void TearDown() override
    {
        if(_corrupted)
        {
            // From the backup on disk rather than from memory, so a crash between here and
            // the write leaves a recoverable copy behind. The in-memory copy, verified
            // non-empty in SetUp, is the fallback: writing an empty read straight back
            // would truncate the shared staged archive to nothing.
            auto restored = readWholeFile(_backupFile);
            if(restored.empty())
            {
                restored = _pristine;
            }

            EXPECT_FALSE(restored.empty()) << "no bytes available to restore " << _victim;
            if(!restored.empty())
            {
                EXPECT_TRUE(writeWholeFile(_victim, restored))
                    << "could not restore the staged archive at " << _victim;
                EXPECT_EQ(readWholeFile(_victim).size(), restored.size())
                    << "the staged archive was only partly restored: " << _victim;
            }

            // Symmetric with the reset in the test body: the restored archive is the
            // pristine one again, but the resident module still holds what was loaded
            // from the corrupt bytes (or nothing at all, if that load failed). Dropping
            // it here means the next case in this binary loads the good archive from
            // disk rather than inheriting this suite's damage. Unconditional, so it runs
            // whether the restore above succeeded or not.
            EXPECT_EQ(resetProviderModuleCaches(), "");
            _corrupted = false;
        }
        _backup.reset();
        IntegrationGpuKernelIngestorKpack::TearDown();
    }

    std::filesystem::path _victim;
    std::filesystem::path _backupFile;
    std::vector<char> _pristine;
    std::unique_ptr<ScopedDirectory> _backup;
    bool _corrupted = false;
};

TEST_F(IntegrationGpuKernelIngestorKpackBroken, SurvivesABrokenArchive)
{
    ASSERT_TRUE(writeWholeFile(_victim, std::vector<char>(CORRUPTION_BYTE_COUNT, '\0')))
        << "could not write the corrupt archive at " << _victim;

    // After the corruption, not in SetUp: a reset before the bytes change would be
    // undone by anything that builds a plan in between, and the cache would be warm
    // again by the time build_plans() below runs. Dropping the resident modules here
    // is what forces the packaged engine to actually re-read the damaged archive --
    // without it a module left over from an earlier case serves the plan, no diagnostic
    // is emitted, and every EXPECT below fails for a reason that is not the product's.
    ASSERT_EQ(resetProviderModuleCaches(), "");

    // No preferred engine here, unlike ExecutesAPackagedKernelOnDevice. Both engines claim
    // this graph, and the point of this case is what happens when one of them is broken:
    // BuildPlanPolicy::ALL below attempts every ranked plan, so the packaged engine reads
    // the corrupt bytes and fails at whatever rank it holds, and the shipped engine is
    // still there to serve. A pin would decide the outcome instead of observing it.
    auto graph = buildPointwiseAddGraph();

    auto result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    std::vector<int64_t> rankedEngineIds;
    result = graph->get_ranked_engine_ids(rankedEngineIds);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    ASSERT_NE(std::find(rankedEngineIds.begin(), rankedEngineIds.end(), packedEngineId()),
              rankedEngineIds.end())
        << "the packaged engine did not offer itself, so the corrupt bytes are read by nothing";
    ASSERT_NE(std::find(rankedEngineIds.begin(), rankedEngineIds.end(), shippedPointwiseEngineId()),
              rankedEngineIds.end())
        << "the shipped " << SHIPPED_POINTWISE_ENGINE_NAME
        << " engine did not offer itself, so there is nothing left to serve the graph";

    result = graph->create_execution_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    hipdnnSeverity_t savedLogLevel = HIPDNN_SEV_OFF;
    ASSERT_EQ(getGlobalLogLevel(savedLogLevel).code, ErrorCode::OK);

    // The diagnostics are the deliverable here as much as the fallback is: a failure the
    // engine swallows silently is indistinguishable from one that never happened.
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_WARN);
    ASSERT_EQ(setUserLogCallback(IsolatedLogRecorder::getIsolatedUserRecordingCallback(),
                                 HIPDNN_SEV_WARN,
                                 LogCallbackMode::SYNC,
                                 this)
                  .code,
              ErrorCode::OK);
    ASSERT_EQ(setGlobalLogLevel(HIPDNN_SEV_WARN).code, ErrorCode::OK);

    result = graph->build_plans(BuildPlanPolicy::ALL);

    setUserLogCallback(IsolatedLogRecorder::getIsolatedUserRecordingCallback(),
                       HIPDNN_SEV_OFF,
                       LogCallbackMode::SYNC,
                       this);
    setGlobalLogLevel(savedLogLevel);

    ASSERT_EQ(result.code, ErrorCode::OK)
        << "the shipped " << SHIPPED_POINTWISE_ENGINE_NAME
        << " engine must still serve this graph when the packaged archive is unreadable. "
        << result.err_msg << "\nRecorded logs:\n"
        << recorder.getRecordedLogsAsString();

    // An unreadable archive is reported at ERROR against the engine that owns it. If this
    // fails while the plan below still builds, the packaged engine was never asked -- a
    // resident module served it, and the corrupt bytes were read by nothing. That is the
    // failure mode the suite's position at the top of this file exists to prevent, so read
    // this assertion as the detector for a registration-order regression as well as for a
    // swallowed diagnostic.
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR,
                                          std::string("engine '") + PACKED_ENGINE_NAME
                                              + "' could not build a plan"))
        << "no ERROR reports the packaged engine's failure. Recorded logs:\n"
        << recorder.getRecordedLogsAsString();

    // The kernel-level detail -- which archive, and why it could not be read -- is carried
    // by the per-candidate record the plan builder emits as it walks past each failure.
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, _victim.filename().string()))
        << "no diagnostic names the archive that failed. Recorded logs:\n"
        << recorder.getRecordedLogsAsString();
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, "could not be read"))
        << "no diagnostic says the archive could not be read. Recorded logs:\n"
        << recorder.getRecordedLogsAsString();

    int64_t servingEngineId = 0;
    ASSERT_EQ(graph->get_execution_plan_engine_id(servingEngineId).code, ErrorCode::OK);
    EXPECT_EQ(servingEngineId, shippedPointwiseEngineId())
        << "engine id " << servingEngineId << " served the graph, not the shipped "
        << SHIPPED_POINTWISE_ENGINE_NAME << " engine";

    result = graph->check_support();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Routing to the surviving engine is not enough: it must still compute the right answer.
    int64_t workspaceSize = 0;
    ASSERT_EQ(graph->get_workspace_size(workspaceSize).code, ErrorCode::OK);
    ASSERT_GE(workspaceSize, 0);
    const hipdnn_data_sdk::utilities::Workspace workspace(static_cast<size_t>(workspaceSize));
    executeAndVerify(*graph, workspace.get(), /*seed=*/0);
}

// ---------------------------------------------------------------------------
// The artifact executes
// ---------------------------------------------------------------------------

TEST_F(IntegrationGpuKernelIngestorKpack, ExecutesAPackagedKernelOnDevice)
{
    auto graph = buildPointwiseAddGraph();
    ASSERT_NO_FATAL_FAILURE(buildAndCompilePacked(*graph));

    // The frontend's route into the engine's getMaxWorkspaceSize().
    int64_t workspaceSize = 0;
    auto result = graph->get_workspace_size(workspaceSize);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    ASSERT_GE(workspaceSize, 0);
    const hipdnn_data_sdk::utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

    executeAndVerify(*graph, workspace.get(), /*seed=*/0);
}

// ---------------------------------------------------------------------------
// A trained UHD decides which kernel runs
// ---------------------------------------------------------------------------
//
// The end of the loop RFC 0019 describes: a model trained offline by uhd_gen,
// packaged with its engine, loaded at plan build, and deciding a dispatch that
// then produces the right answer.
//
// The two packaged pointwise engines exist to make that attributable. Same
// kernels, same matchers, same dispatch handler; one ranks with
// `hipkernel.pointwise.score` (returns block_size, so 256 always wins) and one
// with a UHD trained to prefer 64. Every difference in outcome between them is
// the heuristic, because nothing else differs.

TEST_F(IntegrationGpuKernelIngestorKpack, AModelRankedEngineOffersItself)
{
    // Separated from the ordering assertion below so a packaging failure and a
    // ranking failure are distinguishable. If model.bin did not reach the
    // packed tree, the engine still loads -- it degrades to declared
    // order -- and only this case's sibling would fail, with no clue why.
    auto graph = buildPointwiseAddGraph();
    ASSERT_EQ(graph->build_operation_graph(_handle).code, ErrorCode::OK);

    std::vector<int64_t> rankedEngineIds;
    ASSERT_EQ(graph->get_ranked_engine_ids(rankedEngineIds).code, ErrorCode::OK);

    EXPECT_NE(std::find(rankedEngineIds.begin(), rankedEngineIds.end(), modelEngineId()),
              rankedEngineIds.end())
        << "the model-ranked engine did not offer itself for a single-node FLOAT add";
}

TEST_F(IntegrationGpuKernelIngestorKpack, TheTrainedModelPicksTheKernelItsScoreDisagreesWith)
{
    // The knob default is the top-ranked kernel, which is how the ordering is
    // observable through the public API without reaching into the engine.
    //
    // Native-ranked engine: block_size 256. Model-ranked engine: 64. Same
    // catalog. A UHD that failed to load would leave the model engine on
    // declared order -- priority, then descriptor id -- and give some order that
    // is not this one.
    auto recorder
        = hipdnn_test_sdk::utilities::IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);

    auto graph = buildPointwiseAddGraph();
    ASSERT_EQ(graph->build_operation_graph(_handle).code, ErrorCode::OK);

    const auto blockSizeDefault = [&](int64_t engineId) -> int64_t {
        std::vector<Knob> knobs;
        EXPECT_EQ(graph->get_knobs_for_engine(engineId, knobs).code, ErrorCode::OK);
        const auto knob = std::find_if(knobs.begin(), knobs.end(), [](const Knob& candidate) {
            return candidate.knobId() == BLOCK_SIZE_KNOB;
        });
        EXPECT_NE(knob, knobs.end());
        if(knob == knobs.end())
        {
            return -1;
        }
        const auto* value = std::get_if<int64_t>(&knob->defaultValue());
        EXPECT_NE(value, nullptr);
        return value != nullptr ? *value : -1;
    };

    EXPECT_EQ(blockSizeDefault(packedEngineId()), 256)
        << "hipkernel.pointwise.score returns block_size, so the wide tile must win";
    EXPECT_EQ(blockSizeDefault(modelEngineId()), 64)
        << "the trained UHD prefers the narrow tile; 256 means it did not rank at all";

    // A UHD that could not be brought up says so. Asserting the absence of that
    // diagnostic is what separates "the model chose 64" from "the model was
    // missing and declared order happened to start at 64".
    EXPECT_FALSE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "uhd:"))
        << "the heuristic reported a fault while loading. Recorded logs:\n"
        << recorder.getRecordedLogsAsString();
}

TEST_F(IntegrationGpuKernelIngestorKpack, TheModelsChoiceExecutesCorrectly)
{
    // Ranking that picks an unrunnable kernel is worse than no ranking. The
    // reference comparison is the same one every other case in this suite uses.
    auto graph = buildPointwiseAddGraph();
    graph->set_preferred_engine_id_ext(modelEngineId());

    ASSERT_EQ(graph->build_operation_graph(_handle).code, ErrorCode::OK);
    ASSERT_EQ(graph->create_execution_plans().code, ErrorCode::OK);
    ASSERT_EQ(graph->check_support().code, ErrorCode::OK);
    ASSERT_EQ(graph->build_plans().code, ErrorCode::OK);

    int64_t servingEngineId = 0;
    ASSERT_EQ(graph->get_execution_plan_engine_id(servingEngineId).code, ErrorCode::OK);
    ASSERT_EQ(servingEngineId, modelEngineId())
        << "engine id " << servingEngineId << " served the graph, not " << MODEL_ENGINE_NAME;

    int64_t workspaceSize = 0;
    ASSERT_EQ(graph->get_workspace_size(workspaceSize).code, ErrorCode::OK);
    ASSERT_GE(workspaceSize, 0);
    const hipdnn_data_sdk::utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

    executeAndVerify(*graph, workspace.get(), /*seed=*/0);
}
// The reset sweep reaches the attention packs too
//
// resetIngestorModuleCachesForTesting() is driven off the SAME ingestorPacks() table
// TestIngestorPacksModuleCacheOwnership (host-side, tests/engines/kernel_ingestor_engine/
// TestIngestorPacks.cpp) checks for internal consistency. That host-side test proves the
// table's two fields agree; it cannot prove the reset actually clears a LIVE module,
// because it links no device and calls no pointer. This tier closes that gap for the two
// attention packs specifically: PACKED_ENGINE_NAME above only ever exercised
// hipkernel:pointwise_packed, so a broken reset wire on either attention pack's entry
// would pass every existing suite, GPU and host alike.
// ---------------------------------------------------------------------------

namespace
{

/// One shipped, aligned (non-ragged), causal BF16 variant per pack -- small enough to
/// build quickly, and its `causal_mask`/BSHD shape is exactly what each matcher's
/// graph_match derives TOP_LEFT_CAUSAL from. Picked from the descriptors each pack
/// actually ships (Gfx*AttentionDenseNative.cpp's kernelMatches requires an EXACT
/// shape match against baked metadata), not invented: a shape neither pack shipped
/// would report "no engine offered this graph" here, which is a different failure
/// than the one this tier exists to catch.
struct AttentionDenseFixture
{
    std::string engineName;
    int64_t batch;
    int64_t numQueryHeads;
    int64_t numKvHeads;
    int64_t seqLen;
    int64_t headSize;
};

/// Builds a single-node SDPA graph in the BSHD layout both attention packs require
/// (Gfx*AttentionDenseNative.cpp's hasBshdStrides) and pins it to `fixture.engineName`
/// so the graph is served by that pack specifically rather than whichever engine wins
/// the ranking -- the same pinning discipline ExecutesAPackagedKernelOnDevice above
/// uses for the pointwise pack.
std::shared_ptr<Graph> buildAttentionDenseGraph(const AttentionDenseFixture& fixture)
{
    auto graph = std::make_shared<Graph>();
    graph->set_io_data_type(DataType::BFLOAT16)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    const std::vector<int64_t> qDims{
        fixture.batch, fixture.numQueryHeads, fixture.seqLen, fixture.headSize};
    const std::vector<int64_t> kvDims{
        fixture.batch, fixture.numKvHeads, fixture.seqLen, fixture.headSize};

    auto q = Graph::tensor(TensorAttributes()
                               .set_name("Q")
                               .set_dim(qDims)
                               .set_stride(generateStrides(qDims))
                               .set_data_type(DataType::BFLOAT16));
    auto k = Graph::tensor(TensorAttributes()
                               .set_name("K")
                               .set_dim(kvDims)
                               .set_stride(generateStrides(kvDims))
                               .set_data_type(DataType::BFLOAT16));
    auto v = Graph::tensor(TensorAttributes()
                               .set_name("V")
                               .set_dim(kvDims)
                               .set_stride(generateStrides(kvDims))
                               .set_data_type(DataType::BFLOAT16));

    SdpaAttributes attrs;
    attrs.set_causal_mask(true).set_attn_scale(1.0F
                                               / std::sqrt(static_cast<float>(fixture.headSize)));
    auto [o, stats] = graph->sdpa(q, k, v, attrs);
    static_cast<void>(stats);
    o->set_name("O").set_output(true).set_data_type(DataType::BFLOAT16);

    graph->set_preferred_engine_id_ext(fixture.engineName);
    return graph;
}

} // namespace

class IntegrationGpuKernelIngestorAttentionDenseResetP
    : public hip_kernel_provider::test_utilities::
          IntegrationGraphVerificationHarness<float, AttentionDenseFixture>
{
protected:
    /// Runs the pinned graph to completion, asserting the named engine actually served
    /// it -- the same attributability discipline buildAndCompilePacked() uses for the
    /// pointwise pack above, so a silently-wrong pin cannot be mistaken for a passing
    /// reset case.
    void executeOnPinnedAttentionEngine(Graph& graph, const std::string& engineName)
    {
        auto result = graph.build_operation_graph(_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        const auto engineId = hipdnn_data_sdk::utilities::engineNameToId(engineName);
        std::vector<int64_t> rankedEngineIds;
        result = graph.get_ranked_engine_ids(rankedEngineIds);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_NE(std::find(rankedEngineIds.begin(), rankedEngineIds.end(), engineId),
                  rankedEngineIds.end())
            << engineName << " did not offer itself for the pinned graph";

        result = graph.create_execution_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        result = graph.check_support();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        result = graph.build_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        int64_t servingEngineId = 0;
        ASSERT_EQ(graph.get_execution_plan_engine_id(servingEngineId).code, ErrorCode::OK);
        ASSERT_EQ(servingEngineId, engineId)
            << "engine id " << servingEngineId << " served the pinned graph, not " << engineName;

        int64_t workspaceSize = 0;
        result = graph.get_workspace_size(workspaceSize);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_GE(workspaceSize, 0);
        const hipdnn_data_sdk::utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

        executeAndVerify(graph, workspace.get(), /*seed=*/0);
    }
};

/// Proves resetIngestorModuleCachesForTesting() reaches the attention packs, not just
/// hipkernel:pointwise_packed -- the coverage gap FINDING B1 named. Runs the graph
/// once to populate the pack's module cache, resets every pack's cache (the same call
/// ...SurvivesABrokenArchive makes), then runs it again: a reset that silently missed
/// this pack's entry would be unobservable here (the resident module would just keep
/// serving), but a reset that crashed, corrupted the cache, or dropped a module a live
/// plan still needed would fail the second run. Combined with the host-side
/// TestIngestorPacksModuleCacheOwnership test, which proves the table entry for each
/// pack is wired at all, this is the device-side half: proof the wire actually works.
TEST_P(IntegrationGpuKernelIngestorAttentionDenseResetP, SurvivesAResetAfterFirstDispatch)
{
    const auto& fixture = GetParam();
    const auto arch = hip_kernel_provider_common::getDeviceString(_stream);

    const auto archContentRoot = packagedDescriptorRoot().parent_path();
    if(findKpackArchivesForArch(archContentRoot, arch).empty())
    {
        GTEST_SKIP() << "nothing was packaged for this device (" << arch
                     << "): " << archContentRoot / arch
                     << " does not exist. Environmental -- the build packs per arch and "
                        "this device is outside GPU_TARGETS.";
    }

    {
        auto graph = buildAttentionDenseGraph(fixture);
        ASSERT_NO_FATAL_FAILURE(executeOnPinnedAttentionEngine(*graph, fixture.engineName));
    }

    ASSERT_EQ(resetProviderModuleCaches(), "");

    {
        auto graph = buildAttentionDenseGraph(fixture);
        ASSERT_NO_FATAL_FAILURE(executeOnPinnedAttentionEngine(*graph, fixture.engineName));
    }
}

INSTANTIATE_TEST_SUITE_P(
    ,
    IntegrationGpuKernelIngestorAttentionDenseResetP,
    ::testing::Values(
        // hipkernel:gfx942_attention_dense.kdp.json: BF16, causal, batch=1, Sq=Skv=256,
        // Hq=8, Hkv=1, D=128, block_m=256 (256 % 256 == 0, aligned).
        AttentionDenseFixture{"hipkernel:Gfx942AttentionDense", 1, 8, 1, 256, 128},
        // hipkernel:gfx950_attention_dense.kdp.json: BF16, causal, batch=1, Sq=Skv=512,
        // Hq=32, Hkv=8, D=128, ragged=0 (512 % 256 == 0 and 512 % block_n(64) == 0).
        AttentionDenseFixture{"hipkernel:Gfx950AttentionDense", 1, 32, 8, 512, 128}),
    [](const ::testing::TestParamInfo<AttentionDenseFixture>& info) {
        return info.param.engineName == "hipkernel:Gfx942AttentionDense" ? "Gfx942" : "Gfx950";
    });

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
