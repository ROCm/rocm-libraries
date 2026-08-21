// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Logging.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>

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

/// The engine that ships with the provider. Its kernels are `embedded_source`, so it is
/// reachable whatever state the packaged archive is in, and it claims the same single-node
/// FLOAT add. That makes it the fallback ...SurvivesABrokenArchive requires to still serve.
constexpr const char* SHIPPED_POINTWISE_ENGINE_NAME = "hipkernel:Pointwise";

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

        _archives = findKpackArchives(packagedDescriptorRoot());
        if(_archives.empty())
        {
            GTEST_SKIP() << "packaged artifact absent -- packaging did not run. Looked for "
                            "*.kpack under "
                         << packagedDescriptorRoot()
                         << ". Configure with -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON and a hipcc "
                            "the packaging rule can find.";
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

    /// Executes on device and compares against CpuReferenceGraphExecutor. Routing alone
    /// would pass on a kernel that ran the wrong archive member, or swapped its operands.
    void executeAndVerify(Graph& graph, void* workspace, unsigned int seed)
    {
        GraphTensorBundle gpuBundle;
        GraphTensorBundle cpuBundle;
        graph.visit([&](const INode& node) {
            for(const auto& tensorAttr : node.getNodeOutputTensorAttributes())
            {
                gpuBundle.addTensor(*tensorAttr, createTensorFromAttribute(*tensorAttr));
                cpuBundle.addTensor(*tensorAttr, createTensorFromAttribute(*tensorAttr));
            }
            for(const auto& tensorAttr : node.getNodeInputTensorAttributes())
            {
                if(gpuBundle.tensors.find(tensorAttr->get_uid()) == gpuBundle.tensors.end())
                {
                    gpuBundle.addTensor(*tensorAttr, createTensorFromAttribute(*tensorAttr));
                    cpuBundle.addTensor(*tensorAttr, createTensorFromAttribute(*tensorAttr));
                }
            }
        });

        for(auto& [uid, tensor] : gpuBundle.tensors)
        {
            // Per-uid offset so the operands are never byte-identical: a+a and a+b are
            // indistinguishable to allClose() when both operands hold the same bytes.
            const auto tensorSeed = seed + static_cast<unsigned int>(uid);
            gpuBundle.randomizeTensor(uid, -4.0f, 4.0f, tensorSeed);
            cpuBundle.randomizeTensor(uid, -4.0f, 4.0f, tensorSeed);
        }

        auto deviceVariantPack = gpuBundle.toDeviceVariantPack();
        auto result = graph.execute(_handle, deviceVariantPack, workspace);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);

        auto [serializedGraph, serErr] = graph.to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
        CpuReferenceGraphExecutor().execute(
            serializedGraph.data(), serializedGraph.size(), cpuBundle.toHostVariantPack());

        auto& gpuOut = gpuBundle.getTensor(3);
        auto& cpuOut = cpuBundle.getTensor(3);
        gpuOut.markDeviceModified();
        // A one-term pointwise add is bit-exact on both sides.
        const auto tolerance = std::numeric_limits<float>::epsilon();
        EXPECT_TRUE(CpuFpReferenceValidation<float>(tolerance, tolerance).allClose(cpuOut, gpuOut));
    }

    std::vector<std::filesystem::path> _archives;
};

// ---------------------------------------------------------------------------
// The artifact fails without taking the process with it
//
// First in the file, and that is load-bearing. Gtest runs suites in order of first
// registration, which within one translation unit is definition order, so this suite runs
// before IntegrationGpuKernelIngestorKpack below. It has to: the module cache is
// process-lifetime, so once the packaged kernel has been executed once, a resident module
// serves the plan and the corrupt bytes on disk are read by nothing. Nothing in this repo
// or in the generated build passes --gtest_shuffle, which is what makes definition order a
// sound guarantee rather than a coincidence.
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

        _victim = _archives.front();

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

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
