// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The kernel log tap and what it puts in an outcome record. The engines write one
// `kernel: <engine> <kernel>` INFO line per built plan; the tap collects the distinct names
// between a case's `started` and `finished` records. The log facilities are a fake that
// behaves like the real ones where it matters: a plugin's line reaches a callback only when one
// is registered and the global level lets INFO through.

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

#include "BundleFixtureFiles.hpp"
#include "HarnessTestSupport.hpp"
#include "harness/bundle/CaseOutcome.hpp"
#include "harness/bundle/IntegrationBundleVerificationHarness.hpp"
#include "harness/bundle/KernelLogTap.hpp"

using namespace hipdnn_integration_tests;
using namespace hipdnn_integration_tests::bundle;
using namespace hipdnn_integration_tests::bundle::testing_support;
using hipdnn_test_sdk::utilities::claimScratchDirectory;
using hipdnn_test_sdk::utilities::ScopedDirectory;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

// A line as the user callback receives it: the backend's pattern, then the plugin's component.
std::string kernelLine(const std::string& engine, const std::string& kernel)
{
    return "[2026-10-30 10:00:00.000] [tid 7] [info] [hip_kernel_provider] kernel: " + engine + " "
           + kernel;
}

const std::string ASM_ENGINE = "ASM_SDPA_ENGINE";

/// The logging facilities, with the calls recorded in order.
struct FakeKernelLogControl : public IKernelLogControl
{
    std::vector<std::string> calls;
    std::optional<hipdnnSeverity_t> level = HIPDNN_SEV_OFF;
    bool acceptCallbacks = true;
    hipdnnUserLogCallback_t callback = nullptr;
    hipdnnUserLogCallbackHandle_t handle = nullptr;

    bool addUserCallback(hipdnnUserLogCallback_t cb, hipdnnUserLogCallbackHandle_t h) override
    {
        calls.emplace_back("add");
        if(!acceptCallbacks)
        {
            return false;
        }
        callback = cb;
        handle = h;
        return true;
    }

    void removeUserCallback(hipdnnUserLogCallback_t /*cb*/,
                            hipdnnUserLogCallbackHandle_t /*h*/) override
    {
        calls.emplace_back("remove");
        callback = nullptr;
        handle = nullptr;
    }

    std::optional<hipdnnSeverity_t> globalLevel() override
    {
        calls.emplace_back("get");
        return level;
    }

    bool setGlobalLevel(hipdnnSeverity_t newLevel) override
    {
        calls.push_back("set:" + std::to_string(static_cast<int>(newLevel)));
        level = newLevel;
        return true;
    }

    /// A plugin writing one line at `severity`: it is delivered only to a registered callback
    /// and only when the global level lets the severity through, as in the real backend.
    void log(const std::string& line, hipdnnSeverity_t severity = HIPDNN_SEV_INFO) const
    {
        if(callback == nullptr || !level.has_value() || severity < *level)
        {
            return;
        }
        callback(handle, severity, line.c_str());
    }
};

const std::string SET_INFO = "set:" + std::to_string(static_cast<int>(HIPDNN_SEV_INFO));
const std::string SET_OFF = "set:" + std::to_string(static_cast<int>(HIPDNN_SEV_OFF));

} // namespace

// ---------------------------------------------------------------------------
// What a kernel line is
// ---------------------------------------------------------------------------

TEST(TestKernelLogTap, ReadsTheKernelNameAfterThePatternPrefix)
{
    EXPECT_EQ(KernelLogTap::kernelNameOf(kernelLine(ASM_ENGINE, "fmha_fwd_d128")), "fmha_fwd_d128");
    EXPECT_EQ(KernelLogTap::kernelNameOf("kernel: ASM_SDPA_ENGINE fmha_fwd_d128"), "fmha_fwd_d128");
}

TEST(TestKernelLogTap, KeepsTheWholeNameAndDropsTheLineEnd)
{
    EXPECT_EQ(
        KernelLogTap::kernelNameOf(kernelLine("HIP_FLASH2_ENGINE", "f2_fwd_d64/w4q1k4") + "\n"),
        "f2_fwd_d64/w4q1k4");
    EXPECT_EQ(
        KernelLogTap::kernelNameOf(
            kernelLine("hipkernel:Gfx950AttentionDense", "hipkernel:name/0123-4567") + "  \r\n"),
        "hipkernel:name/0123-4567");
}

// Bug caught: the ASM_SDPA stage lines ("Using bwd dq_convert kernel: ...") and other messages
// that merely contain "kernel: " counted as the case's kernel.
TEST(TestKernelLogTap, OtherLinesAreNotKernelLines)
{
    const std::string prefix = "[2026-10-30 10:00:00.000] [tid 7] [info] [hip_kernel_provider] ";
    EXPECT_FALSE(KernelLogTap::kernelNameOf(
        prefix + "Using bwd dq_convert kernel: tocKey :: dq_convert_name"));
    EXPECT_FALSE(KernelLogTap::kernelNameOf(prefix + "Using bwd odo kernel: tocKey :: odo_name"));
    EXPECT_FALSE(KernelLogTap::kernelNameOf(prefix + "Using kernel with tocKey: abc arch: gfx942"));
    EXPECT_FALSE(KernelLogTap::kernelNameOf(prefix + "failed to load kernel: fmha"));
    EXPECT_FALSE(KernelLogTap::kernelNameOf(""));
}

// Bug caught: a line with an engine and no kernel recorded as a kernel named by the engine.
TEST(TestKernelLogTap, ALineWithoutAKernelNameIsNotAKernelLine)
{
    EXPECT_FALSE(KernelLogTap::kernelNameOf(kernelLine(ASM_ENGINE, "")));
    EXPECT_FALSE(KernelLogTap::kernelNameOf("kernel: ASM_SDPA_ENGINE"));
    EXPECT_FALSE(KernelLogTap::kernelNameOf("kernel: "));
}

// ---------------------------------------------------------------------------
// Between begin() and finish()
// ---------------------------------------------------------------------------

TEST(TestKernelLogTap, CollectsDistinctKernelsSorted)
{
    auto control = std::make_shared<FakeKernelLogControl>();
    KernelLogTap tap(control);

    tap.begin();
    control->log(kernelLine(ASM_ENGINE, "kernel_b"));
    control->log("[2026-10-30 10:00:00.000] [tid 7] [info] [harness] Validating 1 outputs");
    control->log(kernelLine(ASM_ENGINE, "kernel_a"));
    control->log(kernelLine(ASM_ENGINE, "kernel_b"));

    EXPECT_EQ(tap.finish(), (std::vector<std::string>{"kernel_a", "kernel_b"}));
}

// Bug caught: a case recording the kernels of the case before it, or of the engine's lines
// between two cases, so a kernel is credited to a case that never ran it.
TEST(TestKernelLogTap, LinesOutsideACaseAreNeverCollected)
{
    auto control = std::make_shared<FakeKernelLogControl>();
    KernelLogTap tap(control);

    control->log(kernelLine(ASM_ENGINE, "before_any_case"));
    tap.begin();
    control->log(kernelLine(ASM_ENGINE, "first_case"));
    EXPECT_EQ(tap.finish(), (std::vector<std::string>{"first_case"}));

    control->log(kernelLine(ASM_ENGINE, "between_cases"));
    tap.begin();
    control->log(kernelLine(ASM_ENGINE, "second_case"));
    EXPECT_EQ(tap.finish(), (std::vector<std::string>{"second_case"}));
}

// Bug caught: a case that never reached its `finished` record (the process kept going) leaving
// its lines in the next case's record.
TEST(TestKernelLogTap, ABeginOnAnActiveTapDropsTheEarlierCase)
{
    auto control = std::make_shared<FakeKernelLogControl>();
    KernelLogTap tap(control);

    tap.begin();
    control->log(kernelLine(ASM_ENGINE, "abandoned"));
    tap.begin();
    control->log(kernelLine(ASM_ENGINE, "next_case"));

    EXPECT_EQ(tap.finish(), (std::vector<std::string>{"next_case"}));
    EXPECT_EQ(control->callback, nullptr);
}

TEST(TestKernelLogTap, FinishWithoutABeginReportsNothing)
{
    auto control = std::make_shared<FakeKernelLogControl>();
    KernelLogTap tap(control);

    EXPECT_TRUE(tap.finish().empty());
    EXPECT_TRUE(control->calls.empty());
}

TEST(TestKernelLogTap, ANullControlMakesTheTapInert)
{
    KernelLogTap tap(nullptr);

    tap.begin();
    EXPECT_TRUE(tap.finish().empty());
}

// ---------------------------------------------------------------------------
// The log level
// ---------------------------------------------------------------------------

// Bug caught: no kernel ever reaches the record because the level the plugins log at is off
// (the default). Plugins log only at the level the global setter pushes into them.
TEST(TestKernelLogTap, RaisesTheLevelToInfoSoThePluginsLog)
{
    auto control = std::make_shared<FakeKernelLogControl>();
    control->level = HIPDNN_SEV_OFF;
    KernelLogTap tap(control);

    control->log(kernelLine(ASM_ENGINE, "while_off"));
    tap.begin();
    EXPECT_EQ(control->level, HIPDNN_SEV_INFO);
    control->log(kernelLine(ASM_ENGINE, "while_info"));

    EXPECT_EQ(tap.finish(), (std::vector<std::string>{"while_info"}));
}

// Bug caught: registering after the level went up, which makes the backend's first
// registration create a stderr console sink for a level that was off, and the run's stderr fill
// with every INFO line. The callback goes in first.
TEST(TestKernelLogTap, RegistersTheCallbackBeforeItRaisesTheLevel)
{
    auto control = std::make_shared<FakeKernelLogControl>();
    KernelLogTap tap(control);

    tap.begin();
    EXPECT_EQ(control->calls, (std::vector<std::string>{"get", "add", SET_INFO}));
    static_cast<void>(tap.finish());
}

// Bug caught: a run left logging at INFO after the first case, which turns every later case
// into an INFO flood (and the benchmark into a logging benchmark).
TEST(TestKernelLogTap, PutsTheEarlierLevelBackAndUnregisters)
{
    auto control = std::make_shared<FakeKernelLogControl>();
    control->level = HIPDNN_SEV_OFF;
    KernelLogTap tap(control);

    tap.begin();
    static_cast<void>(tap.finish());

    EXPECT_EQ(control->level, HIPDNN_SEV_OFF);
    EXPECT_EQ(control->callback, nullptr);
    EXPECT_EQ(control->calls,
              (std::vector<std::string>{"get", "add", SET_INFO, SET_OFF, "remove"}));
}

// Bug caught: a level the run set with HIPDNN_LOG_LEVEL replaced by off after the first case.
// The level put back is the one that was found, whatever it was.
TEST(TestKernelLogTap, RestoresTheExactLevelItFound)
{
    auto control = std::make_shared<FakeKernelLogControl>();
    control->level = HIPDNN_SEV_WARN;
    KernelLogTap tap(control);

    tap.begin();
    control->log(kernelLine(ASM_ENGINE, "kernel_a"));
    EXPECT_EQ(tap.finish(), (std::vector<std::string>{"kernel_a"}));

    EXPECT_EQ(control->level, HIPDNN_SEV_WARN);
}

// Bug caught: a run already at INFO being set to INFO and then to something less verbose.
TEST(TestKernelLogTap, AnInfoLevelAlreadySetIsLeftAlone)
{
    auto control = std::make_shared<FakeKernelLogControl>();
    control->level = HIPDNN_SEV_INFO;
    KernelLogTap tap(control);

    tap.begin();
    control->log(kernelLine(ASM_ENGINE, "kernel_a"));
    EXPECT_EQ(tap.finish(), (std::vector<std::string>{"kernel_a"}));

    EXPECT_EQ(control->calls, (std::vector<std::string>{"get", "add", "remove"}));
    EXPECT_EQ(control->level, HIPDNN_SEV_INFO);
}

// Bug caught: setting a level that cannot be put back, which would leave the run at INFO.
TEST(TestKernelLogTap, AnUnreadableLevelIsNeverChanged)
{
    auto control = std::make_shared<FakeKernelLogControl>();
    control->level = std::nullopt;
    KernelLogTap tap(control);

    tap.begin();
    static_cast<void>(tap.finish());

    EXPECT_EQ(control->calls, (std::vector<std::string>{"get", "add", "remove"}));
}

// Bug caught: a refused registration still raising the level, with no callback to read it.
TEST(TestKernelLogTap, ARefusedRegistrationChangesNothing)
{
    auto control = std::make_shared<FakeKernelLogControl>();
    control->acceptCallbacks = false;
    KernelLogTap tap(control);

    tap.begin();
    EXPECT_TRUE(tap.finish().empty());

    EXPECT_EQ(control->calls, (std::vector<std::string>{"get", "add"}));
    EXPECT_EQ(control->level, HIPDNN_SEV_OFF);
}

// Bug caught: a tap that goes away mid-case leaving a callback that points at freed memory and
// the run at INFO.
TEST(TestKernelLogTap, ADestroyedTapLeavesNoCallbackAndTheEarlierLevel)
{
    auto control = std::make_shared<FakeKernelLogControl>();
    control->level = HIPDNN_SEV_ERROR;
    {
        KernelLogTap tap(control);
        tap.begin();
        EXPECT_NE(control->callback, nullptr);
    }

    EXPECT_EQ(control->callback, nullptr);
    EXPECT_EQ(control->level, HIPDNN_SEV_ERROR);
}

// ---------------------------------------------------------------------------
// Through the harness: the record a case publishes
// ---------------------------------------------------------------------------

namespace
{

class TestKernelLogHarness : public ::testing::Test
{
protected:
    std::optional<ScopedDirectory> _scopedDir;
    std::filesystem::path _tempDir;
    HarnessMocks _mocks;
    std::shared_ptr<FakeKernelLogControl> _control = std::make_shared<FakeKernelLogControl>();
    std::vector<CaseOutcomeRecord> _finished;
    // What the engine logs while it executes the next case, and the level it saw then.
    std::vector<std::string> _engineLines;
    std::optional<hipdnnSeverity_t> _levelWhileExecuting;

    void SetUp() override
    {
        ensureTestConfigInitialized();
        _scopedDir.emplace(claimScratchDirectory("kernel_log_tap"));
        _tempDir = _scopedDir->path();

        using ::testing::_;
        ON_CALL(_mocks.reporter, recordCaseOutcome(_))
            .WillByDefault(
                [this](const CaseOutcomeRecord& record) { _finished.push_back(record); });
        ON_CALL(_mocks.engineRunner, execute(_, _, _))
            .WillByDefault([this](GraphSession&,
                                  const std::optional<LoadedEngine>&,
                                  VariantPack& variantPack) {
                _levelWhileExecuting = _control->level;
                for(const auto& line : _engineLines)
                {
                    _control->log(line);
                }
                fixtures::writeOutput(variantPack, fixtures::K_OUTPUT_VALUE);
                return EngineOpResult::succeeded();
            });
    }

    HarnessDependencies dependencies()
    {
        auto deps = _mocks.dependencies(hostPolicy());
        deps.kernelLog = nonOwning<IKernelLogControl>(*_control);
        return deps;
    }

    std::shared_ptr<IntegrationTestBundle> bundle() const
    {
        return fixtures::loadBundle(_tempDir, "Bundle", true);
    }
};

} // namespace

TEST_F(TestKernelLogHarness, TheRecordNamesTheKernelsTheCaseRan)
{
    _engineLines = {kernelLine(ASM_ENGINE, "kernel_b"), kernelLine(ASM_ENGINE, "kernel_a")};
    IntegrationBundleVerificationHarness harness(dependencies(), LoadedEngine{11, ASM_ENGINE});
    ::testing::TestPartResultArray results;
    harness.setBundle(bundle(), "test/bundle");
    driveHarness(harness, &results);

    ASSERT_EQ(_finished.size(), 1u);
    EXPECT_EQ(_finished.front().outcome, CaseOutcome::PASS);
    EXPECT_EQ(_finished.front().kernels, (std::vector<std::string>{"kernel_a", "kernel_b"}));
    const auto json = nlohmann::json::parse(finishedRecordJson(_finished.front()));
    EXPECT_EQ(json.at("kernels"), (nlohmann::json::array({"kernel_a", "kernel_b"})));
    // The engine ran at INFO, and the run is back at its own level after the record.
    EXPECT_EQ(_levelWhileExecuting, HIPDNN_SEV_INFO);
    EXPECT_EQ(_control->level, HIPDNN_SEV_OFF);
    EXPECT_EQ(_control->callback, nullptr);
}

// Bug caught: kernels of the first case in the record of the second, or a case that never
// reached the engine carrying a kernel.
TEST_F(TestKernelLogHarness, EachCaseGetsOnlyItsOwnKernels)
{
    IntegrationBundleVerificationHarness harness(dependencies(), LoadedEngine{11, ASM_ENGINE});
    harness.setBundle(bundle(), "test/bundle");

    _engineLines = {kernelLine(ASM_ENGINE, "first_case_kernel")};
    ::testing::TestPartResultArray first;
    driveHarness(harness, &first);

    _engineLines = {kernelLine(ASM_ENGINE, "second_case_kernel")};
    ::testing::TestPartResultArray second;
    driveHarness(harness, &second);

    _engineLines.clear();
    ::testing::TestPartResultArray third;
    driveHarness(harness, &third);

    ASSERT_EQ(_finished.size(), 3u);
    EXPECT_EQ(_finished[0].kernels, (std::vector<std::string>{"first_case_kernel"}));
    EXPECT_EQ(_finished[1].kernels, (std::vector<std::string>{"second_case_kernel"}));
    EXPECT_TRUE(_finished[2].kernels.empty());
    EXPECT_EQ(_control->level, HIPDNN_SEV_OFF);
}

TEST_F(TestKernelLogHarness, ACaseTheEngineDeclinesHasNoKernels)
{
    using ::testing::_;
    ON_CALL(_mocks.engineRunner, openGraph(_, _))
        .WillByDefault([](const IntegrationTestBundle&, const std::optional<LoadedEngine>&) {
            return declinedSession();
        });
    _engineLines = {kernelLine(ASM_ENGINE, "never_executed")};
    IntegrationBundleVerificationHarness harness(dependencies(), LoadedEngine{11, ASM_ENGINE});
    ::testing::TestPartResultArray results;
    harness.setBundle(bundle(), "test/bundle");
    driveHarness(harness, &results);

    ASSERT_EQ(_finished.size(), 1u);
    EXPECT_EQ(_finished.front().outcome, CaseOutcome::DECLINED);
    EXPECT_TRUE(_finished.front().kernels.empty());
    EXPECT_FALSE(nlohmann::json::parse(finishedRecordJson(_finished.front())).contains("kernels"));
    EXPECT_EQ(_control->level, HIPDNN_SEV_OFF);
    EXPECT_EQ(_control->callback, nullptr);
}

// Bug caught: a harness with no log control (every unit suite) registering a callback or
// changing the level.
TEST_F(TestKernelLogHarness, WithoutALogControlTheRecordHasNoKernels)
{
    _engineLines = {kernelLine(ASM_ENGINE, "kernel_a")};
    auto deps = dependencies();
    deps.kernelLog = nullptr;
    IntegrationBundleVerificationHarness harness(std::move(deps), LoadedEngine{11, ASM_ENGINE});
    ::testing::TestPartResultArray results;
    harness.setBundle(bundle(), "test/bundle");
    driveHarness(harness, &results);

    ASSERT_EQ(_finished.size(), 1u);
    EXPECT_TRUE(_finished.front().kernels.empty());
    EXPECT_TRUE(_control->calls.empty());
}

// NOLINTEND(readability-identifier-naming)
