// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <miopen/kernel_tuning_mode.hpp>
#include <miopen/env.hpp>

#include <string>
#include <vector>

namespace {

class PerfLogEnv
{
public:
    explicit PerfLogEnv(unsigned long long level)
    {
        miopen::env::update(MIOPEN_PERFORMANCE_LOGS, level);
    }
    PerfLogEnv(const PerfLogEnv&)            = delete;
    PerfLogEnv& operator=(const PerfLogEnv&) = delete;
    ~PerfLogEnv() { miopen::env::clear(MIOPEN_PERFORMANCE_LOGS); }
};

void ResetAccumulator()
{
    miopen::GetJsonAccumulator().Clear();
    miopen::GetLastPrintedSolutionName().clear();
    miopen::GetLastPrintedSolverId() = 0;
    miopen::SetKernelPhase(miopen::KernelPhase::Unknown);
}

class CPU_KernelTuningModeLogs_NONE : public ::testing::Test
{
protected:
    void SetUp() override { ResetAccumulator(); }
    void TearDown() override
    {
        ResetAccumulator();
        miopen::env::clear(MIOPEN_PERFORMANCE_LOGS);
    }
};

} // namespace

// ---------------------------------------------------------------------------
// Pure helper tests
// ---------------------------------------------------------------------------

TEST_F(CPU_KernelTuningModeLogs_NONE, PhaseToStringMapping)
{
    using miopen::KernelPhase;
    EXPECT_STREQ(miopen::KernelPhaseToString(KernelPhase::Unknown), "unknown");
    EXPECT_STREQ(miopen::KernelPhaseToString(KernelPhase::Execution), "execution");
    EXPECT_STREQ(miopen::KernelPhaseToString(KernelPhase::Validation), "validation");
    EXPECT_STREQ(miopen::KernelPhaseToString(KernelPhase::SolverTuning), "solver_tuning");
    EXPECT_STREQ(miopen::KernelPhaseToString(KernelPhase::Tuning), "tuning");
}

TEST_F(CPU_KernelTuningModeLogs_NONE, ScopedKernelPhaseRestoresPrevious)
{
    using miopen::KernelPhase;
    miopen::SetKernelPhase(KernelPhase::Validation);
    {
        miopen::ScopedKernelPhase scoped(KernelPhase::Execution);
        EXPECT_EQ(miopen::GetKernelPhase(), KernelPhase::Execution);
        {
            miopen::ScopedKernelPhase nested(KernelPhase::SolverTuning);
            EXPECT_EQ(miopen::GetKernelPhase(), KernelPhase::SolverTuning);
        }
        EXPECT_EQ(miopen::GetKernelPhase(), KernelPhase::Execution);
    }
    EXPECT_EQ(miopen::GetKernelPhase(), KernelPhase::Validation);
}

TEST_F(CPU_KernelTuningModeLogs_NONE, IsTransposeOrTransformKernelDetectsPatterns)
{
    using miopen::IsTransposeOrTransformKernel;
    EXPECT_TRUE(IsTransposeOrTransformKernel("transposeNCHW"));
    EXPECT_TRUE(IsTransposeOrTransformKernel("SubTensorOpWithTransform"));
    EXPECT_TRUE(IsTransposeOrTransformKernel("Im2d2ColKernel"));
    EXPECT_TRUE(IsTransposeOrTransformKernel("Col2Im3d_v2"));
    EXPECT_TRUE(IsTransposeOrTransformKernel("NCHW2CNHW_pack"));
    EXPECT_TRUE(IsTransposeOrTransformKernel("MN2NM_pad"));
    EXPECT_FALSE(IsTransposeOrTransformKernel("gemm_kernel"));
    EXPECT_FALSE(IsTransposeOrTransformKernel(""));
    EXPECT_FALSE(IsTransposeOrTransformKernel("MIOpenConvDirUni"));
}

TEST_F(CPU_KernelTuningModeLogs_NONE, JsonEscapeHandlesSpecialChars)
{
    EXPECT_EQ(miopen::JsonEscape("plain"), "plain");
    EXPECT_EQ(miopen::JsonEscape("a\"b"), "a\\\"b");
    EXPECT_EQ(miopen::JsonEscape("a\\b"), "a\\\\b");
    EXPECT_EQ(miopen::JsonEscape("line\n"), "line\\n");
    EXPECT_EQ(miopen::JsonEscape("tab\tend"), "tab\\tend");
    // Non-printable ASCII gets escaped as \\u00XX
    EXPECT_EQ(miopen::JsonEscape(std::string(1, '\x01')), "\\u0001");
}

// ---------------------------------------------------------------------------
// Env-var gating tests for IsLoggingKernel / IsPerformanceLoggingEnabled
// ---------------------------------------------------------------------------

TEST_F(CPU_KernelTuningModeLogs_NONE, LoggingDisabledWhenEnvUnset)
{
    miopen::env::clear(MIOPEN_PERFORMANCE_LOGS);
    miopen::SetKernelPhase(miopen::KernelPhase::Execution);
    EXPECT_FALSE(miopen::IsLoggingKernel());
    EXPECT_FALSE(miopen::IsPerformanceLoggingEnabled());
}

TEST_F(CPU_KernelTuningModeLogs_NONE, LoggingDisabledAtLevelZero)
{
    PerfLogEnv guard{0};
    miopen::SetKernelPhase(miopen::KernelPhase::Execution);
    EXPECT_FALSE(miopen::IsLoggingKernel());
    EXPECT_FALSE(miopen::IsPerformanceLoggingEnabled());
}

TEST_F(CPU_KernelTuningModeLogs_NONE, LowLevelLogsOnlyExecutionPhase)
{
    PerfLogEnv guard{1};
    EXPECT_TRUE(miopen::IsPerformanceLoggingEnabled());

    miopen::SetKernelPhase(miopen::KernelPhase::Execution);
    EXPECT_TRUE(miopen::IsLoggingKernel());

    for(auto p : {miopen::KernelPhase::Unknown,
                  miopen::KernelPhase::Validation,
                  miopen::KernelPhase::SolverTuning,
                  miopen::KernelPhase::Tuning})
    {
        miopen::SetKernelPhase(p);
        EXPECT_FALSE(miopen::IsLoggingKernel())
            << "phase=" << miopen::KernelPhaseToString(p) << " should not log at level 1";
    }
}

TEST_F(CPU_KernelTuningModeLogs_NONE, LevelTwoStillExecutionOnly)
{
    PerfLogEnv guard{2};
    miopen::SetKernelPhase(miopen::KernelPhase::Execution);
    EXPECT_TRUE(miopen::IsLoggingKernel());
    miopen::SetKernelPhase(miopen::KernelPhase::Tuning);
    EXPECT_FALSE(miopen::IsLoggingKernel());
}

TEST_F(CPU_KernelTuningModeLogs_NONE, HighLevelLogsAllPhases)
{
    PerfLogEnv guard{3};
    for(auto p : {miopen::KernelPhase::Unknown,
                  miopen::KernelPhase::Execution,
                  miopen::KernelPhase::Validation,
                  miopen::KernelPhase::SolverTuning,
                  miopen::KernelPhase::Tuning})
    {
        miopen::SetKernelPhase(p);
        EXPECT_TRUE(miopen::IsLoggingKernel())
            << "phase=" << miopen::KernelPhaseToString(p) << " should log at level 3";
    }
}

// ---------------------------------------------------------------------------
// Accumulator behavior — no output when disabled, JSON to stderr when enabled
// ---------------------------------------------------------------------------

TEST_F(CPU_KernelTuningModeLogs_NONE, AccumulatorNoOpWhenLoggingDisabled)
{
    miopen::env::clear(MIOPEN_PERFORMANCE_LOGS);
    miopen::SetKernelPhase(miopen::KernelPhase::Execution);

    testing::internal::CaptureStderr();
    miopen::LogSolutionName("solver_x", 42, 1024);
    miopen::AddPerformanceConfig("kern_x", "desc");
    miopen::AddInvokerTimes({1.0f, 2.0f});
    miopen::AddKernelToJsonAccumulator("kern_x", 1.5f, false);
    miopen::FlushJsonAccumulator();
    const std::string captured = testing::internal::GetCapturedStderr();

    EXPECT_TRUE(captured.empty()) << "captured: " << captured;

    auto& data = miopen::GetJsonAccumulator();
    EXPECT_TRUE(data.solution_name.empty());
    EXPECT_TRUE(data.performance_configs.empty());
    EXPECT_TRUE(data.current_config.config_name.empty());
}

TEST_F(CPU_KernelTuningModeLogs_NONE, AddInvokerTimesWarnsWithoutConfig)
{
    PerfLogEnv guard{1};
    miopen::SetKernelPhase(miopen::KernelPhase::Execution);

    testing::internal::CaptureStderr();
    miopen::AddInvokerTimes({1.0f});
    const std::string captured = testing::internal::GetCapturedStderr();
    EXPECT_NE(captured.find("WARNING"), std::string::npos);
    EXPECT_NE(captured.find("AddInvokerTimes"), std::string::npos);
}

TEST_F(CPU_KernelTuningModeLogs_NONE, FlushEmitsJsonAtLevelOne)
{
    PerfLogEnv guard{1};
    miopen::SetKernelPhase(miopen::KernelPhase::Execution);

    testing::internal::CaptureStderr();
    miopen::LogSolutionName("ConvSolverTest", 1234, 2048);
    miopen::AddPerformanceConfig("MyKernel", "BS=128");
    miopen::AddInvokerTimes({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    miopen::FlushJsonAccumulator();
    const std::string captured = testing::internal::GetCapturedStderr();

    EXPECT_NE(captured.find("\"solution\":\"ConvSolverTest\""), std::string::npos);
    EXPECT_NE(captured.find("\"solver_id\":1234"), std::string::npos);
    EXPECT_NE(captured.find("\"workspace_bytes\":2048"), std::string::npos);
    EXPECT_NE(captured.find("\"phase\":\"execution\""), std::string::npos);
    EXPECT_NE(captured.find("\"config_descriptor\":\"BS=128\""), std::string::npos);
    EXPECT_NE(captured.find("\"performance_configs\":["), std::string::npos);
    // At level 1 the kernels array is null
    EXPECT_NE(captured.find("\"kernels\":null"), std::string::npos);
}

TEST_F(CPU_KernelTuningModeLogs_NONE, FlushIncludesKernelsArrayAtLevelTwo)
{
    PerfLogEnv guard{2};
    miopen::SetKernelPhase(miopen::KernelPhase::Execution);

    testing::internal::CaptureStderr();
    miopen::LogSolutionName("SolverY", 7, 0);
    miopen::AddPerformanceConfig("kern_y", "");
    miopen::AddKernelToJsonAccumulator("kern_y", 0.5f, false);
    miopen::AddKernelToJsonAccumulator("Im2ColHelper", 0.1f, true);
    miopen::FlushJsonAccumulator();
    const std::string captured = testing::internal::GetCapturedStderr();

    EXPECT_NE(captured.find("\"kernels\":["), std::string::npos);
    EXPECT_NE(captured.find("\"kernel_name\":\"kern_y\""), std::string::npos);
    EXPECT_NE(captured.find("\"kernel_name\":\"Im2ColHelper\""), std::string::npos);
    EXPECT_NE(captured.find("\"is_transformation\":true"), std::string::npos);
    EXPECT_NE(captured.find("\"is_transformation\":false"), std::string::npos);
    EXPECT_NE(captured.find("\"number_of_transformations\":1"), std::string::npos);
}

TEST_F(CPU_KernelTuningModeLogs_NONE, LogSolutionNameOnlyEmitsOnChange)
{
    PerfLogEnv guard{1};
    miopen::SetKernelPhase(miopen::KernelPhase::Execution);

    // First call: just records the solution, nothing flushed yet
    testing::internal::CaptureStderr();
    miopen::LogSolutionName("SolverA", 1, 0);
    miopen::AddPerformanceConfig("kA", "");
    miopen::AddInvokerTimes({1.0f});
    // Re-logging the same solution should not flush
    miopen::LogSolutionName("SolverA", 1, 0);
    std::string captured = testing::internal::GetCapturedStderr();
    EXPECT_TRUE(captured.empty()) << "captured: " << captured;

    // Logging a different solution flushes the previous one
    testing::internal::CaptureStderr();
    miopen::LogSolutionName("SolverB", 2, 0);
    captured = testing::internal::GetCapturedStderr();
    EXPECT_NE(captured.find("\"solution\":\"SolverA\""), std::string::npos);
    EXPECT_EQ(captured.find("\"solution\":\"SolverB\""), std::string::npos);

    // Final flush emits SolverB
    miopen::AddPerformanceConfig("kB", "");
    miopen::AddInvokerTimes({1.0f});
    testing::internal::CaptureStderr();
    miopen::FlushJsonAccumulator();
    captured = testing::internal::GetCapturedStderr();
    EXPECT_NE(captured.find("\"solution\":\"SolverB\""), std::string::npos);
}

TEST_F(CPU_KernelTuningModeLogs_NONE, FlushWithoutDataEmitsNothing)
{
    PerfLogEnv guard{2};
    miopen::SetKernelPhase(miopen::KernelPhase::Execution);

    testing::internal::CaptureStderr();
    miopen::FlushJsonAccumulator();
    EXPECT_TRUE(testing::internal::GetCapturedStderr().empty());
}

TEST_F(CPU_KernelTuningModeLogs_NONE, NonExecutionPhaseSuppressedAtLowLevel)
{
    PerfLogEnv guard{1};
    miopen::SetKernelPhase(miopen::KernelPhase::Tuning);

    testing::internal::CaptureStderr();
    miopen::LogSolutionName("ShouldNotAppear", 99, 0);
    miopen::AddPerformanceConfig("k", "");
    miopen::AddInvokerTimes({1.0f});
    miopen::AddKernelToJsonAccumulator("k", 1.0f, false);
    miopen::FlushJsonAccumulator();
    EXPECT_TRUE(testing::internal::GetCapturedStderr().empty());
}

TEST_F(CPU_KernelTuningModeLogs_NONE, HighLevelEmitsTuningPhase)
{
    PerfLogEnv guard{4};
    miopen::SetKernelPhase(miopen::KernelPhase::Tuning);

    testing::internal::CaptureStderr();
    miopen::LogSolutionName("TuningSolver", 5, 0);
    miopen::AddPerformanceConfig("k", "cfg");
    miopen::AddKernelToJsonAccumulator("k", 1.0f, false);
    miopen::FlushJsonAccumulator();
    const std::string captured = testing::internal::GetCapturedStderr();

    EXPECT_NE(captured.find("\"phase\":\"tuning\""), std::string::npos);
    // Level 4 shows the kernels array
    EXPECT_NE(captured.find("\"kernels\":["), std::string::npos);
}
