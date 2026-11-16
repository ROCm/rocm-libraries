// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "compilation/CompileCounters.hpp"
#include "compilation/ProgramFetchRecording.hpp"

#include <thread>

using namespace hip_kernel_provider::compilation;

TEST(TestPlanBuildSummary, NamesTheEngineConfigAndEachProgramsSource)
{
    const std::vector<ProgramFetch> fetches
        = {{"kernels/ConvIgemm.cpp", CodeSource::COMPILED, 812.44},
           {"kernels/ConvCopy.cpp", CodeSource::DISK_CACHE, 1.06},
           {"kernels/ConvReduce.cpp", CodeSource::IN_PROCESS, 0.01},
           {"kernels/ConvSplit.cpp", CodeSource::COMPILED, 100.0},
           {"kernels/ConvStored.cpp", CodeSource::PRECOMPILED_STORE, 0.5}};
    EXPECT_EQ(
        formatPlanBuildSummary("hipkernel:CONV_IGEMM_FWD", "igemm.f16_128x128", false, fetches),
        "plan build: engine 'hipkernel:CONV_IGEMM_FWD', config 'igemm.f16_128x128', "
        "5 program(s): kernels/ConvIgemm.cpp compiled 812.4 ms, kernels/ConvCopy.cpp "
        "disk-cache 1.1 ms, kernels/ConvReduce.cpp in-process 0.0 ms, kernels/ConvSplit.cpp "
        "compiled 100.0 ms, kernels/ConvStored.cpp store 0.5 ms; compile 912.4 ms");
}

TEST(TestPlanBuildSummary, SaysWhenAutotuneChoosesOrTheEngineHasNoConfigName)
{
    EXPECT_EQ(formatPlanBuildSummary("hipkernel:CONV_WRW", "", true, {}),
              "plan build: engine 'hipkernel:CONV_WRW', config chosen by autotune at first "
              "execution, 0 program(s); compile 0.0 ms");
    EXPECT_EQ(formatPlanBuildSummary(
                  "HIP_MLOPS_ENGINE", "", false, {{"batchnorm.cpp", CodeSource::DISK_CACHE, 2.0}}),
              "plan build: engine 'HIP_MLOPS_ENGINE', config -, 1 program(s): batchnorm.cpp "
              "disk-cache 2.0 ms; compile 0.0 ms");
}

TEST(TestProgramFetchRecording, CollectsOnItsOwnThreadAndNests)
{
    const auto before = ProgramFetchRecording::recordedOnThisThread();
    // Outside any recording a fetch is counted but kept nowhere.
    ProgramFetchRecording::record({"a.cpp", CodeSource::COMPILED, 1.0});
    EXPECT_EQ(ProgramFetchRecording::recordedOnThisThread(), before + 1);

    const ProgramFetchRecording outer;
    ProgramFetchRecording::record({"b.cpp", CodeSource::DISK_CACHE, 1.0});
    {
        const ProgramFetchRecording inner;
        ProgramFetchRecording::record({"c.cpp", CodeSource::COMPILED, 1.0});
        // Another thread's fetches never land here.
        std::thread([] {
            ProgramFetchRecording::record({"d.cpp", CodeSource::COMPILED, 1.0});
        }).join();
        ASSERT_EQ(inner.fetches().size(), 1u);
        EXPECT_EQ(inner.fetches()[0].program, "c.cpp");
    }
    ProgramFetchRecording::record({"e.cpp", CodeSource::IN_PROCESS, 1.0});
    ASSERT_EQ(outer.fetches().size(), 2u);
    EXPECT_EQ(outer.fetches()[0].program, "b.cpp");
    EXPECT_EQ(outer.fetches()[1].program, "e.cpp");
    EXPECT_EQ(ProgramFetchRecording::recordedOnThisThread(), before + 4);
}

TEST(TestCompileCounters, CountEachSourceAndFormatAsOneLine)
{
    const auto before = compileCounters();
    ProgramFetchRecording::record({"a.cpp", CodeSource::COMPILED, 1500.0});
    ProgramFetchRecording::record({"b.cpp", CodeSource::DISK_CACHE, 1.0});
    ProgramFetchRecording::record({"c.cpp", CodeSource::IN_PROCESS, 0.0});
    ProgramFetchRecording::record({"c.cpp", CodeSource::IN_PROCESS, 0.0});
    countCompileFailure();
    countCorruptEntry();
    const auto after = compileCounters();
    EXPECT_EQ(after.compiles - before.compiles, 1u);
    EXPECT_NEAR(after.compileSeconds - before.compileSeconds, 1.5, 1e-6);
    EXPECT_EQ(after.diskHits - before.diskHits, 1u);
    EXPECT_EQ(after.inProcessHits - before.inProcessHits, 2u);
    EXPECT_EQ(after.compileFailures - before.compileFailures, 1u);
    EXPECT_EQ(after.corruptEntries - before.corruptEntries, 1u);
    EXPECT_EQ(after.storeHits, before.storeHits);

    CompileCounters counters;
    counters.compiles = 3;
    counters.compileSeconds = 2.414;
    counters.compileFailures = 1;
    counters.diskHits = 12;
    counters.corruptEntries = 2;
    counters.inProcessHits = 40;
    EXPECT_EQ(formatCompileCounters(counters),
              "hipRTC: 3 compiled (2.41 s), 1 failed, 12 disk-cache hits, 2 corrupt entries "
              "replaced, 40 in-process hits, 0 store hits");
}
